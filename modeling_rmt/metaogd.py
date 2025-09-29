import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import transformers
from copy import copy

from accelerate.logging import get_logger
logger = get_logger('')
    

class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, init_inner_lr=1.0, learn_lr=False, inner_steps=10, **kwargs):
        super().__init__()
        self.model = base_model
        self.create_memory(num_mem_tokens)

        self.inner_steps = inner_steps
        self.inner_clip_value = kwargs.get('inner_clip_value', None)
        self.inner_clip_norm = kwargs.get('inner_clip_norm', None)
        # meta-learnable parameters (log-space for power scaling)
        self.log_inner_lr = torch.log(torch.tensor(init_inner_lr))
        if learn_lr:
            self.log_inner_lr = nn.Parameter(self.log_inner_lr)

        self.r_max = 16  # tiny basis rank (8–16 is plenty)
        # OGD buffers (created lazily for the current B,T,D)
        self.ogd_Q = self.ogd_r_used = None

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        self.memory_dim = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, self.memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def _sgd_step(self, p, g, clip_value=None, clip_norm=None):
        g = g.reshape_as(p)

        if clip_value is not None:
            # simple element-wise clamp
            g = torch.clamp(g, -clip_value, clip_value)

        if clip_norm is not None:
            # scale gradient if its 2-norm is too large
            # check grad for each sample separately as we do per-sample optimization
            g_norm = g.norm(dim=[1, 2], keepdim=True)                 # (B,1,1)
            scale = clip_norm / (g_norm + 1e-6)
            g = torch.where(g_norm > clip_norm, g * scale, g)

        inner_lr = torch.exp(self.log_inner_lr)
        return p - inner_lr * g

    @torch.enable_grad()
    def inner_loop(self, input_ids, **seg_kwargs):
        seg_kwargs = copy(seg_kwargs)
        inputs_embeds = seg_kwargs['inputs_embeds']
        B = inputs_embeds.size(0)
        memory_state = inputs_embeds[:, :self.num_mem_tokens]
        orig_embeds = inputs_embeds[:, self.num_mem_tokens:]
        mem_live = memory_state.requires_grad_(True)

        device = inputs_embeds.device
        inner_loop_stats = {'inner_grad_norm_mean': torch.tensor(0.0, device=device)}
        
        for _ in range(self.inner_steps):
            ctx_emb = torch.cat([mem_live, orig_embeds], dim=1)
            seg_kwargs['inputs_embeds'] = ctx_emb
            out = self.model(**seg_kwargs)
            
            logits = out.logits[:, self.num_mem_tokens:]
            inner_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1),)
            g = torch.autograd.grad(inner_loss, mem_live, create_graph=True)[0]
            g = self._ogd_project_grad(g)
            
            g_norm = g.reshape(B, -1).norm(dim=1).detach()
            inner_loop_stats['inner_grad_norm_mean'] += g_norm.mean()

            mem_live = self._sgd_step(mem_live, g, clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)

        self.ogd_update_after_segment(g)

        mem_norm = mem_live.norm(dim=[1, 2]).detach()  # B
        inner_loop_stats['inner_mem_norm_mean'] = mem_norm.mean()
        inner_loop_stats['inner_grad_norm_mean'] /= self.inner_steps
        inner_loop_stats['inner_final_loss'] = inner_loss

        return out, mem_live, inner_loop_stats

    def _ogd_project_grad(self, g) -> torch.Tensor:
        # g_3d: [B,T,D] -> project per item: g - Q(Q^T g)
        B, T, D = g.shape
        g = g.reshape(B, -1)                                # [B, D_tot]
        cols = torch.arange(self.r_max, device=g.device).unsqueeze(0).expand(B, -1)  # [B,R]
        active = (cols < self.ogd_r_used.unsqueeze(1))                       # [B,R]
        Qact = self.ogd_Q * active.unsqueeze(-1)               # [B,R,D_tot]
        coeffs = torch.einsum('brd,bd->br', Qact.type_as(g), g)           # [B,R]  = Q^T g
        proj = torch.einsum('brd,br->bd', Qact.type_as(g), coeffs)      # [B,D]  = Q coeffs
        return (g - proj).view(B, T, D)

    @torch.no_grad()
    def ogd_update_after_segment(self, g: torch.Tensor, novelty_tau=0.2):
        """
        Append at most one new basis vector per item using a single Gram–Schmidt step:
        v_res = v - Q(Q^T v); if ||v_res||/||v|| > tau and r_used<R -> store v_res/||v_res||.
        """
        B, _, _ = g.shape
        v = g.reshape(B, -1)                            # [B, D_tot]
        v_norm = v.norm(dim=1, keepdim=True).clamp_min(1e-12)   # [B,1]

        cols = torch.arange(self.r_max, device=v.device).unsqueeze(0).expand(B, -1)
        active = (cols < self.ogd_r_used.unsqueeze(1))  # [B,R]
        Qact = self.ogd_Q * active.unsqueeze(-1)                # [B,R,D_tot]
        c = torch.einsum('brd,bd->br', Qact.type_as(g), v)                 # [B,R]
        proj = torch.einsum('brd,br->bd', Qact.type_as(g), c)              # [B,D_tot]
        v_res = v - proj
        v_res_norm = v_res.norm(dim=1, keepdim=True).clamp_min(1e-12)

        novelty = (v_res_norm / v_norm).squeeze(1)              # [B]
        take = (novelty > novelty_tau) & (self.ogd_r_used < self.r_max)

        if take.any():
            v_store = (v_res / v_res_norm)                      # normalized residual
            rows = torch.nonzero(take, as_tuple=False).squeeze(1)          # [K]
            idx  = self.ogd_r_used[rows]                                     # [K]
            self.ogd_Q[rows.int(), idx.int(), :] = v_store[rows.int(), :]
            self.ogd_r_used[rows.int()] = self.ogd_r_used[rows.int()] + 1

    def reset_ogd(self, input_ids):
        B, _ = input_ids.shape
        self.ogd_Q = torch.zeros(B, self.r_max, self.num_mem_tokens * self.memory_dim, dtype=self.model.dtype, device=self.model.device)
        self.ogd_r_used = torch.zeros(B, device=self.model.device)

    def forward(self, input_ids, memory_state=None, is_last_segment=False, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)
        
        seg_kwargs = self.process_input(input_ids, memory_state, **kwargs)
        
        if is_last_segment:
            for p in self.model.parameters():
                p.requires_grad = True

            inner_stats = {}
            out = self.model(**seg_kwargs)
        else:
            for p in self.model.parameters():
                p.requires_grad = False
            seg_kwargs['input_ids'] = input_ids
            out, memory_state, inner_stats = self.inner_loop(**seg_kwargs)

        out = self.process_output(out, inner_stats, **kwargs)
        return out, memory_state
    
    def generate(self, input_ids, memory_state, attention_mask=None, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        if self.num_mem_tokens > 0:
            inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
        seg_kwargs['output_hidden_states'] = True
        return seg_kwargs
    
    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, self.num_mem_tokens:] = attention_mask
            return mask
    
    def process_output(self, model_outputs, inner_stats, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            out['logits'] = model_outputs.logits[:, self.num_mem_tokens:]

            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, self.num_mem_tokens:-self.num_mem_tokens] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
            
            for k, v in inner_stats.items():
                out[k] = v
            return out
        
        return model_outputs    


class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=None, output_hidden_states=None):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        self.memory_cell.reset_ogd(input_ids)
        cell_outputs = []
        # print('\n\n\nForward: ', [s['input_ids'].shape for s in segmented])
        for seg_num, segment in enumerate(segmented):
            is_last_segment = seg_num == len(segmented) - 1
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, is_last_segment=is_last_segment, output_hidden_states=True)
            cell_outputs.append(cell_out)
            memory_state = self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states)
        return out
    
    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        # print('\n\n\nGenerate: ', [s['input_ids'].shape for s in segmented])
        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)

        final_segment = segmented[-1]
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

        return out

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments
    
    def split_tensor(self, tensor):
        align = self.rmt_config.get('segment_alignment')
        segment_size = self.rmt_config.get('segment_size')
        if align in {'left', None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align in {'right', None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align == 'center':
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]

                # print(flat_labels)
                # print(flat_logits.argmax(dim=-1))
     
            out['loss'] = loss_fct(flat_logits, flat_labels)
            if out['loss'] is None:
                raise ValueError
        else:
            out['loss'] = 0

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits', 'inner']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f'{key}_{seg_num}'] = value
        
        out['inner_lr'] = torch.exp(self.memory_cell.log_inner_lr).item()

        return out 
        
    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return memory_state
        
        memory_state = memory_state.detach()
        return memory_state
    
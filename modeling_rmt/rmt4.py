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


class MemoryAttention(torch.nn.Module):
    def __init__(self, memory_dim, hidden_dim=None, num_heads=4, embd_std=0.02):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = memory_dim * 4

        self.attention = nn.MultiheadAttention(memory_dim, num_heads, batch_first=True)
        self.q_norm = nn.LayerNorm(memory_dim)
        self.kv_norm = nn.LayerNorm(memory_dim)
        # optional FFN (pre-LN)
        self.ff_norm = nn.LayerNorm(memory_dim)
        self.ff = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, memory_dim),
        )

    def _init_weights(self, embd_std):
        for n, p in self.named_parameters():
            if "weight" in n:
                if "attention" in n:
                    nn.init.normal_(p, mean=0.0, std=embd_std)
                elif "norm" in n:
                    nn.init.ones_(p)
                else:
                    nn.init.xavier_uniform_(p)
            elif "bias" in n:
                nn.init.zeros_(p)

    def forward(self, m_prev, m_new):
        kv = torch.cat([m_prev, m_new], dim=1)
        q  = self.q_norm(m_prev)
        kv = self.kv_norm(kv)

        attn_out, _ = self.attention(query=q, key=kv, value=kv, need_weights=False)
        x = m_prev + attn_out                     # no post-LN here
        x = x + self.ff(self.ff_norm(x))          # FFN as pre-LN residual
        return x
    

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

        self.use_mem_attn = kwargs.get('use_mem_attn', False)
        self.use_agem = kwargs.get('use_agem', False)
        self.use_ogd = kwargs.get('use_ogd', False)

        if self.use_mem_attn:
            self.mem_attn = MemoryAttention(memory_dim=self.model.config.hidden_size)
        if self.use_agem or self.use_ogd:
            self.prev_g = None

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim =  getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
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

    def a_gem_project(self, g_cur, g_ref, eps=1e-12):
        if g_ref is None:
            return g_cur

        B, M, H = g_cur.shape
        g_cur = g_cur.reshape(B, -1)
        g_ref = g_ref.reshape(B, -1)

        dot = (g_cur * g_ref).sum(dim=1, keepdim=True)          # [B,1]
        den = g_ref.pow(2).sum(dim=1, keepdim=True) + eps   # [B,1]
        # mask = (dot < 0).float()
        g_cur = g_cur - (dot < 0) * (dot / den) * g_ref
        return g_cur.view(B, M, H)

    def ogd_project(self, g, B, ridge=1e-4):
        if g is None:
            return g
        
        # g: [D], B: [D, r] where r << D
        if B is None or B.numel() == 0:
            return g
        BtB = B.T @ B
        # small ridge for stability
        proj = B @ torch.linalg.solve(BtB + ridge*torch.eye(B.shape[1], device=B.device), B.T @ g)
        return g - proj

    @torch.enable_grad()
    def inner_loop(self, input_ids, **seg_kwargs):
        seg_kwargs = copy(seg_kwargs)
        inputs_embeds = seg_kwargs['inputs_embeds']
        B = inputs_embeds.size(0)
        memory_state = inputs_embeds[:, :self.num_mem_tokens]
        orig_embeds = inputs_embeds[:, self.num_mem_tokens:]

        # mem_live = memory_state.clone().requires_grad_(True)
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

            if self.use_agem or self.use_ogd:
                if self.prev_g is not None:
                    if self.use_agem:
                        g = self.a_gem_project(g, self.prev_g)
                    if self.use_ogd:
                        g = self.ogd_project(g, self.prev_g)
            
            g_norm = g.reshape(B, -1).norm(dim=1).detach()

            inner_loop_stats['inner_grad_norm_mean'] += g_norm.mean()
            # inner_loop_stats['inner_grad_norm_max'] = max(inner_loop_stats['inner_grad_norm_max'], g_norm.max())
            # inner_loop_stats['inner_grad_norm_min'] = min(inner_loop_stats['inner_grad_norm_min'], g_norm.min())
            mem_live = self._sgd_step(mem_live, g, clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)

        self.prev_g = g.detach()

        mem_norm = mem_live.norm(dim=[1, 2]).detach()  # B
        inner_loop_stats['inner_mem_norm_mean'] = mem_norm.mean()
        inner_loop_stats['inner_grad_norm_mean'] /= self.inner_steps
        inner_loop_stats['inner_final_loss'] = inner_loss

        return out, mem_live, inner_loop_stats

    def forward(self, input_ids, memory_state=None, is_last_segment=False, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)
        
        seg_kwargs = self.process_input(input_ids, memory_state, **kwargs)
        
        if is_last_segment:
            for p in self.model.parameters():
                p.requires_grad = True
            # for p in self.mem_attn.parameters():
            #     p.requires_grad = True
            inner_stats = {}
            out = self.model(**seg_kwargs)
            if self.use_agem or self.use_ogd:
                self.prev_g = None
        else:
            for p in self.model.parameters():
                p.requires_grad = False
            seg_kwargs['input_ids'] = input_ids
            out, mem_live, inner_stats = self.inner_loop(**seg_kwargs)
            
            if self.use_mem_attn:
                memory_state = self.mem_attn(memory_state, mem_live)
            else:
                memory_state = mem_live
        
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


import random
class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=None, output_hidden_states=None):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

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
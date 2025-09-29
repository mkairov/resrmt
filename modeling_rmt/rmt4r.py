import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from copy import copy

from accelerate.logging import get_logger
logger = get_logger('')


class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, init_inner_lr=1.0, learn_lr=False, inner_steps=10, inner_optim='sgd', **kwargs):
        super().__init__()
        self.model = base_model
        self.create_memory(num_mem_tokens)

        self.num_mem_tokens = num_mem_tokens
        self.inner_steps = inner_steps
        self.inner_optim = inner_optim
        assert self.inner_optim != 'muon' or self.num_mem_tokens > 1, "Muon works with 2+ memory tokens only"

        self.inner_clip_value = kwargs.get('inner_clip_value', None)
        self.inner_clip_norm = kwargs.get('inner_clip_norm', None)
        self.use_write_head = kwargs.get('use_write_head', None)
        # meta-learnable parameters (log-space for power scaling)
        self.log_inner_lr = torch.log(torch.tensor(init_inner_lr))
        if learn_lr:
            self.log_inner_lr = nn.Parameter(self.log_inner_lr)

        if self.use_write_head:
            V = self.model.config.vocab_size
            n_embd = self.model.config.hidden_size
            self.write_head = nn.Linear(n_embd, V, bias=False)

            if hasattr(self.model, 'get_output_embeddings'):
                head_params = self.model.get_output_embeddings().weight
            else:  # fallback to input embeddings
                head_params = self.model.get_input_embeddings().weight
            with torch.no_grad():
                self.write_head.weight.copy_(head_params.detach())

        self.model.tie_weights()
        self.mem_list = []

    def create_memory(self, num_mem_tokens):
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

    @staticmethod
    def _zeropower_via_newtonschulz5(g, steps: int):
        assert g.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = g.bfloat16()
        if g.size(-2) > g.size(-1):
            X = X.mT

        # Ensure spectral norm is at most 1
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X
        
        if g.size(-2) > g.size(-1):
            X = X.mT
        return X
    
    def _muon_step(self, p, g, clip_value=None, clip_norm=None, ns_steps=5):
        if clip_value is not None:
            # simple element-wise clamp
            g = torch.clamp(g, -clip_value, clip_value)

        if clip_norm is not None:
            # scale gradient if its 2-norm is too large
            # check grad for each sample separately as we do per-sample optimization
            g_norm = g.norm(dim=[1, 2], keepdim=True)                 # (B,1,1)
            scale = clip_norm / (g_norm + 1e-6)
            g = torch.where(g_norm > clip_norm, g * scale, g)
        
        update = self._zeropower_via_newtonschulz5(g, steps=ns_steps)
        update *= max(1, g.size(-2) / g.size(-1))**0.5

        inner_lr = torch.exp(self.log_inner_lr)
        return p - inner_lr * update

    @torch.enable_grad()
    def inner_loop(self, input_ids, **seg_kwargs):
        seg_kwargs = copy(seg_kwargs)
        inputs_embeds = seg_kwargs['inputs_embeds']
        B = inputs_embeds.size(0)
        mem_live = inputs_embeds[:, :self.num_mem_tokens]
        mem_live = mem_live.requires_grad_(True)
        orig_embeds = inputs_embeds[:, self.num_mem_tokens:]

        device = inputs_embeds.device
        inner_loop_stats = {'inner_grad_norm_mean': torch.tensor(0.0, device=device)}
        
        for _ in range(self.inner_steps):
            ctx_emb = torch.cat([mem_live, orig_embeds], dim=1)
            seg_kwargs['inputs_embeds'] = ctx_emb
            out = self.model(**seg_kwargs)

            if self.use_write_head:
                h = out.hidden_states[-1]
                h = h[:, self.num_mem_tokens:, :]
                logits = self.write_head(h)
                del h
            else:
                logits = out.logits
                logits = logits[:, self.num_mem_tokens:, :]
            
            inner_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                reduction='none'
            ).view(B, -1)
            inner_loss = (inner_loss.sum(1) / inner_loss.size(1)).sum()
            del logits

            g = torch.autograd.grad(inner_loss, mem_live, create_graph=True)[0]
            g_norm = g.reshape(B, -1).norm(dim=1).detach()

            inner_loop_stats['inner_grad_norm_mean'] += g_norm.mean()

            if self.inner_optim == 'sgd':
                mem_live = self._sgd_step(mem_live, g, clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)
            elif self.inner_optim == 'muon':
                mem_live = self._muon_step(mem_live, g, clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)
            else:
                raise "Unknown optimizer for inner loop"

        mem_norm = mem_live.norm(dim=[1, 2]).detach()  # B
        inner_loop_stats['inner_mem_norm_mean'] = mem_norm.mean()
        inner_loop_stats['inner_grad_norm_mean'] /= self.inner_steps
        inner_loop_stats['inner_final_loss'] = inner_loss.detach() / B
        del inner_loss

        return out, mem_live, inner_loop_stats

    def forward(self, input_ids, memory_state=None, is_last_segment=False, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)
        
        seg_kwargs = self.process_input(input_ids, memory_state, is_last_segment=is_last_segment, **kwargs)
        
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
            self.mem_list.append(memory_state)
            
        out = self.process_output(out, inner_stats, is_last_segment=is_last_segment, **kwargs)
        return out, memory_state
    
    def generate(self, input_ids, memory_state, attention_mask=None, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, is_last_segment, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        if self.num_mem_tokens > 0:
            if is_last_segment:
                inputs_embeds = torch.cat(self.mem_list + [inputs_embeds], dim=1)
            else:
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
            mask[:, -attention_mask.shape[-1]:] = attention_mask
            return mask
    
    def process_output(self, model_outputs, inner_stats, is_last_segment, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            log_start = self.num_mem_tokens * (len(self.mem_list) if is_last_segment else 1)

            out['logits'] = model_outputs.logits[:, log_start:]

            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, log_start:] for lh in model_outputs.hidden_states]
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
        self.memory_cell.mem_list.clear()

        cell_outputs = []
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
        self.memory_cell.mem_list.clear()

        for _, segment in enumerate(segmented[:-1]):
            _, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)

        final_segment = segmented[-1]
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, is_last_segment=True, **generate_kwargs)

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
        
        return out 
        
    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return memory_state
        
        memory_state = memory_state.detach()
        return memory_state
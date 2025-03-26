import math
import random

import torch
from torch import nn

from lm_experiments_tools.utils import RMTOutput


class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, **kwargs):
        super().__init__()
        self.model = base_model
        self.num_mem_tokens = num_mem_tokens
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim =  getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, write_mem=True, **kwargs)
        out = self.model(**seg_kwargs)
        out, new_memory_state = self.process_output(out, **kwargs)

        return out, new_memory_state
    
    def generate(self, input_ids, memory_state, attention_mask=None, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask, write_mem=False)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, write_mem, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        if self.num_mem_tokens > 0:
            if write_mem:
                inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)
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
            mask[:, self.num_mem_tokens: self.num_mem_tokens + attention_mask.shape[1]] = attention_mask
            return mask
    
    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = RMTOutput()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens:]
            out['logits'] = model_outputs.logits[:, self.num_mem_tokens:-self.num_mem_tokens]
            
            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, self.num_mem_tokens:-self.num_mem_tokens] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            memory_state = None
            out = model_outputs
            
        return out, memory_state 


class MemoryAttention(torch.nn.Module):
    def __init__(self, memory_dim, res_mem_count, hidden_dim=None, num_heads=1, embd_std=0.02):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = memory_dim * 4

        self.memory_dim = memory_dim
        self.res_mem_count = res_mem_count
        self.attention = torch.nn.MultiheadAttention(embed_dim=memory_dim, num_heads=num_heads, batch_first=True)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(memory_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, memory_dim)
        )
        self.norm1 = torch.nn.LayerNorm(memory_dim)
        self.norm2 = torch.nn.LayerNorm(memory_dim)

        self._init_weights(embd_std)

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

    def forward(self, current_memory, prev_memories):
        if len(prev_memories) == 0 or self.res_mem_count == 0:
            return current_memory
        
        elif self.res_mem_count > 0:
            memory_dropout = random.sample(prev_memories, min(len(prev_memories), self.res_mem_count))
            memory_cat = torch.cat(memory_dropout, dim=1)
            attn_output, attn_map = self.attention(query=current_memory, key=memory_cat, value=memory_cat) # K [memory_dim, n_tokens * n_memory] @ V^T

        else:
            memory_cat = torch.cat(prev_memories, dim=1)
            attn_output, attn_map = self.attention(query=current_memory, key=memory_cat, value=memory_cat)

        updated_memory = current_memory + attn_output 
        updated_memory = self.norm1(updated_memory)
        updated_memory = updated_memory + self.feedforward(updated_memory)
        updated_memory = self.norm2(updated_memory)

        return updated_memory


def apply_rope_with_segments(memory_state, segment_num):
    batch_size, seq_length, d_model = memory_state.shape

    theta = torch.exp(-1j * (torch.arange(0, d_model, 2, device=memory_state.device) / d_model))
    theta = theta.unsqueeze(0)  #  (1, d_model // 2)
    positions = torch.ones(seq_length, device=memory_state.device, dtype=torch.float) * segment_num
    positions = positions.unsqueeze(-1) # (seq_length, 1)
    complex_pos = torch.exp(1j * positions * theta) # (seq_length, d_model // 2)

    memory_state = memory_state.view(batch_size, seq_length, d_model // 2, 2)
    embeddings_complex = memory_state[..., 0] + 1j * memory_state[..., 1]  # (batch_size, seq_length, d_model // 2)

    embeddings_rotated = embeddings_complex * complex_pos  # (batch_size, seq_length, d_model // 2)

    embeddings_rotated = torch.stack([embeddings_rotated.real, embeddings_rotated.imag], dim=-1)
    return embeddings_rotated.view(batch_size, seq_length, d_model)


class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs

        memory_dim = self.memory_cell.memory.shape[-1]

        self.res_mem_count = rmt_kwargs.get("res_mem_count")
        if self.res_mem_count not in {0, None}:
            self.memory_aggregator = MemoryAttention(memory_dim, res_mem_count=rmt_kwargs.get('res_mem_count'))
        else:
            self.aggregator = None

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=False, output_hidden_states=True):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        cell_outputs = []
        past_memory_states = []
        for seg_num, segment in enumerate(segmented):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
            cell_outputs.append(cell_out)

            if self.memory_cell.num_mem_tokens > 0 and self.aggregator is not None:
                memory_pos_embeds = apply_rope_with_segments(memory_state, seg_num).type_as(memory_state)
                past_memory_states.append(memory_pos_embeds)
                memory_state = self.aggregator(memory_pos_embeds, past_memory_states)
                memory_state = self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states)
        past_memory_states.clear()
        
        return out
    
    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        past_memory_states = []
        for seg_num, segment in enumerate(segmented[:-1]):
            _, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)
            memory_state = apply_rope_with_segments(memory_state, seg_num)
            past_memory_states.append(memory_state)
            memory_state = self.memory_aggregator(memory_state, past_memory_states)

        final_segment = segmented[-1]
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)
        past_memory_states.clear()

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
        out = RMTOutput()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = nn.CrossEntropyLoss()
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
        segment_keys = ['loss', 'logits']
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

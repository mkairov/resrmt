from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2PreTrainedModel, GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
import torch.nn as nn


class MemoryAggregator(torch.nn.Module):
    def __init__(self, memory_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = memory_dim

        self.aggr1 = torch.nn.Linear(2 * memory_dim, hidden_dim, bias=False)
        self.activation = torch.nn.ReLU()
        self.aggr2 = torch.nn.Linear(hidden_dim, memory_dim, bias=False)
        self.norm = torch.nn.LayerNorm(memory_dim)
    
    def forward(self, prev_memory, current_memory):
        combined_memory = torch.cat([prev_memory, current_memory], dim=-1)
        updated_memory = self.aggr1(combined_memory)
        updated_memory = self.activation(updated_memory)
        updated_memory = self.aggr2(updated_memory)
        updated_memory = self.norm(updated_memory)
        return updated_memory


class GPT2MemoryBlock(nn.Module):
    def __init__(self, config):
        super(GPT2MemoryBlock, self).__init__(config)
        self.config = config

        # Original GPT-2 transformer blocks
        self.transformer = GPT2Block(config)
        self.aggr_layer = MemoryAggregator(config.memory_dim)
        self.create_memory(config.num_mem_tokens)

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

    def forward(self, hidden_states, *args, memory_state=None, **kwargs):
        # Pass through the original GPT2Block
        outputs = self.gpt2block(hidden_states, *args, **kwargs)
        # block_output = outputs[0]  # Hidden states from GPT2Block

        # # Pass the block output through the Aggregator layer
        # aggregator_output = self.aggregator(block_output)

        # # Combine the Aggregator output with the block output
        # combined_output = aggregator_output + block_output  # You can choose a different combination method

        # # Return combined output along with any other outputs from GPT2Block
        # if len(outputs) > 1:
        #     return (combined_output,) + outputs[1:]
        # else:
        #     return (combined_output,)
        
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
            out = CausalLMOutputWithCrossAttentions()
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

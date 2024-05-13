# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/mbart/modeling_mbart.py
# 
# This code is adapted from The Huggingface Inc. and Facebook AI Research Team's code for
# Pytorch based MBART Model. The original license is attached below.
#
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
from torch import nn
from torch import functional as F

from transformers import MBartConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)

from typing import Optional, List


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, 
    and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` 
    in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

# Copied from 
# transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding 
# with Bart->MBart
class MBartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # MBart is set up so that if padding_idx is specified 
        # then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. 
        # Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)
    
# Copied from 
# transformers.models.bart.modeling_bart.BartScaledWordEmbedding
# with Bart->MBart
class MBartScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' 
    forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale

# Copied from transformers.models.bart.modeling_bart.BartAttention
# with Bart->MBart
# and modified for vllm
class MBartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # This will be overwritten by model initialization if we are using it.
        # N.B. currently we only support per tensor scalar scaling factors
        # & only applicable to ROCm (AMD GPU).
        # The scaling factor convention we are assuming is
        # quantized_value * scaling_factor ~= true_value
        # which is consistent with the practice of setting
        # scaling_factor = tensor_amax / FPtype_max
        self.kv_scale = 1.0

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads
        )

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        attn_output, _ = self.c_proj(attn_output)
        return attn_output


class MBartEncoderLayer(nn.Module):
    def __init__(self, config: MBartConfig, 
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.embed_dim = config.d_model
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = MBartAttention(
            hidden_size=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            bias=attention_bias,
            quant_config=quant_config
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = get_act_fn(
            config.activation_function, 
            quant_config=quant_config
        )
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: Optional[torch.Tensor],
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
        ):
            assert kv_cache is None
            residual = hidden_states
            hidden_states = self.self_attn_layer_norm(hidden_states)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout)
            hidden_states = residual + hidden_states

            residual = hidden_states

            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.activation_fn(self.fc1(hidden_states))
            hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
            hidden_states = self.fc2(hidden_states)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
            ):
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(
                    hidden_states,
                    min=-clamp_value, 
                    max=clamp_value
                )
            
            return hidden_states


class MBartEncoder(nn.Module):
    def __init__(self, config: MBartConfig) -> None:

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = MBartScaledWordEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
            embed_scale=self.embed_scale
        )

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embedding_dim=embed_dim
        )

        self.layers = nn.ModuleList([MBartEncoderLayer(config) for _ in range(config.encoder_layers)])

        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.layer_norm = nn.LayerNorm(config.d_model)


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        hidden_states = self.embed_tokens(input_ids) + \
            self.embed_positions(positions)
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                kv_cache=kv_caches[i],
                attn_metadata=attn_metadata
            )

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states
    
class MBartDecoderLayer(nn.Module):
    def __init__(self, config: MBartConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = MBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            # dropout=config.attention_dropout,
            # is_decoder=True,
            # is_causal=True,
            # config=config,

        )
        self.dropout = config.dropout
        self.activation_fn = get_act_fn(
            config.activation_function,
            quant_config=quant_config
        )
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)


    


# class MBartModel(MBartPreTrainedModel):
#     _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

#     def __init__(self, config: MBartConfig):
#         super().__init__(config)

#         padding_idx, vocab_size = config.pad_token_id, config.vocab_size
#         self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

#         self.encoder = MBartEncoder(config, self.shared)
#         self.decoder = MBartDecoder(config, self.shared)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.shared

#     def set_input_embeddings(self, value):
#         self.shared = value
#         self.encoder.embed_tokens = self.shared
#         self.decoder.embed_tokens = self.shared

#     def get_encoder(self):
#         return self.encoder

#     def get_decoder(self):
#         return self.decoder

#     def _tie_weights(self):
#         if self.config.tie_word_embeddings:
#             self._tie_or_clone_weights(self.encoder.embed_tokens, self.get_input_embeddings())
#             self._tie_or_clone_weights(self.decoder.embed_tokens, self.get_input_embeddings())

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         decoder_head_mask: Optional[torch.Tensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Seq2SeqModelOutput, Tuple[torch.FloatTensor]]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # different to other models, MBart automatically creates decoder_input_ids from
#         # input_ids if no decoder_input_ids are provided
#         if decoder_input_ids is None and decoder_inputs_embeds is None:
#             decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

#         if encoder_outputs is None:
#             encoder_outputs = self.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 head_mask=head_mask,
#                 inputs_embeds=inputs_embeds,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )
#         # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
#         elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#             encoder_outputs = BaseModelOutput(
#                 last_hidden_state=encoder_outputs[0],
#                 hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                 attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#             )

#         # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
#         decoder_outputs = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             encoder_hidden_states=encoder_outputs[0],
#             encoder_attention_mask=attention_mask,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         if not return_dict:
#             return decoder_outputs + encoder_outputs

#         return Seq2SeqModelOutput(
#             last_hidden_state=decoder_outputs.last_hidden_state,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )
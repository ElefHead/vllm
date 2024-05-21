# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/mbart/modeling_mbart.py
# 
# This code is adapted from The Huggingface Inc. 
# and Facebook AI Research Team's code for
# Pytorch based MBART Model. The original license is attached below.
#
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. 
# All rights reserved.
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

from transformers import MBartConfig

from vllm.config import LoRAConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput

from typing import Optional, List, Iterable, Tuple


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

    index_of_eos = (
        prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1
    ).unsqueeze(-1)
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
        decoder: bool = False,
        causal: bool = True,
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
        self.decoder = decoder
        self.causal = causal
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
            causal=self.causal,
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
            hidden_states = residual + hidden_states

            residual = hidden_states

            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.activation_fn(self.fc1(hidden_states))
            hidden_states = self.fc2(hidden_states)
            hidden_states = residual + hidden_states

            if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or 
                torch.isnan(hidden_states).any()
            ):
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(
                    hidden_states,
                    min=-clamp_value, 
                    max=clamp_value
                )
            
            return hidden_states


class MBartEncoder(nn.Module):
    def __init__(
            self, 
            config: MBartConfig,
            embed_tokens: Optional[VocabParallelEmbedding] = None,
            ) -> None:
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) \
            if config.scale_embedding else 1.0

        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=self.padding_idx
        )

        if self.embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = MBartLearnedPositionalEmbedding(
            self.max_source_positions,
            embedding_dim=embed_dim
        )

        self.layers = nn.ModuleList(
            [
                MBartEncoderLayer(config) for _ in range(config.encoder_layers)
            ]
        )

        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.layer_norm = nn.LayerNorm(config.d_model)


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale + \
            self.embed_positions(positions)
        hidden_states = self.layernorm_embedding(hidden_states)

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
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)

        self.self_attn = MBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            is_decoder=True,
            is_causal=True,
            bias=attention_bias,
            quant_config=quant_config
        )
        self.dropout = config.dropout
        self.activation_fn = get_act_fn(
            config.activation_function,
            quant_config=quant_config
        )
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.encoder_attn = MBartAttention(
            self.embed_dim,
            num_heads=config.decoder_attention_heads,
            is_decoder=True,
            bias=attention_bias,
            quant_config=quant_config
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: Optional[torch.Tensor],
            encoder_hidden_states: Optional[torch.Tensor],
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
    ) -> None: 
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # causal attention
        hidden_states = self.self_attn(

        )
        hidden_states = residual + hidden_states

        # cross attention
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        hidden_states = self.encoder_attn(

        )
        hidden_states = residual + hidden_states

        # fully connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    

class MBartDecoder(nn.Module):
    def __init__(
            self,
            config: MBartConfig,
            embed_tokens: Optional[VocabParallelEmbedding] = None,
        ) -> None:
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) \
            if config.scale_embedding else 1.0
        
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=self.padding_idx
        )

        if self.embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )

        self.layers = nn.ModuleList(
            [
                MBartDecoderLayer(config)
                for _ in range(config.decoder_layers)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
            self, 
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            encoder_hidden_state: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
    ):
        hidden_state = self.embed_tokens(input_ids) * self.embed_scale + \
            self.embed_positions(positions)
        hidden_state = self.layernorm_embedding(hidden_state)

        for i, decoder_layer in enumerate(self.layers):
            hidden_state = decoder_layer(

            )

        hidden_state = self.layer_norm(hidden_state)
        return hidden_state


class MBartModel(nn.Module):
    def __init__(
            self,
            config: MBartConfig,
            quant_config: Optional[QuantizationConfig] = None
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.shared = VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=self.padding_idx,
        )

        self.encoder = MBartEncoder(config=config, embed_tokens=self.shared)
        self.decoder = MBartDecoder(config=config, embed_tokens=self.shared)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.shared(input_ids)
    
    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def forward(
            self,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        pass


class MBartForConditionalGeneration(nn.Module):
    def __init__(
            self, 
            config: MBartConfig,
            quant_config: Optional[QuantizationConfig] = None,
            lora_config: Optional[LoRAConfig] = None,
        ) -> None:
        super().__init__()
        self.config = config
        self.model = MBartModel(config=config, quant_config=quant_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.d_model,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE 
            if not lora_config else lora_config.lora_vocab_padding_size
        )

        # todo: handle tie weights
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(
            self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(
            self, 
            weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> None:
        pass
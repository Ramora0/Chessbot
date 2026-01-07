"""
Conformer layers for chess model with 2D spatial convolutions.

This module implements conformer-style layers that apply 2D convolutions
specifically to the chess board tokens (first 64 tokens as 8×8 grid) while
allowing metadata tokens (turn, castling, en passant) to pass through unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class SpatialConvolutionModule(nn.Module):
    """
    Lightweight spatial 2D convolution module for chess board tokens.

    Takes 64 sequential board tokens, reshapes them to an 8×8 spatial grid,
    applies depthwise Conv2d for spatial reasoning, then reshapes back to sequence.

    This module ONLY processes board tokens - metadata tokens must be handled separately.
    Uses NO channel expansion to minimize parameters.
    """

    def __init__(self, hidden_size: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        """
        Args:
            hidden_size: Dimension of hidden states
            kernel_size: Size of Conv2d kernel (default: 3 for 3×3)
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        # Pre-normalization (LayerNorm for token-level normalization)
        self.norm = nn.LayerNorm(hidden_size)

        # Depthwise 2D convolution: spatial mixing on 8×8 board
        # groups=channels means depthwise (each channel convolved independently)
        # NO channel expansion - keeps parameters minimal
        padding = (kernel_size - 1) // 2  # Same padding to preserve 8×8 dimensions
        self.depthwise_conv = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_size,  # Depthwise: each channel processes spatial info independently
            bias=False
        )

        # Batch normalization for Conv2d (stabilizes training)
        self.batch_norm = nn.BatchNorm2d(hidden_size)

        # Activation function (SiLU/Swish is conformer standard)
        self.activation = nn.SiLU()

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, board_tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D spatial convolution to board tokens.

        Args:
            board_tokens: [batch_size, 64, hidden_size] - sequential board tokens

        Returns:
            output: [batch_size, 64, hidden_size] - spatially convolved tokens
        """
        batch_size = board_tokens.size(0)

        # Normalize
        x = self.norm(board_tokens)  # [batch, 64, hidden]

        # CRITICAL: Reshape to 2D spatial grid
        # [batch, 64, hidden] → [batch, hidden, 8, 8]
        # Token order is row-major: rank 8→1, files a→h
        x = x.transpose(1, 2)  # [batch, hidden, 64]
        x = x.view(batch_size, self.hidden_size, 8, 8)  # Reshape to 8×8 grid

        # Apply 2D depthwise convolution for spatial reasoning
        x = self.depthwise_conv(x)  # [batch, hidden, 8, 8]
        x = self.batch_norm(x)
        x = self.activation(x)

        # CRITICAL: Reshape back to sequence
        # [batch, hidden, 8, 8] → [batch, 64, hidden]
        x = x.view(batch_size, self.hidden_size, 64)  # Flatten 8×8 back to 64
        x = x.transpose(1, 2)  # [batch, 64, hidden]

        # Dropout
        x = self.dropout(x)

        return x


class ConformerFeedForward(nn.Module):
    """
    Feedforward module with configurable expansion factor.

    For conformer: use expansion_factor=0.5 for each half-step FFN.
    Uses SwiGLU activation (same as Llama MLP).
    """

    def __init__(self, config, expansion_factor: float = 0.5) -> None:
        """
        Args:
            config: Model configuration
            expansion_factor: Multiplier for intermediate size (0.5 for half-step)
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        # Use config's intermediate_size as base, then apply expansion_factor
        # For macaron-net: each half-step FFN gets expansion_factor * intermediate_size
        # (0.5 for each half means 1.0 total = same as standard transformer)
        base_intermediate = getattr(config, 'intermediate_size', config.hidden_size * 4)
        intermediate_size = int(base_intermediate * expansion_factor)

        # SwiGLU gating (same as Llama)
        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # SwiGLU: gate(x) * up(x) then project down
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return self.dropout(down)


class ChessConformerLayer(nn.Module):
    """
    Conformer layer adapted for chess with spatial convolutions.

    Architecture:
      FFN (half-step) → Multi-Head Self-Attention → Spatial Convolution → FFN (half-step)

    CRITICAL: Convolution applies ONLY to first 64 board tokens (reshaped to 8×8 grid).
    Metadata tokens (64-69) bypass the convolution step and are concatenated back.
    """

    def __init__(self, config, layer_idx: int) -> None:
        """
        Args:
            config: Model configuration
            layer_idx: Index of this layer (for attention)
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Token structure constants
        self.board_tokens = 64  # 8×8 chess board (tokens 0-63)
        self.metadata_tokens = 6  # turn, castling, en passant (tokens 64-69)

        # Component 1: First half-step feedforward (Macaron-Net style)
        self.ffn1_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn1 = ConformerFeedForward(config, expansion_factor=0.5)

        # Component 2: Self-attention (reuse Llama's attention implementation)
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.attn_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embeddings for position encoding
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Component 3: Spatial convolution module (BOARD TOKENS ONLY)
        self.conv_module = SpatialConvolutionModule(
            hidden_size=config.hidden_size,
            kernel_size=3,  # 3×3 convolution
            dropout=config.attention_dropout
        )

        # Component 4: Second half-step feedforward
        self.ffn2_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn2 = ConformerFeedForward(config, expansion_factor=0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through conformer layer.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - expects seq_len=70
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for attention

        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Step 1: First FFN (half-step) - applies to ALL tokens
        residual = hidden_states
        hidden_states = self.ffn1_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = residual + hidden_states

        # Step 2: Self-attention - applies to ALL tokens
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Compute position embeddings from position IDs
        if position_ids is None:
            # Create default position IDs if not provided
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs
        )
        # Handle tuple return from LlamaAttention (output, attention_weights, ...)
        if isinstance(attn_output, tuple):
            hidden_states = attn_output[0]
        else:
            hidden_states = attn_output
        hidden_states = residual + hidden_states

        # Step 3: Convolution module - CRITICAL SPLIT: board vs metadata
        residual = hidden_states

        # CRITICAL: Split at token index 64
        board_tokens = hidden_states[:, :self.board_tokens, :]      # [batch, 64, hidden]
        metadata_tokens = hidden_states[:, self.board_tokens:, :]   # [batch, 6, hidden]

        # Apply spatial convolution ONLY to board tokens (reshaped to 8×8 internally)
        board_tokens_conv = self.conv_module(board_tokens)

        # CRITICAL: Recombine board (post-conv) + metadata (unchanged)
        hidden_states = torch.cat([board_tokens_conv, metadata_tokens], dim=1)
        hidden_states = residual + hidden_states

        # Step 4: Second FFN (half-step) - applies to ALL tokens
        residual = hidden_states
        hidden_states = self.ffn2_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ChessConformerModel(LlamaPreTrainedModel):
    """
    Custom conformer model for chess, replacing LlamaModel.

    Uses ChessConformerLayer instead of LlamaDecoderLayer to incorporate
    2D spatial convolutions for chess board reasoning.
    """

    def __init__(self, config) -> None:
        """
        Args:
            config: LlamaConfig with chess-specific settings
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding layer (same as LlamaModel)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        # Replace transformer layers with conformer layers
        self.layers = nn.ModuleList([
            ChessConformerLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final layer normalization (same as LlamaModel)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> BaseModelOutputWithPast:
        """
        Forward pass through conformer model.

        Args:
            input_ids: [batch_size, seq_len] - token IDs
            inputs_embeds: [batch_size, seq_len, hidden_size] - embeddings (alternative to input_ids)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs

        Returns:
            BaseModelOutputWithPast with last_hidden_state
        """
        # Get embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Pass through all conformer layers
        for layer_idx, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **kwargs
                )

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Return in same format as LlamaModel for compatibility
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None
        )

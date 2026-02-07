from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel

from loss_weights import (
    POLICY_LOSS_WEIGHT,
    WINRATE_LOSS_WEIGHT,
    MASKED_TOKEN_LOSS_WEIGHT,
    MOVE_WINRATE_LOSS_WEIGHT,
    ILLEGALITY_LOSS_WEIGHT,
    CONTROL_MAP_LOSS_WEIGHT,
    MATE_LOSS_WEIGHT,
    TEMPERATURE,
)


# ── Gradient explosion diagnostics ─────────────────────────────────────
# Register backward hooks on intermediate tensors to catch gradient explosions.
# Hooks fire during backward and print when grad norm exceeds threshold.
_GRAD_DEBUG_THRESHOLD = 100.0  # Print when any grad norm > this


def _make_grad_hook(name: str):
    """Create a backward hook that checks gradient health."""
    def hook(grad):
        if grad is None:
            return
        grad_norm = grad.norm().item()
        grad_max = grad.abs().max().item()
        has_nan = bool(torch.isnan(grad).any())
        has_inf = bool(torch.isinf(grad).any())
        if has_nan or has_inf or grad_norm > _GRAD_DEBUG_THRESHOLD:
            print(f"  GRAD [{name}]: norm={grad_norm:.2e} max={grad_max:.2e} "
                  f"shape={list(grad.shape)} nan={has_nan} inf={has_inf}")
    return hook


def _make_rpe_factorizer():
    """Build the fixed 225->4096 factorizer matrix (lc0 style).

    Maps 225 relative displacement parameters (15x15 grid of rank_diff x file_diff,
    each ranging -7..+7) to all 4096 square pairs (64x64).  This is a binary matrix:
    factorizer[disp_idx, sq_pair_idx] = 1 iff that square pair has that displacement.
    """
    out = np.zeros((225, 64 * 64), dtype=np.float32)
    for r1 in range(8):
        for f1 in range(8):
            for r2 in range(8):
                for f2 in range(8):
                    disp = 15 * (r1 - r2 + 7) + (f1 - f2 + 7)
                    pair = 64 * (r1 * 8 + f1) + (r2 * 8 + f2)
                    out[disp, pair] = 1.0
    return out


# Precomputed once at import time (same as lc0)
_RPE_FACTORIZER_NP = _make_rpe_factorizer()


class Lc0RPELogits(nn.Module):
    """Content-dependent relative position encoding for Q or K (lc0 style).

    Learns a weight matrix [head_dim * num_heads, 225] that is expanded to full
    64x64 via the fixed factorizer, then interacts with Q or K content via einsum.
    Each feature dimension learns its own spatial attention pattern.
    """

    def __init__(self, num_heads: int, head_dim: int, rpe_type: str):
        super().__init__()
        assert rpe_type in ('q', 'k')
        self.rpe_type = rpe_type
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Learnable RPE parameters: [head_dim * num_heads, 225]
        self.rpe = nn.Parameter(torch.zeros(head_dim * num_heads, 225))
        # Fixed factorizer (registered as buffer so it moves with .to(device))
        self.register_buffer(
            'factorizer',
            torch.from_numpy(_RPE_FACTORIZER_NP),  # [225, 4096]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Q or K tensor, shape [batch, num_heads, 64, head_dim]
        Returns:
            Attention logit bias [batch, num_heads, 64, 64]
        """
        # Expand 225 params -> full 64x64: [D*H, 225] @ [225, 4096] -> [D*H, 4096]
        rpe = self.rpe @ self.factorizer.to(self.rpe.dtype)
        # Reshape to [head_dim, num_heads, 64, 64]
        rpe = rpe.view(self.head_dim, self.num_heads, 64, 64)

        if self.rpe_type == 'q':
            # Content-dependent: each feature dim of Q selects its spatial pattern
            return torch.einsum('bhqd,dhqk->bhqk', x, rpe)
        else:
            return torch.einsum('bhkd,dhqk->bhqk', x, rpe)


class Lc0RPEValue(nn.Module):
    """Content-dependent relative position encoding for values (lc0 style).

    Generates a position-dependent value contribution weighted by attention.
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rpe_value = nn.Parameter(torch.zeros(head_dim * num_heads, 225))
        self.register_buffer(
            'factorizer',
            torch.from_numpy(_RPE_FACTORIZER_NP),
        )

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn_weights: [batch, num_heads, 64, 64] (post-softmax)
        Returns:
            Value bias [batch, num_heads, 64, head_dim]
        """
        rpe_v = self.rpe_value @ self.factorizer.to(self.rpe_value.dtype)
        rpe_v = rpe_v.view(self.head_dim, self.num_heads, 64, 64)
        return torch.einsum('bhqk,dhqk->bhqd', attn_weights, rpe_v)


class Lc0StyleRPE(nn.Module):
    """Per-layer container for lc0-style RPE (Q, K, V components).

    All three are zero-initialized so the model starts with pure content-based
    attention and gradually learns positional signal.
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.rpe_q = Lc0RPELogits(num_heads, head_dim, rpe_type='q')
        self.rpe_k = Lc0RPELogits(num_heads, head_dim, rpe_type='k')
        self.rpe_v = Lc0RPEValue(num_heads, head_dim)


def relative_position_attention_forward(
    query: torch.Tensor,          # [batch, num_heads, seq_len, head_dim]
    key: torch.Tensor,            # [batch, num_heads, seq_len, head_dim]
    value: torch.Tensor,          # [batch, num_heads, seq_len, head_dim]
    rpe: Lc0StyleRPE,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    training: bool = False,
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention with lc0-style content-dependent relative position encodings.

    attn_logits = Q @ K^T + RPE_q(Q) + RPE_k(K)
    attn_logits = attn_logits / sqrt(d_k)
    attn_weights = softmax(attn_logits)
    output = attn_weights @ V + RPE_v(attn_weights)

    RPE only covers board squares (0-63). For sequences longer than 64,
    Q/K are sliced to board-only for RPE computation, and the bias is
    zero-padded back to full sequence length.
    """
    batch_size, num_heads, seq_len, head_dim = query.shape

    # Term 1: Q @ K^T (standard attention)
    attn_scores = torch.matmul(query, key.transpose(-2, -1))

    # RPE terms: content-dependent positional bias (board squares only)
    board_len = min(seq_len, 64)
    q_board = query[:, :, :board_len, :]  # [B, H, <=64, D]
    k_board = key[:, :, :board_len, :]

    # Term 2: RPE_q(Q) -- query content selects spatial pattern
    rpe_q_bias = rpe.rpe_q(q_board)  # [B, H, 64, 64]
    # Term 3: RPE_k(K) -- key content selects spatial pattern
    rpe_k_bias = rpe.rpe_k(k_board)  # [B, H, 64, 64]

    if seq_len > 64:
        # Zero-pad RPE bias to cover metadata tokens
        # Board-to-board region gets RPE, everything else stays zero
        full_bias = attn_scores.new_zeros(batch_size, num_heads, seq_len, seq_len)
        full_bias[:, :, :board_len, :board_len] = rpe_q_bias + rpe_k_bias
        attn_scores = attn_scores + full_bias
    else:
        attn_scores = attn_scores + rpe_q_bias + rpe_k_bias

    # Scale all terms together (lc0 style: 1/sqrt(d_k))
    attn_scores = attn_scores * scaling

    # Soft-cap attention logits to prevent score explosion (à la Gemini 1.5).
    attn_scores = torch.tanh(attn_scores / 50.0) * 50.0

    # Apply attention mask
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)

    # Standard attention output: attn_weights @ V
    attn_output = torch.matmul(attn_weights, value)

    # RPE value bias: position-dependent value contribution weighted by attention
    board_attn = attn_weights[:, :, :board_len, :board_len]  # [B, H, 64, 64]
    rpe_v_out = rpe.rpe_v(board_attn)  # [B, H, 64, D]
    if seq_len > 64:
        # Zero-pad value RPE for metadata positions
        full_rpe_v = torch.zeros_like(attn_output)
        full_rpe_v[:, :, :board_len, :] = rpe_v_out
        attn_output = attn_output + full_rpe_v
    else:
        attn_output = attn_output + rpe_v_out

    # --- Numerical health check (only on first batch element, cheap) ---
    if training and _rpe_diag_enabled:
        with torch.no_grad():
            b0 = slice(0, 1)

            score_max = attn_scores[b0].max().item()
            score_min = attn_scores[b0].min().item()
            score_abs_max = max(abs(score_max), abs(score_min))

            # Per-term breakdown (pre-scaling)
            term1_only = torch.matmul(query[b0], key[b0].transpose(-2, -1))
            term1_max = (term1_only * scaling).abs().max().item()
            rpe_q_max = (rpe_q_bias[:1] * scaling).abs().max().item()
            rpe_k_max = (rpe_k_bias[:1] * scaling).abs().max().item()

            # Per-head stats
            per_head_max = attn_scores[b0, :, :, :].amax(dim=(-2, -1))
            aw_b0 = attn_weights[b0]
            log_aw = torch.log(aw_b0 + 1e-10)
            per_head_entropy = -(aw_b0 * log_aw).sum(dim=-1).mean(dim=-1).squeeze(0)
            per_head_max_weight = aw_b0.amax(dim=(-2, -1)).squeeze(0)

            # Q, K, V vector norms
            q_vec_norm = query[b0].norm(dim=-1).mean().item()
            k_vec_norm = key[b0].norm(dim=-1).mean().item()
            v_vec_norm = value[b0].norm(dim=-1).mean().item()
            q_vec_max_norm = query[b0].norm(dim=-1).max().item()
            k_vec_max_norm = key[b0].norm(dim=-1).max().item()

            _rpe_diagnostics.setdefault('layers', {})[layer_idx] = {
                'score_max': score_max,
                'score_min': score_min,
                'score_abs_max': score_abs_max,
                'term1_max': term1_max,
                'rpe_q_max': rpe_q_max,
                'rpe_k_max': rpe_k_max,
                'per_head_max': per_head_max.tolist(),
                'per_head_entropy': per_head_entropy.tolist(),
                'per_head_max_weight': per_head_max_weight.tolist(),
                'q_vec_norm': q_vec_norm,
                'k_vec_norm': k_vec_norm,
                'v_vec_norm': v_vec_norm,
                'q_vec_max_norm': q_vec_max_norm,
                'k_vec_max_norm': k_vec_max_norm,
                'attn_output_max': attn_output[b0].abs().max().item(),
                'rpe_v_max': rpe_v_out[:1].abs().max().item(),
                'attn_has_nan': bool(torch.isnan(attn_output[b0]).any()),
                'attn_has_inf': bool(torch.isinf(attn_output[b0]).any()),
            }

    # Transpose for output projection: [batch, seq, heads, dim]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ── RPE diagnostics toggle ──────────────────────────────────────────────
# Set _rpe_diag_enabled = True before a forward pass to collect per-layer
# stats into _rpe_diagnostics.  Cheap (first sample only, no grad).
_rpe_diag_enabled: bool = False
_rpe_diagnostics: dict = {}


def make_relative_attention_forward(rpe: Lc0StyleRPE, layer_idx: int = -1):
    """Create a patched attention forward that uses lc0-style content-dependent RPE."""

    def forward_with_relative_pos(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # NOTE: RoPE is disabled -- lc0-style RPE is applied inside attention

        attn_output, attn_weights = relative_position_attention_forward(
            query_states,
            key_states,
            value_states,
            rpe,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            training=self.training,
            layer_idx=layer_idx,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward_with_relative_pos


class MultiTaskAttentionPooling(nn.Module):
    """Multi-task attention pooling with shared K/V projections.

    Computes multiple task-specific outputs in a single forward pass by using
    separate learnable queries for each task while sharing the K/V projections.
    """

    def __init__(self, hidden_size: int, task_output_dims: dict[str, int]) -> None:
        """
        Args:
            hidden_size: Dimension of input hidden states
            task_output_dims: Dict mapping task names to output dimensions
                             e.g., {'policy': 1958, 'winrate': 3}
        """
        super().__init__()
        self.task_names = list(task_output_dims.keys())
        self.num_tasks = len(self.task_names)

        # Shared K/V projections across all tasks
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** -0.5

        # Task-specific queries, norms, and output projections
        self.queries = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, hidden_size))
            for name in self.task_names
        })
        self.norms = nn.ModuleDict({
            name: nn.LayerNorm(hidden_size)
            for name in self.task_names
        })
        self.output_projs = nn.ModuleDict({
            name: nn.Linear(hidden_size, output_dim)
            for name, output_dim in task_output_dims.items()
        })

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            Dict mapping task names to outputs: {task_name: [batch_size, output_dim]}
        """
        # Shared K/V projections (computed once)
        k = self.key_proj(hidden_states)    # [batch, seq_len, hidden]
        v = self.value_proj(hidden_states)  # [batch, seq_len, hidden]

        outputs = {}
        for task_name in self.task_names:
            # Task-specific query
            q = self.queries[task_name].unsqueeze(0)  # [1, 1, hidden]

            # Compute attention weights
            attn_weights = torch.matmul(q, k.transpose(
                1, 2)) * self.scale  # [batch, 1, seq_len]
            attn_weights = F.softmax(attn_weights, dim=-1)

            # Weighted sum of values
            pooled = torch.matmul(attn_weights, v).squeeze(
                1)  # [batch, hidden]

            # Task-specific normalization and projection
            pooled = self.norms[task_name](pooled)
            outputs[task_name] = self.output_projs[task_name](pooled)

        return outputs


DEFAULT_POLICY_LOSS_WEIGHT = POLICY_LOSS_WEIGHT
DEFAULT_WINRATE_LOSS_WEIGHT = WINRATE_LOSS_WEIGHT
DEFAULT_MASKED_TOKEN_LOSS_WEIGHT = MASKED_TOKEN_LOSS_WEIGHT
DEFAULT_MOVE_WINRATE_LOSS_WEIGHT = MOVE_WINRATE_LOSS_WEIGHT
DEFAULT_ILLEGALITY_LOSS_WEIGHT = ILLEGALITY_LOSS_WEIGHT
DEFAULT_CONTROL_MAP_LOSS_WEIGHT = CONTROL_MAP_LOSS_WEIGHT
DEFAULT_MATE_LOSS_WEIGHT = MATE_LOSS_WEIGHT


@dataclass
class ChessPolicyValueOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    policy_logits: torch.Tensor = None
    winrate_logits: torch.Tensor = None
    control_logits: torch.Tensor = None
    illegality_logits: torch.Tensor = None
    move_winrate_logits: torch.Tensor = None
    mate_logits: torch.Tensor = None
    policy_loss: Optional[torch.Tensor] = None
    winrate_loss: Optional[torch.Tensor] = None
    control_map_loss: Optional[torch.Tensor] = None
    illegality_loss: Optional[torch.Tensor] = None
    masked_token_loss: Optional[torch.Tensor] = None
    move_winrate_loss: Optional[torch.Tensor] = None
    mate_loss: Optional[torch.Tensor] = None
    # Metrics (not losses)
    illegality_rate: Optional[torch.Tensor] = None
    illegality_head_accuracy: Optional[torch.Tensor] = None
    masked_token_accuracy: Optional[torch.Tensor] = None
    top1_agreement: Optional[torch.Tensor] = None
    value_mae: Optional[torch.Tensor] = None
    move_winrate_mae: Optional[torch.Tensor] = None
    control_map_mae: Optional[torch.Tensor] = None
    model_entropy: Optional[torch.Tensor] = None
    mate_accuracy: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class ChessPolicyValueModel(LlamaPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.policy_dim = config.policy_dim
        # Get empty token IDs (can be list or None)
        empty_token_ids_list = getattr(config, 'empty_token_ids', None)
        self.empty_token_ids = set(
            empty_token_ids_list) if empty_token_ids_list else None
        self.transformer = LlamaModel(config)
        self._disable_causal_mask()
        self._disable_rope()
        hidden_size = config.hidden_size

        # Per-layer lc0-style content-dependent relative position embeddings.
        # Each layer learns [head_dim * num_heads, 225] weights for Q, K, V RPE.
        # 225 = 15x15 relative displacements on the 8x8 board.
        # Board squares only (0-63); metadata tokens get zero RPE.
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        num_layers = config.num_hidden_layers
        self.rel_pos_embs = nn.ModuleList([
            Lc0StyleRPE(num_heads=num_heads, head_dim=head_dim)
            for _ in range(num_layers)
        ])

        # Replace attention forward in all layers to use relative position embeddings
        self._install_relative_attention()

        # Multi-task attention pooling (shared K/V, task-specific queries)
        # Winrate head predicts win% in 128 bins (0.0 to 1.0)
        # Mate head predicts 5 classes per move (no_mate, mate_in_1, mate_in_2, mate_in_3, mate_in_4_plus)
        self.num_value_bins = 128
        self.num_mate_classes = 5
        self.task_head = MultiTaskAttentionPooling(
            hidden_size=hidden_size,
            task_output_dims={
                'policy': self.policy_dim,  # Used for both softmax policy loss and sigmoid win% loss (includes illegality)
                'winrate': self.num_value_bins,  # 128 bins for win probability
                'mate': self.policy_dim * self.num_mate_classes,  # 5 classes per move, reshaped in forward()
            }
        )

        # Per-square control head: each of the 64 board squares predicts its own attacker counts
        # Applied to hidden states at positions 0-63 (the board tokens)
        # Output: 2 values per square (white attackers, black attackers)
        self.control_head = nn.Linear(hidden_size, 2)

        # Language modeling head for masked token prediction
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)

        self.policy_loss_weight = float(DEFAULT_POLICY_LOSS_WEIGHT)
        self.winrate_loss_weight = float(DEFAULT_WINRATE_LOSS_WEIGHT)
        self.masked_token_loss_weight = float(DEFAULT_MASKED_TOKEN_LOSS_WEIGHT)
        self.move_winrate_loss_weight = float(DEFAULT_MOVE_WINRATE_LOSS_WEIGHT)
        self.illegality_loss_weight = float(DEFAULT_ILLEGALITY_LOSS_WEIGHT)
        self.control_map_loss_weight = float(DEFAULT_CONTROL_MAP_LOSS_WEIGHT)
        self.mate_loss_weight = float(DEFAULT_MATE_LOSS_WEIGHT)
        self.temperature = float(TEMPERATURE)

        # Illegality penalty annealing: start with -5 penalty, anneal to 0 over first 10% of epoch
        # Uses quadratic decay so learning pressure is spread evenly (compensates for exp in softmax)
        self.illegality_penalty_start = -5.0
        self.illegality_penalty_annealing_steps = getattr(config, 'illegality_penalty_annealing_steps', 0)

        self.post_init()

        # Zero-init RPE weights AFTER post_init() to ensure the model starts
        # with pure content-based attention (matches lc0's approach).
        for rpe in self.rel_pos_embs:
            nn.init.zeros_(rpe.rpe_q.rpe)
            nn.init.zeros_(rpe.rpe_k.rpe)
            nn.init.zeros_(rpe.rpe_v.rpe_value)

    # type: ignore[override]
    def load_state_dict(self, state_dict, strict: bool = False):
        # Handle legacy checkpoints with absolute position embeddings
        legacy_key = 'position_embeddings.weight'
        if legacy_key in state_dict:
            print("Warning: Ignoring legacy absolute position embeddings from checkpoint")
            del state_dict[legacy_key]

        # Handle legacy checkpoints with old DeBERTa-style RPE (nn.Embedding based)
        # Old keys look like: rel_pos_embs.0.rel_query_board.weight, rel_pos_embs.0.rel_key_board.weight, etc.
        # Also handle shared rel_pos_emb.* and metadata keys
        old_rpe_keys = [k for k in state_dict if 'rel_pos_emb' in k and (
            'rel_query_board' in k or 'rel_key_board' in k or 'rel_value_board' in k or '_meta' in k
        )]
        if old_rpe_keys:
            print(f"Warning: Ignoring {len(old_rpe_keys)} legacy DeBERTa-style RPE keys from checkpoint "
                  "(model now uses lc0-style RPE, will start with zero-init)")
            for key in old_rpe_keys:
                del state_dict[key]

        # Handle legacy shared rel_pos_emb -> ignored (incompatible with lc0-style)
        shared_keys = [k for k in state_dict if k.startswith('rel_pos_emb.') and 'rel_pos_embs.' not in k]
        if shared_keys:
            print(f"Warning: Ignoring {len(shared_keys)} legacy shared RPE keys from checkpoint")
            for key in list(shared_keys):
                del state_dict[key]

        # Check for lc0-style RPE keys
        lc0_rpe_keys = [k for k in state_dict if 'rel_pos_embs' in k and ('rpe_q.' in k or 'rpe_k.' in k or 'rpe_v.' in k)]
        if not lc0_rpe_keys:
            print("Note: Checkpoint has no lc0-style RPE weights, using zero initialization")

        result = super().load_state_dict(state_dict, strict=strict)

        missing = list(getattr(result, "missing_keys", ()))
        unexpected = list(getattr(result, "unexpected_keys", ()))
        if missing:
            print(
                "load_state_dict: missing keys while loading checkpoint:",
                ", ".join(missing),
            )
        if unexpected:
            print(
                "load_state_dict: unexpected keys while loading checkpoint:",
                ", ".join(unexpected),
            )

        return result

    @classmethod
    def from_pretrained_compiled(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load a model that was saved with torch.compile() applied.

        This handles the _orig_mod. prefix that torch.compile() adds to state dict keys.
        """
        import os
        from transformers import AutoConfig

        # Load config
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # Initialize model
        model = cls(config)

        # Load state dict
        state_dict_path = os.path.join(
            pretrained_model_name_or_path, "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            # Try model.safetensors
            state_dict_path = os.path.join(
                pretrained_model_name_or_path, "model.safetensors")
            if os.path.exists(state_dict_path):
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
            else:
                raise FileNotFoundError(
                    f"Could not find model weights in {pretrained_model_name_or_path}")
        else:
            import torch
            state_dict = torch.load(state_dict_path, map_location="cpu")

        # Strip _orig_mod. prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod."):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        # Load the cleaned state dict
        model.load_state_dict(new_state_dict, strict=False)

        return model

    def _disable_causal_mask(self) -> None:
        for block in self.transformer.layers:
            block.self_attn.is_causal = False

    def _disable_rope(self) -> None:
        """Disable rotary position embeddings - we use relative position embeddings instead."""
        import transformers.models.llama.modeling_llama as llama_module

        def no_op_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return q, k  # Return unchanged

        llama_module.apply_rotary_pos_emb = no_op_rotary

    def _install_relative_attention(self) -> None:
        """Replace LlamaAttention forward methods to use per-layer relative position embeddings."""
        import types

        for i, block in enumerate(self.transformer.layers):
            forward_fn = make_relative_attention_forward(self.rel_pos_embs[i], layer_idx=i)
            block.self_attn.forward = types.MethodType(forward_fn, block.self_attn)

    def forward(
        self,
        input_ids: torch.Tensor,
        policy: Optional[torch.Tensor] = None,
        winrate: Optional[torch.Tensor] = None,
        control_map: Optional[torch.Tensor] = None,
        true_value: Optional[torch.Tensor] = None,
        masked_positions: Optional[torch.Tensor] = None,
        original_input_ids: Optional[torch.Tensor] = None,
        legal_move_mask: Optional[torch.Tensor] = None,
        endgame_weights: Optional[torch.Tensor] = None,
        mate_classes: Optional[torch.Tensor] = None,
        training_step: Optional[int] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> ChessPolicyValueOutput:
        # Convert input_ids to embeddings
        batch_size = input_ids.size(0)
        input_embeds = self.transformer.embed_tokens(input_ids)

        # Relative position embeddings are handled inside attention layers
        # via lc0-style content-dependent RPE (no absolute position embedding addition)

        # Enable per-layer RPE diagnostics during training
        global _rpe_diag_enabled, _rpe_diagnostics
        if self.training:
            _rpe_diag_enabled = True
            _rpe_diagnostics.clear()

        # Process all tokens through transformer
        # Enable output_hidden_states during training for per-layer norm diagnostics (H2)
        if self.training:
            kwargs['output_hidden_states'] = True
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds, **kwargs)
        hidden_states = transformer_outputs.last_hidden_state

        # Collect per-layer hidden state norms for H2 diagnostics
        if self.training:
            _rpe_diag_enabled = False
            with torch.no_grad():
                # H2: Per-layer hidden state norms (tracks residual accumulation)
                if transformer_outputs.hidden_states is not None:
                    per_layer_hs_norms = []
                    for layer_hs in transformer_outputs.hidden_states:
                        # Mean norm across positions for first batch element
                        per_layer_hs_norms.append(layer_hs[:1].norm(dim=-1).mean().item())
                    _rpe_diagnostics['per_layer_hs_norms'] = per_layer_hs_norms

                hs_abs_max = hidden_states.abs().max().item()
                hs_has_nan = bool(torch.isnan(hidden_states).any())
                hs_has_inf = bool(torch.isinf(hidden_states).any())
                if hs_has_nan or hs_has_inf or hs_abs_max > 60000:
                    print(f"\n{'='*70}")
                    print(f"NUMERICAL ISSUE in hidden states: max={hs_abs_max:.1f} nan={hs_has_nan} inf={hs_has_inf}")
                    print(f"{'='*70}\n")

        # Multi-task attention pooling (single forward pass)
        task_outputs = self.task_head(hidden_states)
        # Used for softmax policy loss, sigmoid win% loss, AND illegality prediction
        policy_logits = task_outputs['policy']
        winrate_logits = task_outputs['winrate']

        # Per-square control prediction: apply control head to each board square's hidden state
        # hidden_states[:, :64, :] are the 64 board position tokens
        # Output: [batch, 64, 2] -> reshape to [batch, 128] for compatibility
        board_hidden_states = hidden_states[:, :64, :]  # [batch, 64, hidden]
        control_logits_per_square = self.control_head(board_hidden_states)  # [batch, 64, 2]
        # Reshape to [batch, 128]: first 64 = white counts, last 64 = black counts
        control_logits = torch.cat([
            control_logits_per_square[:, :, 0],  # [batch, 64] white attackers
            control_logits_per_square[:, :, 1],  # [batch, 64] black attackers
        ], dim=1)  # [batch, 128]

        # Mate prediction: 5-class classification per move (from shared attention pooling)
        # Reshape from [batch, policy_dim * 5] to [batch, policy_dim, 5]
        mate_logits_flat = task_outputs['mate']  # [batch, policy_dim * num_mate_classes]
        mate_logits = mate_logits_flat.view(batch_size, self.policy_dim, self.num_mate_classes)

        target_device = policy_logits.device

        # Register backward hooks on key intermediate tensors (training only)
        if self.training:
            hidden_states.register_hook(_make_grad_hook("hidden_states"))
            policy_logits.register_hook(_make_grad_hook("policy_logits"))
            winrate_logits.register_hook(_make_grad_hook("winrate_logits"))
            control_logits.register_hook(_make_grad_hook("control_logits"))
            mate_logits_flat.register_hook(_make_grad_hook("mate_logits_flat"))

        # Forward-pass health check on task head outputs
        if self.training:
            with torch.no_grad():
                _fwd_issues = []
                for _name, _t in [("policy_logits", policy_logits),
                                   ("winrate_logits", winrate_logits),
                                   ("control_logits", control_logits),
                                   ("mate_logits_flat", mate_logits_flat)]:
                    _t_max = _t.abs().max().item()
                    if torch.isnan(_t).any():
                        _fwd_issues.append(f"{_name} has NaN")
                    if torch.isinf(_t).any():
                        _fwd_issues.append(f"{_name} has Inf")
                    if _t_max > 1000:
                        _fwd_issues.append(f"{_name} max={_t_max:.1f}")
                if _fwd_issues:
                    print(f"\n{'='*70}")
                    print(f"FORWARD HEALTH CHECK — task head outputs (step {training_step}):")
                    for _issue in _fwd_issues:
                        print(f"  {_issue}")
                    print(f"{'='*70}\n")

        # Use passed endgame weights for loss upweighting, or default to uniform weights
        if endgame_weights is None:
            endgame_weights = torch.ones(batch_size, device=target_device)
        elif endgame_weights.device != target_device:
            endgame_weights = endgame_weights.to(target_device)

        # Policy head has TWO losses on the SAME logits:
        # Loss 1: Cross-entropy with softmax target distribution (from un-sigmoid'd Stockfish win%)
        # Loss 2: Sigmoid-based win% prediction (encourages correct ranking of all moves + illegality detection)
        policy_loss: Optional[torch.Tensor] = None
        move_winrate_loss: Optional[torch.Tensor] = None
        illegality_loss: Optional[torch.Tensor] = None
        move_winrate_mae: Optional[torch.Tensor] = None
        policy_mask_bool: Optional[torch.Tensor] = None
        model_entropy: Optional[torch.Tensor] = None

        if policy is not None and true_value is not None:
            if policy.device != target_device:
                policy = policy.to(target_device)
            if true_value.device != target_device:
                true_value = true_value.to(target_device)

            # policy contains normalized win%: best move = 0, others negative, illegal = -1
            # Identify legal moves (> -0.99 to distinguish from -1 illegal marker)
            policy_mask_bool = (policy > -0.99).to(dtype=torch.bool)

            # Loss 1: Cross-entropy with softmax target distribution
            # Create target distribution by un-sigmoid'ing Stockfish win% and applying softmax
            # This aligns the softmax and MCE losses to target the same underlying values

            # Convert relative to absolute win%: absolute_win%[move] = true_value + policy[move]
            absolute_winrates = true_value.unsqueeze(1) + policy  # [batch, policy_dim]

            # Un-sigmoid (logit transform): logit(p) = log(p / (1-p))
            # Clamp to avoid log(0) and division by zero
            eps = 1e-7
            clamped_winrates = torch.clamp(absolute_winrates, eps, 1 - eps)

            # Compute target logits (un-sigmoid)
            # For illegal moves, set to very negative value
            target_logits = torch.where(
                policy_mask_bool,
                torch.log(clamped_winrates / (1 - clamped_winrates)),
                torch.full_like(clamped_winrates, -1e9)
            )

            # Apply softmax with temperature to get target probability distribution
            target_probs = F.softmax(target_logits / self.temperature, dim=-1)

            # Apply annealing penalty to illegal moves (helps both policy and BCE losses)
            annealed_logits = policy_logits.clone()

            # Annealing: Start by adding -5 penalty to illegal moves, gradually reduce to 0
            # Uses quadratic decay so learning pressure is spread evenly (compensates for exp in softmax)
            if training_step is not None and self.illegality_penalty_annealing_steps > 0:
                if training_step < self.illegality_penalty_annealing_steps:
                    # Quadratic annealing from -5 to 0 (penalty drops faster initially)
                    progress = training_step / self.illegality_penalty_annealing_steps  # 0 to 1
                    illegality_penalty = self.illegality_penalty_start * (1.0 - progress) ** 2  # -5 to 0
                    # Use torch.where instead of boolean indexing to avoid torch.compile graph break
                    annealed_logits = torch.where(
                        policy_mask_bool,
                        annealed_logits,
                        annealed_logits + illegality_penalty
                    )

            # For softmax: also enforce floor at -1e9 for numerical stability
            # Use torch.where instead of boolean indexing to avoid torch.compile graph break
            masked_logits = torch.where(
                policy_mask_bool,
                annealed_logits,
                torch.clamp(annealed_logits, max=-1e9)
            )

            # Register gradient hook on annealed_logits (feeds both policy CE and BCE losses)
            if self.training:
                annealed_logits.register_hook(_make_grad_hook("annealed_logits"))

            # Use log_softmax (numerically stable — never computes log(0))
            # then derive probs only where needed for metrics.
            # The old pattern `softmax() + log() + 1e-10` is unsafe in bf16:
            # 1e-10 rounds to 0 in bf16 (min positive ~6e-8), so log(0) = -inf.
            log_model_probs = F.log_softmax(masked_logits / self.temperature, dim=-1)
            model_probs = log_model_probs.exp()  # only for metrics (entropy, top1)

            # Cross-entropy loss: -sum(target_probs * log_softmax(model_logits))
            # Compute per-sample loss, apply endgame weights, then take weighted mean
            per_sample_policy_loss = -(target_probs * log_model_probs).sum(dim=-1)  # [batch]
            raw_policy_loss = (per_sample_policy_loss * endgame_weights).sum() / endgame_weights.sum()
            policy_loss = self.policy_loss_weight * raw_policy_loss

            # Compute policy entropy for monitoring saturation (use log_probs we already have)
            model_entropy = -(model_probs * log_model_probs).sum(dim=-1).mean()

            # Loss 2: Sigmoid-based absolute win% prediction for LEGAL moves only
            # Legal moves: target = their absolute win% (e.g., 0.52, 0.48, etc.)
            # absolute_winrates already computed above in Loss 1
            # Temperature is NOT applied here - sigmoid uses the raw logits

            # Compute BCE loss on LEGAL moves only
            # Uses annealed_logits for consistency with policy loss
            per_move_bce = F.binary_cross_entropy_with_logits(
                annealed_logits, absolute_winrates, reduction='none'
            )  # [batch, policy_dim]
            legal_count = policy_mask_bool.float().sum(dim=-1).clamp(min=1)  # [batch]
            per_sample_winrate_loss = (per_move_bce * policy_mask_bool.float()).sum(dim=-1) / legal_count
            raw_move_winrate_loss = (per_sample_winrate_loss * endgame_weights).sum() / endgame_weights.sum()
            move_winrate_loss = self.move_winrate_loss_weight * raw_move_winrate_loss

            # Loss 3: Hinge loss to push illegal move logits below -margin
            # This provides explicit illegality signal without drowning out the legal move BCE
            margin = 5.0
            illegal_mask = ~policy_mask_bool
            # relu(logit + margin) = 0 when logit < -margin, linear penalty otherwise
            per_move_illegality = F.relu(annealed_logits + margin)
            illegal_count = illegal_mask.float().sum(dim=-1).clamp(min=1)  # [batch]
            per_sample_illegality_loss = (per_move_illegality * illegal_mask.float()).sum(dim=-1) / illegal_count
            raw_illegality_loss = (per_sample_illegality_loss * endgame_weights).sum() / endgame_weights.sum()
            illegality_loss = self.illegality_loss_weight * raw_illegality_loss

            # MAE metric for win% predictions - ONLY on legal moves for monitoring
            pred_winrates = torch.sigmoid(annealed_logits)
            mae_per_move = torch.abs(pred_winrates - absolute_winrates)
            total_legal_count = policy_mask_bool.float().sum()
            move_winrate_mae = (mae_per_move * policy_mask_bool.float()).sum() / total_legal_count.clamp(min=1)

            # Forward health check on targets and loss intermediates
            if self.training:
                with torch.no_grad():
                    _fwd_issues = []
                    # Check absolute_winrates range on LEGAL moves only (should be [0,1])
                    # Illegal moves have policy=-1.0 so their absolute_winrates is expected to be negative
                    _legal_aw = absolute_winrates[policy_mask_bool]
                    if _legal_aw.numel() > 0:
                        aw_min = _legal_aw.min().item()
                        aw_max = _legal_aw.max().item()
                        if aw_min < -0.1 or aw_max > 1.1:
                            _fwd_issues.append(f"absolute_winrates (legal) out of range [{aw_min:.4f}, {aw_max:.4f}]")
                    # Check target_logits on legal moves
                    _legal_tl = target_logits[policy_mask_bool]
                    if _legal_tl.numel() > 0:
                        tl_max = _legal_tl.abs().max().item()
                        if tl_max > 50:
                            _fwd_issues.append(f"target_logits (legal) max={tl_max:.1f}")
                    # Check annealed_logits range
                    _legal_al = annealed_logits[policy_mask_bool]
                    if _legal_al.numel() > 0:
                        al_max = _legal_al.abs().max().item()
                        if al_max > 100:
                            _fwd_issues.append(f"annealed_logits (legal) max={al_max:.1f}")
                    # Check individual loss values
                    for _ln, _lv in [("policy_loss", policy_loss), ("move_winrate_loss", move_winrate_loss),
                                     ("illegality_loss", illegality_loss)]:
                        if _lv is not None:
                            _lval = _lv.item()
                            if not math.isfinite(_lval) or _lval > 100:
                                _fwd_issues.append(f"{_ln}={_lval:.4f}")
                    if _fwd_issues:
                        print(f"\n{'='*70}")
                        print(f"FORWARD HEALTH CHECK — targets/losses (step {training_step}):")
                        for _issue in _fwd_issues:
                            print(f"  {_issue}")
                        print(f"{'='*70}\n")

        winrate_loss: Optional[torch.Tensor] = None
        value_mae: Optional[torch.Tensor] = None
        if winrate is not None:
            # Winrate head predicts win% distribution over 128 bins
            # Use cross-entropy loss on smoothed target distribution
            if winrate.device != target_device:
                winrate = winrate.to(target_device)

            # Cross-entropy: -sum(target * log(pred))
            # Target (winrate) is already a normalized distribution from preprocessing
            log_probs = F.log_softmax(winrate_logits, dim=-1)
            per_sample_winrate_loss = -(winrate * log_probs).sum(dim=-1)  # [batch]
            raw_winrate_loss = (per_sample_winrate_loss * endgame_weights).sum() / endgame_weights.sum()
            winrate_loss = self.winrate_loss_weight * raw_winrate_loss

            # MAE metric on expected values for monitoring
            bin_centers = torch.linspace(
                0, 1, self.num_value_bins, device=target_device)
            winrate_probs = F.softmax(winrate_logits, dim=-1)
            predicted_value = (winrate_probs * bin_centers).sum(dim=-1)
            target_value = (winrate * bin_centers).sum(dim=-1)
            value_mae = torch.abs(predicted_value - target_value).mean()

        # Control map loss: predict attacker counts per square for each side
        control_map_loss: Optional[torch.Tensor] = None
        control_map_mae: Optional[torch.Tensor] = None
        if control_map is not None:
            if control_map.device != target_device:
                control_map = control_map.to(target_device)

            # Huber loss on attacker count predictions
            # control_logits: [batch, 128] (64 white + 64 black counts)
            # control_map: [batch, 128] (ground truth counts)
            per_sample_control_loss = F.huber_loss(
                control_logits, control_map, delta=1.0, reduction='none'
            ).mean(dim=-1)  # [batch]
            raw_control_loss = (per_sample_control_loss * endgame_weights).sum() / endgame_weights.sum()
            control_map_loss = self.control_map_loss_weight * raw_control_loss

            # MAE metric for monitoring
            control_map_mae = torch.abs(control_logits - control_map).mean()

        # Mate prediction loss: 5-class classification per legal move
        # Trained on all positions, with 10x weight for actual mate classes to handle imbalance
        mate_loss: Optional[torch.Tensor] = None
        mate_accuracy: Optional[torch.Tensor] = None
        if mate_classes is not None and policy is not None:
            if mate_classes.device != target_device:
                mate_classes = mate_classes.to(target_device)

            # Legal move mask (same as policy > -0.99)
            legal_mask = (policy > -0.99)  # [batch, policy_dim]

            # Flatten for cross-entropy
            mate_logits_flat = mate_logits.view(-1, self.num_mate_classes)  # [batch * policy_dim, 5]
            mate_targets_flat = mate_classes.view(-1)  # [batch * policy_dim]
            legal_mask_flat = legal_mask.view(-1)  # [batch * policy_dim]

            # Only compute loss on legal moves
            if legal_mask_flat.any():
                legal_logits = mate_logits_flat[legal_mask_flat]  # [num_legal, 5]
                legal_targets = mate_targets_flat[legal_mask_flat]  # [num_legal]

                # Class weights: 10x for mate classes (1-4) to handle imbalance
                # ~1/30 legal moves lead to mate, so upweight to make gradient contribution comparable
                mate_class_weights = torch.tensor(
                    [1.0, 10.0, 10.0, 10.0, 10.0], device=target_device
                )
                raw_mate_loss = F.cross_entropy(legal_logits, legal_targets, weight=mate_class_weights)
                mate_loss = self.mate_loss_weight * raw_mate_loss

                # Accuracy metric
                mate_preds = legal_logits.argmax(dim=-1)
                mate_accuracy = (mate_preds == legal_targets).float().mean()

        # Masked token prediction loss (language modeling objective)
        masked_token_loss: Optional[torch.Tensor] = None
        masked_token_accuracy: Optional[torch.Tensor] = None
        if masked_positions is not None and original_input_ids is not None:
            # Only compute loss on positions that were masked
            if masked_positions.any():
                # Get logits for all input tokens
                lm_logits = self.lm_head(hidden_states)

                # Move tensors to same device if needed
                if original_input_ids.device != target_device:
                    original_input_ids = original_input_ids.to(target_device)
                if masked_positions.device != target_device:
                    masked_positions = masked_positions.to(target_device)

                # Flatten for loss computation
                # [batch*seq, vocab]
                lm_logits_flat = lm_logits.view(-1, lm_logits.size(-1))
                original_ids_flat = original_input_ids.view(-1)  # [batch*seq]
                # [batch*seq]
                masked_positions_flat = masked_positions.view(-1)

                # Only compute loss on masked positions
                masked_lm_logits = lm_logits_flat[masked_positions_flat]
                masked_labels = original_ids_flat[masked_positions_flat]

                if masked_labels.numel() > 0:
                    # Compute per-token loss without reduction
                    per_token_loss = F.cross_entropy(
                        masked_lm_logits, masked_labels, reduction='none'
                    )  # [num_masked_tokens]

                    # Reshape to [batch, seq_len] and compute per-sample mean
                    # masked_positions is [batch, seq_len] bool tensor
                    seq_len = masked_positions.size(1)
                    per_sample_masked_loss = torch.zeros(batch_size, device=target_device)

                    # Scatter per-token losses back to samples and average
                    # Create sample indices for each masked token
                    sample_indices = torch.arange(batch_size, device=target_device).unsqueeze(1).expand(-1, seq_len)
                    sample_indices_flat = sample_indices.reshape(-1)[masked_positions_flat]

                    # Aggregate losses per sample
                    per_sample_masked_loss.scatter_add_(0, sample_indices_flat, per_token_loss)
                    tokens_per_sample = masked_positions.float().sum(dim=1).clamp(min=1)
                    per_sample_masked_loss = per_sample_masked_loss / tokens_per_sample

                    # Apply endgame weighting
                    raw_masked_token_loss = (per_sample_masked_loss * endgame_weights).sum() / endgame_weights.sum()
                    masked_token_loss = self.masked_token_loss_weight * raw_masked_token_loss

                    # Compute accuracy only on masked positions that are pieces (not empty squares)
                    masked_preds = masked_lm_logits.argmax(dim=-1)
                    if self.empty_token_ids is not None:
                        # Filter out empty squares - only count accuracy on piece squares
                        # Create tensor of empty token IDs for efficient comparison
                        empty_ids_tensor = torch.tensor(
                            list(self.empty_token_ids),
                            device=masked_labels.device,
                            dtype=masked_labels.dtype
                        )
                        # Create mask: True for non-empty squares
                        non_empty_mask = ~torch.isin(
                            masked_labels, empty_ids_tensor)
                        if non_empty_mask.any():
                            masked_token_accuracy = (
                                (masked_preds[non_empty_mask] ==
                                 masked_labels[non_empty_mask])
                                .float().mean()
                            )
                        # If all masked tokens are empty squares, don't report accuracy
                    else:
                        # Fallback: compute accuracy on all masked positions
                        masked_token_accuracy = (
                            masked_preds == masked_labels).float().mean()


        # Compute metrics for reporting (not used in loss)
        illegality_rate: Optional[torch.Tensor] = None
        top1_agreement: Optional[torch.Tensor] = None

        if policy is not None and policy_mask_bool is not None:
            # Illegality rate: fraction of probability mass on illegal moves (from policy head softmax)
            illegal_mask = (~policy_mask_bool).to(dtype=policy_logits.dtype)
            illegal_probs = F.softmax(policy_logits, dim=-1)
            summed_illegal_prob = (illegal_probs * illegal_mask).sum(dim=-1)
            illegality_rate = summed_illegal_prob.mean()

            # Top-1 agreement: % of time model's top move matches Stockfish's best move
            model_best_move_idx = model_probs.argmax(dim=-1)
            stockfish_best_move_idx = policy.argmax(dim=-1)
            top1_agreement = (model_best_move_idx == stockfish_best_move_idx).float().mean()

        # Register backward hooks on individual loss tensors
        if self.training:
            for loss_name, loss_tensor in [
                ("policy_loss", policy_loss),
                ("move_winrate_loss", move_winrate_loss),
                ("illegality_loss", illegality_loss),
                ("winrate_loss", winrate_loss),
                ("control_map_loss", control_map_loss),
                ("masked_token_loss", masked_token_loss),
                ("mate_loss", mate_loss),
            ]:
                if loss_tensor is not None and loss_tensor.requires_grad:
                    loss_tensor.register_hook(_make_grad_hook(loss_name))

        loss_components = [
            component
            for component in (
                policy_loss,
                move_winrate_loss,
                illegality_loss,
                winrate_loss,
                control_map_loss,
                masked_token_loss,
                mate_loss,
            )
            if component is not None
        ]
        loss: Optional[torch.Tensor] = None
        if loss_components:
            loss = sum(loss_components)

        if not return_dict:
            outputs = (
                policy_logits,
                winrate_logits,
                transformer_outputs.hidden_states,
                transformer_outputs.attentions,
            )
            return ((loss,) + outputs) if loss is not None else outputs

        return ChessPolicyValueOutput(
            loss=loss,
            # Used for softmax policy loss, sigmoid win% loss, AND illegality prediction
            policy_logits=policy_logits,
            winrate_logits=winrate_logits,
            control_logits=control_logits,
            illegality_logits=policy_logits,  # Now unified with policy_logits - sigmoid predicts both win% and legality
            move_winrate_logits=policy_logits,  # Alias - sigmoid of policy_logits used for win% and legality
            mate_logits=mate_logits,  # [batch, policy_dim, 5] mate class predictions per move
            policy_loss=policy_loss,  # Cross-entropy with softmax target from un-sigmoid'd Stockfish win%
            winrate_loss=winrate_loss,
            control_map_loss=control_map_loss,
            illegality_loss=illegality_loss,  # Hinge loss to push illegal logits below margin
            masked_token_loss=masked_token_loss,
            move_winrate_loss=move_winrate_loss,  # Sigmoid win% prediction on legal moves only
            mate_loss=mate_loss,  # Cross-entropy on mate class prediction (legal moves only)
            # Metrics
            illegality_rate=illegality_rate,
            illegality_head_accuracy=None,  # Removed - illegality now implicit in move_winrate_loss
            masked_token_accuracy=masked_token_accuracy,
            top1_agreement=top1_agreement,
            value_mae=value_mae,
            move_winrate_mae=move_winrate_mae,  # Still computed only on legal moves
            control_map_mae=control_map_mae,
            model_entropy=model_entropy,
            mate_accuracy=mate_accuracy,  # Accuracy on mate class prediction
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

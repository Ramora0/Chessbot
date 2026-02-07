from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

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


class ChessRelativePositionEmbedding(nn.Module):
    """Chess-aware relative position embeddings with full Q, K, V biases.

    Board positions (0-63) use 2D relative coordinates (rank_diff, file_diff).
    Metadata positions (64-71) get zero bias (rely on token embeddings for identity).
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        board_size: int = 64,
        max_seq_len: int = 72,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.board_size = board_size
        self.max_seq_len = max_seq_len

        # Board-to-board: 15x15 = 225 unique (rank_diff, file_diff) pairs
        # rank_diff and file_diff each range from -7 to +7
        self.rel_query_board = nn.Embedding(225, num_heads * head_dim)
        self.rel_key_board = nn.Embedding(225, num_heads * head_dim)
        self.rel_value_board = nn.Embedding(225, num_heads * head_dim)

        # Initialize to zeros for stability (matches lc0's approach).
        # Zero init means the model starts with no positional signal and
        # gradually learns it, preventing RPE from dominating content-based
        # attention and causing gradient spikes in deep (20-layer) models.
        nn.init.zeros_(self.rel_query_board.weight)
        nn.init.zeros_(self.rel_key_board.weight)
        nn.init.zeros_(self.rel_value_board.weight)

        # Precompute relative position indices (fixed for chess board)
        self._precompute_indices()

    def _precompute_indices(self):
        """Precompute relative position index matrix for board squares only."""
        # Board indices: convert (rank_diff, file_diff) to single index
        # rank_diff + 7 gives 0-14, file_diff + 7 gives 0-14
        # index = (rank_diff + 7) * 15 + (file_diff + 7)
        board_indices = torch.zeros(self.board_size, self.board_size, dtype=torch.long)
        for i in range(self.board_size):
            rank_i, file_i = i // 8, i % 8
            for j in range(self.board_size):
                rank_j, file_j = j // 8, j % 8
                rank_diff = rank_j - rank_i + 7  # 0-14
                file_diff = file_j - file_i + 7  # 0-14
                board_indices[i, j] = rank_diff * 15 + file_diff

        self.register_buffer('board_indices', board_indices)

    def get_bias_matrices(self, seq_len: int, device: torch.device):
        """Get relative position bias matrices for queries, keys, and values.

        Board-to-board interactions use learned relative position embeddings.
        All other interactions (involving metadata) get zero bias.

        Returns:
            rel_query_bias: [seq_len, seq_len, num_heads, head_dim]
            rel_key_bias: [seq_len, seq_len, num_heads, head_dim]
            rel_value_bias: [seq_len, seq_len, num_heads, head_dim]
        """
        seq_len = min(seq_len, self.max_seq_len)
        board_size = min(self.board_size, seq_len)
        emb_dim = self.num_heads * self.head_dim

        # Initialize full bias matrices with zeros
        rel_query_bias = torch.zeros(seq_len, seq_len, emb_dim, device=device)
        rel_key_bias = torch.zeros(seq_len, seq_len, emb_dim, device=device)
        rel_value_bias = torch.zeros(seq_len, seq_len, emb_dim, device=device)

        # Fill in board-to-board region with learned embeddings
        if board_size > 0:
            board_idx = self.board_indices[:board_size, :board_size]
            rel_query_bias[:board_size, :board_size] = self.rel_query_board(board_idx)
            rel_key_bias[:board_size, :board_size] = self.rel_key_board(board_idx)
            rel_value_bias[:board_size, :board_size] = self.rel_value_board(board_idx)

        # Reshape to [seq, seq, num_heads, head_dim]
        rel_query_bias = rel_query_bias.view(seq_len, seq_len, self.num_heads, self.head_dim)
        rel_key_bias = rel_key_bias.view(seq_len, seq_len, self.num_heads, self.head_dim)
        rel_value_bias = rel_value_bias.view(seq_len, seq_len, self.num_heads, self.head_dim)

        return rel_query_bias, rel_key_bias, rel_value_bias


def relative_position_attention_forward(
    query: torch.Tensor,          # [batch, num_heads, seq_len, head_dim]
    key: torch.Tensor,            # [batch, num_heads, seq_len, head_dim]
    value: torch.Tensor,          # [batch, num_heads, seq_len, head_dim]
    rel_query_bias: torch.Tensor, # [seq_len, seq_len, num_heads, head_dim]
    rel_key_bias: torch.Tensor,   # [seq_len, seq_len, num_heads, head_dim]
    rel_value_bias: torch.Tensor, # [seq_len, seq_len, num_heads, head_dim]
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    training: bool = False,
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention with full Q, K, V relative position representations.

    e_ij = (q_i + a_ij^Q) @ (k_j + a_ij^K)^T * scale
         = (q_i @ k_j^T + q_i @ a_ij^K + a_ij^Q @ k_j^T + a_ij^Q @ a_ij^K) * scale
    z_i = sum_j softmax(e_ij) * (v_j + a_ij^V)
    """
    batch_size, num_heads, seq_len, head_dim = query.shape

    # Permute bias tensors for efficient computation: [heads, seq_q, seq_k, dim]
    rel_query_bias_t = rel_query_bias.permute(2, 0, 1, 3)
    rel_key_bias_t = rel_key_bias.permute(2, 0, 1, 3)
    rel_value_bias_t = rel_value_bias.permute(2, 0, 1, 3)

    # Term 1: q_i @ k_j^T (standard attention)
    # [batch, heads, seq_q, dim] @ [batch, heads, dim, seq_k] -> [batch, heads, seq_q, seq_k]
    attn_scores = torch.matmul(query, key.transpose(-2, -1))

    # Term 2: q_i @ a_ij^K (content-to-position)
    # [batch, heads, seq_q, dim] einsum [heads, seq_q, seq_k, dim] -> [batch, heads, seq_q, seq_k]
    term2 = torch.einsum('bhqd,hqkd->bhqk', query, rel_key_bias_t)
    attn_scores = attn_scores + term2

    # Term 3: a_ij^Q @ k_j^T (position-to-content)
    # [heads, seq_q, seq_k, dim] einsum [batch, heads, seq_k, dim] -> [batch, heads, seq_q, seq_k]
    term3 = torch.einsum('hqkd,bhkd->bhqk', rel_query_bias_t, key)
    attn_scores = attn_scores + term3

    # Term 4 (position-to-position, a^Q @ a^K) deliberately omitted:
    # it's input-independent and universally dropped (DeBERTa, Transformer-XL)

    # Standard 1/sqrt(d) scaling, matching lc0's approach: all three terms
    # (QK^T, Q@RPE_K, RPE_Q@K) are summed then scaled together.
    # With zero-init RPE, only the QK^T term is active early in training,
    # so the 1/sqrt(3) DeBERTa correction would over-dampen initial attention.
    attn_scores = attn_scores * scaling

    # Soft-cap attention logits to prevent score explosion (à la Gemini 1.5).
    # Bounds total attention score regardless of which term (QK, Q·RPE, RPE·K) produces it.
    attn_scores = torch.tanh(attn_scores / 50.0) * 50.0

    # Apply attention mask
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)

    # Standard attention output: attn_weights @ V
    attn_output = torch.matmul(attn_weights, value)  # [batch, heads, seq_q, dim]

    # Relative position value bias term: attn_weights @ a_ij^V
    rel_value_output = torch.einsum('bhqk,hqkd->bhqd', attn_weights, rel_value_bias_t)
    attn_output = attn_output + rel_value_output

    # --- Numerical health check (only on first batch element, cheap) ---
    if training and _rpe_diag_enabled:
        with torch.no_grad():
            _rpe_diagnostics.setdefault('layers', {})[layer_idx] = {
                'qk_max': attn_scores[:1].abs().max().item(),
                'term2_max': term2[:1].abs().max().item(),
                'term3_max': term3[:1].abs().max().item(),
                'attn_output_max': attn_output[:1].abs().max().item(),
                'rel_value_max': rel_value_output[:1].abs().max().item(),
                'value_max': value[:1].abs().max().item(),
                'attn_has_nan': bool(torch.isnan(attn_output[:1]).any()),
                'attn_has_inf': bool(torch.isinf(attn_output[:1]).any()),
            }

    # Transpose for output projection: [batch, seq, heads, dim]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ── RPE diagnostics toggle ──────────────────────────────────────────────
# Set _rpe_diag_enabled = True before a forward pass to collect per-layer
# stats into _rpe_diagnostics.  Cheap (first sample only, no grad).
_rpe_diag_enabled: bool = False
_rpe_diagnostics: dict = {}


def make_relative_attention_forward(rel_pos_emb: ChessRelativePositionEmbedding, layer_idx: int = -1):
    """Create a patched attention forward that uses full Q, K, V relative positions."""

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

        # NOTE: RoPE is disabled, so we skip apply_rotary_pos_emb
        # Full Q, K, V relative position bias is computed here instead

        seq_len = hidden_states.size(1)
        rel_query_bias, rel_key_bias, rel_value_bias = rel_pos_emb.get_bias_matrices(
            seq_len, hidden_states.device
        )

        attn_output, attn_weights = relative_position_attention_forward(
            query_states,
            key_states,
            value_states,
            rel_query_bias,
            rel_key_bias,
            rel_value_bias,
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

        # Per-layer relative position embeddings with full Q, K, V biases
        # Board positions (0-63) use 2D relative coordinates (rank_diff, file_diff)
        # Each layer gets its own embeddings to avoid 20x gradient accumulation
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        num_layers = config.num_hidden_layers
        self.rel_pos_embs = nn.ModuleList([
            ChessRelativePositionEmbedding(
                num_heads=num_heads,
                head_dim=head_dim,
                board_size=64,
                max_seq_len=72,
            )
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
        
        # Zero-init RPE embeddings AFTER post_init(), which re-inits all
        # nn.Embedding modules with normal(0, initializer_range).
        # Zero init is critical: matches lc0, lets the model start with
        # pure content-based attention and gradually learn positional signal.
        for rpe in self.rel_pos_embs:
            nn.init.zeros_(rpe.rel_query_board.weight)
            nn.init.zeros_(rpe.rel_key_board.weight)
            nn.init.zeros_(rpe.rel_value_board.weight)

    # type: ignore[override]
    def load_state_dict(self, state_dict, strict: bool = False):
        # Handle legacy checkpoints with absolute position embeddings
        legacy_key = 'position_embeddings.weight'
        if legacy_key in state_dict:
            print("Warning: Ignoring legacy absolute position embeddings from checkpoint")
            del state_dict[legacy_key]

        # Handle legacy checkpoints with metadata relative position embeddings (now removed)
        legacy_meta_keys = [k for k in state_dict if 'rel_pos_emb' in k and '_meta' in k]
        for key in legacy_meta_keys:
            print(f"Warning: Ignoring legacy metadata position embedding: {key}")
            del state_dict[key]

        # Handle legacy shared rel_pos_emb -> broadcast to per-layer rel_pos_embs
        shared_keys = [k for k in state_dict if k.startswith('rel_pos_emb.') and 'rel_pos_embs.' not in k]
        if shared_keys:
            num_layers = len(self.rel_pos_embs)
            print(f"Broadcasting legacy shared position embeddings to {num_layers} per-layer embeddings")
            for key in list(shared_keys):
                suffix = key[len('rel_pos_emb.'):]  # e.g. 'rel_query_board.weight'
                for i in range(num_layers):
                    state_dict[f'rel_pos_embs.{i}.{suffix}'] = state_dict[key].clone()
                del state_dict[key]

        # Check for relative position keys
        rel_pos_keys = [k for k in state_dict if 'rel_pos_emb' in k]
        if not rel_pos_keys:
            print("Note: Checkpoint has no relative position embeddings, using random initialization")

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

        # Relative position embeddings are now handled inside attention layers
        # via ChessRelativePositionEmbedding (no absolute position embedding addition)

        # Enable per-layer RPE diagnostics during training
        global _rpe_diag_enabled, _rpe_diagnostics
        if self.training:
            _rpe_diag_enabled = True
            _rpe_diagnostics.clear()

        # Process all tokens through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds, **kwargs)
        hidden_states = transformer_outputs.last_hidden_state

        # Check for numerical issues in hidden states coming out of the transformer
        if self.training:
            _rpe_diag_enabled = False
            with torch.no_grad():
                hs_abs_max = hidden_states.abs().max().item()
                hs_has_nan = bool(torch.isnan(hidden_states).any())
                hs_has_inf = bool(torch.isinf(hidden_states).any())
                if hs_has_nan or hs_has_inf or hs_abs_max > 60000:
                    print(f"\n{'='*70}")
                    print(f"NUMERICAL ISSUE in hidden states: max={hs_abs_max:.1f} nan={hs_has_nan} inf={hs_has_inf}")
                    print(f"Input embed max: {input_embeds.abs().max().item():.4f}")
                    # Dump per-layer attention diagnostics
                    for layer_id in sorted(_rpe_diagnostics.get('layers', {}).keys()):
                        d = _rpe_diagnostics['layers'][layer_id]
                        print(f"  Layer {layer_id:2d}: "
                              f"qk_max={d['qk_max']:.1f}  "
                              f"term2_max={d['term2_max']:.1f}  "
                              f"term3_max={d['term3_max']:.1f}  "
                              f"attn_out_max={d['attn_output_max']:.1f}  "
                              f"rpe_v_max={d['rel_value_max']:.1f}  "
                              f"V_max={d['value_max']:.1f}  "
                              f"nan={d['attn_has_nan']}  inf={d['attn_has_inf']}")
                    # RPE weight norms
                    for i, rpe in enumerate(self.rel_pos_embs):
                        q_norm = rpe.rel_query_board.weight.norm().item()
                        k_norm = rpe.rel_key_board.weight.norm().item()
                        v_norm = rpe.rel_value_board.weight.norm().item()
                        print(f"  RPE weights layer {i:2d}: Q_norm={q_norm:.4f}  K_norm={k_norm:.4f}  V_norm={v_norm:.4f}")
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

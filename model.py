from __future__ import annotations

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
    WDL_LOSS_WEIGHT,
    ILLEGALITY_HEAD_LOSS_WEIGHT,
    MASKED_TOKEN_LOSS_WEIGHT,
    MOVE_WINRATE_LOSS_WEIGHT,
)


def compute_geometric_bias_matrix(
    bias_params: torch.Tensor,  # [num_heads, 15, 15]
    device: torch.device
) -> torch.Tensor:  # [num_heads, 70, 70]
    """
    Pre-compute 70×70 attention bias matrix from 15×15 displacement biases.

    For board squares i→j, computes displacement (Δrow, Δcol) and looks up
    bias_params[:, Δrow+7, Δcol+7]. Metadata tokens (64-69) get zero bias.

    Example: e2 (pos 54) → e4 (pos 38): Δrow=-2, Δcol=0 → bias_params[:, 5, 7]
    """
    num_heads = bias_params.shape[0]
    bias_matrix = torch.zeros(num_heads, 70, 70, device=device)

    # For board squares (0-63): compute geometric biases
    for i in range(64):
        row_i, col_i = i // 8, i % 8  # Position 0 = (0,0) = a8

        for j in range(64):
            row_j, col_j = j // 8, j % 8

            # Displacement: how to get from square i to square j
            # Directional: i→j uses different bias than j→i
            row_offset = row_j - row_i  # Range: [-7, 7]
            col_offset = col_j - col_i  # Range: [-7, 7]

            # Map to bias matrix indices [0, 14]
            bias_row = row_offset + 7
            bias_col = col_offset + 7

            # Lookup bias for all heads
            bias_matrix[:, i, j] = bias_params[:, bias_row, bias_col]

    # Metadata tokens (64-69): remain zero (no geometric constraints)
    # This allows turn/castling/en passant to attend freely

    return bias_matrix


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
                             e.g., {'policy': 1858, 'wdl': 3}
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
DEFAULT_WDL_LOSS_WEIGHT = WDL_LOSS_WEIGHT
DEFAULT_ILLEGALITY_HEAD_LOSS_WEIGHT = ILLEGALITY_HEAD_LOSS_WEIGHT
DEFAULT_MASKED_TOKEN_LOSS_WEIGHT = MASKED_TOKEN_LOSS_WEIGHT
DEFAULT_MOVE_WINRATE_LOSS_WEIGHT = MOVE_WINRATE_LOSS_WEIGHT


@dataclass
class ChessPolicyValueOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    policy_logits: torch.Tensor = None
    wdl_logits: torch.Tensor = None
    illegality_logits: torch.Tensor = None
    move_winrate_logits: torch.Tensor = None
    policy_loss: Optional[torch.Tensor] = None
    wdl_loss: Optional[torch.Tensor] = None
    illegality_head_loss: Optional[torch.Tensor] = None
    masked_token_loss: Optional[torch.Tensor] = None
    move_winrate_loss: Optional[torch.Tensor] = None
    # Metrics (not losses)
    illegality_rate: Optional[torch.Tensor] = None
    illegality_head_accuracy: Optional[torch.Tensor] = None
    masked_token_accuracy: Optional[torch.Tensor] = None
    best_move_prob: Optional[torch.Tensor] = None
    value_mae: Optional[torch.Tensor] = None
    move_winrate_mae: Optional[torch.Tensor] = None
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
        hidden_size = config.hidden_size

        # Learned positional embeddings for all tokens
        # Tokens 0-63: board positions (a8, b8, ..., h1)
        # Token 64: turn (w/b)
        # Tokens 65-68: castling rights (K, Q, k, q)
        # Token 69: en passant square
        self.position_embeddings = nn.Embedding(70, hidden_size)

        # Geometric attention biases: 15×15 for all displacement vectors
        # Per-head biases allow each head to learn different spatial relationships
        # (e.g., diagonal moves, knight moves, rank/file relationships)
        num_heads = config.num_attention_heads  # 8
        self.geometric_attention_bias = nn.Parameter(
            torch.zeros(num_heads, 15, 15)
        )

        # Buffer to store pre-computed 70×70 bias matrix (recomputed if device changes)
        self.register_buffer(
            '_geometric_bias_matrix',
            torch.zeros(num_heads, 70, 70),
            persistent=False  # Don't save to checkpoint, recompute on load
        )
        self._bias_matrix_initialized = False

        # Multi-task attention pooling (shared K/V, task-specific queries)
        # WDL head now predicts win% in 128 bins (0.0 to 1.0)
        self.num_value_bins = 128
        self.task_head = MultiTaskAttentionPooling(
            hidden_size=hidden_size,
            task_output_dims={
                'policy': self.policy_dim,  # Used for both softmax policy loss and sigmoid win% loss
                'wdl': self.num_value_bins,  # 128 bins for win probability
                'illegality': self.policy_dim,  # Separate head: predict legality for each move
            }
        )

        # Language modeling head for masked token prediction
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)

        self.policy_loss_weight = float(DEFAULT_POLICY_LOSS_WEIGHT)
        self.wdl_loss_weight = float(DEFAULT_WDL_LOSS_WEIGHT)
        self.illegality_head_loss_weight = float(
            DEFAULT_ILLEGALITY_HEAD_LOSS_WEIGHT)
        self.masked_token_loss_weight = float(DEFAULT_MASKED_TOKEN_LOSS_WEIGHT)
        self.move_winrate_loss_weight = float(DEFAULT_MOVE_WINRATE_LOSS_WEIGHT)
        self.post_init()

    # type: ignore[override]
    def load_state_dict(self, state_dict, strict: bool = False):
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
        """Disable causal masking and inject geometric attention biases."""
        for layer_idx, block in enumerate(self.transformer.layers):
            block.self_attn.is_causal = False

            # Store reference to inject biases in attention forward pass
            # We'll monkey-patch the attention module's forward method
            original_forward = block.self_attn.forward

            def forward_with_geometric_bias(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                **kwargs
            ):
                # Get model's geometric bias matrix
                # Access parent model via closure (self refers to ChessPolicyValueModel)
                geometric_bias = self._geometric_bias_matrix  # [num_heads, 70, 70]

                # Add geometric bias to attention_mask
                # attention_mask shape: [batch, 1, seq_len, seq_len] or None
                # geometric_bias shape: [num_heads, 70, 70]

                if attention_mask is None:
                    # Create attention_mask from geometric bias
                    batch_size = hidden_states.shape[0]
                    seq_len = hidden_states.shape[1]
                    # Broadcast: [1, num_heads, seq_len, seq_len]
                    attention_mask = geometric_bias.unsqueeze(0).expand(
                        batch_size, -1, seq_len, seq_len
                    )
                else:
                    # Add geometric bias to existing mask
                    # attention_mask is [batch, 1, seq_len, seq_len]
                    # Expand to [batch, num_heads, seq_len, seq_len]
                    attention_mask = attention_mask.expand(
                        -1, geometric_bias.shape[0], -1, -1
                    )
                    attention_mask = attention_mask + geometric_bias.unsqueeze(0)

                # Call original attention forward with modified mask
                return original_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs
                )

            # Replace the forward method
            block.self_attn.forward = forward_with_geometric_bias

    def forward(
        self,
        input_ids: torch.Tensor,
        policy: Optional[torch.Tensor] = None,
        wdl: Optional[torch.Tensor] = None,
        true_value: Optional[torch.Tensor] = None,
        masked_positions: Optional[torch.Tensor] = None,
        original_input_ids: Optional[torch.Tensor] = None,
        legal_move_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> ChessPolicyValueOutput:
        # Convert input_ids to embeddings
        batch_size = input_ids.size(0)
        input_embeds = self.transformer.embed_tokens(input_ids)
        original_seq_len = input_embeds.size(1)

        # Add learned positional embeddings to all tokens
        # This allows the model to instantly know each token's position (e.g., knight on f1 = position 61)
        position_ids = torch.arange(original_seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
        position_embeds = self.position_embeddings(position_ids)  # [batch_size, seq_len, hidden_size]
        input_embeds = input_embeds + position_embeds

        # Lazily initialize geometric bias matrix on first forward pass
        if not self._bias_matrix_initialized:
            self._geometric_bias_matrix = compute_geometric_bias_matrix(
                self.geometric_attention_bias,
                device=input_embeds.device
            )
            self._bias_matrix_initialized = True

        # Ensure bias matrix is on correct device (handles multi-GPU)
        if self._geometric_bias_matrix.device != input_embeds.device:
            self._geometric_bias_matrix = self._geometric_bias_matrix.to(input_embeds.device)
            self._bias_matrix_initialized = True  # Mark as initialized for new device

        # Process all tokens through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds, **kwargs)
        hidden_states = transformer_outputs.last_hidden_state

        # Multi-task attention pooling (single forward pass)
        task_outputs = self.task_head(hidden_states)
        # Used for both softmax policy loss and sigmoid win% loss
        policy_logits = task_outputs['policy']
        wdl_logits = task_outputs['wdl']
        # Separate head for legality prediction
        illegality_logits = task_outputs['illegality']

        target_device = policy_logits.device

        # Policy head has TWO losses on the SAME logits:
        # Loss 1: Softmax-based expected regret (original policy loss)
        # Loss 2: Sigmoid-based win% prediction (encourages correct ranking of all moves)
        policy_loss: Optional[torch.Tensor] = None
        move_winrate_loss: Optional[torch.Tensor] = None
        move_winrate_mae: Optional[torch.Tensor] = None
        policy_mask_bool: Optional[torch.Tensor] = None

        if policy is not None:
            if policy.device != target_device:
                policy = policy.to(target_device)

            # policy contains normalized win%: best move = 0, others negative, illegal = -1
            # Identify legal moves (> -0.99 to distinguish from -1 illegal marker)
            policy_mask_bool = (policy > -0.99).to(dtype=torch.bool)

            # Loss 1: Original policy loss (softmax-based expected regret)
            # Illegal moves get large negative value (treated as very bad)
            win_values = torch.where(
                policy_mask_bool, policy, torch.full_like(policy, -1.0))

            # Compute model's move probabilities (softmax over all moves)
            # Mask illegal moves with large negative logits before softmax
            masked_logits = policy_logits.clone()
            masked_logits[~policy_mask_bool] = -1e9
            model_probs = F.softmax(masked_logits, dim=-1)

            # Expected win% lost = sum(model_prob * win_value) for each position
            # win_value is 0 for best move, negative for others
            # Loss = -expected_value = expected regret (positive when not picking best move)
            expected_win_value = (model_probs * win_values).sum(dim=-1)
            raw_policy_loss = -expected_win_value.mean()
            policy_loss = self.policy_loss_weight * raw_policy_loss

            # Loss 2: Sigmoid-based win% prediction (only on legal moves)
            # This encourages the model to rank ALL moves correctly, not just pick the best one
            # Recover original win% values from normalized policy
            # Best move has value 0
            max_p_win = -policy[policy_mask_bool].min()
            true_winrates = policy.clone()
            true_winrates[policy_mask_bool] = policy[policy_mask_bool] + max_p_win

            # Compute BCE loss only on legal moves using logits (safe for autocast)
            legal_true_winrates = true_winrates[policy_mask_bool]
            legal_policy_logits = policy_logits[policy_mask_bool]

            raw_move_winrate_loss = F.binary_cross_entropy_with_logits(
                legal_policy_logits, legal_true_winrates, reduction='mean'
            )
            move_winrate_loss = self.move_winrate_loss_weight * raw_move_winrate_loss

            # MAE metric for win% predictions (apply sigmoid for metric only)
            legal_pred_winrates = torch.sigmoid(legal_policy_logits)
            move_winrate_mae = torch.abs(
                legal_pred_winrates - legal_true_winrates).mean()

        wdl_loss: Optional[torch.Tensor] = None
        value_mae: Optional[torch.Tensor] = None
        if wdl is not None:
            # WDL head now predicts win% distribution over 128 bins
            # Use Huber loss on expected values (smooth, distance-aware)
            if wdl.device != target_device:
                wdl = wdl.to(target_device)

            # Compute predicted win% as weighted average over bins
            bin_centers = torch.linspace(
                0, 1, self.num_value_bins, device=target_device)
            wdl_probs = F.softmax(wdl_logits, dim=-1)
            predicted_value = (wdl_probs * bin_centers).sum(dim=-1)

            # Use true_value if provided, otherwise fall back to distribution center
            if true_value is not None:
                if true_value.device != target_device:
                    true_value = true_value.to(target_device)
                target_value = true_value
            else:
                # Fallback: compare to center of target distribution
                target_value = (wdl * bin_centers).sum(dim=-1)

            # Huber loss: smooth L1 that's quadratic for small errors, linear for large
            # This is smooth AND cares about distance between bins
            raw_wdl_loss = F.huber_loss(
                predicted_value, target_value, delta=0.1)
            wdl_loss = self.wdl_loss_weight * raw_wdl_loss

            # MAE metric (same as before)
            value_mae = torch.abs(predicted_value - target_value).mean()

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
                    raw_masked_token_loss = F.cross_entropy(
                        masked_lm_logits, masked_labels, reduction='mean'
                    )
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

        # Illegality head loss (SEPARATE head - binary cross-entropy for legality prediction)
        illegality_head_loss: Optional[torch.Tensor] = None
        illegality_head_accuracy: Optional[torch.Tensor] = None
        if policy is not None and policy_mask_bool is not None:
            # Target: 1 for legal moves, 0 for illegal moves
            illegality_target = policy_mask_bool.float()

            # Compute BCE loss with logits on SEPARATE illegality head
            raw_illegality_head_loss = F.binary_cross_entropy_with_logits(
                illegality_logits, illegality_target, reduction='mean'
            )
            illegality_head_loss = self.illegality_head_loss_weight * raw_illegality_head_loss

            # Compute % of legal moves marked as legal (recall for legal class)
            illegality_preds = (torch.sigmoid(illegality_logits) > 0.5).float()
            legal_mask = (illegality_target == 1)
            if legal_mask.any():
                illegality_head_accuracy = illegality_preds[legal_mask].mean()

        # Compute metrics for reporting (not used in loss)
        illegality_rate: Optional[torch.Tensor] = None
        best_move_prob: Optional[torch.Tensor] = None

        if policy is not None and policy_mask_bool is not None:
            # Illegality rate: fraction of probability mass on illegal moves (from policy head softmax)
            illegal_mask = (~policy_mask_bool).to(dtype=policy_logits.dtype)
            illegal_probs = F.softmax(policy_logits, dim=-1)
            summed_illegal_prob = (illegal_probs * illegal_mask).sum(dim=-1)
            illegality_rate = summed_illegal_prob.mean()

            # Best move probability: average probability assigned to Stockfish's best move
            stockfish_best_move_idx = policy.argmax(dim=-1)
            best_move_prob = model_probs.gather(
                1, stockfish_best_move_idx.unsqueeze(1)).squeeze(1).mean()

        loss_components = [
            component
            for component in (policy_loss, move_winrate_loss, wdl_loss, illegality_head_loss, masked_token_loss)
            if component is not None
        ]
        loss: Optional[torch.Tensor] = None
        if loss_components:
            loss = sum(loss_components)

        if not return_dict:
            outputs = (
                policy_logits,
                wdl_logits,
                illegality_logits,
                transformer_outputs.hidden_states,
                transformer_outputs.attentions,
            )
            return ((loss,) + outputs) if loss is not None else outputs

        return ChessPolicyValueOutput(
            loss=loss,
            # Used for both softmax policy loss and sigmoid win% loss
            policy_logits=policy_logits,
            wdl_logits=wdl_logits,
            illegality_logits=illegality_logits,  # SEPARATE head for legality prediction
            move_winrate_logits=policy_logits,  # Alias - sigmoid of policy_logits used for win%
            policy_loss=policy_loss,  # Original softmax-based expected regret loss
            wdl_loss=wdl_loss,
            illegality_head_loss=illegality_head_loss,
            masked_token_loss=masked_token_loss,
            move_winrate_loss=move_winrate_loss,  # New sigmoid-based win% prediction loss
            illegality_rate=illegality_rate,
            illegality_head_accuracy=illegality_head_accuracy,
            masked_token_accuracy=masked_token_accuracy,
            best_move_prob=best_move_prob,
            value_mae=value_mae,
            move_winrate_mae=move_winrate_mae,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

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
    WINRATE_LOSS_WEIGHT,
    MASKED_TOKEN_LOSS_WEIGHT,
    MOVE_WINRATE_LOSS_WEIGHT,
    ILLEGALITY_LOSS_WEIGHT,
    CONTROL_MAP_LOSS_WEIGHT,
    MATE_LOSS_WEIGHT,
    TEMPERATURE,
)


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

        # Learned positional embeddings for all tokens
        # Tokens 0-63: board positions (a8, b8, ..., h1)
        # Token 64: turn (w/b)
        # Tokens 65-68: castling rights (K, Q, k, q)
        # Token 69: en passant square
        # Token 70: halfmove clock
        # Token 71: fullmove number
        self.position_embeddings = nn.Embedding(72, hidden_size)

        # Multi-task attention pooling (shared K/V, task-specific queries)
        # Winrate head predicts win% in 128 bins (0.0 to 1.0)
        # Mate head predicts 7 classes per move (mated_4+, mated_2-3, mated_1, no_mate, mate_1, mate_2-3, mate_4+)
        self.num_value_bins = 128
        self.num_mate_classes = 7
        self.task_head = MultiTaskAttentionPooling(
            hidden_size=hidden_size,
            task_output_dims={
                'policy': self.policy_dim,  # Used for both softmax policy loss and sigmoid win% loss (includes illegality)
                'winrate': self.num_value_bins,  # 128 bins for win probability
                'mate': 128,  # Bottleneck dim, expanded by self.mate_expand
            }
        )
        self.mate_expand = nn.Linear(128, self.policy_dim * self.num_mate_classes)

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
        for block in self.transformer.layers:
            block.self_attn.is_causal = False

    def _disable_rope(self) -> None:
        """Disable rotary position embeddings - we use learned positional embeddings instead."""
        import transformers.models.llama.modeling_llama as llama_module

        def no_op_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return q, k  # Return unchanged

        llama_module.apply_rotary_pos_emb = no_op_rotary

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
        mate_depths: Optional[torch.Tensor] = None,
        training_step: Optional[int] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> ChessPolicyValueOutput:
        # Convert input_ids to embeddings
        batch_size = input_ids.size(0)
        input_embeds = self.transformer.embed_tokens(input_ids)
        original_seq_len = input_embeds.size(1)

        # Add learned positional embeddings to all tokens
        # This allows the model to instantly know each token's position (e.g., knight on f1 = position 61)
        position_ids = torch.arange(
            original_seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(
            batch_size, -1)  # [batch_size, seq_len]
        position_embeds = self.position_embeddings(
            position_ids)  # [batch_size, seq_len, hidden_size]
        input_embeds = input_embeds + position_embeds

        # Process all tokens through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds, **kwargs)
        hidden_states = transformer_outputs.last_hidden_state

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

        # Mate prediction: 7-class classification per move (from shared attention pooling via bottleneck)
        mate_bottleneck = task_outputs['mate']  # [batch, 128]
        mate_logits_flat = self.mate_expand(mate_bottleneck)  # [batch, policy_dim * num_mate_classes]
        mate_logits = mate_logits_flat.reshape(batch_size, self.policy_dim, self.num_mate_classes) # .contiguous()

        target_device = policy_logits.device

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
            eps = 1e-3
            clamped_winrates = torch.clamp(absolute_winrates, eps, 1 - eps)

            # Un-sigmoid to get base target logits
            base_target_logits = torch.log(clamped_winrates / (1 - clamped_winrates))

            # Add mate bonus: 10.0 / mate_depth for non-zero depths
            if mate_depths is not None:
                if mate_depths.device != target_device:
                    mate_depths = mate_depths.to(target_device)
                md_float = mate_depths.float()
                has_mate = mate_depths != 0
                safe_depths = torch.where(has_mate, md_float, torch.ones_like(md_float))
                mate_bonus = torch.where(has_mate, 10.0 / safe_depths, torch.zeros_like(md_float))
                base_target_logits = base_target_logits + mate_bonus

            # Apply illegal move mask
            target_logits = torch.where(
                policy_mask_bool,
                base_target_logits,
                torch.full_like(base_target_logits, -1e9)
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

            model_probs = F.softmax(masked_logits / self.temperature, dim=-1)

            # Cross-entropy loss: -sum(target_probs * log(model_probs))
            # Compute per-sample loss, apply endgame weights, then take weighted mean
            per_sample_policy_loss = -(target_probs * torch.log(model_probs + 1e-10)).sum(dim=-1)  # [batch]
            raw_policy_loss = (per_sample_policy_loss * endgame_weights).sum() / endgame_weights.sum()
            policy_loss = self.policy_loss_weight * raw_policy_loss

            # Compute policy entropy for monitoring saturation
            model_entropy = -(model_probs * torch.log(model_probs + 1e-10)).sum(dim=-1).mean()

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

        # Mate prediction loss: 7-class classification per legal move
        # Classes: 0=mated_4+, 1=mated_2-3, 2=mated_1, 3=no_mate, 4=mate_1, 5=mate_2-3, 6=mate_4+
        mate_loss: Optional[torch.Tensor] = None
        mate_accuracy: Optional[torch.Tensor] = None
        if mate_depths is not None and policy is not None:
            if mate_depths.device != target_device:
                mate_depths = mate_depths.to(target_device)

            # Vectorized class conversion from raw depths
            md = mate_depths.long()
            mate_classes = torch.full_like(md, 3)  # default: no_mate
            mate_classes = torch.where(md <= -4, torch.tensor(0, device=target_device), mate_classes)
            mate_classes = torch.where((md >= -3) & (md <= -2), torch.tensor(1, device=target_device), mate_classes)
            mate_classes = torch.where(md == -1, torch.tensor(2, device=target_device), mate_classes)
            mate_classes = torch.where(md == 1, torch.tensor(4, device=target_device), mate_classes)
            mate_classes = torch.where((md >= 2) & (md <= 3), torch.tensor(5, device=target_device), mate_classes)
            mate_classes = torch.where(md >= 4, torch.tensor(6, device=target_device), mate_classes)

            # Legal move mask (same as policy > -0.99)
            legal_mask = (policy > -0.99)  # [batch, policy_dim]

            # Flatten for cross-entropy
            mate_logits_flat = mate_logits.reshape(-1, self.num_mate_classes)  # [batch * policy_dim, 7]
            mate_targets_flat = mate_classes.reshape(-1)  # [batch * policy_dim]
            legal_mask_flat = legal_mask.reshape(-1)  # [batch * policy_dim]

            # Only compute loss on legal moves
            if legal_mask_flat.any():
                legal_logits = mate_logits_flat[legal_mask_flat]  # [num_legal, 7]
                legal_targets = mate_targets_flat[legal_mask_flat]  # [num_legal]

                # Class weights: 10x for mate/mated classes to handle imbalance
                mate_class_weights = torch.tensor(
                    [10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0], device=target_device
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
                lm_logits_flat = lm_logits.reshape(-1, lm_logits.size(-1))
                original_ids_flat = original_input_ids.reshape(-1)  # [batch*seq]
                # [batch*seq]
                masked_positions_flat = masked_positions.reshape(-1)

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
            mate_logits=mate_logits,  # [batch, policy_dim, 7] mate class predictions per move
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

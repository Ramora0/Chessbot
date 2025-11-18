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
            attn_weights = torch.matmul(q, k.transpose(1, 2)) * self.scale  # [batch, 1, seq_len]
            attn_weights = F.softmax(attn_weights, dim=-1)

            # Weighted sum of values
            pooled = torch.matmul(attn_weights, v).squeeze(1)  # [batch, hidden]

            # Task-specific normalization and projection
            pooled = self.norms[task_name](pooled)
            outputs[task_name] = self.output_projs[task_name](pooled)

        return outputs


DEFAULT_POLICY_LOSS_WEIGHT = POLICY_LOSS_WEIGHT
DEFAULT_WDL_LOSS_WEIGHT = WDL_LOSS_WEIGHT
DEFAULT_ILLEGALITY_HEAD_LOSS_WEIGHT = ILLEGALITY_HEAD_LOSS_WEIGHT
DEFAULT_MASKED_TOKEN_LOSS_WEIGHT = MASKED_TOKEN_LOSS_WEIGHT


@dataclass
class ChessPolicyValueOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    policy_logits: torch.Tensor = None
    wdl_logits: torch.Tensor = None
    illegality_logits: torch.Tensor = None
    policy_loss: Optional[torch.Tensor] = None
    wdl_loss: Optional[torch.Tensor] = None
    illegality_head_loss: Optional[torch.Tensor] = None
    masked_token_loss: Optional[torch.Tensor] = None
    # Metrics (not losses)
    illegality_rate: Optional[torch.Tensor] = None
    illegality_head_accuracy: Optional[torch.Tensor] = None
    masked_token_accuracy: Optional[torch.Tensor] = None
    top1_agreement: Optional[torch.Tensor] = None
    model_entropy: Optional[torch.Tensor] = None
    white_advantage_mae: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class ChessPolicyValueModel(LlamaPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.policy_dim = config.policy_dim
        # Get empty token IDs (can be list or None)
        empty_token_ids_list = getattr(config, 'empty_token_ids', None)
        self.empty_token_ids = set(empty_token_ids_list) if empty_token_ids_list else None
        self.transformer = LlamaModel(config)
        self._disable_causal_mask()
        hidden_size = config.hidden_size

        # Thinking tokens - learnable embeddings for additional compute
        # Set to 0 to disable thinking tokens
        self.num_thinking_tokens = getattr(config, 'num_thinking_tokens', 64)
        if self.num_thinking_tokens > 0:
            self.thinking_tokens = nn.Parameter(
                torch.randn(1, self.num_thinking_tokens, hidden_size) * 0.02
            )
            self._thinking_tokens_logged = False  # For one-time logging

        # Multi-task attention pooling (shared K/V, task-specific queries)
        self.task_head = MultiTaskAttentionPooling(
            hidden_size=hidden_size,
            task_output_dims={
                'policy': self.policy_dim,
                'wdl': 3,
                'illegality': self.policy_dim,  # Predict legality for each move
            }
        )

        # Language modeling head for masked token prediction
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)

        self.policy_loss_weight = float(DEFAULT_POLICY_LOSS_WEIGHT)
        self.wdl_loss_weight = float(DEFAULT_WDL_LOSS_WEIGHT)
        self.illegality_head_loss_weight = float(DEFAULT_ILLEGALITY_HEAD_LOSS_WEIGHT)
        self.masked_token_loss_weight = float(DEFAULT_MASKED_TOKEN_LOSS_WEIGHT)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        policy: Optional[torch.Tensor] = None,
        wdl: Optional[torch.Tensor] = None,
        masked_positions: Optional[torch.Tensor] = None,
        original_input_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> ChessPolicyValueOutput:
        # Convert input_ids to embeddings
        batch_size = input_ids.size(0)
        input_embeds = self.transformer.embed_tokens(input_ids)
        original_seq_len = input_embeds.size(1)

        # Append thinking tokens BEFORE transformer processing (if enabled)
        if self.num_thinking_tokens > 0:
            thinking_tokens = self.thinking_tokens.expand(batch_size, -1, -1)
            input_embeds = torch.cat([input_embeds, thinking_tokens], dim=1)

            # Sanity check: verify thinking tokens were added
            expected_seq_len = original_seq_len + self.num_thinking_tokens
            actual_seq_len = input_embeds.size(1)
            if actual_seq_len != expected_seq_len:
                raise RuntimeError(
                    f"Thinking tokens concatenation failed! "
                    f"Expected seq_len={expected_seq_len} (board={original_seq_len} + thinking={self.num_thinking_tokens}), "
                    f"but got {actual_seq_len}. Shape: {input_embeds.shape}"
                )

            # One-time log to confirm thinking tokens are active
            if not self._thinking_tokens_logged and self.training:
                print(f"✓ Thinking tokens active: {original_seq_len} board tokens + {self.num_thinking_tokens} thinking tokens = {actual_seq_len} total")
                print(f"✓ All {actual_seq_len} tokens will be processed through transformer's {len(self.transformer.layers)} layers")
                self._thinking_tokens_logged = True

        # Process ALL tokens (board + thinking) through transformer
        transformer_outputs = self.transformer(inputs_embeds=input_embeds, **kwargs)
        hidden_states = transformer_outputs.last_hidden_state

        # Multi-task attention pooling (single forward pass)
        task_outputs = self.task_head(hidden_states)
        policy_logits = task_outputs['policy']
        wdl_logits = task_outputs['wdl']
        illegality_logits = task_outputs['illegality']

        target_device = policy_logits.device

        policy_loss: Optional[torch.Tensor] = None
        policy_mask_bool: Optional[torch.Tensor] = None
        if policy is not None:
            if policy.device != target_device:
                policy = policy.to(target_device)

            policy_mask_bool = (policy >= 0).to(dtype=torch.bool)
            policy = policy.masked_fill(~policy_mask_bool, 0)

            # NEW: Mask illegal moves before computing softmax (clean policy loss)
            # masked_logits = policy_logits.masked_fill(~policy_mask_bool, -1e9)
            # model_probs = F.softmax(masked_logits, dim=-1)
            # CHANGED: No masking - model is punished for putting probability on illegal moves
            model_probs = F.softmax(policy_logits, dim=-1)

            # Expected quality under model's distribution
            expected_quality = (
                model_probs * policy).sum(dim=-1)  # Range: [0, 1]

            # Minimize negative log expected quality (stronger gradients early training)
            raw_policy_loss = -torch.log(expected_quality + 1e-9).mean()
            policy_loss = self.policy_loss_weight * raw_policy_loss

        wdl_loss: Optional[torch.Tensor] = None
        white_advantage_mae: Optional[torch.Tensor] = None
        if wdl is not None:
            if wdl.device != target_device:
                wdl = wdl.to(target_device)

            wdl_log_probs = F.log_softmax(wdl_logits, dim=-1)
            raw_wdl_loss = -(wdl * wdl_log_probs).sum(dim=-1).mean()
            wdl_loss = self.wdl_loss_weight * raw_wdl_loss

            # White advantage MAE: mean absolute error of (win - loss) prediction
            wdl_probs = F.softmax(wdl_logits, dim=-1)
            predicted_advantage = wdl_probs[:, 0] - wdl_probs[:, 2]  # win - loss
            target_advantage = wdl[:, 0] - wdl[:, 2]  # win - loss
            white_advantage_mae = torch.abs(predicted_advantage - target_advantage).mean()

        # NEW: Illegality head loss (binary cross-entropy for legality prediction)
        illegality_head_loss: Optional[torch.Tensor] = None
        illegality_head_accuracy: Optional[torch.Tensor] = None
        if policy is not None and policy_mask_bool is not None:
            # Target: 1 for legal moves, 0 for illegal moves
            illegality_target = policy_mask_bool.float()

            # Compute BCE loss with logits
            raw_illegality_head_loss = F.binary_cross_entropy_with_logits(
                illegality_logits, illegality_target, reduction='mean'
            )
            illegality_head_loss = self.illegality_head_loss_weight * raw_illegality_head_loss

            # Compute % of legal moves marked as legal (recall for legal class)
            illegality_preds = (torch.sigmoid(illegality_logits) > 0.5).float()
            legal_mask = (illegality_target == 1)
            illegality_head_accuracy = illegality_preds[legal_mask].mean()

        # Masked token prediction loss (language modeling objective)
        masked_token_loss: Optional[torch.Tensor] = None
        masked_token_accuracy: Optional[torch.Tensor] = None
        if masked_positions is not None and original_input_ids is not None:
            # Only compute loss on positions that were masked
            if masked_positions.any():
                # Get logits only for board tokens (not thinking tokens)
                # hidden_states[:, :original_input_ids.size(1)] extracts first 72 tokens
                board_hidden = hidden_states[:, :original_input_ids.size(1), :]
                lm_logits = self.lm_head(board_hidden)

                # Move tensors to same device if needed
                if original_input_ids.device != target_device:
                    original_input_ids = original_input_ids.to(target_device)
                if masked_positions.device != target_device:
                    masked_positions = masked_positions.to(target_device)

                # Flatten for loss computation
                lm_logits_flat = lm_logits.view(-1, lm_logits.size(-1))  # [batch*seq, vocab]
                original_ids_flat = original_input_ids.view(-1)  # [batch*seq]
                masked_positions_flat = masked_positions.view(-1)  # [batch*seq]

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
                        non_empty_mask = ~torch.isin(masked_labels, empty_ids_tensor)
                        if non_empty_mask.any():
                            masked_token_accuracy = (
                                (masked_preds[non_empty_mask] == masked_labels[non_empty_mask])
                                .float().mean()
                            )
                        # If all masked tokens are empty squares, don't report accuracy
                    else:
                        # Fallback: compute accuracy on all masked positions
                        masked_token_accuracy = (masked_preds == masked_labels).float().mean()

        # Compute metrics for reporting (not used in loss)
        illegality_rate: Optional[torch.Tensor] = None
        top1_agreement: Optional[torch.Tensor] = None
        model_entropy: Optional[torch.Tensor] = None

        if policy is not None and policy_mask_bool is not None:
            # Illegality rate: fraction of probability mass on illegal moves (from policy head)
            illegal_mask = (~policy_mask_bool).to(dtype=policy_logits.dtype)
            illegal_probs = F.softmax(policy_logits, dim=-1)
            summed_illegal_prob = (illegal_probs * illegal_mask).sum(dim=-1)
            illegality_rate = summed_illegal_prob.mean()

            # Top-1 agreement: how often model's top move matches Stockfish's
            # Use masked logits for fair comparison (only legal moves)
            masked_logits_for_top1 = policy_logits.masked_fill(~policy_mask_bool, -1e9)
            model_top_move = masked_logits_for_top1.argmax(dim=-1)
            stockfish_top_move = policy.argmax(dim=-1)
            top1_agreement = (model_top_move == stockfish_top_move).float().mean()

            # Model entropy: measure of model's confidence/diversity
            masked_logits_for_entropy = policy_logits.masked_fill(~policy_mask_bool, -1e9)
            model_probs_for_entropy = F.softmax(masked_logits_for_entropy, dim=-1)
            model_entropy = -(model_probs_for_entropy *
                              torch.log(model_probs_for_entropy + 1e-9)).sum(dim=-1).mean()

        loss_components = [
            component
            for component in (policy_loss, wdl_loss, illegality_head_loss, masked_token_loss)
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
            policy_logits=policy_logits,
            wdl_logits=wdl_logits,
            illegality_logits=illegality_logits,
            policy_loss=policy_loss,
            wdl_loss=wdl_loss,
            illegality_head_loss=illegality_head_loss,
            masked_token_loss=masked_token_loss,
            illegality_rate=illegality_rate,
            illegality_head_accuracy=illegality_head_accuracy,
            masked_token_accuracy=masked_token_accuracy,
            top1_agreement=top1_agreement,
            model_entropy=model_entropy,
            white_advantage_mae=white_advantage_mae,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

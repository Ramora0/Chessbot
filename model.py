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
    ILLEGALITY_LOSS_WEIGHT,
)


DEFAULT_POLICY_LOSS_WEIGHT = POLICY_LOSS_WEIGHT
DEFAULT_WDL_LOSS_WEIGHT = WDL_LOSS_WEIGHT
DEFAULT_ILLEGALITY_LOSS_WEIGHT = ILLEGALITY_LOSS_WEIGHT


@dataclass
class ChessPolicyValueOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    policy_logits: torch.Tensor = None
    wdl_logits: torch.Tensor = None
    policy_loss: Optional[torch.Tensor] = None
    wdl_loss: Optional[torch.Tensor] = None
    illegality_loss: Optional[torch.Tensor] = None
    # Metrics (not losses)
    illegality_rate: Optional[torch.Tensor] = None
    top1_agreement: Optional[torch.Tensor] = None
    model_entropy: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class ChessPolicyValueModel(LlamaPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.policy_dim = config.policy_dim
        self.transformer = LlamaModel(config)
        self._disable_causal_mask()
        hidden_size = config.hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.policy_head = nn.Linear(hidden_size, self.policy_dim)
        self.wdl_head = nn.Linear(hidden_size, 3)
        self.policy_loss_weight = float(DEFAULT_POLICY_LOSS_WEIGHT)
        self.wdl_loss_weight = float(DEFAULT_WDL_LOSS_WEIGHT)
        self.illegality_loss_weight = float(DEFAULT_ILLEGALITY_LOSS_WEIGHT)
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
        state_dict_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            # Try model.safetensors
            state_dict_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            if os.path.exists(state_dict_path):
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
            else:
                raise FileNotFoundError(f"Could not find model weights in {pretrained_model_name_or_path}")
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
        return_dict: bool = True,
        **kwargs,
    ) -> ChessPolicyValueOutput:
        transformer_outputs = self.transformer(input_ids=input_ids, **kwargs)
        hidden_states = transformer_outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        pooled = self.norm(pooled)
        policy_logits = self.policy_head(pooled)
        wdl_logits = self.wdl_head(pooled)

        target_device = policy_logits.device

        policy_loss: Optional[torch.Tensor] = None
        policy_mask_bool: Optional[torch.Tensor] = None
        if policy is not None:
            if policy.device != target_device:
                policy = policy.to(target_device)

            policy_mask_bool = (policy >= 0).to(dtype=torch.bool)
            policy = policy.masked_fill(~policy_mask_bool, 0)

            # Old wide-reaching cross-entropy loss:
            # masked_logits = policy_logits.masked_fill(~policy_mask_bool, -1e9)
            # policy_log_probs = F.log_softmax(masked_logits, dim=-1)
            # raw_policy_loss = -(policy * policy_log_probs).sum(dim=-1).mean()
            # policy_loss = self.policy_loss_weight * raw_policy_loss

            # Old mode-centered MSE loss on probabilities:
            # masked_logits = policy_logits.masked_fill(~policy_mask_bool, -1e9)
            # policy_probs = F.softmax(masked_logits, dim=-1)
            # legal_pred = policy_probs[policy_mask_bool]
            # legal_target = policy[policy_mask_bool]
            # raw_policy_loss = 100.0 * F.mse_loss(legal_pred, legal_target, reduction='mean')
            # policy_loss = self.policy_loss_weight * raw_policy_loss

            # Log reward maximization: maximize dot product with Stockfish qualities
            masked_logits = policy_logits.masked_fill(~policy_mask_bool, -1e9)
            model_probs = F.softmax(masked_logits, dim=-1)

            # Expected quality under model's distribution
            expected_quality = (
                model_probs * policy).sum(dim=-1)  # Range: [0, 1]

            # Minimize negative log expected quality (stronger gradients early training)
            raw_policy_loss = -torch.log(expected_quality + 1e-9).mean()
            policy_loss = self.policy_loss_weight * raw_policy_loss

        wdl_loss: Optional[torch.Tensor] = None
        if wdl is not None:
            if wdl.device != target_device:
                wdl = wdl.to(target_device)

            wdl_log_probs = F.log_softmax(wdl_logits, dim=-1)
            raw_wdl_loss = -(wdl * wdl_log_probs).sum(dim=-1).mean()
            wdl_loss = self.wdl_loss_weight * raw_wdl_loss

        # Compute metrics for reporting (not used in loss)
        illegality_rate: Optional[torch.Tensor] = None
        top1_agreement: Optional[torch.Tensor] = None
        model_entropy: Optional[torch.Tensor] = None

        if policy is not None and policy_mask_bool is not None:
            # Compute illegality metrics (shared computation for rate and loss)
            illegal_mask = (~policy_mask_bool).to(dtype=policy_logits.dtype)
            illegal_probs = F.softmax(policy_logits, dim=-1)
            summed_illegal_prob = (illegal_probs * illegal_mask).sum(dim=-1)

            # Illegality rate: fraction of probability mass on illegal moves
            illegality_rate = summed_illegal_prob.mean()

            # Top-1 agreement: how often model's top move matches Stockfish's
            model_top_move = policy_logits.argmax(dim=-1)
            stockfish_top_move = policy.argmax(dim=-1)
            top1_agreement = (model_top_move ==
                              stockfish_top_move).float().mean()

            # Model entropy: measure of model's confidence/diversity
            masked_logits_for_entropy = policy_logits.masked_fill(
                ~policy_mask_bool, -1e9)
            model_probs_for_entropy = F.softmax(
                masked_logits_for_entropy, dim=-1)
            model_entropy = -(model_probs_for_entropy *
                              torch.log(model_probs_for_entropy + 1e-9)).sum(dim=-1).mean()

        # Illegality loss (using precomputed illegality rate)
        illegality_loss: Optional[torch.Tensor] = None
        if policy is not None and self.illegality_loss_weight > 0 and illegality_rate is not None:
            # Use squared illegality rate as loss
            illegality_loss = self.illegality_loss_weight * \
                (illegality_rate ** 2)

        loss_components = [
            component
            for component in (policy_loss, wdl_loss, illegality_loss)
            if component is not None
        ]
        loss: Optional[torch.Tensor] = None
        if loss_components:
            loss = sum(loss_components)

        if not return_dict:
            outputs = (
                policy_logits,
                wdl_logits,
                transformer_outputs.hidden_states,
                transformer_outputs.attentions,
            )
            return ((loss,) + outputs) if loss is not None else outputs

        return ChessPolicyValueOutput(
            loss=loss,
            policy_logits=policy_logits,
            wdl_logits=wdl_logits,
            policy_loss=policy_loss,
            wdl_loss=wdl_loss,
            illegality_loss=illegality_loss,
            illegality_rate=illegality_rate,
            top1_agreement=top1_agreement,
            model_entropy=model_entropy,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model
from transformers.modeling_outputs import ModelOutput
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel

from loss_weights import (
    POLICY_LOSS_WEIGHT,
    WDL_LOSS_WEIGHT,
    ILLEGALITY_LOSS_WEIGHT,
)


DEFAULT_POLICY_LOSS_WEIGHT = POLICY_LOSS_WEIGHT
DEFAULT_WDL_LOSS_WEIGHT = WDL_LOSS_WEIGHT
DEFAULT_ILLEGALITY_LOSS_WEIGHT = ILLEGALITY_LOSS_WEIGHT


@dataclass
class ChessGPT2Output(ModelOutput):
    loss: Optional[torch.Tensor] = None
    policy_logits: torch.Tensor = None
    wdl_logits: torch.Tensor = None
    policy_loss: Optional[torch.Tensor] = None
    wdl_loss: Optional[torch.Tensor] = None
    illegality_loss: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class ChessGPT2PolicyValue(GPT2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.policy_dim = config.policy_dim
        self.transformer = GPT2Model(config)
        self._disable_causal_mask()
        self.norm = nn.LayerNorm(config.n_embd)
        self.policy_head = nn.Linear(config.n_embd, self.policy_dim)
        self.wdl_head = nn.Linear(config.n_embd, 3)
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

    def _disable_causal_mask(self) -> None:
        for block in self.transformer.h:
            attn = block.attn
            ones_bias = torch.ones_like(attn.bias)
            attn.register_buffer("bias", ones_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        policy: Optional[torch.Tensor] = None,
        wdl: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> ChessGPT2Output:
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

            masked_logits = policy_logits.masked_fill(~policy_mask_bool, -1e9)
            policy_log_probs = F.log_softmax(masked_logits, dim=-1)
            raw_policy_loss = -(policy * policy_log_probs).sum(dim=-1).mean()
            policy_loss = self.policy_loss_weight * raw_policy_loss

        wdl_loss: Optional[torch.Tensor] = None
        if wdl is not None:
            if wdl.device != target_device:
                wdl = wdl.to(target_device)

            wdl_log_probs = F.log_softmax(wdl_logits, dim=-1)
            raw_wdl_loss = -(wdl * wdl_log_probs).sum(dim=-1).mean()
            wdl_loss = self.wdl_loss_weight * raw_wdl_loss

        illegality_loss: Optional[torch.Tensor] = None
        if policy is not None and self.illegality_loss_weight > 0:
            if policy_mask_bool is None:
                raise ValueError("Policy mask missing while computing illegality loss")
            illegal_mask = (~policy_mask_bool).to(dtype=policy_logits.dtype)
            if illegal_mask.ndim == policy_logits.ndim:
                illegal_probs = F.softmax(policy_logits, dim=-1)
                summed_illegal_prob = (illegal_probs * illegal_mask).sum(dim=-1)
                illegality_loss = self.illegality_loss_weight * (
                    summed_illegal_prob.pow(2).mean()
                )
            else:
                raise ValueError("Illegal mask shape mismatch when computing illegality loss")
        elif policy is None and self.illegality_loss_weight > 0:
            illegality_loss = None

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

        return ChessGPT2Output(
            loss=loss,
            policy_logits=policy_logits,
            wdl_logits=wdl_logits,
            policy_loss=policy_loss,
            wdl_loss=wdl_loss,
            illegality_loss=illegality_loss,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

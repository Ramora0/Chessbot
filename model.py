from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model
from transformers.modeling_outputs import ModelOutput
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel


@dataclass
class ChessGPT2Output(ModelOutput):
    loss: Optional[torch.Tensor] = None
    policy_logits: torch.Tensor = None
    wdl_logits: torch.Tensor = None
    policy_loss: Optional[torch.Tensor] = None
    wdl_loss: Optional[torch.Tensor] = None
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
        self.post_init()

    def load_state_dict(self, state_dict, strict: bool = False):  # type: ignore[override]
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
        if policy is not None:
            if policy.device != target_device:
                policy = policy.to(target_device)

            policy_mask = (policy >= 0).to(dtype=torch.bool)
            policy = policy.masked_fill(~policy_mask, 0)

            masked_logits = policy_logits.masked_fill(~policy_mask, -1e9)
            policy_log_probs = F.log_softmax(masked_logits, dim=-1)
            raw_policy_loss = -(policy * policy_log_probs).sum(dim=-1).mean()
            policy_loss = 0.8 * raw_policy_loss

        wdl_loss: Optional[torch.Tensor] = None
        if wdl is not None:
            if wdl.device != target_device:
                wdl = wdl.to(target_device)

            wdl_log_probs = F.log_softmax(wdl_logits, dim=-1)
            raw_wdl_loss = -(wdl * wdl_log_probs).sum(dim=-1).mean()
            wdl_loss = 0.2 * raw_wdl_loss

        loss: Optional[torch.Tensor] = None
        if policy_loss is not None and wdl_loss is not None:
            loss = policy_loss + wdl_loss
        elif policy_loss is not None:
            loss = policy_loss
        elif wdl_loss is not None:
            loss = wdl_loss

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
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

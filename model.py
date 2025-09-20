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
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class ChessGPT2PolicyValue(GPT2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.policy_dim = config.policy_dim
        self.transformer = GPT2Model(config)
        self.norm = nn.LayerNorm(config.n_embd)
        self.policy_head = nn.Linear(config.n_embd, self.policy_dim)
        self.wdl_head = nn.Linear(config.n_embd, 3)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        policy: Optional[torch.Tensor] = None,
        policy_mask: Optional[torch.Tensor] = None,
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

        loss = None
        if policy is not None and policy_mask is not None:
            masked_logits = policy_logits.masked_fill(~policy_mask, -1e9)
            policy_log_probs = F.log_softmax(masked_logits, dim=-1)
            policy_loss = -(policy * policy_log_probs).sum(dim=-1).mean()
            loss = policy_loss

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
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

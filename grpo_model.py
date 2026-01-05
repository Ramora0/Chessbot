import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, TaskType

# --- Model Paths ---
CHESS_ENC_HF_PATH = 'jrahn/ROOK-CLF-9m'
LLM_HF_PATH = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

# Special token to mark where chess embeddings should be injected
CHESS_PLACEHOLDER = "<|chess_position|>"


class ChessLMConfig(PretrainedConfig):
    """Config for ChessLM that wraps an LLM config."""
    model_type = "chess_lm"

    def __init__(
        self,
        chess_enc_path: str = CHESS_ENC_HF_PATH,
        llm_path: str = LLM_HF_PATH,
        chess_dim: int = 256,
        freeze_chess_enc: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.chess_enc_path = chess_enc_path
        self.llm_path = llm_path
        self.chess_dim = chess_dim
        self.freeze_chess_enc = freeze_chess_enc


class ChessLMWrapper(PreTrainedModel):
    """
    Wrapper that makes ChessLM compatible with HuggingFace Trainer/GRPOTrainer.

    The key insight: we use a special placeholder token in the input_ids.
    During forward/generate, we detect these placeholders and replace them
    with chess encoder embeddings stored in a buffer.

    Usage:
    1. Store chess embeddings: model.set_chess_embeddings(batch_embeddings)
    2. Include CHESS_PLACEHOLDER tokens in your prompts
    3. Call forward/generate normally - embeddings are injected automatically
    """
    config_class = ChessLMConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: ChessLMConfig, lora_config: LoraConfig = None):
        super().__init__(config)

        # Chess encoder (frozen)
        self.c_enc = AutoModelForSequenceClassification.from_pretrained(
            config.chess_enc_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).model
        for param in self.c_enc.parameters():
            param.requires_grad = not config.freeze_chess_enc

        # Language model
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # Apply LoRA if provided
        if lora_config is not None:
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()

        self.chess_dim = config.chess_dim
        self.llm_dim = self.llm.config.hidden_size

        # Projector: chess encoder -> LLM embedding space
        self.projector = nn.Sequential(
            nn.Linear(self.chess_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        ).to(torch.bfloat16)

        # Buffer to store pre-computed chess embeddings for current batch
        self._chess_embeddings = None
        self._chess_attention_mask = None

    def set_chess_embeddings(self, chess_input_ids: torch.Tensor, chess_attention_mask: torch.Tensor):
        """
        Pre-compute and store chess embeddings for the current batch.
        Call this before forward/generate.

        Args:
            chess_input_ids: (batch_size, chess_seq_len) tokenized FEN positions
            chess_attention_mask: (batch_size, chess_seq_len) attention mask
        """
        with torch.no_grad():
            chess_outputs = self.c_enc(
                input_ids=chess_input_ids,
                attention_mask=chess_attention_mask
            )
            chess_feats = chess_outputs.last_hidden_state
            chess_feats = chess_feats.to(self.projector[0].weight.dtype)

        # Project to LLM space (this part can be trained)
        self._chess_embeddings = self.projector(chess_feats)
        self._chess_attention_mask = chess_attention_mask

    def clear_chess_embeddings(self):
        """Clear the stored chess embeddings."""
        self._chess_embeddings = None
        self._chess_attention_mask = None

    def _inject_chess_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Replace placeholder tokens with chess embeddings.

        Returns:
            inputs_embeds: (batch_size, new_seq_len, hidden_size)
            new_attention_mask: (batch_size, new_seq_len)
        """
        if self._chess_embeddings is None:
            # No chess embeddings set, just return normal text embeddings
            return self.llm.get_input_embeddings()(input_ids), attention_mask

        batch_size = input_ids.shape[0]
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Prepend chess embeddings to text embeddings
        # Shape: (batch, chess_seq + text_seq, hidden)
        combined_embeds = torch.cat([self._chess_embeddings, text_embeds], dim=1)

        if attention_mask is not None:
            combined_mask = torch.cat([self._chess_attention_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        return combined_embeds, combined_mask

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        **kwargs
    ):
        """Forward pass that injects chess embeddings."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # If we have chess embeddings and input_ids (not inputs_embeds), inject them
        if self._chess_embeddings is not None and input_ids is not None and inputs_embeds is None:
            inputs_embeds, attention_mask = self._inject_chess_embeddings(input_ids, attention_mask)
            input_ids = None  # Use inputs_embeds instead

            # Adjust labels if provided (prepend -100 for chess tokens)
            if labels is not None:
                chess_seq_len = self._chess_embeddings.shape[1]
                ignore_labels = torch.full(
                    (labels.shape[0], chess_seq_len),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                labels = torch.cat([ignore_labels, labels], dim=1)

        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def generate(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        **generate_kwargs
    ):
        """Generate with chess embeddings injected."""
        # Inject chess embeddings if available
        if self._chess_embeddings is not None and input_ids is not None and inputs_embeds is None:
            inputs_embeds, attention_mask = self._inject_chess_embeddings(input_ids, attention_mask)
            input_ids = None

        return self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **generate_kwargs
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    @property
    def device(self):
        return next(self.parameters()).device

    def save_pretrained(self, output_dir, **kwargs):
        """Save LoRA weights and projector."""
        import os
        super().save_pretrained(output_dir, **kwargs)
        # Save projector separately
        torch.save(self.projector.state_dict(), os.path.join(output_dir, "projector.pt"))

    def load_projector(self, path):
        """Load projector weights."""
        self.projector.load_state_dict(torch.load(path))


# Keep the original ChessLM for backward compatibility
class ChessLM(nn.Module):
    """Original ChessLM with explicit dual inputs (for custom training loops)."""

    def __init__(
        self,
        chess_enc: str = CHESS_ENC_HF_PATH,
        llm_model: str = LLM_HF_PATH,
        freeze_chess_enc: bool = True,
        lora_config: LoraConfig = None
    ):
        super().__init__()

        # Chess encoder (frozen)
        self.c_enc = AutoModelForSequenceClassification.from_pretrained(
            chess_enc,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).model
        for param in self.c_enc.parameters():
            param.requires_grad = not freeze_chess_enc

        # Language model with PEFT/LoRA
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        if lora_config is not None:
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()

        self.chess_dim = 256
        self.llm_dim = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(self.chess_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        ).to(torch.bfloat16)

    def forward(self, chess_input_ids, chess_attn_mask, text_input_ids, text_attn_mask, labels=None):
        chess_outputs = self.c_enc(
            input_ids=chess_input_ids, attention_mask=chess_attn_mask)
        chess_feats = chess_outputs.last_hidden_state
        chess_feats = chess_feats.to(self.projector[0].weight.dtype)
        chess_embeds = self.projector(chess_feats)

        text_embeds = self.llm.get_input_embeddings()(text_input_ids)

        full_input = torch.cat([chess_embeds, text_embeds], dim=1)
        full_mask = torch.cat([chess_attn_mask, text_attn_mask], dim=1)

        combined_labels = None
        if labels is not None:
            c_ignore = torch.full(
                (chess_embeds.shape[0], chess_embeds.shape[1]),
                -100, dtype=torch.long, device=labels.device
            )
            combined_labels = torch.cat([c_ignore, labels], dim=1)

        return self.llm(
            inputs_embeds=full_input,
            attention_mask=full_mask,
            labels=combined_labels
        )

    def generate(self, chess_input_ids, chess_attn_mask, text_input_ids, text_attn_mask, **kwargs):
        chess_outputs = self.c_enc(
            input_ids=chess_input_ids, attention_mask=chess_attn_mask)
        chess_feats = chess_outputs.last_hidden_state
        chess_feats = chess_feats.to(self.projector[0].weight.dtype)
        chess_embeds = self.projector(chess_feats)

        text_embeds = self.llm.get_input_embeddings()(text_input_ids)

        full_input = torch.cat([chess_embeds, text_embeds], dim=1)
        full_mask = torch.cat([chess_attn_mask, text_attn_mask], dim=1)

        return self.llm.generate(
            inputs_embeds=full_input,
            attention_mask=full_mask,
            **kwargs
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    @property
    def config(self):
        return self.llm.config

    @property
    def device(self):
        return next(self.parameters()).device

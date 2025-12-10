class ChessLM(nn.Module):
    def __init__(
        self,
        chess_enc: str = CHESS_ENC_HF_PATH,
        llm_model: str = LLM_HF_PATH,
        freeze_chess_enc: bool = True,
        freeze_llm: bool = True
    ):
        super().__init__()
        # chess encoder
        self.c_enc = AutoModelForSequenceClassification.from_pretrained(
            chess_enc,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).model
        for param in self.c_enc.parameters():
            param.requires_grad = not freeze_chess_enc

        # language model
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
            quantization_config=bnb_cfg
        )
        self.llm = prepare_model_for_kbit_training(self.llm)
        for param in self.llm.parameters():
            param.requires_grad = not freeze_llm

        self.chess_dim = 256  # TODO not chess_enc drop in safe
        self.llm_dim = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(self.chess_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        ).to(torch.bfloat16)

    def forward(self, chess_input_ids, chess_attn_mask, text_input_ids, text_attn_mask, labels=None):
        # get chess embeddings
        chess_outputs = self.c_enc(
            input_ids=chess_input_ids, attention_mask=chess_attn_mask)
        chess_feats = chess_outputs.last_hidden_state
        chess_feats = chess_feats.to(self.projector[0].weight.dtype)
        chess_embeds = self.projector(chess_feats)

        # get text embeddings
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)

        # get X,Y
        full_input = torch.cat([chess_embeds, text_embeds], dim=1)
        full_mask = torch.cat([chess_attn_mask, text_attn_mask], dim=1)
        combined_labels = None
        if labels is not None:
            c_ignore = torch.full(
                (chess_embeds.shape[0], chess_embeds.shape[1]),
                -100,
                dtype=torch.long,
                device=labels.device
            )
            combined_labels = torch.cat([c_ignore, labels], dim=1)

        return self.llm(
            inputs_embeds=full_input,
            attention_mask=full_mask,
            labels=combined_labels
        )

    # needed in HF trainer
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def get_input_embeddings(self): return self.llm.get_input_embeddings()

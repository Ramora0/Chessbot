from __future__ import annotations

import os
from pathlib import Path

from transformers import (
    GPT2Config,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, load_from_disk

from data import ChessPolicyCollator, ChessPolicyDataset
from model import ChessGPT2PolicyValue
from policy_index import policy_index
from tokenizer import create_tokenizer


OUTPUT_DIR = "outputs"
DROPOUT = 0.1
MAX_SEQ_LENGTH = 71
PROCESSED_DATASET_DIR = "processed_chessfens"
RAW_DATASET_NAME = "Maxlegrec/ChessFENS"
RAW_DATASET_SPLIT = "train"


def train() -> None:
    print("Starting chess transformer training...")
    os.environ["WANDB_PROJECT"] = "chessformer"

    print("Creating tokenizer...")
    tokenizer = create_tokenizer()
    vocab_size = len(tokenizer.get_vocab())
    pad_token_id = tokenizer.token_to_id("[PAD]")
    print(
        f"Tokenizer created - vocab size: {vocab_size}, pad token id: {pad_token_id}")

    processed_path = Path(PROCESSED_DATASET_DIR)
    if processed_path.exists():
        print(f"Loading preprocessed dataset from '{processed_path}'...")
        hf_dataset = load_from_disk(str(processed_path))
    else:
        print(
            f"Preprocessed dataset not found at '{processed_path}'. "
            f"Falling back to raw dataset '{RAW_DATASET_NAME}'."
        )
        hf_dataset = load_dataset(RAW_DATASET_NAME, split=RAW_DATASET_SPLIT)

    train_dataset = ChessPolicyDataset(hf_dataset)
    print(f"Dataset loaded - training samples: {len(train_dataset)}")

    print("Creating model configuration...")
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=MAX_SEQ_LENGTH,
        n_ctx=MAX_SEQ_LENGTH,
        n_embd=512,
        n_layer=8,
        n_head=8,
        resid_pdrop=DROPOUT,
        embd_pdrop=DROPOUT,
        attn_pdrop=DROPOUT,
        pad_token_id=pad_token_id,
    )
    config.policy_dim = len(policy_index)
    print(f"Model config created - policy dimension: {config.policy_dim}")

    print("Initializing ChessGPT2 model...")
    model = ChessGPT2PolicyValue(config)
    model.config.use_cache = False
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,

        per_device_train_batch_size=512,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        report_to=["wandb"],
        run_name="chessformer-gpt2-policy",
        remove_unused_columns=False,

        dataloader_num_workers=8,
        dataloader_prefetch_factor=1,
        dataloader_pin_memory=True,
    )
    print(f"Training config: {training_args.num_train_epochs} epochs, batch size {training_args.per_device_train_batch_size}, lr {training_args.learning_rate}")

    print("Creating trainer...")
    data_collator = ChessPolicyCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    # trainer.save_model(OUTPUT_DIR)
    # trainer.save_state()
    print(f"Training complete. Final model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()

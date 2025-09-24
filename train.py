from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from transformers import (
    GPT2Config,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_from_disk, load_dataset
from datasets import IterableDataset as HFIterableDataset

from data import ChessPolicyCollator, ChessPolicyDataset
from model import ChessGPT2PolicyValue
from policy_index import policy_index
from tokenizer import create_tokenizer
from evaluation import evaluate_model_elo, DEFAULT_EVAL_DATASET_DIR


OUTPUT_DIR = "outputs"
DROPOUT = 0.1
MAX_SEQ_LENGTH = 71
PROCESSED_DATASET_DIR = "/fs/scratch/PAS3150/lees_stuff/processed_chessfens"
ELO_EVAL_STEPS = 2000
EVAL_BATCH_SIZE = 4096
TRAIN_MAX_STEPS_ENV = "TRAIN_MAX_STEPS"


class EloEvaluationCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset,
        frequency: int,
        batch_size: int = EVAL_BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.eval_dataset = eval_dataset
        self.frequency = max(0, int(frequency))
        self.batch_size = batch_size
        self.trainer: Optional[Trainer] = None
        self._last_step_logged: int = -1

    def attach_trainer(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def _should_run(self, step: int) -> bool:
        if self.trainer is None:
            return False
        if self.frequency <= 0:
            return False
        if step <= 0:
            return False
        if step == self._last_step_logged:
            return False
        return step % self.frequency == 0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if not self._should_run(step):
            return control

        model = self.trainer.model
        was_training = model.training
        elo, elo_se = evaluate_model_elo(
            model=model,
            batch_size=self.batch_size,
            dataset=self.eval_dataset,
        )
        if was_training:
            model.train()

        metrics = {
            "eval/elo": float(elo),
            "eval/elo_se": float(elo_se),
            "step": step,
        }
        self.trainer.log(metrics)
        self._last_step_logged = step

        return control


def train() -> None:
    print("Starting chess transformer training...")
    os.environ["WANDB_PROJECT"] = "chessformer"
    # Avoid W&B from uploading checkpoints while keeping metric logging enabled.
    os.environ["WANDB_LOG_MODEL"] = "false"

    print("Creating tokenizer...")
    tokenizer = create_tokenizer()
    vocab_size = len(tokenizer.get_vocab())
    pad_token_id = tokenizer.token_to_id("[PAD]")
    act_token_id = tokenizer.token_to_id("<ACT>")
    if act_token_id is None:
        raise ValueError("Tokenizer is missing the <ACT> token")
    print(
        f"Tokenizer created - vocab size: {vocab_size}, pad token id: {pad_token_id}")

    processed_path = Path(PROCESSED_DATASET_DIR)
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at '{processed_path}'. "
            "Run process.py first to generate it."
        )

    print(f"Loading preprocessed dataset from '{processed_path}'...")

    # hf_dataset = load_from_disk(str(processed_path))
    data_files = sorted(str(path)
                        for path in processed_path.glob("data-*.arrow"))

    hf_dataset = load_dataset(
        "arrow",
        data_files=data_files,
        split="train",
        streaming=True,
    )

    # optional: approximate shuffle with a buffer
    hf_dataset = hf_dataset.shuffle(buffer_size=100_000)

    if not isinstance(hf_dataset, HFIterableDataset):
        raise TypeError(
            "Expected streaming dataset when loading training data")

    train_dataset = ChessPolicyDataset(
        hf_dataset,
        act_token_id=act_token_id,
    )

    max_steps = 2_800_000
    print(
        f"Streaming dataset detected. Training will run for {max_steps} steps.")
    eval_dataset = None
    eval_path = Path(DEFAULT_EVAL_DATASET_DIR)
    if eval_path.exists():
        print(
            f"Loading evaluation dataset from '{eval_path}' once for Elo tracking...")
        eval_dataset = load_from_disk(str(eval_path))
    else:
        print(
            f"Evaluation dataset not found at '{eval_path}'. "
            "Elo evaluations during training will be skipped."
        )

    print("Creating model configuration...")
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=MAX_SEQ_LENGTH,
        n_ctx=MAX_SEQ_LENGTH,
        n_embd=768,
        n_layer=18,
        n_head=12,
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
        # num_train_epochs=1,

        per_device_train_batch_size=256,
        learning_rate=2e-4,
        weight_decay=0,
        max_grad_norm=1.0,
        bf16=True,
        save_strategy="steps",
        save_steps=10000,
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
        run_name="chessformer-gpt2-policy",
        remove_unused_columns=False,

        dataloader_num_workers=8,
        dataloader_prefetch_factor=1,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        max_steps=max_steps,
    )
    print(f"Training config: {training_args.num_train_epochs} epochs, batch size {training_args.per_device_train_batch_size}, lr {training_args.learning_rate}")

    print("Creating trainer...")
    data_collator = ChessPolicyCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if eval_dataset is not None and ELO_EVAL_STEPS > 0:
        print(
            f"Registering Elo evaluation callback every {ELO_EVAL_STEPS} steps "
            f"(batch size {EVAL_BATCH_SIZE})"
        )
        elo_callback = EloEvaluationCallback(
            eval_dataset=eval_dataset,
            frequency=ELO_EVAL_STEPS,
            batch_size=EVAL_BATCH_SIZE,
        )
        elo_callback.attach_trainer(trainer)
        trainer.add_callback(elo_callback)

    print("Starting training...")
    trainer.train()
    # trainer.save_model(OUTPUT_DIR)
    # trainer.save_state()
    print(f"Training complete. Final model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()

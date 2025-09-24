from __future__ import annotations

import os
import torch
from dataclasses import dataclass
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
ELO_EVAL_STEPS = 4000
EVAL_BATCH_SIZE = 4096
TRAIN_MAX_STEPS_ENV = "TRAIN_MAX_STEPS"
BASE_BATCH_SIZE = 256
BASE_LEARNING_RATE = 5e-5
BASE_MAX_STEPS = 2_800_000
BASE_SAVE_STEPS = 10_000
BASE_LOGGING_STEPS = 200
BASE_ELO_EVAL_STEPS = ELO_EVAL_STEPS
POLICY_LOSS_WEIGHT = 0.9
WDL_LOSS_WEIGHT = 0.1


@dataclass
class TrainingSchedule:
    learning_rate: float
    max_steps: int
    save_steps: int
    logging_steps: int
    elo_eval_steps: int


def build_training_schedule(batch_size: int) -> TrainingSchedule:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    scale = batch_size / BASE_BATCH_SIZE
    inv_scale = BASE_BATCH_SIZE / batch_size

    learning_rate = BASE_LEARNING_RATE * scale
    max_steps = max(1, int(BASE_MAX_STEPS * inv_scale))
    save_steps = max(1, int(BASE_SAVE_STEPS * inv_scale))
    logging_steps = max(1, int(BASE_LOGGING_STEPS * inv_scale))
    elo_eval_steps = max(1, int(BASE_ELO_EVAL_STEPS * inv_scale))

    return TrainingSchedule(
        learning_rate=learning_rate,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        elo_eval_steps=elo_eval_steps,
    )


class TrackingTrainer(Trainer):
    """Custom Trainer that logs individual loss components."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_policy_loss: Optional[float] = None
        self._last_wdl_loss: Optional[float] = None
        self._last_total_loss: Optional[float] = None

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):  # type: ignore[override]
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise ValueError(
                "Model did not return a loss tensor during training")

        self._last_total_loss = float(loss.detach().item())

        policy_loss = getattr(outputs, "policy_loss", None)
        if policy_loss is not None:
            policy_weight = float(getattr(model, "policy_loss_weight", 0.0))
            policy_value = float(policy_loss.detach().item())
            self._last_policy_loss = (
                policy_value / policy_weight if policy_weight > 0 else policy_value
            )
        else:
            self._last_policy_loss = None

        wdl_loss = getattr(outputs, "wdl_loss", None)
        if wdl_loss is not None:
            wdl_weight = float(getattr(model, "wdl_loss_weight", 0.0))
            wdl_value = float(wdl_loss.detach().item())
            self._last_wdl_loss = (
                wdl_value / wdl_weight if wdl_weight > 0 else wdl_value
            )
        else:
            self._last_wdl_loss = None

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs, *args, **kwargs):  # type: ignore[override]
        logs = dict(logs)
        if "loss" in logs:
            if self._last_total_loss is not None:
                logs.setdefault("total_loss", self._last_total_loss)
            if self._last_policy_loss is not None:
                logs.setdefault("policy_loss", self._last_policy_loss)
            if self._last_wdl_loss is not None:
                logs.setdefault("wdl_loss", self._last_wdl_loss)
        super().log(logs, *args, **kwargs)


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
            "eval_elo": float(elo),
            "eval_elo_se": float(elo_se),
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

    per_device_batch_size = 512
    schedule = build_training_schedule(per_device_batch_size)

    print(
        f"Streaming dataset detected. Training will run for {schedule.max_steps} steps.")
    print(
        "Training schedule:",
        f"batch_size={per_device_batch_size}",
        f"learning_rate={schedule.learning_rate}",
        f"save_steps={schedule.save_steps}",
        f"logging_steps={schedule.logging_steps}",
        f"elo_eval_steps={schedule.elo_eval_steps}",
    )
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
    config.policy_loss_weight = POLICY_LOSS_WEIGHT
    config.wdl_loss_weight = WDL_LOSS_WEIGHT
    print(f"Model config created - policy dimension: {config.policy_dim}")

    print("Initializing ChessGPT2 model...")
    model = ChessGPT2PolicyValue(config)
    model.config.use_cache = False
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except RuntimeError as err:
            print(f"torch.compile unavailable at runtime: {err}")

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        # num_train_epochs=1,

        per_device_train_batch_size=per_device_batch_size,
        learning_rate=schedule.learning_rate,
        weight_decay=0,
        max_grad_norm=1.0,
        bf16=True,
        save_strategy="steps",
        save_steps=schedule.save_steps,
        logging_strategy="steps",
        logging_steps=schedule.logging_steps,
        report_to=["wandb"],
        run_name="chessformer-gpt2-policy",
        remove_unused_columns=False,

        dataloader_num_workers=8,
        dataloader_prefetch_factor=1,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        max_steps=schedule.max_steps,
    )
    print(
        f"Training config: {training_args.num_train_epochs} epochs, batch size {training_args.per_device_train_batch_size}, "
        f"lr {training_args.learning_rate}")
    print(
        "Effective schedule:",
        f"max_steps={training_args.max_steps}",
        f"save_steps={training_args.save_steps}",
        f"logging_steps={training_args.logging_steps}",
        f"elo_eval_steps={schedule.elo_eval_steps}",
    )

    print("Creating trainer...")
    data_collator = ChessPolicyCollator()

    trainer = TrackingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if eval_dataset is not None and schedule.elo_eval_steps > 0:
        print(
            f"Registering Elo evaluation callback every {schedule.elo_eval_steps} steps "
            f"(batch size {EVAL_BATCH_SIZE})"
        )
        elo_callback = EloEvaluationCallback(
            eval_dataset=eval_dataset,
            frequency=schedule.elo_eval_steps,
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

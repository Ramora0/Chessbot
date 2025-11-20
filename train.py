from __future__ import annotations

import os
import math
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import (
    LlamaConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_from_disk

from data import ChessPolicyCollator
from action_value_dataset import ActionValueDataset
from model import ChessPolicyValueModel
from policy_index import policy_index
from tokenizer import create_tokenizer
from evaluation_puzzle import evaluate_model_elo, DEFAULT_EVAL_CSV_PATH
from loss_weights import MASKED_TOKEN_LOSS_WEIGHT


OUTPUT_DIR = "new"
DROPOUT = 0.1
NUM_THINKING_TOKENS = 0  # Set to 0 to disable thinking tokens (72 board tokens + thinking tokens)
MAX_SEQ_LENGTH = 256  # Must be >= 72 + NUM_THINKING_TOKENS
PROCESSED_DATASET_DIR = "/fs/scratch/PAS2836/lees_stuff/action_value"
ELO_EVAL_STEPS = 4000
EVAL_BATCH_SIZE = 4096
TRAIN_MAX_STEPS_ENV = "TRAIN_MAX_STEPS"
BASE_BATCH_SIZE = 256
BASE_LEARNING_RATE = 2e-5
BASE_MAX_STEPS = 2_800_000
BASE_SAVE_STEPS = 10_000
BASE_LOGGING_STEPS = 200
BASE_ELO_EVAL_STEPS = ELO_EVAL_STEPS

# Set to a checkpoint path to resume training (e.g., "./outputs/checkpoint-45000")
# Set to None to start from scratch
# RESUME_FROM_CHECKPOINT = "./outputs/checkpoint-90000"
RESUME_FROM_CHECKPOINT = None


@dataclass
class TrainingSchedule:
    learning_rate: float
    max_steps: int
    save_steps: int
    logging_steps: int
    elo_eval_steps: int
    warmup_steps: int


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
    warmup_steps = max(1, int(max_steps * 0.02))  # 1% of total steps

    return TrainingSchedule(
        learning_rate=learning_rate,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        elo_eval_steps=elo_eval_steps,
        warmup_steps=warmup_steps,
    )


class TrackingTrainer(Trainer):
    """Custom Trainer that logs individual loss components."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_policy_loss: Optional[float] = None
        self._last_wdl_loss: Optional[float] = None
        self._last_total_loss: Optional[float] = None
        self._last_illegality_head_loss: Optional[float] = None
        self._last_masked_token_loss: Optional[float] = None
        self._last_illegality_rate: Optional[float] = None
        self._last_illegality_head_accuracy: Optional[float] = None
        self._last_masked_token_accuracy: Optional[float] = None
        self._last_top1_agreement: Optional[float] = None
        self._last_model_entropy: Optional[float] = None
        self._last_value_mae: Optional[float] = None

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

        illegality_head_loss = getattr(outputs, "illegality_head_loss", None)
        if illegality_head_loss is not None:
            illegality_head_weight = float(
                getattr(model, "illegality_head_loss_weight", 0.0))
            illegality_head_value = float(illegality_head_loss.detach().item())
            self._last_illegality_head_loss = (
                illegality_head_value / illegality_head_weight
                if illegality_head_weight > 0
                else illegality_head_value
            )
        else:
            self._last_illegality_head_loss = None

        masked_token_loss = getattr(outputs, "masked_token_loss", None)
        if masked_token_loss is not None:
            masked_token_weight = float(
                getattr(model, "masked_token_loss_weight", 0.0))
            masked_token_value = float(masked_token_loss.detach().item())
            self._last_masked_token_loss = (
                masked_token_value / masked_token_weight
                if masked_token_weight > 0
                else masked_token_value
            )
        else:
            self._last_masked_token_loss = None

        # Extract metrics (not losses)
        illegality_rate = getattr(outputs, "illegality_rate", None)
        self._last_illegality_rate = (
            float(illegality_rate.detach().item()
                  ) if illegality_rate is not None else None
        )

        illegality_head_accuracy = getattr(
            outputs, "illegality_head_accuracy", None)
        self._last_illegality_head_accuracy = (
            float(illegality_head_accuracy.detach().item()
                  ) if illegality_head_accuracy is not None else None
        )

        masked_token_accuracy = getattr(outputs, "masked_token_accuracy", None)
        self._last_masked_token_accuracy = (
            float(masked_token_accuracy.detach().item()
                  ) if masked_token_accuracy is not None else None
        )

        top1_agreement = getattr(outputs, "top1_agreement", None)
        self._last_top1_agreement = (
            float(top1_agreement.detach().item()
                  ) if top1_agreement is not None else None
        )

        model_entropy = getattr(outputs, "model_entropy", None)
        self._last_model_entropy = (
            float(model_entropy.detach().item()
                  ) if model_entropy is not None else None
        )

        value_mae = getattr(outputs, "value_mae", None)
        self._last_value_mae = (
            float(value_mae.detach().item()
                  ) if value_mae is not None else None
        )

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs, *args, **kwargs):  # type: ignore[override]
        logs = dict(logs)
        if "loss" in logs:
            # Log losses
            if self._last_total_loss is not None:
                logs.setdefault("total_loss", self._last_total_loss)
            if self._last_policy_loss is not None:
                logs.setdefault("policy_loss", self._last_policy_loss)
            if self._last_wdl_loss is not None:
                logs.setdefault("wdl_loss", self._last_wdl_loss)
            if self._last_illegality_head_loss is not None:
                logs.setdefault("illegality_head_loss",
                                self._last_illegality_head_loss)
            if self._last_masked_token_loss is not None:
                logs.setdefault("masked_token_loss",
                                self._last_masked_token_loss)

            # Log metrics
            if self._last_illegality_rate is not None:
                logs.setdefault("illegality_rate",
                                self._last_illegality_rate)
            if self._last_illegality_head_accuracy is not None:
                logs.setdefault("illegality_head_accuracy",
                                self._last_illegality_head_accuracy)
            if self._last_masked_token_accuracy is not None:
                logs.setdefault("masked_token_accuracy",
                                self._last_masked_token_accuracy)
            if self._last_top1_agreement is not None:
                logs.setdefault("top1_agreement",
                                self._last_top1_agreement)
            if self._last_model_entropy is not None:
                logs.setdefault("model_entropy",
                                self._last_model_entropy)
            if self._last_value_mae is not None:
                logs.setdefault("value_mae",
                                self._last_value_mae)
        super().log(logs, *args, **kwargs)


class EloEvaluationCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset,
        frequency: int,
        tokenizer,
        batch_size: int = EVAL_BATCH_SIZE,
        csv_path=None,
    ) -> None:
        super().__init__()
        self.eval_dataset = eval_dataset
        self.frequency = max(0, int(frequency))
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.csv_path = csv_path
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
        elo, elo_se, solve_percentage = evaluate_model_elo(
            model=model,
            batch_size=self.batch_size,
            dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            csv_path=self.csv_path,
        )
        if was_training:
            model.train()

        metrics = {
            "eval_elo": float(elo),
            "eval_elo_se": float(elo_se),
        }
        if not math.isnan(solve_percentage):
            metrics["eval_puzzle_accuracy"] = float(solve_percentage)
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
    mask_token_id = tokenizer.token_to_id("[MASK]")

    # Load action-value dataset and tokenize at runtime
    # This ensures the tokenizer used during training matches inference
    print(f"Loading action-value dataset from '{PROCESSED_DATASET_DIR}'...")

    train_dataset = ActionValueDataset(
        dataset_path=PROCESSED_DATASET_DIR,
        tokenizer=tokenizer,
        streaming=True,
    )

    per_device_batch_size = 1024
    schedule = build_training_schedule(per_device_batch_size)

    print(
        f"Streaming dataset detected. Training will run for {schedule.max_steps} steps.")
    print(
        "Training schedule:",
        f"batch_size={per_device_batch_size}",
        f"learning_rate={schedule.learning_rate}",
        f"warmup_steps={schedule.warmup_steps}",
        f"save_steps={schedule.save_steps}",
        f"logging_steps={schedule.logging_steps}",
        f"elo_eval_steps={schedule.elo_eval_steps}",
    )
    eval_dataset = None
    csv_path = DEFAULT_EVAL_CSV_PATH
    if csv_path.exists():
        print(
            f"Loading evaluation puzzles from '{csv_path}'...")
        eval_dataset = None  # Will load from CSV
    else:
        print(
            f"CSV file not found at '{csv_path}'. "
            "Elo evaluations during training will be skipped."
        )

    # Load model from checkpoint or create new
    if RESUME_FROM_CHECKPOINT:
        print(f"Loading model from checkpoint: {RESUME_FROM_CHECKPOINT}")
        model = ChessPolicyValueModel.from_pretrained_compiled(
            RESUME_FROM_CHECKPOINT)
        model.config.use_cache = False
        print(
            f"Model loaded from checkpoint with {sum(p.numel() for p in model.parameters()):,} parameters")
    else:
        print("Creating model configuration...")
        config = LlamaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=MAX_SEQ_LENGTH,
            hidden_size=768,
            intermediate_size=768,
            num_hidden_layers=20,
            num_attention_heads=8,
            num_key_value_heads=8,
            attention_dropout=DROPOUT,
            hidden_dropout=DROPOUT,
            pad_token_id=pad_token_id,
        )
        config.policy_dim = len(policy_index)
        config.num_thinking_tokens = NUM_THINKING_TOKENS

        # Sanity check: ensure MAX_SEQ_LENGTH is large enough
        min_seq_length = 72 + NUM_THINKING_TOKENS
        if MAX_SEQ_LENGTH < min_seq_length:
            raise ValueError(
                f"MAX_SEQ_LENGTH ({MAX_SEQ_LENGTH}) is too small for "
                f"72 board tokens + {NUM_THINKING_TOKENS} thinking tokens = {min_seq_length} tokens"
            )

        print(f"Model config created - policy dimension: {config.policy_dim}")
        if NUM_THINKING_TOKENS > 0:
            print(f"✓ Thinking tokens enabled: {NUM_THINKING_TOKENS} tokens (total sequence length: {72 + NUM_THINKING_TOKENS})")
        else:
            print("✓ Thinking tokens disabled (sequence length: 72)")

        print("Initializing Chess LLaMA model...")
        model = ChessPolicyValueModel(config)
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
        warmup_steps=schedule.warmup_steps,
        weight_decay=0,
        max_grad_norm=1.0,
        bf16=True,
        save_strategy="steps",
        save_steps=schedule.save_steps,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=schedule.logging_steps,
        report_to=["wandb"],
        run_name="testz",
        remove_unused_columns=False,

        dataloader_num_workers=1,
        dataloader_prefetch_factor=1,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        max_steps=schedule.max_steps,
        ignore_data_skip=True,  # Skip dataloader state restoration for streaming datasets
    )
    print(
        f"Training config: {training_args.num_train_epochs} epochs, batch size {training_args.per_device_train_batch_size}, "
        f"lr {training_args.learning_rate}")
    print(
        "Effective schedule:",
        f"max_steps={training_args.max_steps}",
        f"warmup_steps={training_args.warmup_steps}",
        f"save_steps={training_args.save_steps}",
        f"logging_steps={training_args.logging_steps}",
        f"elo_eval_steps={schedule.elo_eval_steps}",
    )

    print("Creating trainer...")
    # Only enable token masking if MASKED_TOKEN_LOSS_WEIGHT > 0
    effective_mask_token_id = mask_token_id if MASKED_TOKEN_LOSS_WEIGHT > 0 else None
    data_collator = ChessPolicyCollator(mask_token_id=effective_mask_token_id)

    trainer = TrackingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if csv_path.exists() and schedule.elo_eval_steps > 0:
        print(
            f"Registering Elo evaluation callback every {schedule.elo_eval_steps} steps "
            f"(batch size {EVAL_BATCH_SIZE})"
        )
        elo_callback = EloEvaluationCallback(
            eval_dataset=eval_dataset,
            frequency=schedule.elo_eval_steps,
            batch_size=EVAL_BATCH_SIZE,
            tokenizer=tokenizer,
            csv_path=csv_path,
        )
        elo_callback.attach_trainer(trainer)
        trainer.add_callback(elo_callback)

    print("Starting training...")
    if RESUME_FROM_CHECKPOINT:
        print(f"Resuming from checkpoint: {RESUME_FROM_CHECKPOINT}")
        trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    else:
        trainer.train()
    # trainer.save_model(OUTPUT_DIR)
    # trainer.save_state()
    print(f"Training complete. Final model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()

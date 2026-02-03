"""
Fine-tune a pretrained chess model on the mate-refined dataset.

Adapted from train.py for the smaller mating dataset (~2M positions).
Runs 3 epochs with frequent validation.
"""

from __future__ import annotations

import os
import math
import torch
import argparse
from pathlib import Path
from typing import Optional

from transformers import Trainer, TrainingArguments, TrainerCallback

from data import ChessPolicyCollator, build_material_lookup
from action_value_dataset import create_action_value_dataset
from model import ChessPolicyValueModel
from tokenizer import create_tokenizer
from evaluation_puzzle import evaluate_model_elo, DEFAULT_EVAL_CSV_PATH
from loss_weights import MASKED_TOKEN_LOSS_WEIGHT


MATE_DATASET_PATH = os.getenv(
    "MATE_DATASET_PATH",
    "/fs/scratch/PAS2836/lees_stuff/action_value_mates"
)
PRETRAINED_MODEL_PATH = os.getenv(
    "PRETRAINED_MODEL_PATH",
    "./final-model/checkpoint-515268"
)
OUTPUT_DIR = "./mating-finetuned"

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision("high")


class TrackingTrainer(Trainer):
    """Custom Trainer that logs individual loss components."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_policy_loss: Optional[float] = None
        self._last_winrate_loss: Optional[float] = None
        self._last_total_loss: Optional[float] = None
        self._last_move_winrate_loss: Optional[float] = None
        self._last_illegality_loss: Optional[float] = None
        self._last_mate_loss: Optional[float] = None
        self._last_top1_agreement: Optional[float] = None
        self._last_value_mae: Optional[float] = None
        self._last_mate_accuracy: Optional[float] = None

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: Optional[int] = None):
        inputs['training_step'] = self.state.global_step
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise ValueError("Model did not return a loss tensor")

        self._last_total_loss = float(loss.detach().item())

        for name, attr in [
            ("policy_loss", "_last_policy_loss"),
            ("winrate_loss", "_last_winrate_loss"),
            ("move_winrate_loss", "_last_move_winrate_loss"),
            ("illegality_loss", "_last_illegality_loss"),
            ("mate_loss", "_last_mate_loss"),
        ]:
            val = getattr(outputs, name, None)
            if val is not None:
                weight = float(getattr(model, f"{name}_weight", 1.0))
                setattr(self, attr, float(val.detach().item()) / weight if weight > 0 else float(val.detach().item()))

        for name, attr in [
            ("top1_agreement", "_last_top1_agreement"),
            ("value_mae", "_last_value_mae"),
            ("mate_accuracy", "_last_mate_accuracy"),
        ]:
            val = getattr(outputs, name, None)
            setattr(self, attr, float(val.detach().item()) if val is not None else None)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        if "loss" in logs:
            for attr, key in [
                ("_last_total_loss", "total_loss"),
                ("_last_policy_loss", "policy_loss"),
                ("_last_winrate_loss", "winrate_loss"),
                ("_last_move_winrate_loss", "move_winrate_loss"),
                ("_last_illegality_loss", "illegality_loss"),
                ("_last_mate_loss", "mate_loss"),
                ("_last_top1_agreement", "top1_agreement"),
                ("_last_value_mae", "value_mae"),
                ("_last_mate_accuracy", "mate_accuracy"),
            ]:
                val = getattr(self, attr, None)
                if val is not None:
                    logs.setdefault(key, val)
        super().log(logs, *args, **kwargs)


class EloEvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, frequency: int = 400) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.frequency = frequency
        self.trainer: Optional[Trainer] = None
        self._last_step: int = -1

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step <= 0 or step == self._last_step or step % self.frequency != 0:
            return control

        model = self.trainer.model
        was_training = model.training

        elo, _, solve_pct, elo_greedy, _, solve_pct_greedy = evaluate_model_elo(
            model=model,
            batch_size=4096,
            dataset=None,
            tokenizer=self.tokenizer,
            csv_path=DEFAULT_EVAL_CSV_PATH,
            compute_both_sampling_modes=True,
        )

        if was_training:
            model.train()

        metrics = {"eval_elo": float(elo), "eval_elo_greedy": float(elo_greedy)}
        if not math.isnan(solve_pct):
            metrics["eval_puzzle_accuracy"] = float(solve_pct)
        if not math.isnan(solve_pct_greedy):
            metrics["eval_puzzle_accuracy_greedy"] = float(solve_pct_greedy)

        self.trainer.log(metrics)
        self._last_step = step
        return control


def train_mates(run_name: Optional[str] = None) -> None:
    print("=" * 60)
    print("FINE-TUNING ON MATING POSITIONS")
    print("=" * 60)

    os.environ["WANDB_PROJECT"] = "chessformer-mates"
    os.environ["WANDB_LOG_MODEL"] = "false"

    if not Path(PRETRAINED_MODEL_PATH).exists():
        raise FileNotFoundError(f"Pretrained model not found: {PRETRAINED_MODEL_PATH}")
    if not Path(MATE_DATASET_PATH).exists():
        raise FileNotFoundError(f"Mate dataset not found: {MATE_DATASET_PATH}")

    print(f"Model: {PRETRAINED_MODEL_PATH}")
    print(f"Dataset: {MATE_DATASET_PATH}")
    print("=" * 60)

    tokenizer = create_tokenizer()
    vocab = tokenizer.get_vocab()
    material_lookup = build_material_lookup(len(vocab), {t: i for t, i in vocab.items()})

    train_dataset = create_action_value_dataset(
        dataset_path=MATE_DATASET_PATH,
        tokenizer=tokenizer,
        shuffle=True,
        seed=42,
    )

    print(f"Dataset: {len(train_dataset):,} positions")
    print(f"Steps/epoch: ~{len(train_dataset) // 1024:,}")

    model = ChessPolicyValueModel.from_pretrained_compiled(PRETRAINED_MODEL_PATH)
    model.config.use_cache = False
    model.config.illegality_penalty_annealing_steps = 0
    model.illegality_penalty_annealing_steps = 0
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Compiled with torch.compile")
        except RuntimeError as e:
            print(f"torch.compile failed: {e}")

    # Hardcoded for batch 1024, ~2M positions, 3 epochs (~6000 steps)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1024,
        learning_rate=2e-5,
        warmup_steps=300,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        logging_strategy="steps",
        logging_steps=10,
        report_to=["wandb"],
        run_name=run_name or "mate-finetune",
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,
    )

    mask_token_id = tokenizer.token_to_id("[MASK]") if MASKED_TOKEN_LOSS_WEIGHT > 0 else None
    data_collator = ChessPolicyCollator(mask_token_id=mask_token_id, material_lookup=material_lookup)

    trainer = TrackingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if DEFAULT_EVAL_CSV_PATH.exists():
        print("Registering ELO callback (every 400 steps)")
        elo_callback = EloEvaluationCallback(tokenizer=tokenizer, frequency=400)
        elo_callback.trainer = trainer
        trainer.add_callback(elo_callback)

    print("\nStarting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\nDone! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    train_mates(run_name=args.run_name)

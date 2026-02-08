from __future__ import annotations

import os
import math
import torch
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import (
    LlamaConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from data import ChessPolicyCollator, build_material_lookup
from action_value_dataset import create_action_value_dataset
from model import ChessPolicyValueModel
from policy_index import policy_index
from tokenizer import create_tokenizer
from evaluation_puzzle import evaluate_model_elo, DEFAULT_EVAL_CSV_PATH
from evaluation_regret import evaluate_regret, print_regret_results, NUM_EVAL_POSITIONS
from loss_weights import MASKED_TOKEN_LOSS_WEIGHT


OUTPUT_DIR = "/users/PAS2836/leedavis/personal/Chessbot/positional-final"
DROPOUT = 0.1
MAX_SEQ_LENGTH = 256  # Board tokens: 72
DATASET_PATH = os.getenv("DATASET_PATH", "fuck")
SHUFFLE_DATASET = True
ELO_EVAL_STEPS = 16000
EVAL_BATCH_SIZE = 512
TRAIN_MAX_STEPS_ENV = "TRAIN_MAX_STEPS"
BASE_BATCH_SIZE = 256
BASE_LEARNING_RATE = 4e-5
BASE_MAX_STEPS = 2_800_000
BASE_SAVE_STEPS = 10_000
BASE_LOGGING_STEPS = 200
BASE_ELO_EVAL_STEPS = ELO_EVAL_STEPS

# Regret evaluation configuration
REGRET_EVAL_STEPS = 16000  # Same frequency as ELO evaluation
BASE_REGRET_EVAL_STEPS = REGRET_EVAL_STEPS

# Game-based evaluation configuration
# Base steps between game evaluations (30k at batch_size=1024)
GAME_EVAL_STEPS = 120000
GAME_EVAL_NUM_GAMES = 250            # Number of games to play per evaluation
GAME_EVAL_BATCH_SIZE = 128           # Parallel games during evaluation
GAME_EVAL_OPPONENT_ELO = 1350        # Stockfish ELO level
GAME_EVAL_STOCKFISH_PATH = "/users/PAS2836/leedavis/stockfish/src/stockfish"
BASE_GAME_EVAL_STEPS = GAME_EVAL_STEPS

# Set to a checkpoint path to resume training (e.g., "./outputs/checkpoint-45000")
# Set to None to start from scratch
# RESUME_FROM_CHECKPOINT = "./outputs/checkpoint-90000"
# RESUME_FROM_CHECKPOINT = "./checkpoints/final/checkpoint-197500"
RESUME_FROM_CHECKPOINT = None


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

torch.set_float32_matmul_precision("high")


@dataclass
class TrainingSchedule:
    learning_rate: float
    max_steps: int
    save_steps: int
    logging_steps: int
    elo_eval_steps: int
    regret_eval_steps: int
    game_eval_steps: int
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
    regret_eval_steps = max(1, int(BASE_REGRET_EVAL_STEPS * inv_scale))
    game_eval_steps = max(1, int(BASE_GAME_EVAL_STEPS * inv_scale))
    warmup_steps = max(1, int(max_steps * 0.02))  # 1% of total steps

    return TrainingSchedule(
        learning_rate=learning_rate,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        elo_eval_steps=elo_eval_steps,
        regret_eval_steps=regret_eval_steps,
        game_eval_steps=game_eval_steps,
        warmup_steps=warmup_steps,
    )


class TrackingTrainer(Trainer):
    """Custom Trainer that logs individual loss components."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_policy_loss: Optional[float] = None
        self._last_winrate_loss: Optional[float] = None
        self._last_total_loss: Optional[float] = None
        self._last_masked_token_loss: Optional[float] = None
        self._last_move_winrate_loss: Optional[float] = None
        self._last_illegality_loss: Optional[float] = None
        self._last_illegality_rate: Optional[float] = None
        self._last_masked_token_accuracy: Optional[float] = None
        self._last_top1_agreement: Optional[float] = None
        self._last_value_mae: Optional[float] = None
        self._last_move_winrate_mae: Optional[float] = None
        self._last_model_entropy: Optional[float] = None
        # Attention policy head losses
        self._last_attention_policy_loss: Optional[float] = None
        self._last_attention_move_winrate_loss: Optional[float] = None

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):  # type: ignore[override]
        # Pass the current training step to the model for annealing
        inputs['training_step'] = self.state.global_step
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

        winrate_loss = getattr(outputs, "winrate_loss", None)
        if winrate_loss is not None:
            winrate_weight = float(getattr(model, "winrate_loss_weight", 0.0))
            winrate_value = float(winrate_loss.detach().item())
            self._last_winrate_loss = (
                winrate_value / winrate_weight if winrate_weight > 0 else winrate_value
            )
        else:
            self._last_winrate_loss = None

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

        move_winrate_loss = getattr(outputs, "move_winrate_loss", None)
        if move_winrate_loss is not None:
            move_winrate_weight = float(
                getattr(model, "move_winrate_loss_weight", 0.0))
            move_winrate_value = float(move_winrate_loss.detach().item())
            self._last_move_winrate_loss = (
                move_winrate_value / move_winrate_weight
                if move_winrate_weight > 0
                else move_winrate_value
            )
        else:
            self._last_move_winrate_loss = None

        illegality_loss = getattr(outputs, "illegality_loss", None)
        if illegality_loss is not None:
            illegality_weight = float(
                getattr(model, "illegality_loss_weight", 0.0))
            illegality_value = float(illegality_loss.detach().item())
            self._last_illegality_loss = (
                illegality_value / illegality_weight
                if illegality_weight > 0
                else illegality_value
            )
        else:
            self._last_illegality_loss = None

        attention_policy_loss = getattr(outputs, "attention_policy_loss", None)
        if attention_policy_loss is not None:
            weight = float(getattr(model, "attention_policy_loss_weight", 0.0))
            value = float(attention_policy_loss.detach().item())
            self._last_attention_policy_loss = (
                value / weight if weight > 0 else value
            )
        else:
            self._last_attention_policy_loss = None

        attention_move_winrate_loss = getattr(
            outputs, "attention_move_winrate_loss", None)
        if attention_move_winrate_loss is not None:
            weight = float(
                getattr(model, "attention_move_winrate_loss_weight", 0.0))
            value = float(attention_move_winrate_loss.detach().item())
            self._last_attention_move_winrate_loss = (
                value / weight if weight > 0 else value
            )
        else:
            self._last_attention_move_winrate_loss = None

        # Extract metrics (not losses)
        illegality_rate = getattr(outputs, "illegality_rate", None)
        self._last_illegality_rate = (
            float(illegality_rate.detach().item()
                  ) if illegality_rate is not None else None
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

        value_mae = getattr(outputs, "value_mae", None)
        self._last_value_mae = (
            float(value_mae.detach().item())
        ) if value_mae is not None else None

        move_winrate_mae = getattr(outputs, "move_winrate_mae", None)
        self._last_move_winrate_mae = (
            float(move_winrate_mae.detach().item())
        ) if move_winrate_mae is not None else None

        model_entropy = getattr(outputs, "model_entropy", None)
        self._last_model_entropy = (
            float(model_entropy.detach().item())
        ) if model_entropy is not None else None

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
            if self._last_winrate_loss is not None:
                logs.setdefault("winrate_loss", self._last_winrate_loss)
            if self._last_masked_token_loss is not None:
                logs.setdefault("masked_token_loss",
                                self._last_masked_token_loss)
            if self._last_move_winrate_loss is not None:
                logs.setdefault("move_winrate_loss",
                                self._last_move_winrate_loss)
            if self._last_illegality_loss is not None:
                logs.setdefault("illegality_loss",
                                self._last_illegality_loss)
            if self._last_attention_policy_loss is not None:
                logs.setdefault("attention_policy_loss",
                                self._last_attention_policy_loss)
            if self._last_attention_move_winrate_loss is not None:
                logs.setdefault("attention_move_winrate_loss",
                                self._last_attention_move_winrate_loss)

            # Log metrics
            if self._last_illegality_rate is not None:
                logs.setdefault("illegality_rate",
                                self._last_illegality_rate)
            if self._last_masked_token_accuracy is not None:
                logs.setdefault("masked_token_accuracy",
                                self._last_masked_token_accuracy)
            if self._last_top1_agreement is not None:
                logs.setdefault("top1_agreement",
                                self._last_top1_agreement)
            if self._last_value_mae is not None:
                logs.setdefault("value_mae",
                                self._last_value_mae)
            if self._last_move_winrate_mae is not None:
                logs.setdefault("move_winrate_mae",
                                self._last_move_winrate_mae)
            if self._last_model_entropy is not None:
                logs.setdefault("model_entropy",
                                self._last_model_entropy)

        super().log(logs, *args, **kwargs)


class EloEvaluationCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset,
        frequency: int,
        tokenizer,
        batch_size: int = EVAL_BATCH_SIZE,
        csv_path=None,
        compute_both_sampling_modes: bool = True,
    ) -> None:
        super().__init__()
        self.eval_dataset = eval_dataset
        self.frequency = max(0, int(frequency))
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.csv_path = csv_path
        self.compute_both_sampling_modes = compute_both_sampling_modes
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
        torch.cuda.empty_cache()
        elo, elo_se, solve_percentage, elo_greedy, elo_se_greedy, solve_percentage_greedy = evaluate_model_elo(
            model=model,
            batch_size=self.batch_size,
            dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            csv_path=self.csv_path,
            compute_both_sampling_modes=self.compute_both_sampling_modes,
        )
        if was_training:
            model.train()
        torch.cuda.empty_cache()

        metrics = {
            "eval_elo": float(elo),
        }
        if not math.isnan(solve_percentage):
            metrics["eval_puzzle_accuracy"] = float(solve_percentage)

        # Log greedy metrics
        if self.compute_both_sampling_modes and elo_greedy is not None:
            metrics["eval_elo_greedy"] = float(elo_greedy)
            if not math.isnan(solve_percentage_greedy):
                metrics["eval_puzzle_accuracy_greedy"] = float(
                    solve_percentage_greedy)

        self.trainer.log(metrics)
        self._last_step_logged = step

        return control


class GameEvaluationCallback(TrainerCallback):
    """Callback to run game-based evaluation against Stockfish during training.

    The opponent ELO dynamically adjusts after each evaluation: it is set to the
    model's estimated ELO from the previous evaluation (clamped to Stockfish's
    supported range).  This ensures the model always faces a challenging opponent
    rather than being stuck playing against a fixed, potentially inferior level.
    """

    # Stockfish UCI_Elo supported range
    STOCKFISH_ELO_MIN = 1320
    STOCKFISH_ELO_MAX = 3190

    def __init__(
        self,
        frequency: int,
        tokenizer,
        stockfish_path: str,
        num_games: int = GAME_EVAL_NUM_GAMES,
        batch_size: int = GAME_EVAL_BATCH_SIZE,
        opponent_elo: int = GAME_EVAL_OPPONENT_ELO,
    ) -> None:
        super().__init__()
        self.frequency = max(0, int(frequency))
        self.tokenizer = tokenizer
        self.stockfish_path = stockfish_path
        self.num_games = num_games
        self.batch_size = batch_size
        self.opponent_elo = opponent_elo
        self.trainer: Optional[Trainer] = None
        self._last_step_logged: int = -1

    def attach_trainer(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def _should_run(self, step: int) -> bool:
        """Check if evaluation should run at this step."""
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
        torch.cuda.empty_cache()

        # Import here to avoid circular dependency
        from evaluation_game import evaluate_model_against_stockfish

        try:
            print(f"\n{'='*80}")
            print(f"Running game evaluation at step {step} (opponent ELO: {self.opponent_elo})...")
            print(f"{'='*80}")

            # Run game evaluation
            estimated_elo, std_error = evaluate_model_against_stockfish(
                model=model,
                stockfish_path=self.stockfish_path,
                num_games=self.num_games,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                opponent_elo=self.opponent_elo,
                verbose=True,
            )

            # Restore training mode
            if was_training:
                model.train()
            torch.cuda.empty_cache()

            # Log metrics to wandb
            metrics = {
                "game_elo": float(estimated_elo),
                "game_elo_se": float(std_error),
                "game_opponent_elo": float(self.opponent_elo),
            }

            self.trainer.log(metrics)
            self._last_step_logged = step

            # Dynamically adjust opponent ELO for next evaluation
            if not math.isnan(estimated_elo):
                old_elo = self.opponent_elo
                new_elo = int(round(estimated_elo))
                new_elo = max(self.STOCKFISH_ELO_MIN, min(self.STOCKFISH_ELO_MAX, new_elo))
                self.opponent_elo = new_elo
                print(f"Opponent ELO updated: {old_elo} -> {new_elo} (model estimated: {estimated_elo:.0f})")

            print(
                f"Game evaluation complete: ELO={estimated_elo:.1f} ± {std_error:.1f}")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"ERROR during game evaluation at step {step}: {e}")
            # Continue training even if evaluation fails
            if was_training:
                model.train()
            torch.cuda.empty_cache()

        return control


class RegretEvaluationCallback(TrainerCallback):
    """Callback to evaluate regret by game phase during training."""

    def __init__(
        self,
        frequency: int,
        dataset_path: str,
        tokenizer,
        num_positions: int = NUM_EVAL_POSITIONS,
    ) -> None:
        super().__init__()
        self.frequency = max(0, int(frequency))
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.num_positions = num_positions
        self.trainer: Optional[Trainer] = None
        self._last_step_logged: int = -1

    def attach_trainer(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def _should_run(self, step: int) -> bool:
        """Check if evaluation should run at this step."""
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
        torch.cuda.empty_cache()

        try:
            print(f"\n{'='*80}")
            print(f"Running regret evaluation at step {step}...")
            print(f"{'='*80}")

            results = evaluate_regret(
                model=model,
                dataset_path=self.dataset_path,
                tokenizer=self.tokenizer,
                num_positions=self.num_positions,
                verbose=True,
            )

            if was_training:
                model.train()
            torch.cuda.empty_cache()

            print_regret_results(results)

            # Log metrics to wandb
            metrics = {
                "regret_overall": float(results.overall_regret),
                "regret_opening": float(results.opening_regret),
                "regret_middlegame": float(results.middlegame_regret),
                "regret_endgame": float(results.endgame_regret),
                "regret_opening_count": results.opening_count,
                "regret_middlegame_count": results.middlegame_count,
                "regret_endgame_count": results.endgame_count,
            }

            self.trainer.log(metrics)
            self._last_step_logged = step

        except Exception as e:
            print(f"ERROR during regret evaluation at step {step}: {e}")
            import traceback
            traceback.print_exc()
            if was_training:
                model.train()
            torch.cuda.empty_cache()

        return control


def train(
    run_name: Optional[str] = None,
    hidden_dim: int = 768,
    ffn_dim: Optional[int] = None,
    depth: int = 20,
    heads: int = 8,
    lr: Optional[float] = None,
    beta2: float = 0.999,
    dropout: float = DROPOUT,
    weight_decay: float = 0.01,
) -> None:
    # ffn_dim defaults to hidden_dim when not explicitly set
    effective_ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim

    print("Starting chess transformer training...")

    os.environ["WANDB_PROJECT"] = "chessformer"
    # Avoid W&B from uploading checkpoints while keeping metric logging enabled.
    os.environ["WANDB_LOG_MODEL"] = "false"

    print("Creating tokenizer...")
    tokenizer = create_tokenizer()
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    pad_token_id = tokenizer.token_to_id("[PAD]")
    mask_token_id = tokenizer.token_to_id("[MASK]")

    # Build material lookup for endgame detection (computed in dataloader workers)
    token_to_id = {token: id for token, id in vocab.items()}
    material_lookup = build_material_lookup(vocab_size, token_to_id)

    print(f"Loading action value dataset from: {DATASET_PATH}...")
    train_dataset = create_action_value_dataset(
        dataset_path=DATASET_PATH,
        tokenizer=tokenizer,
        shuffle=SHUFFLE_DATASET,
        seed=42,
    )

    per_device_batch_size = 1024
    schedule = build_training_schedule(per_device_batch_size)

    # CLI --lr overrides the auto-scaled learning rate
    effective_lr = lr if lr is not None else schedule.learning_rate

    print(f"Training will run for {schedule.max_steps} steps")
    print(
        "Training schedule:",
        f"batch_size={per_device_batch_size}",
        f"learning_rate={effective_lr}",
        f"warmup_steps={schedule.warmup_steps}",
        f"save_steps={schedule.save_steps}",
        f"logging_steps={schedule.logging_steps}",
        f"elo_eval_steps={schedule.elo_eval_steps}",
        f"regret_eval_steps={schedule.regret_eval_steps}",
        f"game_eval_steps={schedule.game_eval_steps}",
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
        # Update annealing steps for resumed training
        model.config.illegality_penalty_annealing_steps = int(
            schedule.max_steps * 0.1)
        model.illegality_penalty_annealing_steps = model.config.illegality_penalty_annealing_steps
        print(
            f"Model loaded from checkpoint with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(
            f"Illegality penalty annealing: {model.illegality_penalty_annealing_steps} steps (10% of epoch)")
    else:
        print("Creating model configuration...")
        config = LlamaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=MAX_SEQ_LENGTH,
            hidden_size=hidden_dim,
            intermediate_size=effective_ffn_dim,
            num_hidden_layers=depth,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            attention_dropout=dropout,
            hidden_dropout=dropout,
            pad_token_id=pad_token_id,
        )
        config.policy_dim = len(policy_index)
        # Anneal illegality penalty over first 10% of epoch
        # Start with -5 penalty on illegal logits, quadratically reduce to 0
        # Quadratic decay spreads learning evenly (compensates for exp in softmax)
        config.illegality_penalty_annealing_steps = int(
            schedule.max_steps * 0.1)

        print(f"Model config created - policy dimension: {config.policy_dim}")
        print(
            f"Illegality penalty annealing: {config.illegality_penalty_annealing_steps} steps (10% of epoch)")

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
        num_train_epochs=1,

        # gradient_accumulation_steps=2,

        per_device_train_batch_size=per_device_batch_size,
        learning_rate=effective_lr,
        warmup_steps=schedule.warmup_steps,
        weight_decay=weight_decay,
        adam_beta2=beta2,
        max_grad_norm=1.0,
        bf16=True,
        # fp16=True,
        save_strategy="steps",
        save_steps=2500,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=schedule.logging_steps,
        report_to=["wandb"],
        run_name=run_name if run_name else "testz",
        remove_unused_columns=False,

        dataloader_num_workers=10,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,

        # dataloader_num_workers=2,
        # dataloader_prefetch_factor=1,
        # dataloader_pin_memory=False,

        # max_steps=schedule.max_steps,
        # ignore_data_skip=False,  # Allow dataloader state restoration when resuming
    )
    print(
        f"Training config: {training_args.num_train_epochs} epochs, "
        f"batch_size={training_args.per_device_train_batch_size}, "
        f"lr={training_args.learning_rate}")
    print(
        "Effective schedule:",
        f"max_steps={training_args.max_steps}",
        f"warmup_steps={training_args.warmup_steps}",
        f"save_steps={training_args.save_steps}",
        f"logging_steps={training_args.logging_steps}",
        f"elo_eval_steps={schedule.elo_eval_steps}",
        f"regret_eval_steps={schedule.regret_eval_steps}",
        f"game_eval_steps={schedule.game_eval_steps}",
    )

    print("Creating trainer...")
    # Only enable token masking if MASKED_TOKEN_LOSS_WEIGHT > 0
    effective_mask_token_id = mask_token_id if MASKED_TOKEN_LOSS_WEIGHT > 0 else None
    data_collator = ChessPolicyCollator(
        mask_token_id=effective_mask_token_id,
        material_lookup=material_lookup,
    )

    trainer = TrackingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if csv_path.exists() and schedule.elo_eval_steps > 0:
        print(
            f"Registering Elo evaluation callback every {schedule.elo_eval_steps} steps "
            f"(batch size {EVAL_BATCH_SIZE}, dual sampling mode enabled)"
        )
        elo_callback = EloEvaluationCallback(
            eval_dataset=eval_dataset,
            frequency=schedule.elo_eval_steps,
            batch_size=EVAL_BATCH_SIZE,
            tokenizer=tokenizer,
            csv_path=csv_path,
            compute_both_sampling_modes=True,
        )
        elo_callback.attach_trainer(trainer)
        trainer.add_callback(elo_callback)

    # Game evaluation callback
    if Path(GAME_EVAL_STOCKFISH_PATH).exists() and schedule.game_eval_steps > 0:
        print(
            f"Registering game evaluation callback every {schedule.game_eval_steps} steps "
            f"(num_games={GAME_EVAL_NUM_GAMES}, batch_size={GAME_EVAL_BATCH_SIZE}, "
            f"opponent_elo={GAME_EVAL_OPPONENT_ELO})"
        )
        print(f"  Stockfish path: {GAME_EVAL_STOCKFISH_PATH}")
        game_callback = GameEvaluationCallback(
            frequency=schedule.game_eval_steps,
            tokenizer=tokenizer,
            stockfish_path=GAME_EVAL_STOCKFISH_PATH,
            num_games=GAME_EVAL_NUM_GAMES,
            batch_size=GAME_EVAL_BATCH_SIZE,
            opponent_elo=GAME_EVAL_OPPONENT_ELO,
        )
        game_callback.attach_trainer(trainer)
        trainer.add_callback(game_callback)
    else:
        if not Path(GAME_EVAL_STOCKFISH_PATH).exists():
            print(
                f"Stockfish not found at {GAME_EVAL_STOCKFISH_PATH}. Game evaluation disabled.")
        if schedule.game_eval_steps <= 0:
            print("Game evaluation frequency set to 0. Game evaluation disabled.")

    # Regret evaluation callback
    if schedule.regret_eval_steps > 0:
        print(
            f"Registering regret evaluation callback every {schedule.regret_eval_steps} steps "
            f"(num_positions={NUM_EVAL_POSITIONS}, phases: opening/middlegame/endgame)"
        )
        regret_callback = RegretEvaluationCallback(
            frequency=schedule.regret_eval_steps,
            dataset_path=DATASET_PATH,
            tokenizer=tokenizer,
            num_positions=NUM_EVAL_POSITIONS,
        )
        regret_callback.attach_trainer(trainer)
        trainer.add_callback(regret_callback)
    else:
        print("Regret evaluation frequency set to 0. Regret evaluation disabled.")

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
    parser = argparse.ArgumentParser(
        description="Train chess transformer model")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the W&B run (default: testz)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=768,
        help="Model hidden dimension (default: 768)"
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=None,
        help="FFN intermediate dimension (default: hidden-dim)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Number of transformer layers (default: 20)"
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate, overrides batch-scaled default (default: ~1.6e-4)"
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.98,
        help="Adam beta2 (default: 0.999)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
        help=f"Dropout rate (default: {DROPOUT})"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="Weight decay (default: 0.01)"
    )
    args = parser.parse_args()
    train(
        run_name=args.run_name,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        depth=args.depth,
        heads=args.heads,
        lr=args.lr,
        beta2=args.beta2,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
    )

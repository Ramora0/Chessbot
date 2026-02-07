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


OUTPUT_DIR = "/users/PAS2836/leedavis/personal/Chessbot/positional-funkiness"
DROPOUT = 0.1
MAX_SEQ_LENGTH = 256  # Board tokens: 72
DATASET_PATH = os.getenv("DATASET_PATH", "fuck")
SHUFFLE_DATASET = True
ELO_EVAL_STEPS = 16000
EVAL_BATCH_SIZE = 4096
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


# Set True to get exact backward traceback for NaN grads (~2x slower)
DETECT_ANOMALY = False

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

torch.set_float32_matmul_precision("high")


def _classify_param_group(name: str) -> str:
    """Classify a parameter name into a human-readable group for grad norm analysis."""
    if 'rel_pos_emb' in name or 'rel_query' in name or 'rel_key' in name or 'rel_value' in name:
        return 'rpe'
    if 'task_head' in name:
        return 'task_heads'
    if 'control_head' in name:
        return 'control_head'
    if 'lm_head' in name:
        return 'lm_head'
    if 'embed_tokens' in name:
        return 'embeddings'
    if 'layers.' in name:
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                return f'transformer_layer_{int(parts[i+1]):02d}'
        return 'transformer_other'
    return 'other'


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
        self._last_preclip_grad_norm: Optional[float] = None
        # NEW: Attention policy head losses
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

        # --- Blowup detection: dump all loss components if loss is bad ---
        loss_val = loss.detach().item()
        if not math.isfinite(loss_val) or loss_val > 1000:
            step = self.state.global_step
            print(f"\n{'!'*70}")
            print(f"LOSS BLOWUP at step {step}: total_loss={loss_val}")
            # Dump every individual loss component (unweighted)
            raw_model = model.module if hasattr(model, 'module') else model
            component_names = [
                'policy_loss', 'move_winrate_loss', 'illegality_loss',
                'winrate_loss', 'control_map_loss', 'masked_token_loss', 'mate_loss',
            ]
            for name in component_names:
                val = getattr(outputs, name, None)
                if val is not None:
                    v = val.detach().item()
                    weight_attr = name.replace('_loss', '_loss_weight')
                    w = float(getattr(raw_model, weight_attr, 1.0))
                    print(f"  {name:25s} = {v:12.4f}  (weight={w}, raw={v/w if w else v:.6f})")
            # RPE weight norms
            if hasattr(raw_model, 'rel_pos_embs'):
                print("  RPE weight norms:")
                for i, rpe in enumerate(raw_model.rel_pos_embs):
                    q_n = rpe.rel_query_board.weight.norm().item()
                    k_n = rpe.rel_key_board.weight.norm().item()
                    v_n = rpe.rel_value_board.weight.norm().item()
                    q_max = rpe.rel_query_board.weight.abs().max().item()
                    k_max = rpe.rel_key_board.weight.abs().max().item()
                    v_max = rpe.rel_value_board.weight.abs().max().item()
                    print(f"    Layer {i:2d}: Q(norm={q_n:.4f}, max={q_max:.4f})  "
                          f"K(norm={k_n:.4f}, max={k_max:.4f})  "
                          f"V(norm={v_n:.4f}, max={v_max:.4f})")
            # Input data stats
            if 'policy' in inputs and inputs['policy'] is not None:
                p = inputs['policy']
                tv = inputs.get('true_value')
                print(f"  Input policy: min={p.min().item():.4f} max={p.max().item():.4f} "
                      f"has_nan={bool(torch.isnan(p).any())}")
                if tv is not None:
                    print(f"  Input true_value: min={tv.min().item():.4f} max={tv.max().item():.4f} "
                          f"has_nan={bool(torch.isnan(tv).any())}")
            print(f"{'!'*70}\n")

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

        # NEW: Attention policy head losses
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to compute per-parameter-group gradient norms after backward."""
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        # Compute pre-clip gradient norms by parameter group
        raw_model = model.module if hasattr(model, 'module') else model
        total_norm_sq = 0.0
        group_norms_sq: dict[str, float] = {}
        for name, param in raw_model.named_parameters():
            if param.grad is not None:
                pn_sq = param.grad.norm().item() ** 2
                total_norm_sq += pn_sq
                group = _classify_param_group(name)
                group_norms_sq.setdefault(group, 0.0)
                group_norms_sq[group] += pn_sq
        total_norm = total_norm_sq ** 0.5
        group_norms = {k: v ** 0.5 for k, v in group_norms_sq.items()}

        self._last_preclip_grad_norm = total_norm

        if total_norm > 100:
            step = self.state.global_step
            print(f"\n{'#'*70}")
            print(f"GRAD EXPLOSION at step {step}: total_norm={total_norm:.2e}")
            print(f"Per-group grad norms (sorted):")
            for group, norm in sorted(group_norms.items(), key=lambda x: -x[1]):
                print(f"  {group:30s}: {norm:.2e}")
            # Top individual parameters
            param_norms = []
            for name, param in raw_model.named_parameters():
                if param.grad is not None:
                    param_norms.append((name, param.grad.norm().item(), param.grad.abs().max().item()))
            param_norms.sort(key=lambda x: -x[1])
            print(f"Top 10 parameters by grad norm:")
            for name, norm, mx in param_norms[:10]:
                print(f"  {name:60s}: norm={norm:.2e} max={mx:.2e}")

            # ── H1 + H3 + H4: Per-layer attention diagnostics ──
            from model import _rpe_diagnostics
            layers_diag = _rpe_diagnostics.get('layers', {})
            if layers_diag:
                print(f"\n--- H1: Per-layer attention score max (post-scale, post-cap) ---")
                for lid in sorted(layers_diag.keys()):
                    d = layers_diag[lid]
                    print(f"  Layer {lid:2d}: score_max={d['score_max']:7.2f}  "
                          f"score_min={d['score_min']:7.2f}  "
                          f"term1(QK)={d['term1_max']:6.2f}  "
                          f"term2(Q·RPE_K)={d['term2_max']:6.2f}  "
                          f"term3(RPE_Q·K)={d['term3_max']:6.2f}")

                print(f"\n--- H3: Per-head breakdown (layer with max score) ---")
                worst_layer = max(layers_diag.keys(), key=lambda l: layers_diag[l]['score_abs_max'])
                wd = layers_diag[worst_layer]
                print(f"  Worst layer: {worst_layer} (score_abs_max={wd['score_abs_max']:.2f})")
                for h in range(len(wd['per_head_max'])):
                    print(f"    Head {h}: max_score={wd['per_head_max'][h]:7.2f}  "
                          f"entropy={wd['per_head_entropy'][h]:5.2f}  "
                          f"max_attn_weight={wd['per_head_max_weight'][h]:.4f}")

                print(f"\n--- H2: Q/K/V vector norms per layer ---")
                for lid in sorted(layers_diag.keys()):
                    d = layers_diag[lid]
                    print(f"  Layer {lid:2d}: "
                          f"Q_mean={d['q_vec_norm']:6.2f}  Q_max={d['q_vec_max_norm']:6.2f}  "
                          f"K_mean={d['k_vec_norm']:6.2f}  K_max={d['k_vec_max_norm']:6.2f}  "
                          f"V_mean={d['v_vec_norm']:6.2f}")

            # ── H2: Per-layer hidden state norms (residual accumulation) ──
            hs_norms = _rpe_diagnostics.get('per_layer_hs_norms', [])
            if hs_norms:
                print(f"\n--- H2: Hidden state norms per layer (residual accumulation) ---")
                for i, norm in enumerate(hs_norms):
                    label = "embed" if i == 0 else f"layer {i-1:2d}"
                    print(f"  After {label:10s}: {norm:.2f}")

            # ── H5: Per-layer weight norms (W_Q, W_K, W_V, W_O) ──
            print(f"\n--- H5: Per-layer attention weight norms ---")
            unwrapped = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
            if hasattr(unwrapped, 'transformer'):
                for i, layer in enumerate(unwrapped.transformer.layers):
                    attn = layer.self_attn
                    q_wn = attn.q_proj.weight.norm().item()
                    k_wn = attn.k_proj.weight.norm().item()
                    v_wn = attn.v_proj.weight.norm().item()
                    o_wn = attn.o_proj.weight.norm().item()
                    print(f"  Layer {i:2d}: W_Q={q_wn:7.2f}  W_K={k_wn:7.2f}  "
                          f"W_V={v_wn:7.2f}  W_O={o_wn:7.2f}")

            # ── H6: Batch characteristics ──
            print(f"\n--- H6: Batch characteristics ---")
            if 'true_value' in inputs and inputs['true_value'] is not None:
                tv = inputs['true_value']
                print(f"  true_value: min={tv.min().item():.4f}  max={tv.max().item():.4f}  "
                      f"mean={tv.mean().item():.4f}  std={tv.std().item():.4f}")
                # Extreme win rates
                extreme_low = (tv < 0.05).sum().item()
                extreme_high = (tv > 0.95).sum().item()
                print(f"  extreme win rates: <0.05: {extreme_low}/{len(tv)}  >0.95: {extreme_high}/{len(tv)}")
            if 'policy' in inputs and inputs['policy'] is not None:
                p = inputs['policy']
                legal_per_sample = (p > -0.99).float().sum(dim=-1)
                print(f"  legal moves: min={legal_per_sample.min().item():.0f}  "
                      f"max={legal_per_sample.max().item():.0f}  "
                      f"mean={legal_per_sample.mean().item():.1f}")
                print(f"  policy has_nan={bool(torch.isnan(p).any())}  "
                      f"has_inf={bool(torch.isinf(p).any())}")
            if 'input_ids' in inputs:
                ids = inputs['input_ids']
                print(f"  input_ids: shape={list(ids.shape)}  "
                      f"has_nan={bool(torch.isnan(ids.float()).any())}")

            print(f"{'#'*70}\n")

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

            # Pre-clip gradient norm (catches explosions before clipping hides them)
            if self._last_preclip_grad_norm is not None:
                logs.setdefault("preclip_grad_norm",
                                self._last_preclip_grad_norm)

            # RPE weight norms (track growth over time in wandb)
            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(raw_model, 'rel_pos_embs'):
                q_norms, k_norms, v_norms = [], [], []
                for rpe in raw_model.rel_pos_embs:
                    q_norms.append(rpe.rel_query_board.weight.norm().item())
                    k_norms.append(rpe.rel_key_board.weight.norm().item())
                    v_norms.append(rpe.rel_value_board.weight.norm().item())
                logs["rpe_q_norm_mean"] = sum(q_norms) / len(q_norms)
                logs["rpe_k_norm_mean"] = sum(k_norms) / len(k_norms)
                logs["rpe_v_norm_mean"] = sum(v_norms) / len(v_norms)
                logs["rpe_q_norm_max"] = max(q_norms)
                logs["rpe_k_norm_max"] = max(k_norms)
                logs["rpe_v_norm_max"] = max(v_norms)

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

        return control


def train(run_name: Optional[str] = None) -> None:
    print("Starting chess transformer training...")

    if DETECT_ANOMALY:
        torch.autograd.set_detect_anomaly(True)
        print("WARNING: torch.autograd.detect_anomaly enabled — training will be ~2x slower")

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

    print(f"Training will run for {schedule.max_steps} steps")
    print(
        "Training schedule:",
        f"batch_size={per_device_batch_size}",
        f"learning_rate={schedule.learning_rate}",
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

        gradient_accumulation_steps=2,

        per_device_train_batch_size=per_device_batch_size // 2,
        learning_rate=schedule.learning_rate,
        warmup_steps=schedule.warmup_steps,
        weight_decay=0.01,
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
    args = parser.parse_args()
    train(run_name=args.run_name)

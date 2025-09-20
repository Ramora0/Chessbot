#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulate puzzle ratings from a triangular distribution, generate model solve
probabilities across a sweep of true model Elos, add noise, estimate Elo back,
and plot deviation vs true rating.

- Puzzle rating distribution: triangular with (left=400, mode=1000, right=2900)
- Number of puzzles: 10,000
- Model ratings evaluated: 400..3500 (step configurable)
- Noise injected in logit space (Gaussian with sigma = noise_sigma)
- Two estimators:
    1) MLE (1-parameter logistic fit)
    2) Inverted-per-puzzle Elo with Fisher weights q*(1-q)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ------------------------------- Config ------------------------------------ #
SEED = 42
N_PUZZLES = 10_000
TRI_LEFT, TRI_MODE, TRI_RIGHT = 400.0, 1000.0, 2900.0
R_MIN, R_MAX, R_STEP = 0, 4000, 1          # sweep true ratings
NOISE_SIGMA = 0.6
CLIP_EPS = 1e-6                                # avoid exact 0/1 probs
DO_TRIM = True                                # optionally trim to informative band
TRIM_LOW, TRIM_HIGH = 0.1, 0.9                 # only used if DO_TRIM=True
# light moving average for final Elo traces
SMOOTH_WINDOW = 5
# --------------------------------------------------------------------------- #


# ------------------------------- Helpers ----------------------------------- #
LOG10 = np.log(10.0)
K = LOG10 / 400.0  # natural-log logistic slope for Elo diff

rng = np.random.default_rng(SEED)


def clip01(x, eps=CLIP_EPS):
    return np.clip(x, eps, 1.0 - eps)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = clip01(p)
    return np.log(p / (1.0 - p))


def smooth_moving_average(values, window=1):
    """Apply a lightweight moving average while preserving array length."""
    if window <= 1:
        return values

    values = np.asarray(values)
    kernel = np.ones(window, dtype=values.dtype) / float(window)
    left_pad = window // 2
    right_pad = window - 1 - left_pad
    padded = np.pad(values, (left_pad, right_pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def elo_solve_prob(model_rating, puzzle_rating):
    """Elo logistic: P(solve) = sigmoid(K * (model - puzzle))."""
    return sigmoid(K * (model_rating - puzzle_rating))


def estimate_elo_mle(puzzle_ratings, q, init=None, max_iter=50, tol=1e-3):
    """
    One-parameter MLE for Elo rating R that best matches soft targets q_i
    under the Elo logistic s_i(R) = sigmoid(K*(R - r_i)).

    Returns (R_hat, SE), where SE is from the observed Fisher information.
    """
    q = clip01(q)
    if init is None:
        init = float(np.median(puzzle_ratings))
    R = float(init)

    for _ in range(max_iter):
        s = sigmoid(K * (R - puzzle_ratings))
        # Gradient (w.r.t. R) and Fisher information (positive)
        g = np.sum((q - s) * K)
        I = np.sum(s * (1 - s) * K * K)
        if I <= 0:
            break
        step = g / I
        R_new = R + step
        if abs(R_new - R) < tol:
            R = R_new
            break
        R = R_new

    s = sigmoid(K * (R - puzzle_ratings))
    I = np.sum(s * (1 - s) * K * K)
    SE = 1.0 / np.sqrt(I) if I > 0 else np.nan
    return R, SE


def estimate_elo_inverted_weighted(puzzle_ratings, q, do_trim=DO_TRIM,
                                   trim_low=TRIM_LOW, trim_high=TRIM_HIGH):
    """
    Simple closed-form estimator:
      R_i = r_i + 400 * log10(q_i / (1 - q_i))
      R_hat = weighted mean of R_i with weights w_i = q_i * (1 - q_i)

    Optionally trims puzzles to an informative band q in [trim_low, trim_high].
    """
    q = clip01(q)
    if do_trim:
        mask = (q >= trim_low) & (q <= trim_high)
        if not np.any(mask):  # fall back if trimming wiped everything
            mask = slice(None)
    else:
        mask = slice(None)

    q_ = q[mask]
    r_ = puzzle_ratings[mask]

    Ri = r_ + 400.0 * np.log10(q_ / (1.0 - q_))
    w = q_ * (1.0 - q_)
    R_hat = np.sum(w * Ri) / np.sum(w)
    return R_hat


# -------------------------- Generate puzzle ratings ------------------------ #
puzzle_ratings = rng.triangular(TRI_LEFT, TRI_MODE, TRI_RIGHT, size=N_PUZZLES)

# ------------------------- Sweep true model ratings ------------------------ #
model_ratings = np.arange(R_MIN, R_MAX + 1, R_STEP)

est_mle = []
se_mle = []
est_inv = []

for R_true in tqdm(model_ratings):
    # True per-puzzle solve probabilities
    p_true = elo_solve_prob(R_true, puzzle_ratings)

    # Add randomness in logit space to simulate calibration/noise
    z = logit(p_true) + rng.normal(0.0, NOISE_SIGMA, size=N_PUZZLES)
    q_obs = clip01(sigmoid(z))

    # Estimate Elo by MLE
    R_mle, se = estimate_elo_mle(
        puzzle_ratings, q_obs, init=np.median(puzzle_ratings))
    est_mle.append(R_mle)
    se_mle.append(se)

    # Estimate Elo by inverted-per-puzzle (weighted)
    R_inv = estimate_elo_inverted_weighted(puzzle_ratings, q_obs)
    est_inv.append(R_inv)

est_mle = np.array(est_mle)  # - 28) * 1.0177852349
se_mle = np.array(se_mle)
est_inv = np.array(est_inv)

est_mle = smooth_moving_average(est_mle, SMOOTH_WINDOW)
se_mle = smooth_moving_average(se_mle, SMOOTH_WINDOW)
est_inv = smooth_moving_average(est_inv, SMOOTH_WINDOW)

dev_mle = est_mle - model_ratings
dev_inv = est_inv - model_ratings

# ------------------------------- Plot -------------------------------------- #
plt.figure(figsize=(8, 5))
# plt.plot(model_ratings, model_ratings, label="True", linestyle="--")
line_mle, = plt.plot(model_ratings, dev_mle, label="MLE (1-parameter logistic)")
plt.fill_between(
    model_ratings,
    dev_mle - se_mle,
    dev_mle + se_mle,
    color=line_mle.get_color(),
    alpha=0.25,
    linewidth=0,
    label="MLE ± SE",
)
# plt.plot(model_ratings, dev_inv, linestyle="--",
#          label="Inverted + Fisher weights")

plt.axhline(0.0, linewidth=1)
plt.title("Elo Estimation Deviation vs True Rating\n(estimated − true Elo)")
plt.xlabel("True Model Rating")
plt.ylabel("Deviation (Elo)")
subtitle = (
    f"Puzzles: {N_PUZZLES}, Tri({int(TRI_LEFT)}, {int(TRI_MODE)}, {int(TRI_RIGHT)}), "
    f"Noise σ={NOISE_SIGMA}, step={R_STEP}"
)
plt.suptitle(subtitle, y=0.96, fontsize=9)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------- Quick console summary -------------------------- #
mae_mle = np.mean(np.abs(dev_mle))
mae_inv = np.mean(np.abs(dev_inv))
print(f"Mean Abs Error (MLE): {mae_mle:.1f} Elo")
print(f"Mean diff: {np.mean(dev_mle)}")
print(f"Mean Abs Error (Inverted+Weighted): {mae_inv:.1f} Elo")

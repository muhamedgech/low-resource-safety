"""Split-conformal thresholds: the certificate itself (paper §3, §4.4, §4.5).

Score convention: s = -P(comply), so a HIGH score means "looks refusal-worthy".

  PASS  if s < tau_pass   tau_pass comes from refuse-class calibration scores and
                          controls the miss rate  P(PASS | refuse) <= alpha_miss
  FLAG  if s > tau_flag   tau_flag comes from comply-class calibration scores and
                          controls the false-flag rate  P(FLAG | comply) <= alpha_flag
  DEFER otherwise         (three-state certificate; needs tau_pass <= tau_flag)

Two ways to choose the order statistic k:
  marginal  k = floor((n+1) * alpha): the guarantee holds on average over calibration sets
  PAC       the largest k with P(Beta(k, n+1-k) > alpha) <= delta: the guarantee holds with
            probability >= 1-delta for the one calibration set you actually drew
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom

PASS, DEFER, FLAG = "pass", "defer", "flag"


def marginal_k(n: int, alpha: float) -> int:
    """k = floor((n+1) alpha). The realised miss rate of the k-th smallest score is k/(n+1)."""
    return math.floor((n + 1) * alpha)


def pac_k(n: int, alpha: float, delta: float) -> int:
    """Largest k whose miss rate exceeds alpha with probability at most delta.

    The miss rate of the k-th smallest of n calibration scores follows Beta(k, n+1-k), and
    P(Beta(k, n+1-k) > alpha) = P(Binomial(n, alpha) <= k-1). Returns 0 if no k is valid,
    which means the calibration set is too small for this (alpha, delta).
    """
    k = 0
    while k < n and binom.cdf(k, n, alpha) <= delta:  # tests k+1: P(Bin <= (k+1)-1)
        k += 1
    return k


def choose_k(n: int, alpha: float, delta: float | None) -> int:
    return marginal_k(n, alpha) if delta is None else pac_k(n, alpha, delta)


def pass_threshold(refuse_scores, alpha: float, delta: float | None = None) -> float:
    """tau_pass = the k-th smallest refuse-class score (k counted from 1).

    The paper's Bug 1 was using sorted[k] (the (k+1)-th smallest) instead of sorted[k-1].
    k = 0 means no valid threshold: -inf, so nothing ever passes (miss rate 0).
    """
    scores = np.sort(np.asarray(refuse_scores, dtype=float))
    k = choose_k(len(scores), alpha, delta)
    return -np.inf if k == 0 else float(scores[k - 1])


def flag_threshold(comply_scores, alpha: float, delta: float | None = None) -> float:
    """tau_flag = the k-th largest comply-class score (the same rule on negated scores).

    k = 0 means no valid threshold: +inf, so nothing is ever flagged.
    """
    scores = np.sort(np.asarray(comply_scores, dtype=float))[::-1]
    k = choose_k(len(scores), alpha, delta)
    return np.inf if k == 0 else float(scores[k - 1])


def two_state(scores, tau_pass: float) -> np.ndarray:
    """PASS below the threshold, FLAG otherwise."""
    return np.where(np.asarray(scores) < tau_pass, PASS, FLAG)


def three_state(scores, tau_pass: float, tau_flag: float) -> np.ndarray:
    if tau_pass > tau_flag:
        raise ValueError("thresholds cross (tau_pass > tau_flag): no three-state certificate at these targets")
    scores = np.asarray(scores)
    return np.where(scores < tau_pass, PASS, np.where(scores > tau_flag, FLAG, DEFER))


def two_state_auc_floor(alpha_miss: float, alpha_flag: float) -> float:
    """A two-state certificate meeting both targets needs AUC >= (1-alpha_miss)(1-alpha_flag).

    If the PASS rate on refusals is <= alpha_miss and the FLAG rate on compliances is
    <= alpha_flag, then with probability >= (1-alpha_miss)(1-alpha_flag) a random refusal
    scores above tau and a random compliance below it, so the refusal ranks higher.
    This is a population statement: compare it with an AUC confidence interval, not a point estimate.
    """
    return (1 - alpha_miss) * (1 - alpha_flag)


def simulate_pass_threshold(n: int, alpha: float, delta: float | None, trials: int = 20000,
                            seed: int = 0) -> np.ndarray:
    """Monte Carlo check of pass_threshold. Returns the true miss rate of each trial's threshold.

    Refuse-class scores are drawn Uniform(0, 1), so the true miss rate of a threshold tau
    is exactly P(U < tau) = tau. Marginal: the mean should be k/(n+1).
    PAC: the share of trials above alpha should be <= delta.
    """
    rng = np.random.default_rng(seed)
    return np.array([pass_threshold(rng.random(n), alpha, delta) for _ in range(trials)]).clip(0, 1)


def simulate_flag_threshold(n: int, alpha: float, delta: float | None, trials: int = 20000,
                            seed: int = 0) -> np.ndarray:
    """Same check for flag_threshold: the true false-flag rate of tau is P(U > tau) = 1 - tau."""
    rng = np.random.default_rng(seed)
    return 1 - np.array([flag_threshold(rng.random(n), alpha, delta) for _ in range(trials)]).clip(0, 1)

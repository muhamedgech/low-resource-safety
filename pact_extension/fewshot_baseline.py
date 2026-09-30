"""Aziz-style few-shot gate, reimplemented faithfully as the PACT baseline.

Their recipe (src/multilingual_latent_safety/probes.py + probe_evaluation.py):
  1. train an L2 logistic probe on English harmful/harmless activations,
  2. draw k target examples per class, append them, refit,
  3. pick the threshold that maximizes macro-F1 on those few examples.

Positive class = harmful, so a high score means "harmful". For a gate:
  FLAG  if score >  threshold
  miss  = harmful prompt with score <= threshold  (a dangerous prompt let through)
  false-flag = harmless prompt with score >  threshold

This module keeps the threshold logic in numpy and pushes the (torch) probe fit
to the caller, so it is testable without their activation cache.
"""
from __future__ import annotations

import numpy as np


def macro_f1_at(threshold: float, harmful: np.ndarray, harmless: np.ndarray) -> float:
    """Macro-F1 of the two classes at a FLAG-if-above threshold."""
    tp = int((harmful > threshold).sum())            # harmful flagged
    fn = int((harmful <= threshold).sum())           # harmful missed
    tn = int((harmless <= threshold).sum())          # harmless passed
    fp = int((harmless > threshold).sum())           # harmless flagged
    f1_harmful = _f1(tp, fp, fn)
    f1_harmless = _f1(tn, fn, fp)                     # swap roles for the negative class
    return 0.5 * (f1_harmful + f1_harmless)


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def fewshot_threshold(harmful: np.ndarray, harmless: np.ndarray) -> float:
    """Their `select_language_thresholds`: the candidate maximizing macro-F1.

    Candidates are the observed scores; ties resolve to the first (lowest) score,
    matching torch.argmax on the F1 curve.
    """
    candidates = np.unique(np.concatenate([harmful, harmless]))
    if candidates.size == 0:
        raise ValueError("cannot choose a threshold without calibration scores")
    f1 = np.array([macro_f1_at(t, harmful, harmless) for t in candidates])
    return float(candidates[int(np.argmax(f1))])


def gate_metrics(threshold: float, harmful: np.ndarray, harmless: np.ndarray) -> dict:
    """Realized miss rate and false-flag rate on a held-out split."""
    miss = float((harmful <= threshold).mean()) if harmful.size else float("nan")
    false_flag = float((harmless > threshold).mean()) if harmless.size else float("nan")
    return {"miss_rate": miss, "false_flag_rate": false_flag,
            "flagged_fraction": float(np.concatenate([harmful, harmless]) .__gt__(threshold).mean())}


if __name__ == "__main__":
    # Synthetic self-test: harmful scores centered high, harmless low.
    rng = np.random.default_rng(0)
    h = rng.normal(1.0, 1.0, 200)
    n = rng.normal(-1.0, 1.0, 200)
    thr = fewshot_threshold(h[:8], n[:8])            # k=8 per class
    m = gate_metrics(thr, h[8:], n[8:])
    print(f"threshold={thr:.3f} miss={m['miss_rate']:.3f} false_flag={m['false_flag_rate']:.3f}")
    assert 0.0 <= m["miss_rate"] <= 0.5 and 0.0 <= m["false_flag_rate"] <= 0.5
    print("ok")

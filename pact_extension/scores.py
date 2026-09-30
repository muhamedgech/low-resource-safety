"""PACT score functions for the PolyRefuse experiment (single fixed layer, numpy in).

Ported/simplified from the safty repo's pact/scores.py. Three scores share one certificate:

  A0  probe trained on English features (gold harmful=1), applied to the target language
  A2  A0 after Procrustes-aligning the target language's space onto English (paired prompts)
  TT  probe trained directly on target-language features (translate-train)

Every score returns s = -P(harmful), so HIGH s = "looks harmful/refusal-worthy".

Exact speed trick: with L2 the logistic weights lie in the span of the training rows,
so we fit in that n-dim span instead of the ~4096-dim space (identical model, ~100x faster).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class Probe:
    mean: np.ndarray
    std: np.ndarray
    basis: np.ndarray
    clf: LogisticRegression

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = ((X.astype(np.float64) - self.mean) / self.std) @ self.basis
        return -self.clf.predict_proba(Z)[:, 1]


def _span(X: np.ndarray):
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    Xs = (X - mean) / std
    _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    basis = Vt[S > 1e-8 * S[0]].T
    return mean, std, basis, Xs @ basis


def fit_probe(X: np.ndarray, y: np.ndarray, C: float = 0.1) -> Probe:
    """L2 logistic probe on standardized features, fitted in the training span."""
    X = X.astype(np.float64)
    mean, std, basis, Z = _span(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(Z, y.astype(int))
    return Probe(mean, std, basis, clf)


@dataclass
class ProcrustesProbe:
    en_mean: np.ndarray
    tgt_mean: np.ndarray
    components: np.ndarray   # (d, K)
    rotation: np.ndarray     # (K, K)
    probe: Probe

    def score(self, X_target: np.ndarray) -> np.ndarray:
        return self.probe.score(((X_target.astype(np.float64) - self.tgt_mean) @ self.components) @ self.rotation)


def fit_procrustes(X_en: np.ndarray, X_tgt: np.ndarray, y: np.ndarray, k: int = 64, C: float = 0.1) -> ProcrustesProbe:
    """Align the target space onto English using paired rows (same prompt, both languages).

    R = argmin ||P_tgt R - P_en|| over orthogonal R (SVD), after projecting both onto the
    top-k principal components of the centred English features. The probe is trained on
    English's own projections; target scoring centers, projects, then rotates.
    """
    X_en, X_tgt = X_en.astype(np.float64), X_tgt.astype(np.float64)
    en_mean, tgt_mean = X_en.mean(0), X_tgt.mean(0)
    _, _, Vt = np.linalg.svd(X_en - en_mean, full_matrices=False)
    comp = Vt[:min(k, X_en.shape[0] - 1)].T
    P_en = (X_en - en_mean) @ comp
    P_tgt = (X_tgt - tgt_mean) @ comp
    U, _, Wt = np.linalg.svd(P_tgt.T @ P_en)
    rotation = U @ Wt
    mean, std, basis, Z = _span(P_en)
    clf = LogisticRegression(C=C, max_iter=5000).fit(Z, y.astype(int))
    return ProcrustesProbe(en_mean, tgt_mean, comp, rotation, Probe(mean, std, basis, clf))

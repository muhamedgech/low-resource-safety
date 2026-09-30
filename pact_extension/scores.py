"""PACT score functions (A0 / A2 / TT) — GPU-accelerated with torch.

Same interface as before (fit_probe / fit_procrustes; .score() takes and returns numpy),
but the SVDs and the logistic fit run on `DEVICE` (cuda if available). Train on the SAFE
class (1 = harmless) so score = -P(safe) is HIGH for harmful (refusal-worthy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to(t):
    return torch.as_tensor(t, dtype=torch.float32, device=DEVICE)


def _fit_logistic(Z: torch.Tensor, y: torch.Tensor, C: float, steps: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """L2 logistic regression via LBFGS on GPU. lam = 1/C matches sklearn's convention scale."""
    w = torch.zeros(Z.shape[1], device=Z.device, requires_grad=True)
    b = torch.zeros(1, device=Z.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")
    lam = 1.0 / float(C)

    def closure():
        opt.zero_grad()
        logits = Z @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y) + lam * w.pow(2).sum() / Z.shape[0]
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def _span(Xs: torch.Tensor):
    """Orthonormal basis of the row span via SVD (economy). Xs already standardized."""
    _, S, Vh = torch.linalg.svd(Xs, full_matrices=False)
    keep = S > 1e-8 * S[0]
    return Vh[keep].T  # (d, r)


@dataclass
class Probe:
    mean: torch.Tensor
    std: torch.Tensor
    basis: torch.Tensor
    w: torch.Tensor
    b: torch.Tensor

    def score(self, X) -> np.ndarray:
        Xt = _to(X)
        Z = ((Xt - self.mean) / self.std) @ self.basis
        p = torch.sigmoid(Z @ self.w + self.b)   # P(safe)
        return (-p).cpu().numpy()                # high = harmful


def fit_probe(X, y, C: float = 0.1) -> Probe:
    Xt = _to(X)
    mean = Xt.mean(0)
    std = Xt.std(0) + 1e-6
    Xs = (Xt - mean) / std
    basis = _span(Xs)
    Z = Xs @ basis
    w, b = _fit_logistic(Z, _to(np.asarray(y)), C)
    return Probe(mean, std, basis, w, b)


@dataclass
class ProcrustesProbe:
    tgt_mean: torch.Tensor
    components: torch.Tensor
    rotation: torch.Tensor
    probe: Probe

    def score(self, X_target) -> np.ndarray:
        Xt = _to(X_target)
        aligned = ((Xt - self.tgt_mean) @ self.components) @ self.rotation
        return self.probe.score(aligned.cpu().numpy())


def fit_procrustes(X_en, X_tgt, y, k: int = 64, C: float = 0.1) -> ProcrustesProbe:
    Xen = _to(X_en)
    Xtg = _to(X_tgt)
    en_mean, tgt_mean = Xen.mean(0), Xtg.mean(0)
    _, _, Vh = torch.linalg.svd(Xen - en_mean, full_matrices=False)
    comp = Vh[:min(k, Xen.shape[0] - 1)].T           # (d, K)
    P_en = (Xen - en_mean) @ comp
    P_tgt = (Xtg - tgt_mean) @ comp
    U, _, Wh = torch.linalg.svd(P_tgt.T @ P_en)
    rotation = U @ Wh                                 # (K, K)
    mean = P_en.mean(0)
    std = P_en.std(0) + 1e-6
    Ps = (P_en - mean) / std
    basis = _span(Ps)
    w, b = _fit_logistic(Ps @ basis, _to(np.asarray(y)), C)
    probe = Probe(mean, std, basis, w, b)
    return ProcrustesProbe(tgt_mean, comp, rotation, probe)

# PACT extension

This folder contains **new work built on top of** Aziz et al.,
*"Low-Resource Safety Failures Are Action Failures, Not Representation Failures"*
(EMNLP 2026 Findings; original repo: https://github.com/rashadaziz/low-resource-safety).
Everything **outside** this folder is the original authors' code, unchanged.

## Why this folder exists

Aziz et al. show that in low-resource languages the harmfulness **representation**
transfers but the refusal **decision** does not, and they repair the decision by
recalibrating a gate with **1–64 labeled target-language examples per class**,
selecting the threshold by macro-F1.

PACT (Parallel-Anchored Conformal Transfer) asks the next question:

> Can we recalibrate the decision **with zero target-language labels** and a
> **finite-sample error guarantee**, instead of a few labels and an F1 point estimate?

PACT does this by transporting the **English model's own decision** to parallel
translations and calibrating a **split-conformal threshold per language**.

## What is theirs vs. ours

| Theirs (this repo, everywhere else) | Ours (this folder only) |
|---|---|
| PolyRefuse dataset, activation extraction | reused as-is, not modified |
| harmfulness direction / logistic probe | reused as the score function |
| few-shot gate: k labels/class + macro-F1 threshold | reimplemented faithfully as the **baseline** |
| steering interventions (VHRL, AdaSteer, CAST) | not used |
| — | **conformal certificate** (miss-rate / false-flag guarantee) |
| — | **transported-label calibration** (0 target labels) |
| — | **PACT vs. few-shot comparison** on the same activations |

## Contents

- `docs/aziz_summary.md` — plain-language summary of their dataset, method and
  results (for the related-work section of the PACT paper).
- `docs/comparison_plan.md` — the experiment that puts Aziz's few-shot gate and
  PACT head-to-head on the identical PolyRefuse activations.
- (code lands here next: `conformal.py`, `fewshot_baseline.py`, `run_comparison.py`)

## Attribution and license

This extension does not modify the original authors' files. The original
LICENSE at the repository root continues to govern their code. Cite Aziz et al.
(arXiv:2606.01196) for anything under `src/`, `scripts/`, `configs/`.

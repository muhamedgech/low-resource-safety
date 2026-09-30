# Aziz et al. (2026) — summary for the PACT related-work section

*"Low-Resource Safety Failures Are Action Failures, Not Representation Failures"*,
EMNLP 2026 Findings, arXiv:2606.01196. Repo: rashadaziz/low-resource-safety.
This summary is derived from reading their code and README directly.

## Thesis

In low-resource languages the model still **represents** that a prompt is harmful,
but fails to **act** on it (does not refuse). The failure is in the decision/routing,
not the representation — so the fix is recalibration, not better features, and needs
only a little target-language data.

## Dataset: PolyRefuse

- From Wang et al. (`mainlp/Multilingual-Refusal`).
- 23 languages, 3 tiers by CommonCrawl share:
  - **high** (>1%): en, de, fr, es, it, nl, pl, ru, zh, ja
  - **mid** (0.1–1%): ar, ko, th, el, he, hi, fa
  - **low** (<0.1%): sw, am, my, km, si, yo  (Amharic is low-resource)
- Each language: `harmful` and `harmless` prompts, split train/val/test.
- English prompts machine-translated to every language.
- Translation quality checked by back-translation → BLEU + SBERT (same idea as
  PACT's §4.3 audit).

## Method (six experiments)

1. **Refusal gap.** Generate completions, judge refusal (OpenRouter judge or guard
   models). Harmful-prompt refusal drops 87.9% (en) → 43.9% (low-resource).
2. **Harmfulness direction.** Residual activations → per-language mean-difference
   direction (mean_harmful − mean_harmless), pooled over high-resource languages.
   Layers: Qwen 15, Gemma 20, **Llama-3.1-8B layer 10**.
3. **Recoverability (key result).** The high-resource direction/probe still
   separates harmful vs harmless in low-resource languages (AUC stays high) even
   where refusal behavior collapsed. Tools: **U-Score** (matched-pair cosine minus
   random-pair baseline; measures representation alignment across languages) and a
   **few-shot gate** (k per class, budgets 1–64, 10 seeds; append to English train,
   refit, pick threshold by macro-F1).
4. **Activation-strength sweep.** Pushing the direction harder during generation
   restores refusal → the direction *causally* controls refusal.
5. **OOD transfer.** Gate tested on MultiJail / IndoSafety to show it is not
   PolyRefuse-specific.
6. **Routing into refusal.** Steering methods (conditional VHRL, directional
   ablation, AdaSteer, CAST) restore refusal at inference, measured against
   Global-MMLU utility. Selectivity 33.6 → 54.5 with utility preserved.

## Results, condensed

| Claim | Number |
|---|---|
| Refusal gap (harmful refusal, en → low-resource) | 87.9% → 43.9% |
| Representation still separates in low-resource | AUC stays high |
| Fix with few labels | 1–4/class: selectivity 33.6 → 54.5 |
| Models | Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B |

## Relationship to PACT

- Same finding shape: representation transfers, the *decision* does not. Aziz frame
  it as **behavior**; PACT frames it as **calibration** (certificate FPR violates 19×).
- Their fix: recalibrate with **1–64 real target labels**, threshold by macro-F1,
  **no guarantee**. PACT: recalibrate with **0 target labels** (transported English
  decision), split-conformal threshold, **finite-sample guarantee**.
- Aziz treat the six low-resource languages uniformly; PACT isolates **Amharic** as a
  distinct near-chance failure even after recalibration.

**Positioning line for the paper:** PACT is the certification layer on Aziz's
diagnosis — "recalibrate" becomes "recalibrate with a label-free finite-sample
guarantee, and here is exactly where it still fails (Amharic)."

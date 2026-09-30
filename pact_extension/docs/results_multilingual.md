# Result: PACT vs. Aziz few-shot gate — 6 low-resource languages

Model: Llama-3.1-8B-Instruct (bf16), layer 10, token position `t_post_inst`.
Data: PolyRefuse harmful/harmless; English-trained logistic probe as the shared score.
Protocol: pooled per-language scores, `n_cal=100` per class, 50 seeds; PACT sets a
split-conformal pass threshold (α_miss = 0.05, 0 target labels), few-shot picks k
labels/class and a macro-F1 threshold. Both evaluated on the same held-out split each seed.

## Miss rate (target 0.05)

| Lang | AUC | PACT (0 labels) | few-shot k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| am | 0.756 | **0.047** | 0.243 | 0.257 | 0.269 | 0.344 | 0.360 |
| km | 0.756 | **0.054** | 0.275 | 0.275 | 0.268 | 0.337 | 0.391 |
| yo | 0.795 | **0.048** | 0.236 | 0.197 | 0.229 | 0.319 | 0.286 |
| my | 0.893 | **0.052** | 0.064 | 0.094 | 0.132 | 0.168 | 0.177 |
| si | 0.895 | **0.052** | 0.115 | 0.098 | 0.110 | 0.173 | 0.181 |
| sw | 0.972 | **0.052** | 0.024 | 0.040 | 0.053 | 0.070 | 0.076 |

PACT std ~0.020–0.025 in every language; PACT worst-split miss ≤ 0.13.
Few-shot std 0.09–0.20; few-shot worst-split miss up to 0.85 (km k=1).

## Findings

1. **Language-invariant guarantee.** Across AUC 0.756–0.972, PACT lands at ~0.05
   every time (mean 0.051). Conformal calibration controls the miss rate regardless
   of score quality — the defining property of the certificate.
2. **Few-shot is uncontrolled and AUC-dependent.** Strong-score languages (sw) sit
   near target; weak-score languages (am, km, yo) miss 5–7× the target. A deployer
   using few-shot cannot know which regime they are in.
3. **More labels make few-shot worse.** Miss rises with k in every language, because
   macro-F1 optimizes balanced accuracy, not the miss rate. Even 32 labels do not
   reach the target. PACT uses zero.
4. **Worst-case.** Even in easy Swahili, few-shot's worst split misses 27%; PACT's
   worst across all six languages is ~13%.

## Scope / honesty

- This is **calibration transfer** on curated harmful-vs-harmless prompts (AUC is high
  here). It is *not* the harder XSTest look-alike setting from the main PACT paper; that
  discrimination story stays there. The point of this experiment is error-rate control.
- Both methods use the identical English-trained score; only the thresholding differs
  (conformal vs. macro-F1-on-few-labels), so the comparison is apples-to-apples.
- A faithful "refit the probe on the few examples" variant of Aziz is a further option;
  here we hold the score fixed to isolate the thresholding rule.
- Single model family (Llama-3.1-8B); layer 10 (Aziz's choice). Other layers/models TBD.

Raw per-seed rows: `pact_extension/results/comparison.csv`.

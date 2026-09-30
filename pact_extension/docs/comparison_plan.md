# Experiment plan: Aziz few-shot gate vs. PACT on PolyRefuse

Goal: on the **same** PolyRefuse activations, answer the reviewer question
"why not just label a few target examples?" by comparing, per language and per
budget, **realized miss rate** and **label cost**.

## Shared inputs (reuse their pipeline, nothing new)

- Model: Llama-3.1-8B-Instruct, layer 10 (their config). Later: Qwen, Gemma.
- Activations: their `scripts/hrl_direction/extract_activations.py`, token position
  `t_post_inst` — the last post-instruction token (their probe's feature).
- Languages: en (source) + low-resource {sw, am, my, km, si, yo}. Amharic is the focus.
- Probe/score: their L2 logistic probe trained on English harmful/harmless.

## Two methods, same score, same splits

For each language and each seed:

1. **Aziz few-shot (baseline, faithful).**
   - Draw k per class from the target `train` split (k ∈ {1,2,4,8,16}).
   - Append to the English training set, refit the probe (their `budgeted_logistic_scores`).
   - Set the threshold to **maximize macro-F1** on those k×2 examples
     (their `select_language_thresholds`).
   - Report realized **miss rate** and **false-flag rate** on the target `test` split.

2. **PACT (ours, 0 labels).**
   - Same English-trained probe (no refit needed).
   - Transported label: the English model's own refuse/comply decision on the English
     prompt, copied to its translation. (On PolyRefuse harmful/harmless this is close
     to the gold label; on a deployed instruct model it is the model's decision.)
   - Split-conformal threshold on the target calibration split, α_miss = 0.05.
   - Report realized miss rate and false-flag rate on the target `test` split.

## Metrics (what the figure shows)

- **Realized miss rate vs. budget k**, per language, with spread over 10 seeds.
- **Label cost**: Aziz needs 2k real labels; PACT needs 0.
- Expected story: at k = 1–4 Aziz's miss rate is unstable (swings well above and
  below target because macro-F1 on a handful of points is noisy and carries no error
  guarantee); PACT holds ≈ α every time. At larger k Aziz's discrimination improves,
  but PACT's advantage is **guaranteed error control at zero label cost**, so frame
  the comparison on miss-rate validity and label cost, not raw F1.

## Honest caveats to report

- On PolyRefuse the transported label ≈ gold label (these are curated harmful/harmless
  prompts, not a deployed model's imperfect decisions), so this experiment tests
  **calibration transfer**, not the full "faithful to a deployed boundary" claim —
  that part stays with the XSTest/OR-Bench experiments in the main PACT paper.
- Amharic is expected to stay weak under both methods; the point is that PACT stays
  *valid* (miss rate controlled) even when discrimination is poor, which is exactly
  what a conformal guarantee gives.

## Code to add here (next)

- `conformal.py` — split-conformal pass/flag thresholds (ported from the safty repo,
  Monte-Carlo verified).
- `fewshot_baseline.py` — thin wrapper over their probe + macro-F1 threshold.
- `run_comparison.py` — loads their activation cache, runs both methods over
  languages × budgets × seeds, writes a tidy CSV.

No model runs are needed beyond their existing activation extraction; both methods
consume the cached activations.

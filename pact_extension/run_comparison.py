"""Aziz few-shot gate vs. PACT conformal gate on the SAME PolyRefuse activations.

Answers the reviewer question "why not just label a few target examples?" by
reporting, per language and budget, the realized miss rate and the label cost.

It reuses the original authors' code for everything upstream of the threshold:
  - ActivationCache.pair(language, split) -> (harmful, harmless) activations
  - logistic_regression_scores            -> the English-trained probe

Two methods, compared on the identical held-out target `test` split:
  few-shot (baseline)  refit the probe with k target examples/class, threshold by
                       macro-F1 on those examples  (needs 2k real target labels)
  PACT (ours)          English-trained probe score, split-conformal threshold on the
                       target calibration split labeled by transported decision
                       (needs 0 target labels)

On PolyRefuse the transported English decision is approximated by the gold
harmful/harmless class (curated prompts). This isolates *calibration transfer*;
the "faithful to a deployed boundary" claim is tested elsewhere (XSTest/OR-Bench).

Run on a machine that has the authors' activation cache:
    python pact_extension/run_comparison.py \
        --activations-root artifacts/activations/meta-llama/Llama-3.1-8B-Instruct \
        --layer 10 --out pact_extension/results/comparison_llama.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import conformal  # noqa: E402
from fewshot_baseline import fewshot_threshold, gate_metrics  # noqa: E402

LOW_RESOURCE = ["sw", "am", "my", "km", "si", "yo"]
BUDGETS = [1, 2, 4, 8, 16]


def english_probe_scores(cache, source_lang, target_langs, splits, layer, seed):
    """Train the authors' logistic probe on English, return {(lang,split): (h,n) scores}."""
    import torch
    from multilingual_latent_safety.probes import binary_labels, logistic_regression_scores

    h_en, n_en = cache.pair(source_lang, "train")
    train_x = torch.cat([h_en, n_en], dim=0)
    train_y = binary_labels(h_en.shape[0], n_en.shape[0])  # harmful=1
    scores = {}
    for lang in target_langs:
        for split in splits:
            h, n = cache.pair(lang, split)
            eval_x = torch.cat([h, n], dim=0)
            logits = logistic_regression_scores(train_x, train_y, eval_x, seed=seed).cpu().numpy()
            scores[(lang, split)] = (logits[: h.shape[0]], logits[h.shape[0]:])
    return scores


def balanced_indices(n_h, n_n, k, seed):
    g = np.random.default_rng(seed)
    return g.choice(n_h, k, replace=False), g.choice(n_n, k, replace=False)


def run(cache, source_lang, target_langs, layer, alpha_miss, alpha_flag, seeds, seed_offset, n_cal=100):
    """Both methods averaged over seeds on the SAME calibration/test split each seed.

    We pool all of a language's harmful and harmless scores, then per seed carve out a
    calibration set (n_cal per class) and use the rest as the test set. PACT sets a
    conformal threshold on the calibration set (0 labels beyond the transported class);
    few-shot picks k of those same calibration points per class and sets a macro-F1
    threshold. Averaging over seeds is what makes PACT's guarantee visible: its MEAN
    miss should sit near alpha_miss, however weak the underlying score is.
    """
    scores = english_probe_scores(cache, source_lang, target_langs, ["train", "val", "test"], layer, seed_offset)
    rows = []
    for lang in target_langs:
        H = np.concatenate([scores[(lang, s)][0] for s in ("train", "val", "test")])  # harmful
        N = np.concatenate([scores[(lang, s)][1] for s in ("train", "val", "test")])  # harmless
        cal_n = min(n_cal, len(H) // 2, len(N) // 2)
        auc_lang = round(_auc(H, N), 4)   # discrimination of the transported score in this language
        for s in range(seeds):
            rng = np.random.default_rng(seed_offset + s)
            hi, ni = rng.permutation(len(H)), rng.permutation(len(N))
            cal_H, test_H = H[hi[:cal_n]], H[hi[cal_n:]]
            cal_N, test_N = N[ni[:cal_n]], N[ni[cal_n:]]
            # PACT: conformal thresholds on the calibration set, 0 target labels.
            tau_pass = conformal.pass_threshold(cal_H, alpha_miss)   # harmful high; PASS/miss if below
            rows.append(dict(lang=lang, method="pact", budget=0, seed=s, label_cost=0, auc=auc_lang,
                             **gate_metrics(tau_pass, test_H, test_N)))
            # Few-shot: k of those calibration points per class, macro-F1 threshold.
            for k in BUDGETS:
                if k > min(len(cal_H), len(cal_N)):
                    continue
                thr = fewshot_threshold(cal_H[rng.choice(len(cal_H), k, replace=False)],
                                        cal_N[rng.choice(len(cal_N), k, replace=False)])
                rows.append(dict(lang=lang, method="fewshot", budget=k, seed=s, label_cost=2 * k, auc=auc_lang,
                                 **gate_metrics(thr, test_H, test_N)))
    return rows


def _auc(harmful, harmless):
    """P(harmful score > harmless score); 0.5 = chance."""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(harmful)), np.zeros(len(harmless))]
    return float(roc_auc_score(y, np.r_[harmful, harmless]))


def summarize(rows):
    """Mean miss rate and its spread, per (method, lang, budget)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["lang"], r["budget"])].append(r["miss_rate"])
    out = []
    for (method, lang, budget), miss in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        arr = np.array(miss)
        out.append(dict(method=method, lang=lang, budget=budget, n=len(arr),
                        miss_mean=round(float(arr.mean()), 4),
                        miss_std=round(float(arr.std()), 4),
                        miss_max=round(float(arr.max()), 4)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations-root", required=True)
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--source-lang", default="en")
    p.add_argument("--target-langs", nargs="+", default=LOW_RESOURCE)
    p.add_argument("--token-position", default="t_post_inst")
    p.add_argument("--alpha-miss", type=float, default=0.05)
    p.add_argument("--alpha-flag", type=float, default=0.10)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--out", default="pact_extension/results/comparison.csv")
    args = p.parse_args()

    from multilingual_latent_safety.activation_store import ActivationCache
    cache = ActivationCache(args.activations_root, args.layer, args.token_position)
    rows = run(cache, args.source_lang, args.target_langs, args.layer,
               args.alpha_miss, args.alpha_flag, args.seeds, args.seed_offset)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ_path = out.with_name(out.stem + "_summary.csv")
    summ = summarize(rows)
    with open(summ_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    print(f"{'method':<9}{'lang':<5}{'budget':>7}{'miss_mean':>11}{'miss_std':>10}{'miss_max':>10}")
    for r in summ:
        print(f"{r['method']:<9}{r['lang']:<5}{r['budget']:>7}{r['miss_mean']:>11}{r['miss_std']:>10}{r['miss_max']:>10}")
    print(f"\nrows -> {out}\nsummary -> {summ_path}")
    print("PACT uses 0 target labels; few-shot uses 2*budget. Compare miss control at equal label cost.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════
# TRACE
#   D-ID     : A3
#   FEEDS    : P1.§4.5 (primary, Adversarial paragraph-shuffle)
#   PRE-REG  : P-A3  (PAPER-P1-HNC-SKELETON §4.5)
#              Trained models evaluated on {val, shuf_paragraph, shuf_sentence,
#              shuf_word} corpora satisfy:
#                Δσ(s₂) significant (q_BH < 0.05) under paragraph shuffle
#                ΔPR(s₂), ΔH₁(s₂) significant (q_BH < 0.05) under paragraph shuffle
#                Δσ(s₀) NOT significant (expected, no hierarchy at s₀)
#              Gradient: paragraph > sentence > word (descending disruption).
#   CARRIL   : A (local CPU, post-D-095 but independent)
#   SEEDS    : n=3 × 1 condition (healthy) × 4 corpora = 12 evaluations
#   DATE     : 2026-04-20  (QUEUED — code ready, not yet executed)
#   DEPENDS  : exp_d095_bicondition.py (classes via import); INFRA-02 corpora
#   LOG      : EXPERIMENTS-LOG.md  §3.3  row A3
# ═══════════════════════════════════════════════════════════════════════
"""
A3 · ADVERSARIAL PARAGRAPH-SHUFFLE EVALUATION
==============================================

Question
--------
If HNC scale s₂ encodes paragraph-level (long-range discourse) structure
while s₀ encodes token-level structure, then:
    Shuffling paragraphs should disrupt s₂ heavily but leave s₀ intact.
    Shuffling words should disrupt s₀ heavily, s₂ only mildly.
    Shuffling sentences should sit in between.

This is the *adversarial control* for P1's claim that scales have
distinct semantic roles.

Pre-registration (P-A3)
-----------------------
For each trained model (healthy, 3 seeds), evaluated on 4 corpora:
    val              (control, in-distribution)
    shuf_paragraph   (INFRA-02, paragraph-level shuffle, SEED=20260420)
    shuf_sentence    (INFRA-02, sentence-level shuffle)
    shuf_word        (INFRA-02, word-level shuffle)

Predictions (at q_BH=0.05):
  P-A3-1  Δσ(s₂)  | paragraph−val     significant, |d| ≥ 0.8
  P-A3-2  ΔH₁(s₂) | paragraph−val     significant, |d| ≥ 0.8
  P-A3-3  ΔPR(s₂) | paragraph−val     significant, |d| ≥ 0.8
  P-A3-4  Δσ(s₀)  | paragraph−val     NOT significant  (|d| < 0.3)
  P-A3-5  Gradient: |Δ(paragraph)| > |Δ(sentence)| > |Δ(word)| at s₂
  P-A3-6  Inverse gradient at s₀:  |Δ(word)| > |Δ(sentence)| > |Δ(paragraph)|

Procedure
---------
1. Train HNC-healthy on wikitext103_val (reuses train_model from D-095).
2. For each seed × each eval-corpus: extract_scale_vectors → σ_k, PR_k,
   H₁_k; also measure BPB.
3. Paired bootstrap (per seed, paragraph − val) for each metric/scale.
4. Cohen's d across seeds.
5. BH-FDR over the family of 3 scales × 3 metrics × 3 shufflings = 27 tests
   at q=0.05.

Outputs
-------
results_A3_shuffled_eval/A3_results.json  with per-(seed, corpus):
  bpb, σ_k, PR_k, H1_k for k∈{0,1,2}
Plus aggregated:
  Δ(corpus − val) per metric × scale, bootstrap CI, Cohen's d, BH-FDR.

CPU-only, ~20 min (1 train ~3 min × 3 seeds + 4 corpora × 3 seeds × eval).
"""

import os, sys, json, time, math
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import exp_d095_bicondition as d095
from exp_d095_bicondition import (
    HNCModel, D_MODEL, NUM_LAYERS, NUM_SCALES, SEQ_LEN, N_SEQ,
    MID_POS, BATCH_SIZE, SEEDS,
    train_model, extract_scale_vectors,
    compute_sigma, compute_h1, compute_pr_cov,
)
from stats_helpers import bootstrap_ci, paired_bootstrap_diff, cohens_d, bh_fdr

# ── A3 config ──────────────────────────────────────────────────────────
DATA_DIR = HERE.parent.parent.parent / "data"
CORPORA = {
    "val":             DATA_DIR / "wikitext103_val.txt",
    "shuf_paragraph":  DATA_DIR / "wikitext103_val_shuf_paragraph.txt",
    "shuf_sentence":   DATA_DIR / "wikitext103_val_shuf_sentence.txt",
    "shuf_word":       DATA_DIR / "wikitext103_val_shuf_word.txt",
}
OUTPUT_DIR = HERE / "results_A3_shuffled_eval"
OUTPUT_DIR.mkdir(exist_ok=True)

SUBSET_MB_EVAL = d095.SUBSET_MB   # keep eval-data size matched to train


# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING (per-corpus)
# ═══════════════════════════════════════════════════════════════════════

def load_corpus(path: Path) -> np.ndarray:
    raw = open(path, "r", encoding="utf-8").read()
    data = np.frombuffer(raw.encode("utf-8"), dtype=np.uint8)
    limit = int(SUBSET_MB_EVAL * 1e6)
    return data[:limit]


# ═══════════════════════════════════════════════════════════════════════
#  METRICS ON A GIVEN CORPUS
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_bpb(model, data: np.ndarray, seed: int = 42) -> float:
    """Mean byte-level cross-entropy (in bits per byte) on the given corpus."""
    model.eval()
    rng = np.random.RandomState(seed)
    max_start = len(data) - SEQ_LEN - 1
    n_batches = 20
    starts = rng.randint(0, max_start, size=n_batches * BATCH_SIZE)

    total_loss, total_tokens = 0.0, 0
    for b0 in range(0, len(starts), BATCH_SIZE):
        bs = starts[b0:b0 + BATCH_SIZE]
        xb = torch.stack([torch.tensor(data[s:s + SEQ_LEN], dtype=torch.long) for s in bs])
        yb = torch.stack([torch.tensor(data[s + 1:s + SEQ_LEN + 1], dtype=torch.long) for s in bs])
        logits, _, _ = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, 256), yb.reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += xb.numel()
    return (total_loss / max(total_tokens, 1)) / math.log(2)


def analyze_on_corpus(model, data, label, seed=42) -> dict:
    """σ, PR, H₁, BPB per scale for a single (model, corpus, seed) tuple."""
    scale_vecs = extract_scale_vectors(model, data, seed=seed)
    bpb = evaluate_bpb(model, data, seed=seed)
    res = {"corpus": label, "bpb": bpb, "scales": {}}
    for k in range(NUM_SCALES):
        vecs = scale_vecs[k]
        sigma_m, _ = compute_sigma(vecs, seed=seed)
        pr = compute_pr_cov(vecs)
        h1 = compute_h1(vecs, seed=seed)
        res["scales"][str(k)] = dict(
            sigma=sigma_m,
            pr=pr,
            h1_n=h1["n_significant"],
            h1_total=h1["total_persistence"],
        )
    return res


# ═══════════════════════════════════════════════════════════════════════
#  STATISTICS (paired: corpus vs val, per seed)
# ═══════════════════════════════════════════════════════════════════════

def _collect(results, corpus_key, metric_path, seeds):
    """Extract a per-seed array for a given metric in a given corpus.
    metric_path is a tuple like ('scales','2','sigma') or ('bpb',)."""
    vals = []
    for s in seeds:
        node = results["seeds"][str(s)]["corpora"][corpus_key]
        for p in metric_path:
            node = node[p]
        vals.append(float(node))
    return np.array(vals)


def compute_deltas_and_fdr(results: dict, seeds) -> dict:
    """For each (scale, metric, shuf_corpus): compute Δ = shuf − val across seeds,
    bootstrap CI, Cohen's d, permutation-free p via bootstrap, then BH-FDR."""
    shuf_corpora = ["shuf_paragraph", "shuf_sentence", "shuf_word"]
    metrics = ["sigma", "pr", "h1_n"]
    scales  = ["0", "1", "2"]

    family = []
    for shuf in shuf_corpora:
        for k in scales:
            for m in metrics:
                family.append((shuf, k, m))

    out = {"family": [], "p_raw": [], "deltas": {}}
    for (shuf, k, m) in family:
        val_vals  = _collect(results, "val",  ("scales", k, m), seeds)
        shuf_vals = _collect(results, shuf,   ("scales", k, m), seeds)
        delta = shuf_vals - val_vals

        ci = bootstrap_ci(delta, seed=42)
        # p-value: fraction of bootstrap samples where sign flips (two-sided, rough)
        rng = np.random.RandomState(42)
        n_boot = 1000
        boots = np.array([np.mean(rng.choice(delta, size=len(delta), replace=True))
                          for _ in range(n_boot)])
        p = 2.0 * min(np.mean(boots >= 0), np.mean(boots <= 0))
        p = float(max(p, 1.0 / n_boot))
        d_eff = cohens_d(shuf_vals, val_vals)

        key = f"{shuf}|s{k}|{m}"
        out["family"].append(key)
        out["p_raw"].append(p)
        out["deltas"][key] = dict(
            mean=float(delta.mean()), std=float(delta.std()),
            ci=ci,
            cohens_d=d_eff,
            p_value=p,
            val_mean=float(val_vals.mean()),
            shuf_mean=float(shuf_vals.mean()),
        )

    fdr = bh_fdr(out["p_raw"], alpha=0.05)
    out["bh_fdr"] = dict(
        alpha=0.05,
        p_adj=fdr.get("p_adjusted", fdr.get("p_adj")),
        reject=fdr.get("reject"),
    )
    return out


def verdict_P_A3(deltas_table: dict) -> dict:
    """Apply the 6 pre-registered predictions P-A3-{1..6} to the delta table."""
    V = {}
    def get(shuf, k, m):
        return deltas_table["deltas"][f"{shuf}|s{k}|{m}"]

    # P-A3-1/2/3 : paragraph shuffle on s₂ (σ, h1, pr) significant, |d| ≥ 0.8
    for idx, m in enumerate(["sigma", "h1_n", "pr"], start=1):
        d_val = get("shuf_paragraph", "2", m)
        sig = d_val["cohens_d"] is not None and abs(d_val["cohens_d"]) >= 0.8
        V[f"P-A3-{idx}"] = dict(metric=f"s2/{m}", cohens_d=d_val["cohens_d"], passes=bool(sig))

    # P-A3-4 : σ(s₀) on paragraph shuffle NOT significant (|d| < 0.3)
    d0 = get("shuf_paragraph", "0", "sigma")
    V["P-A3-4"] = dict(metric="s0/sigma", cohens_d=d0["cohens_d"],
                       passes=bool(d0["cohens_d"] is not None and abs(d0["cohens_d"]) < 0.3))

    # P-A3-5 : at s₂, |Δ(paragraph)| > |Δ(sentence)| > |Δ(word)| for sigma
    s2 = [abs(get(c, "2", "sigma")["mean"]) for c in ("shuf_paragraph", "shuf_sentence", "shuf_word")]
    V["P-A3-5"] = dict(metric="s2/sigma gradient",
                       order=s2, passes=bool(s2[0] >= s2[1] >= s2[2]))

    # P-A3-6 : at s₀, inverse — |Δ(word)| > |Δ(sentence)| > |Δ(paragraph)| for sigma
    s0 = [abs(get(c, "0", "sigma")["mean"]) for c in ("shuf_word", "shuf_sentence", "shuf_paragraph")]
    V["P-A3-6"] = dict(metric="s0/sigma inverse gradient",
                       order=s0, passes=bool(s0[0] >= s0[1] >= s0[2]))
    return V


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("A3 · ADVERSARIAL PARAGRAPH-SHUFFLE EVAL (P1.§4.5, pre-reg P-A3)")
    print("=" * 75)
    print(f"  Corpora: {list(CORPORA.keys())}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Config: {NUM_LAYERS}L, d={D_MODEL}, seq={SEQ_LEN}, scales={NUM_SCALES}")
    print()

    # Preload all corpora
    data_by_corpus = {k: load_corpus(p) for k, p in CORPORA.items()}
    for k, d in data_by_corpus.items():
        print(f"  {k}: {len(d)} bytes")
    print()

    cache = OUTPUT_DIR / "A3_results.json"
    if cache.exists():
        results = json.load(open(cache))
        print(f"  Loaded cache: {len(results.get('seeds', {}))} seeds already done")
    else:
        results = {"seeds": {}, "meta": dict(
            seeds=SEEDS, n_seq=N_SEQ, d_model=D_MODEL,
            num_scales=NUM_SCALES, subset_mb=SUBSET_MB_EVAL,
        )}

    train_data = data_by_corpus["val"]   # always train on in-distribution

    for seed in SEEDS:
        if str(seed) in results["seeds"]:
            print(f"  [SKIP] seed={seed} already in cache")
            continue
        print(f"\n── seed={seed} ──")
        print("  [train healthy on val]")
        # healthy: d_proj=None, mode='standard' (same as D-095 reference arm)
        model, n_params, _hist = train_model(
            d_proj=None, mode='standard', data=train_data, seed=seed)

        seed_results = {"corpora": {}}
        for corpus_name, corpus_data in data_by_corpus.items():
            print(f"  [eval on {corpus_name}]")
            res = analyze_on_corpus(model, corpus_data, corpus_name, seed=seed)
            seed_results["corpora"][corpus_name] = res
            s = res["scales"]
            print(f"    BPB={res['bpb']:.3f}  "
                  f"σ₀={s['0']['sigma']:.1f}° σ₂={s['2']['sigma']:.1f}°  "
                  f"PR₀={s['0']['pr']:.1f} PR₂={s['2']['pr']:.1f}  "
                  f"H₁₀={s['0']['h1_n']} H₁₂={s['2']['h1_n']}")

        results["seeds"][str(seed)] = seed_results
        with open(cache, "w") as f:
            json.dump(results, f, indent=2, default=float)

    # ── Stats ─────────────────────────────────────────────────────────
    print("\n── Statistical analysis (paired Δ vs val, BH-FDR q=0.05) ──")
    deltas_table = compute_deltas_and_fdr(results, SEEDS)
    results["stats"] = deltas_table

    print(f"\nFamily size: {len(deltas_table['family'])} tests")
    reject = deltas_table["bh_fdr"]["reject"]
    n_sig = int(np.sum(reject)) if reject is not None and len(reject) > 0 else 0
    print(f"Significant after BH-FDR: {n_sig}/{len(deltas_table['family'])}")

    verdict = verdict_P_A3(deltas_table)
    results["verdict_P_A3"] = verdict

    print("\nP-A3 verdict:")
    for k, v in verdict.items():
        mark = "✓" if v["passes"] else "✗"
        if "cohens_d" in v:
            print(f"  [{mark}] {k}  {v['metric']}  d={v['cohens_d']}")
        else:
            print(f"  [{mark}] {k}  {v['metric']}  order={v.get('order')}")

    with open(cache, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nA3 DONE  [{time.time() - t0:.1f}s]")
    print(f"  Results → {cache}")


if __name__ == "__main__":
    main()

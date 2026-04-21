#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════
# TRACE
#   D-ID     : A2
#   FEEDS    : P1.§4.2b (primary, Causal permutation test)
#   PRE-REG  : P-A2  (PAPER-P1-HNC-SKELETON §4.2b)
#              Null permutation AR (shuffled scale assignment per example)
#              satisfies:  AR_null ≤ 0.5 × AR_original  with q_BH < 0.05
#              across (condition, seed) in {healthy, dual_reg_d8} × {42,123,456}.
#   CARRIL   : A (local CPU, post-D-095)
#   SEEDS    : n=3 × 2 conditions = 6 models
#   DATE     : 2026-04-20  (QUEUED — code ready, not yet executed)
#   DEPENDS  : exp_d095_bicondition.py (model classes reused via import)
#   LOG      : EXPERIMENTS-LOG.md  §3.3  row A2
# ═══════════════════════════════════════════════════════════════════════
"""
A2 · CAUSAL PERMUTATION TEST — Scale identity as structural, not labeling
=========================================================================

Question
--------
HNC produces an AR = σ_max/σ_min >> 1 gradient across scales (IVM signature).
Is this gradient *structural* (scale identity is causally determined by the
top-down coupling and bottleneck geometry), or is it merely a *labeling*
artifact (any permutation of scale assignments across examples would yield
the same AR distribution)?

The classical permutation-test null: if we shuffle WHICH sample belongs to
WHICH apparent scale (independently per example), the resulting per-scale
cloud should become isotropic → σ_k equal across k → AR_null ≈ 1.0.

Pre-registration (P-A2)
-----------------------
For each trained model in {healthy, dual_reg_d8} × seeds {42, 123, 456}:
    AR_original ≥ 2.0          (bioreg IVM gradient)
    AR_null     ≤ 0.5 × AR_original     (scale identity collapses)
    Permutation p-value < 0.05 (Bonferroni-FWER per-seed)
    BH-FDR over 6 tests at q = 0.05 must retain all as significant.

Functional swap (secondary check)
---------------------------------
Bonus intervention: pass the extracted trained state trajectories through
the model's OutputHead in the WRONG scale order (s2 as s0, s0 as s2) and
measure BPB degradation. Predict BPB_swapped ≥ BPB_normal × 1.2.

Procedure
---------
1. Import HNCModel, train_model, extract_scale_vectors, compute_sigma from
   exp_d095_bicondition (same config, same seeds, matched to D-095).
2. Train only {healthy, dual_reg_d8} (A2 does NOT need all 5 D-095 arms).
3. For each trained model:
     a. extract_scale_vectors → s0, s1, s2 per example (shape [N_SEQ, d]).
     b. Compute σ_k for k∈{0,1,2} normally; AR_original = max/min.
     c. Permutation null (N_PERM = 500):
          For each perm:
             For each example i, assign a random permutation π_i ∈ S_3
             Build null_s_k = np.array([vecs[π_i(k)][i] for i])
             Compute σ_k_null, AR_null = max/min
          p_value = fraction(AR_null >= AR_original)
     d. (Optional) Functional swap test — BPB comparison.
4. Aggregate across seeds via bootstrap_ci.
5. Apply bh_fdr over the family of 6 p-values.

Outputs
-------
results_A2_causal_permutation/A2_results.json with per-(condition, seed):
  AR_original, AR_null_mean, AR_null_ci, p_value, bpb_normal, bpb_swapped
Plus aggregate bootstrap CI for AR_original and AR_null_mean across seeds,
plus BH-FDR table.

CPU-only, ~20 min (two conditions × 3 seeds × ~3 min train + 1 min perm).
"""

import os, sys, json, time, math
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch

# ── Path setup ─────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # FASE 5/
sys.path.insert(0, str(HERE))                   # allow importing sibling scripts
sys.path.insert(0, str(HERE.parent))            # LINEA A/

# Reuse verbatim from D-095 (same architecture, same config)
import exp_d095_bicondition as d095
from exp_d095_bicondition import (
    HNCModel, D_MODEL, NUM_LAYERS, NUM_SCALES, D_PROJ, SEQ_LEN, N_SEQ,
    MID_POS, BATCH_SIZE, SEEDS, load_data, train_model, extract_scale_vectors,
    compute_sigma,
)
from stats_helpers import bootstrap_ci, bh_fdr, cohens_d

# ── A2-specific config ─────────────────────────────────────────────────
N_PERM = 500
OUTPUT_DIR = HERE / "results_A2_causal_permutation"
OUTPUT_DIR.mkdir(exist_ok=True)

# Subset of D-095 conditions that A2 needs (two poles of the design):
CONDITIONS_A2 = [
    ("healthy",       None,    'standard'),   # reference IVM gradient
    ("dual_reg_d8",   D_PROJ,  'dual_reg'),   # dual-reg model (key claim)
]


# ═══════════════════════════════════════════════════════════════════════
#  PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════

def ar_from_scale_vecs(scale_vecs: dict, seed: int = 42) -> tuple:
    """Compute (AR, σ_per_scale) from a dict {k: vecs[N,d]}."""
    sigmas = []
    for k in range(NUM_SCALES):
        sigma_m, _ = compute_sigma(scale_vecs[k], seed=seed)
        sigmas.append(sigma_m)
    sigmas = np.array(sigmas)
    eps = 1e-6
    AR = float(np.max(sigmas) / max(np.min(sigmas), eps))
    return AR, sigmas.tolist()


def permutation_null_ar(scale_vecs: dict, n_perm: int, seed: int = 42) -> dict:
    """Shuffle the scale-label of each example independently; recompute AR.

    For each permutation draw, we construct null_s_k by randomly re-assigning
    each example's scale-identity. The expected AR under this null (scale
    identity is arbitrary labeling) is ≈ 1.0.
    """
    rng = np.random.RandomState(seed)
    N = scale_vecs[0].shape[0]
    d = scale_vecs[0].shape[1]
    stacked = np.stack([scale_vecs[k] for k in range(NUM_SCALES)], axis=0)   # (K, N, d)

    ar_nulls = []
    for p in range(n_perm):
        perms = rng.randint(0, NUM_SCALES, size=(NUM_SCALES, N))  # independent per example
        # Ensure each column is a permutation of {0,…,K-1} per example
        for i in range(N):
            perms[:, i] = rng.permutation(NUM_SCALES)
        null_vecs = {}
        for k in range(NUM_SCALES):
            null_vecs[k] = stacked[perms[k], np.arange(N), :]
        AR_null, _ = ar_from_scale_vecs(null_vecs, seed=seed + p)
        ar_nulls.append(AR_null)

    arr = np.array(ar_nulls)
    return dict(
        ar_null_mean=float(arr.mean()),
        ar_null_std=float(arr.std()),
        ar_null_p05=float(np.percentile(arr, 5)),
        ar_null_p95=float(np.percentile(arr, 95)),
        ar_null_max=float(arr.max()),
        ar_null_samples=arr.tolist(),
    )


@torch.no_grad()
def functional_swap_bpb(model, data, seed: int = 42) -> dict:
    """Secondary test: pass permuted scale trajectories through OutputHead.

    Compares BPB under:
      - normal scale order (s0, s1, s2)
      - swapped scale order (s2, s1, s0)
    If scale identity is structural, the output head should degrade.
    """
    rng = np.random.RandomState(seed)
    max_start = len(data) - SEQ_LEN - 1
    starts = rng.randint(0, max_start, size=min(N_SEQ, 16))
    import torch.nn.functional as F

    bpb_normal_vals, bpb_swapped_vals = [], []

    for b0 in range(0, len(starts), BATCH_SIZE):
        bs = starts[b0:b0 + BATCH_SIZE]
        xb = torch.stack([torch.tensor(data[s:s + SEQ_LEN], dtype=torch.long) for s in bs])
        yb = torch.stack([torch.tensor(data[s + 1:s + SEQ_LEN + 1], dtype=torch.long) for s in bs])

        # normal forward — use model's internal forward
        logits_norm, _, state_trajs = model(xb)
        loss_norm = F.cross_entropy(logits_norm.reshape(-1, 256), yb.reshape(-1))
        bpb_normal_vals.append(loss_norm.item() / math.log(2))

        # swapped: reverse order of state_trajs, feed through OutputHead only
        h = model.embed_norm(model.pos_encoding(model.embedding(xb)))
        # re-run blocks to get same h (avoid recomputation bugs); use stored state_trajs
        swapped_trajs = [state_trajs[NUM_SCALES - 1 - k] for k in range(NUM_SCALES)]
        logits_swap = model.output_head(h, swapped_trajs)
        loss_swap = F.cross_entropy(logits_swap.reshape(-1, 256), yb.reshape(-1))
        bpb_swapped_vals.append(loss_swap.item() / math.log(2))

    return dict(
        bpb_normal=float(np.mean(bpb_normal_vals)),
        bpb_swapped=float(np.mean(bpb_swapped_vals)),
        bpb_ratio=float(np.mean(bpb_swapped_vals) / max(np.mean(bpb_normal_vals), 1e-6)),
    )


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("A2 · CAUSAL PERMUTATION TEST (P1.§4.2b, pre-reg P-A2)")
    print("=" * 75)
    print(f"  Conditions: {[c[0] for c in CONDITIONS_A2]}")
    print(f"  Seeds: {SEEDS}   N_PERM: {N_PERM}")
    print(f"  Config: {NUM_LAYERS}L, d={D_MODEL}, seq={SEQ_LEN}, scales={NUM_SCALES}")
    print()

    data = load_data()
    print(f"  Data: {len(data)} bytes\n")

    results = {"conditions": {}, "meta": dict(
        n_perm=N_PERM, seeds=SEEDS, n_seq=N_SEQ, mid_pos=MID_POS,
        d_model=D_MODEL, num_scales=NUM_SCALES,
    )}

    # Cache file
    cache = OUTPUT_DIR / "A2_results.json"
    if cache.exists():
        results = json.load(open(cache))
        print(f"  Loaded cache: {len(results.get('conditions', {}))} conditions already done\n")

    for (label, d_proj, mode) in CONDITIONS_A2:
        if label in results["conditions"]:
            print(f"  [SKIP] {label} already in cache")
            continue
        print(f"\n── {label}  (d_proj={d_proj}, mode={mode}) ──")
        cond_results = {"seeds": {}}
        for seed in SEEDS:
            print(f"\n  seed={seed}")
            print("   [train]")
            model, n_params, _hist = train_model(d_proj, mode, data, seed=seed)
            print(f"   [extract]")
            scale_vecs = extract_scale_vectors(model, data, seed=seed)
            AR_orig, sigmas_orig = ar_from_scale_vecs(scale_vecs, seed=seed)
            print(f"   AR_original = {AR_orig:.3f}  σ = {['%.1f°' % s for s in sigmas_orig]}")

            print(f"   [permutation null, N={N_PERM}]")
            perm = permutation_null_ar(scale_vecs, n_perm=N_PERM, seed=seed)
            p_value = float(np.mean(np.array(perm["ar_null_samples"]) >= AR_orig))
            print(f"   AR_null  = {perm['ar_null_mean']:.3f} ± {perm['ar_null_std']:.3f}   "
                  f"(p95={perm['ar_null_p95']:.3f}, max={perm['ar_null_max']:.3f})")
            print(f"   p_value (permutation) = {p_value:.4g}")

            print(f"   [functional swap BPB]")
            try:
                swap_bpb = functional_swap_bpb(model, data, seed=seed)
                print(f"   BPB normal={swap_bpb['bpb_normal']:.3f}  "
                      f"swapped={swap_bpb['bpb_swapped']:.3f}  "
                      f"ratio={swap_bpb['bpb_ratio']:.3f}")
            except Exception as e:
                print(f"   [WARN] functional swap failed: {e}")
                swap_bpb = dict(bpb_normal=None, bpb_swapped=None, bpb_ratio=None,
                                error=str(e))

            cond_results["seeds"][str(seed)] = dict(
                AR_original=AR_orig,
                sigmas_original=sigmas_orig,
                AR_null_mean=perm["ar_null_mean"],
                AR_null_std=perm["ar_null_std"],
                AR_null_p95=perm["ar_null_p95"],
                p_value=p_value,
                predicted_pass=(perm["ar_null_mean"] <= 0.5 * AR_orig) and (p_value < 0.05),
                functional_swap=swap_bpb,
                n_params=n_params,
            )

        # aggregate via bootstrap CI across seeds
        ar_origs = [cond_results["seeds"][str(s)]["AR_original"] for s in SEEDS]
        ar_nulls = [cond_results["seeds"][str(s)]["AR_null_mean"] for s in SEEDS]
        cond_results["AR_original_ci"] = bootstrap_ci(ar_origs, seed=42)
        cond_results["AR_null_ci"] = bootstrap_ci(ar_nulls, seed=42)
        cond_results["cohens_d_orig_vs_null"] = cohens_d(ar_origs, ar_nulls)
        results["conditions"][label] = cond_results

        with open(cache, "w") as f:
            json.dump(results, f, indent=2, default=float)

    # BH-FDR over the family of 6 p-values (2 conditions × 3 seeds)
    pvals, labels_pv = [], []
    for lbl, cr in results["conditions"].items():
        for seed in SEEDS:
            pvals.append(cr["seeds"][str(seed)]["p_value"])
            labels_pv.append(f"{lbl}/seed{seed}")
    if pvals:
        fdr = bh_fdr(pvals, alpha=0.05)
        results["bh_fdr"] = dict(
            family=labels_pv,
            p_raw=pvals,
            p_adj=fdr.get("p_adjusted", fdr.get("p_adj")),
            reject=fdr.get("reject"),
            alpha=0.05,
        )

    with open(cache, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print("\n" + "=" * 75)
    print(f"A2 DONE  [{time.time() - t0:.1f}s]")
    print(f"  Results → {cache}")
    print("\nP-A2 verdict per (condition, seed):")
    for lbl, cr in results["conditions"].items():
        for seed in SEEDS:
            d = cr["seeds"][str(seed)]
            mark = "✓" if d["predicted_pass"] else "✗"
            print(f"  [{mark}] {lbl}/seed{seed}: AR_orig={d['AR_original']:.2f}  "
                  f"AR_null={d['AR_null_mean']:.2f}  p={d['p_value']:.3g}")


if __name__ == "__main__":
    main()

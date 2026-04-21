"""
exp_ctrl_A1_A4_untrained_and_null_tda.py
=========================================
Two CONTROL experiments from the 2026-04-20 creative proposals:

A1. UNTRAINED BASELINE — σ/PR/H1 on random-init HNC (no training).
    Question: does the architecture ALONE produce the σ/PR/H1 gradient,
    or does training create it? This separates inductive bias from learning.

A4. NULL TDA (Gaussian null) — for each set of activation vectors,
    generate a Gaussian cloud with matched covariance and compute H1.
    This gives the floor H1 attributable to finite-sample noise in a
    structureless cloud, and lets us report H1_real - H1_null with CIs.
    Critical for defending N_SUBSAMPLE=100 against "your persistent
    loops might just be sampling noise" criticism.

Both run in a single script because they share the model build + data
pipeline. CPU-only, <5 min for 5 seeds.

Outputs: results_ctrl_A1_A4/ctrl_results.json with per-seed, per-scale
σ/PR/H1_real/H1_null arrays + bootstrap CIs via stats_helpers.

Usage:
    python exp_ctrl_A1_A4_untrained_and_null_tda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Reuse the EXACT HNC architecture + metric functions from D-095 so
# comparisons are apples-to-apples.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_d095_bicondition import (  # type: ignore
    HNCModel, load_data,
    compute_sigma, compute_h1, compute_pr_cov,
    D_MODEL, NUM_SCALES, NUM_LAYERS, SEQ_LEN, N_SEQ, MID_POS,
    N_SUBSAMPLE, PCA_DIM,
)
from stats_helpers import bootstrap_ci, bh_fdr, paired_bootstrap_diff

OUT_DIR = Path(__file__).parent / "results_ctrl_A1_A4"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]          # n=5 per design rule post-review
N_NULL_RESAMPLES = 20            # Gaussian nulls per real cloud
DEVICE = "cpu"


# ── Null TDA ─────────────────────────────────────────────────────────────
def gaussian_null_h1(vectors: np.ndarray, n_resamples: int,
                     seed: int) -> dict:
    """Generate matched-covariance Gaussian clouds, report H1 distribution.

    Returns dict(mean, ci_lo, ci_hi) for each H1 sub-metric.
    """
    if vectors.shape[0] < 10:
        return dict(n_sig_mean=0.0, n_sig_ci=(0.0, 0.0),
                    total_pers_mean=0.0, total_pers_ci=(0.0, 0.0))
    mu = vectors.mean(axis=0)
    cov = np.cov(vectors, rowvar=False)
    # Regularize covariance slightly to ensure PSD
    cov = cov + 1e-8 * np.eye(cov.shape[0])
    # Cholesky once; multivariate_normal is slow for many resamples
    try:
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(cov.shape[0]))
    except np.linalg.LinAlgError:
        # Fallback for near-singular cov
        w, V = np.linalg.eigh(cov)
        w = np.maximum(w, 1e-8)
        L = V * np.sqrt(w)
    rng = np.random.RandomState(seed)
    n_sig_list, tot_list = [], []
    for r in range(n_resamples):
        z = rng.standard_normal(size=vectors.shape)
        sample = mu + z @ L.T
        h = compute_h1(sample, seed=seed + r)
        n_sig_list.append(h['n_significant'])
        tot_list.append(h['total_persistence'])
    n_sig = np.array(n_sig_list, dtype=float)
    tot = np.array(tot_list, dtype=float)
    return dict(
        n_sig_mean=float(n_sig.mean()),
        n_sig_ci=(float(np.percentile(n_sig, 2.5)),
                  float(np.percentile(n_sig, 97.5))),
        total_pers_mean=float(tot.mean()),
        total_pers_ci=(float(np.percentile(tot, 2.5)),
                       float(np.percentile(tot, 97.5))),
    )


# ── Extract untrained states ─────────────────────────────────────────────
def collect_states_untrained(seed: int, data: np.ndarray) -> dict:
    """Build HNC with random init, run inference, collect state trajectories."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HNCModel(d_model=D_MODEL, num_layers=NUM_LAYERS,
                     num_scales=NUM_SCALES, d_proj=None,
                     mode='standard').to(DEVICE)
    model.eval()

    by_scale = {k: [] for k in range(NUM_SCALES)}
    with torch.no_grad():
        for i in range(N_SEQ):
            start = i * SEQ_LEN
            if start + SEQ_LEN > len(data):
                break
            seq = torch.tensor(data[start:start + SEQ_LEN].copy(),
                               dtype=torch.long, device=DEVICE).unsqueeze(0)
            _, _, state_trajs = model(seq)
            # state_trajs: list of [B, T, D] per scale (from HNCBlock)
            for k in range(NUM_SCALES):
                by_scale[k].append(state_trajs[k][0, MID_POS, :].cpu().numpy())
    return {k: np.stack(v, axis=0) for k, v in by_scale.items()}


# ── Metrics per scale with null ──────────────────────────────────────────
def metrics_with_null(vectors: np.ndarray, seed: int) -> dict:
    sigma_mean, sigma_std = compute_sigma(vectors, seed=seed)
    pr = compute_pr_cov(vectors)
    h_real = compute_h1(vectors, seed=seed)
    null = gaussian_null_h1(vectors, n_resamples=N_NULL_RESAMPLES, seed=seed)
    return dict(
        sigma_mean=sigma_mean, sigma_std=sigma_std,
        pr=pr,
        h1_real_n_sig=h_real['n_significant'],
        h1_real_total=h_real['total_persistence'],
        h1_null_n_sig_mean=null['n_sig_mean'],
        h1_null_n_sig_ci=null['n_sig_ci'],
        h1_null_total_mean=null['total_pers_mean'],
        h1_null_total_ci=null['total_pers_ci'],
        h1_excess_n_sig=h_real['n_significant'] - null['n_sig_mean'],
        h1_excess_total=h_real['total_persistence'] - null['total_pers_mean'],
    )


# ── Main loop ────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("A1 + A4 CONTROL - untrained HNC + Gaussian null TDA")
    print(f"Seeds: {SEEDS} | Nulls per cloud: {N_NULL_RESAMPLES}")
    print("=" * 72)

    data = load_data()
    print(f"Loaded {len(data):,} bytes")

    per_seed = []
    for s_idx, seed in enumerate(SEEDS):
        print(f"\n[seed {seed} | {s_idx+1}/{len(SEEDS)}] untrained forward pass...")
        states = collect_states_untrained(seed, data)
        seed_result = dict(seed=seed, scales={})
        for k in range(NUM_SCALES):
            m = metrics_with_null(states[k], seed=seed)
            seed_result['scales'][str(k)] = m
            print(f"   s{k}: sigma={m['sigma_mean']:5.1f}  PR={m['pr']:5.2f}  "
                  f"H1_real={m['h1_real_n_sig']:2d}  "
                  f"H1_null={m['h1_null_n_sig_mean']:4.1f} "
                  f"CI[{m['h1_null_n_sig_ci'][0]:.1f},{m['h1_null_n_sig_ci'][1]:.1f}]  "
                  f"EXCESS={m['h1_excess_n_sig']:+.1f}")
        per_seed.append(seed_result)

    # Aggregate per scale across seeds with bootstrap CIs
    agg = {}
    for k in range(NUM_SCALES):
        sig = [r['scales'][str(k)]['sigma_mean'] for r in per_seed]
        pr = [r['scales'][str(k)]['pr'] for r in per_seed]
        h1r = [r['scales'][str(k)]['h1_real_n_sig'] for r in per_seed]
        h1n = [r['scales'][str(k)]['h1_null_n_sig_mean'] for r in per_seed]
        excess = [r['scales'][str(k)]['h1_excess_n_sig'] for r in per_seed]
        agg[str(k)] = dict(
            sigma=bootstrap_ci(sig),
            pr=bootstrap_ci(pr),
            h1_real=bootstrap_ci(h1r),
            h1_null=bootstrap_ci(h1n),
            h1_excess=bootstrap_ci(excess),
        )

    # Family-wise test: for each scale, is H1_excess significantly > 0?
    p_raw = []
    for k in range(NUM_SCALES):
        excess = [r['scales'][str(k)]['h1_excess_n_sig'] for r in per_seed]
        zero = [0.0] * len(excess)
        p_raw.append(paired_bootstrap_diff(excess, zero)['p_boot'])
    fdr = bh_fdr(p_raw, alpha=0.05)

    report = dict(
        config=dict(seeds=SEEDS, n_null=N_NULL_RESAMPLES,
                    d_model=D_MODEL, n_seq=N_SEQ, n_subsample=N_SUBSAMPLE,
                    pca_dim=PCA_DIM),
        per_seed=per_seed,
        aggregate=agg,
        family_test_h1_excess_gt_zero=dict(
            p_raw=[float(p) for p in p_raw],
            p_adj=[float(p) for p in fdr['p_adj']],
            reject=[bool(r) for r in fdr['reject']],
        ),
    )

    out_path = OUT_DIR / "ctrl_A1_A4_results.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n" + "=" * 72)
    print("SUMMARY (untrained HNC, aggregated over {} seeds)".format(len(SEEDS)))
    print("=" * 72)
    print(f"{'scale':>6} {'sigma':>14} {'PR':>12} {'H1_real':>14} "
          f"{'H1_null':>14} {'H1_excess':>16} {'FDR-signif':>12}")
    for k in range(NUM_SCALES):
        a = agg[str(k)]
        print(f"  s{k}   "
              f"{a['sigma']['point']:5.1f}[{a['sigma']['ci_lo']:4.1f},{a['sigma']['ci_hi']:4.1f}] "
              f"{a['pr']['point']:4.2f}[{a['pr']['ci_lo']:4.2f},{a['pr']['ci_hi']:4.2f}] "
              f"{a['h1_real']['point']:4.1f}[{a['h1_real']['ci_lo']:4.1f},{a['h1_real']['ci_hi']:4.1f}] "
              f"{a['h1_null']['point']:4.1f}[{a['h1_null']['ci_lo']:4.1f},{a['h1_null']['ci_hi']:4.1f}] "
              f"{a['h1_excess']['point']:+5.1f}[{a['h1_excess']['ci_lo']:+4.1f},{a['h1_excess']['ci_hi']:+4.1f}]  "
              f"{'YES' if report['family_test_h1_excess_gt_zero']['reject'][k] else 'no':>10}")

    print(f"\nResults saved: {out_path}")
    print("\nInterpretation guide:")
    print("  - H1_excess > 0 with FDR significance → untrained arch alone builds topology.")
    print("  - H1_excess ~ 0 → all observed H1 in trained models is training-induced.")
    print("  - σ gradient σ0<σ1<σ2 in untrained? → hard-coded by architecture.")
    print("  - σ gradient appears only after training → emergent from learning.")


if __name__ == "__main__":
    main()

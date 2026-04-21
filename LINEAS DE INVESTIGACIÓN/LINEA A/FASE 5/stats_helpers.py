"""
stats_helpers.py
================
Rigor statistical utilities for the Tensegrity / HNC project.

Covers the three holes identified in the 2026-04-20 critical review:
  1. Bootstrap 95% CIs instead of naked means (functions: bootstrap_ci, paired_bootstrap).
  2. Effect sizes (Cohen's d, Hedges' g).
  3. Family-wise multiple comparison correction (Benjamini-Hochberg FDR).

All functions are pure NumPy + SciPy (no pandas). Deterministic when given a seed.

Target usage (paper ALPHA):
  - Every scalar result that goes in a table must carry CI95.
  - Every two-condition comparison must report Cohen's d + p-value.
  - Every family of tests (C1...C6) must pass through bh_fdr().
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Sequence


# ── Bootstrap ────────────────────────────────────────────────────────────
def bootstrap_ci(values: Sequence[float],
                 stat: Callable[[np.ndarray], float] = np.mean,
                 n_boot: int = 1000,
                 alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """Percentile bootstrap CI for an arbitrary statistic.

    Returns dict(point, ci_lo, ci_hi, se).
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return dict(point=float('nan'), ci_lo=float('nan'),
                    ci_hi=float('nan'), se=float('nan'), n=0)
    rng = np.random.RandomState(seed)
    n = x.size
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot[b] = stat(x[idx])
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return dict(point=float(stat(x)),
                ci_lo=float(lo), ci_hi=float(hi),
                se=float(np.std(boot, ddof=1)), n=int(n))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float],
                          n_boot: int = 1000, alpha: float = 0.05,
                          seed: int = 42) -> dict:
    """Paired bootstrap on diff = mean(a) - mean(b). Requires same length."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.size == b.size, "paired_bootstrap_diff needs equal length"
    rng = np.random.RandomState(seed)
    n = a.size
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[k] = a[idx].mean() - b[idx].mean()
    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    # Two-sided bootstrap p-value: fraction of boots that crossed zero
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return dict(diff=float(a.mean() - b.mean()),
                ci_lo=float(lo), ci_hi=float(hi),
                p_boot=float(p), n=int(n))


# ── Effect sizes ─────────────────────────────────────────────────────────
def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d with pooled SD. 0.2 small / 0.5 medium / 0.8 large / 1.2+ very large."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float('nan')
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb)
                     / (a.size + b.size - 2))
    if pooled < 1e-12:
        return float('nan')
    return float((a.mean() - b.mean()) / pooled)


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    """Hedges' g: Cohen's d with small-sample correction. Recommended for n<20."""
    d = cohens_d(a, b)
    n = len(a) + len(b)
    if n <= 2 or not np.isfinite(d):
        return float('nan')
    J = 1.0 - 3.0 / (4 * n - 9)
    return float(J * d)


# ── Multiple comparison correction ───────────────────────────────────────
def bh_fdr(pvalues: Sequence[float], alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR. Returns reject mask + adjusted p-values.

    Usage for C1...Cm family: collect raw p's, call bh_fdr(ps, 0.05),
    report reject[i] (True means significant after FDR correction at q=0.05).
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return dict(reject=np.array([]), p_adj=np.array([]), alpha=alpha)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    # Largest i such that p_(i) <= (i/m)*alpha defines the cutoff
    if not passed.any():
        cutoff = -1
    else:
        cutoff = np.where(passed)[0].max()
    reject_sorted = np.zeros(m, dtype=bool)
    if cutoff >= 0:
        reject_sorted[:cutoff + 1] = True
    # BH-adjusted p-values (monotone from the top)
    p_adj_sorted = np.minimum.accumulate(
        (ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    p_adj_sorted = np.clip(p_adj_sorted, 0, 1)
    # Unsort back to original order
    reject = np.empty(m, dtype=bool)
    p_adj = np.empty(m, dtype=float)
    reject[order] = reject_sorted
    p_adj[order] = p_adj_sorted
    return dict(reject=reject, p_adj=p_adj, alpha=alpha)


# ── Power analysis (ex-ante) ─────────────────────────────────────────────
def approx_power_two_sample(d_expected: float, n_per_group: int,
                            alpha: float = 0.05) -> float:
    """Rough two-sample t-test power for an expected Cohen's d.

    Uses normal approximation (ok for n>=5). Returns power in [0,1].
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    se = np.sqrt(2.0 / n_per_group)
    z_power = d_expected / se - z_alpha
    return float(norm.cdf(z_power))


# ── Quick CLI self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, 30)
    b = rng.normal(0.6, 1, 30)  # true d ~= 0.6
    print("bootstrap_ci(a) =", bootstrap_ci(a))
    print("paired_bootstrap_diff(a,b) =", paired_bootstrap_diff(a, b))
    print("cohens_d(a,b) =", cohens_d(a, b))
    print("hedges_g(a,b) =", hedges_g(a, b))
    print("bh_fdr =", bh_fdr([0.001, 0.01, 0.03, 0.20, 0.50, 0.80]))
    print("power(d=0.8, n=5) =", approx_power_two_sample(0.8, 5))
    print("power(d=0.8, n=10) =", approx_power_two_sample(0.8, 10))

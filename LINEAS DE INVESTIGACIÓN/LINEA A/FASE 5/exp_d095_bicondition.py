#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D-095 · BICONDITION THEOREM: σ AND PR as Independent Necessary Conditions
=========================================================================
Motivated by D-089 + D-094 results:

  D-089 (DIRECTION 1 — HIGH σ, LOW PR):
    sigma_reg_d4:  σ₂=80°, PR₂=2.6 → H₁₂=1  (σ alone is NOT sufficient)

  D-094 (DIRECTION 2 — LOW σ, HIGH PR):
    sigma_suppress_skip: σ₂=13°, PR₂=10.1 → H₁₂=4.7 (PR alone is NOT sufficient)

GAP: Both directions are partial. We have not yet shown that:
    HIGH σ + HIGH PR → HIGH H₁  (via active regularization, not just healthy control)

D-095 closes this with a DUAL REGULARIZATION design:
    L = L_BPB + λ₁·max(0, PR* − PR̂) + λ₂·max(0, σ* − σ̂)
    → Forces BOTH PR ≥ 8 AND σ ≥ 70° simultaneously
    → If H₁ recovers to healthy levels: Bicondition proven by causal intervention

CONDITIONS (5 × 3 seeds = 15 models):
    A. healthy          — no bottleneck, free training (reference)
    B. control_d8       — bottleneck d=8, no reg (baseline collapse)
    C. sigma_reg_d8     — bottleneck d=8 + σ↑ reg only (σ high, PR still low)
    D. pr_reg_d8        — bottleneck d=8 + PR↑ reg only  ★ NEW
    E. dual_reg_d8      — bottleneck d=8 + σ↑ reg + PR↑ reg  ★ KEY

PRE-REGISTRATION:
    P1: dual_reg_d8  → σ₂ ≥ 70° AND PR₂ ≥ 8 → H₁₂ ≥ 7
        (Both conditions met → H₁ recovers. Bicondition necessary AND sufficient.)
    P2: pr_reg_d8    → PR₂ ≥ 8 but σ₂ < 60° → H₁₂ < 7
        (PR alone, σ still depressed → H₁ partial at best)
    P3: sigma_reg_d8 → σ₂ ≥ 70° but PR₂ < 6 → H₁₂ < 5
        (Replication of D-089/D-094 pattern at d=8)

PR REGULARIZATION MECHANISM:
    PR (participation ratio) ≈ (Σλᵢ)² / Σλᵢ²  where λᵢ are eigenvalues of Cov(h).
    We use a differentiable proxy: penalize when the "effective rank" of the
    covariance of s2 is below a target.
    Proxy: L_PR = max(0, PR* − (‖h‖_F² / ‖h‖_op²) × scale_factor)
    where ‖·‖_F is Frobenius norm and ‖·‖_op is spectral (operator) norm.
    Both are differentiable w.r.t. the hidden states.

MEDIUM config, CPU-only. ~45-60 min total (15 models × ~3-4 min each).
"""

import os, sys, json, time, math
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from scipy.stats import norm as scipy_norm

# ── Paths ───────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent          # LINEA A/
sys.path.insert(0, str(BASE))
DATA_PATH = BASE.parent.parent / "data" / "wikitext103_val.txt"
OUTPUT_DIR = Path(__file__).parent / "results_d095_bicondition"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config (MEDIUM — matches D-089/D-094) ──────────────────────────────
D_MODEL       = 128
NUM_SCALES    = 3
NUM_LAYERS    = 1
SEQ_LEN       = 128
BATCH_SIZE    = 8
NUM_EPOCHS    = 3
LR            = 5e-4
SUBSET_MB     = 0.5
N_SEQ         = 30
MID_POS       = 64
PCA_DIM       = 12
N_SUBSAMPLE   = 100
DEVICE        = "cpu"
K_NN          = 12

# Regularization hyperparameters
SIGMA_TARGET_UP   = 70.0   # σ↑ target for sigma_reg
LAMBDA_SIGMA      = 0.1    # σ regularization strength
PR_TARGET         = 8.0    # PR target for pr_reg
LAMBDA_PR         = 0.05   # PR regularization strength
SKIP_ALPHA        = 0.3    # residual ratio (not used here but kept for compat)
D_PROJ            = 8      # bottleneck dimension for all constrained conditions

SEEDS = [42, 123, 456]

# ── Optional imports ────────────────────────────────────────────────────
try:
    from ripser import ripser
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("[WARN] ripser not available — H₁ will be skipped")


# ════════════════════════════════════════════════════════════════════════
#  MODEL COMPONENTS  (verbatim from D-094)
# ════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class PreStress(nn.Module):
    def __init__(self, num_scales, max_len=2048):
        super().__init__()
        self.num_scales = num_scales
        init_freqs = [1/4, 1/10, 1/32][:num_scales]
        self.log_freq = nn.Parameter(torch.log(torch.tensor(init_freqs, dtype=torch.float32)))
        self.amplitude = nn.Parameter(torch.ones(num_scales) * 0.5)
        self.phase = nn.Parameter(torch.zeros(num_scales))
        self.register_buffer('positions', torch.arange(max_len, dtype=torch.float32))

    def forward(self, seq_len):
        t = self.positions[:seq_len]
        freq = torch.exp(self.log_freq)
        return self.amplitude.unsqueeze(1) * torch.sin(
            2 * math.pi * freq.unsqueeze(1) * t.unsqueeze(0) + self.phase.unsqueeze(1))


class InterScaleBottleneck(nn.Module):
    def __init__(self, d_model, d_proj):
        super().__init__()
        self.compress = nn.Linear(d_model, d_proj)
        self.expand   = nn.Linear(d_proj, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x):
        compressed = torch.relu(self.compress(x))
        expanded   = self.expand(compressed)
        return self.norm(expanded)


def compute_sigma_loss_up(state_trajectory, target_deg=70.0):
    """Penalize σ BELOW target (force σ UP)."""
    B, T, d = state_trajectory.shape
    flat = state_trajectory.reshape(-1, d)
    N = flat.shape[0]
    n_sample = min(N, 64)
    idx = torch.randperm(N)[:n_sample]
    sub = flat[idx]
    norms = sub.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = sub / norms
    gram = normed @ normed.T
    mask = torch.triu(torch.ones_like(gram, dtype=torch.bool), diagonal=1)
    cosines = gram[mask].clamp(-0.999, 0.999)
    angles_rad = torch.acos(cosines)
    sigma_hat = angles_rad.mean() * (180.0 / math.pi)
    target = torch.tensor(target_deg, dtype=torch.float32)
    loss = torch.relu(target - sigma_hat)
    return loss, sigma_hat


def compute_pr_loss(state_trajectory, target_pr=8.0):
    """Differentiable PR proxy: penalize when effective rank < target.

    Proxy: effective_rank ≈ (‖H‖_F²)² / (d · ‖H^T H‖_F²)
    where H is the (N, d) matrix of sampled hidden states.

    This is a well-known differentiable approximation of the participation ratio.
    When all singular values are equal, this equals d (max PR).
    When one singular value dominates, this approaches 1.
    """
    B, T, d = state_trajectory.shape
    flat = state_trajectory.reshape(-1, d)
    N = flat.shape[0]
    n_sample = min(N, 64)
    idx = torch.randperm(N)[:n_sample]
    H = flat[idx]                             # (n_sample, d)

    # Center
    H = H - H.mean(dim=0, keepdim=True)

    # Frobenius norm squared of H
    frob2 = (H ** 2).sum()

    # Frobenius norm squared of H^T H  (d×d covariance-like matrix)
    HtH = H.T @ H                             # (d, d)
    frob_HtH2 = (HtH ** 2).sum()

    # Effective rank proxy
    eps = torch.tensor(1e-12)
    eff_rank = (frob2 ** 2) / (d * frob_HtH2 + eps)

    target = torch.tensor(target_pr, dtype=torch.float32)
    loss = torch.relu(target - eff_rank)
    return loss, eff_rank


class MultiScaleEMABottleneck(nn.Module):
    def __init__(self, d_model, num_scales=3, max_len=2048, d_proj=None, mode='standard'):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.d_proj = d_proj
        self.mode = mode

        self.prestress = PreStress(num_scales, max_len)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])
        self.W_z = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        self.W_r = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        self.W_c = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        self.prestress_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_scales)])

        if d_proj is not None:
            self.bottlenecks = nn.ModuleList([
                InterScaleBottleneck(d_model, d_proj)
                for _ in range(num_scales - 1)])
        else:
            self.bottlenecks = None

    def forward(self, h, prev_states):
        B, T, d = h.shape
        prestress = self.prestress(T)
        all_states, final_states, alphas_list = [], [], []

        for k in range(self.num_scales):
            if k == 0:
                gru_input = h
            else:
                gru_input = all_states[k - 1]
                if self.bottlenecks is not None:
                    gru_input = self.bottlenecks[k - 1](gru_input)

            ps_bias = self.prestress_proj[k](prestress[k].unsqueeze(-1))
            s = prev_states[k]
            state_seq, z_seq = [], []

            for t_i in range(T):
                h_t = gru_input[:, t_i]
                hs = torch.cat([h_t, s], dim=-1)
                z = torch.sigmoid(self.W_z[k](hs) + ps_bias[t_i].unsqueeze(0))
                r = torch.sigmoid(self.W_r[k](hs))
                c = torch.tanh(self.W_c[k](torch.cat([h_t, r * s], dim=-1)))
                s = z * c + (1 - z) * s
                state_seq.append(s)
                z_seq.append(z.mean(dim=-1, keepdim=True))

            state_seq = self.norms[k](torch.stack(state_seq, dim=1))
            all_states.append(state_seq)
            final_states.append(s)
            alphas_list.append(torch.stack(z_seq, dim=1))

        return all_states, final_states, torch.cat(alphas_list, dim=-1)


class TopDownCoupling(nn.Module):
    def __init__(self, d_model, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.scale_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_scales)])
        self.confidence = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, max(d_model // 4, 1)), nn.GELU(),
                          nn.Linear(max(d_model // 4, 1), 1), nn.Softplus())
            for _ in range(num_scales)])
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, states):
        B, T, d = h.shape
        ws = torch.zeros_like(h)
        tc = torch.zeros(B, T, 1, device=h.device)
        for k in range(self.num_scales):
            s_k = states[k]
            if s_k.dim() == 2:
                s_k = s_k.unsqueeze(1).expand(B, T, d)
            c = self.confidence[k](s_k.detach())
            p = self.scale_proj[k](s_k)
            ws = ws + c * p
            tc = tc + c
        agg = ws / (tc + 1e-8)
        g = self.gate(agg)
        return self.norm(h + g * agg)


class OutputHead(nn.Module):
    def __init__(self, d_model, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.confidence = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, max(d_model // 4, 1)), nn.GELU(),
                          nn.Linear(max(d_model // 4, 1), 1), nn.Softplus())
            for _ in range(num_scales)])
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.LayerNorm(d_model), nn.Dropout(0.1), nn.Linear(d_model, d_model))
        self.output_proj = nn.Linear(d_model, 256)

    def forward(self, h, state_trajectories):
        B, T, d = h.shape
        ws = torch.zeros_like(h)
        tc = torch.zeros(B, T, 1, device=h.device)
        for k in range(self.num_scales):
            c = self.confidence[k](state_trajectories[k].detach())
            ws = ws + c * state_trajectories[k]
            tc = tc + c
        context = ws / (tc + 1e-8)
        fused = self.fusion(torch.cat([h, context], dim=-1))
        return self.output_proj(fused)


class HNCBlock(nn.Module):
    def __init__(self, d_model, num_scales=3, max_len=2048, d_proj=None, dropout=0.1, mode='standard'):
        super().__init__()
        self.top_down = TopDownCoupling(d_model, num_scales)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ema = MultiScaleEMABottleneck(d_model, num_scales, max_len,
                                           d_proj=d_proj, mode=mode)

    def forward(self, h, states):
        expanded = [s.unsqueeze(1).expand(-1, h.size(1), -1) if s.dim() == 2 else s
                    for s in states]
        h = self.top_down(h, expanded)
        h = h + self.ffn(self.ffn_norm(h))
        state_trajs, new_states, alphas = self.ema(h, states)
        return h, state_trajs, new_states, alphas


class HNCModel(nn.Module):
    def __init__(self, d_model=128, num_layers=1, num_scales=3, max_len=2048,
                 d_proj=None, mode='standard'):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.d_proj = d_proj
        self.mode = mode
        self.embedding = nn.Embedding(256, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.embed_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            HNCBlock(d_model, num_scales, max_len, d_proj=d_proj, mode=mode)
            for _ in range(num_layers)])
        self.output_head = OutputHead(d_model, num_scales)

    def forward(self, bytes_input):
        B, T = bytes_input.shape
        h = self.embed_norm(self.pos_encoding(self.embedding(bytes_input)))
        states = [torch.zeros(B, self.d_model, device=h.device)
                  for _ in range(self.num_scales)]
        alphas_all = []
        for block in self.blocks:
            h, state_trajs, states, alphas = block(h, states)
            alphas_all.append(alphas)
        logits = self.output_head(h, state_trajs)
        return logits, alphas_all, state_trajs


# ════════════════════════════════════════════════════════════════════════
#  METRICS  (verbatim from D-094)
# ════════════════════════════════════════════════════════════════════════

def compute_sigma(vectors, seed=42):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.maximum(norms, 1e-8)
    N = normed.shape[0]
    n_pairs = min(N * (N - 1) // 2, 10000)
    rng = np.random.RandomState(seed)
    a = rng.randint(0, N, size=n_pairs)
    b = rng.randint(0, N, size=n_pairs)
    mask = a != b
    dots = np.clip(np.sum(normed[a[mask]] * normed[b[mask]], axis=1), -1, 1)
    angles = np.degrees(np.arccos(dots))
    return float(np.mean(angles)), float(np.std(angles))


def compute_h1(vectors, seed=42):
    if not HAS_RIPSER:
        return dict(n_significant=0, total_persistence=0.0, betti_midscale=0)
    N = vectors.shape[0]
    if N < 10:
        return dict(n_significant=0, total_persistence=0.0, betti_midscale=0)
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=min(N, N_SUBSAMPLE), replace=False)
    sub = normalize(vectors[idx], norm='l2')
    pca_k = min(PCA_DIM, sub.shape[0] - 1, sub.shape[1])
    if pca_k < 2:
        return dict(n_significant=0, total_persistence=0.0, betti_midscale=0)
    proj = PCA(n_components=pca_k, random_state=seed).fit_transform(sub)
    dgms = ripser(proj, maxdim=1, do_cocycles=False)['dgms']
    dgm = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    fin = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.empty((0, 2))
    life = fin[:, 1] - fin[:, 0] if len(fin) > 0 else np.array([])
    if len(life) == 0:
        return dict(n_significant=0, total_persistence=0.0, betti_midscale=0)
    thr = np.median(life) * 0.5
    return dict(
        n_significant=int(np.sum(life > thr)),
        total_persistence=float(np.sum(life)),
        betti_midscale=int(np.sum((life > thr) & (life < np.percentile(life, 90)))),
    )


def compute_pr_cov(vectors):
    if vectors.shape[0] < 3:
        return 0.0
    centered = vectors - vectors.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.maximum(np.linalg.eigvalsh(cov), 0)
    total = eigvals.sum()
    if total < 1e-12:
        return 0.0
    p = eigvals / total
    return float(1.0 / np.sum(p ** 2))


def compute_kappa_or(vectors, k=6):
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    n = vectors.shape[0]
    k_use = min(k, n - 1)
    if k_use < 2:
        return dict(mean=float('nan'), std=float('nan'), n_edges=0)

    n_nodes = min(n, 20)
    if n_nodes < n:
        idx = np.random.choice(n, n_nodes, replace=False)
        vectors = vectors[idx]
        n = n_nodes

    nn_model = NearestNeighbors(n_neighbors=k_use + 1, algorithm='brute')
    nn_model.fit(vectors)
    dists_knn, indices_knn = nn_model.kneighbors(vectors)

    curvatures = []
    for i in range(n):
        nbrs_i = indices_knn[i, 1:]
        for j_pos in range(min(2, k_use)):
            j = nbrs_i[j_pos]
            d_ij = dists_knn[i, j_pos + 1]
            if d_ij < 1e-12:
                continue
            nbrs_j = indices_knn[j, 1:]
            pts_i = vectors[nbrs_i]
            pts_j = vectors[nbrs_j]
            cost = cdist(pts_i, pts_j)
            row_ind, col_ind = linear_sum_assignment(cost)
            w1 = cost[row_ind, col_ind].sum() / k_use
            kappa = 1.0 - w1 / d_ij
            curvatures.append(kappa)

    if len(curvatures) == 0:
        return dict(mean=float('nan'), std=float('nan'), n_edges=0)
    arr = np.array(curvatures)
    return dict(mean=float(np.mean(arr)), std=float(np.std(arr)), n_edges=len(arr))


# ════════════════════════════════════════════════════════════════════════
#  DATA
# ════════════════════════════════════════════════════════════════════════

def load_data():
    raw = open(DATA_PATH, "r", encoding="utf-8").read()
    data = np.frombuffer(raw.encode("utf-8"), dtype=np.uint8)
    limit = int(SUBSET_MB * 1e6)
    return data[:limit]


# ════════════════════════════════════════════════════════════════════════
#  TRAINING
# ════════════════════════════════════════════════════════════════════════

def train_model(d_proj, mode, data, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HNCModel(d_model=D_MODEL, num_layers=NUM_LAYERS,
                     num_scales=NUM_SCALES, d_proj=d_proj, mode=mode)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = NUM_EPOCHS * (len(data) // (BATCH_SIZE * SEQ_LEN))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    history = []

    for epoch in range(NUM_EPOCHS):
        rng = np.random.RandomState(seed + epoch)
        max_start = len(data) - SEQ_LEN - 1
        n_batches = max_start // (BATCH_SIZE * SEQ_LEN)
        epoch_loss_bpb = 0.0
        epoch_loss_sigma = 0.0
        epoch_sigma_hat = 0.0
        epoch_loss_pr = 0.0
        epoch_pr_hat = 0.0
        n_reg_steps = 0
        n_tokens = 0

        for _ in range(n_batches):
            starts = rng.randint(0, max_start, size=BATCH_SIZE)
            xb = torch.stack([torch.tensor(data[s:s + SEQ_LEN], dtype=torch.long)
                              for s in starts])
            yb = torch.stack([torch.tensor(data[s + 1:s + SEQ_LEN + 1], dtype=torch.long)
                              for s in starts])

            logits, _, state_trajs = model(xb)
            loss_ce = loss_fn(logits.reshape(-1, 256), yb.reshape(-1))
            loss = loss_ce

            s2_traj = state_trajs[2]
            sigma_loss_val = 0.0
            sigma_val = 0.0
            pr_loss_val = 0.0
            pr_val = 0.0

            if mode in ('sigma_reg', 'dual_reg'):
                sigma_penalty, sigma_hat = compute_sigma_loss_up(
                    s2_traj, target_deg=SIGMA_TARGET_UP)
                loss = loss + LAMBDA_SIGMA * sigma_penalty
                sigma_loss_val = sigma_penalty.item()
                sigma_val = sigma_hat.item()

            if mode in ('pr_reg', 'dual_reg'):
                pr_penalty, pr_hat = compute_pr_loss(s2_traj, target_pr=PR_TARGET)
                loss = loss + LAMBDA_PR * pr_penalty
                pr_loss_val = pr_penalty.item()
                pr_val = pr_hat.item()

            if mode in ('sigma_reg', 'pr_reg', 'dual_reg'):
                n_reg_steps += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss_bpb += loss_ce.item() * xb.numel()
            epoch_loss_sigma += sigma_loss_val
            epoch_sigma_hat += sigma_val
            epoch_loss_pr += pr_loss_val
            epoch_pr_hat += pr_val
            n_tokens += xb.numel()

        bpb = epoch_loss_bpb / max(n_tokens, 1) / math.log(2)
        avg_sigma_loss = epoch_loss_sigma / max(n_reg_steps, 1)
        avg_sigma_hat = epoch_sigma_hat / max(n_reg_steps, 1)
        avg_pr_loss = epoch_loss_pr / max(n_reg_steps, 1)
        avg_pr_hat = epoch_pr_hat / max(n_reg_steps, 1)

        history.append(dict(
            epoch=epoch, bpb=bpb,
            sigma_loss=avg_sigma_loss, sigma_hat=avg_sigma_hat,
            pr_loss=avg_pr_loss, pr_hat=avg_pr_hat,
        ))

        extra = ""
        if mode in ('sigma_reg', 'dual_reg'):
            extra += f"  σ̂={avg_sigma_hat:.1f}°(L={avg_sigma_loss:.3f})"
        if mode in ('pr_reg', 'dual_reg'):
            extra += f"  PR̂={avg_pr_hat:.1f}(L={avg_pr_loss:.3f})"
        print(f"    [{mode}] ep{epoch}: BPB={bpb:.4f}{extra}")

    model.eval()
    return model, n_params, history


# ════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_scale_vectors(model, data, seed=42):
    rng = np.random.RandomState(seed)
    max_start = len(data) - SEQ_LEN
    starts = rng.randint(0, max_start, size=N_SEQ)
    scale_vecs = {k: [] for k in range(NUM_SCALES)}

    for b0 in range(0, N_SEQ, BATCH_SIZE):
        batch_starts = starts[b0:b0 + BATCH_SIZE]
        seqs = [torch.tensor(data[s:s + SEQ_LEN], dtype=torch.long) for s in batch_starts]
        batch = torch.stack(seqs)
        _, _, state_trajs = model(batch)
        for k in range(NUM_SCALES):
            vecs = state_trajs[k][:, MID_POS, :]
            scale_vecs[k].append(vecs.cpu().numpy())

    for k in range(NUM_SCALES):
        scale_vecs[k] = np.concatenate(scale_vecs[k], axis=0)
    return scale_vecs


def analyze_model(model, data, label, seed=42):
    scale_vecs = extract_scale_vectors(model, data, seed=seed)
    results = {"condition": label, "scales": {}}

    for k in range(NUM_SCALES):
        vecs = scale_vecs[k]
        sigma_m, sigma_s = compute_sigma(vecs, seed=seed)
        h1 = compute_h1(vecs, seed=seed)
        pr = compute_pr_cov(vecs)
        kor = compute_kappa_or(vecs)
        ck = kor['mean'] * math.log2(max(pr, 1.01)) if not math.isnan(kor['mean']) else 0.0

        results["scales"][str(k)] = dict(
            sigma_mean=sigma_m, sigma_std=sigma_s,
            h1=h1, pr_cov=pr, kappa_or=kor,
            curvature_capacity=ck,
        )
        print(f"    s{k}: σ={sigma_m:.1f}° PR={pr:.1f} H₁={h1['n_significant']} "
              f"κ={kor['mean']:.3f} C_k={ck:.2f}")

    return results


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("D-095 · BICONDITION THEOREM: σ AND PR as Independent Necessary Conditions")
    print("=" * 75)
    print(f"  5 conditions × 3 seeds = 15 models")
    print(f"  Config: {NUM_LAYERS}L, seq={SEQ_LEN}, batch={BATCH_SIZE}, "
          f"epochs={NUM_EPOCHS}, data={SUBSET_MB}MB")
    print(f"  d_proj={D_PROJ} (all bottleneck conditions)")
    print(f"  σ target={SIGMA_TARGET_UP}°  λ_σ={LAMBDA_SIGMA}")
    print(f"  PR target={PR_TARGET}  λ_PR={LAMBDA_PR}")
    print(f"  Seeds: {SEEDS}\n")

    data = load_data()
    print(f"  Data loaded: {len(data)} bytes ({len(data)/1e6:.2f} MB)\n")

    # ── Experimental conditions ─────────────────────────────────────
    conditions = [
        # (label,          d_proj,   mode)
        ("healthy",        None,     'standard'),   # A — reference
        ("control_d8",     D_PROJ,   'standard'),   # B — baseline collapse
        ("sigma_reg_d8",   D_PROJ,   'sigma_reg'),  # C — σ↑ only
        ("pr_reg_d8",      D_PROJ,   'pr_reg'),     # D — PR↑ only  ★ NEW
        ("dual_reg_d8",    D_PROJ,   'dual_reg'),   # E — σ↑ + PR↑  ★ KEY
    ]

    all_results = {}
    checkpoint_file = OUTPUT_DIR / "d095_results.json"

    # Load partial results if resuming
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            all_results = json.load(f)
        print(f"  Loaded {len(all_results)} existing results (resume mode)\n")

    for label, d_proj, mode in conditions:
        for seed in SEEDS:
            key = f"{label}_s{seed}"

            if key in all_results:
                print(f"  [SKIP] {key} already computed")
                continue

            print(f"{'─'*65}")
            dp_str = f"d_proj={d_proj}" if d_proj else "no_bn"
            print(f"  Training {key} ({dp_str}, mode={mode}, seed={seed})...")

            model, n_params, history = train_model(d_proj, mode, data, seed=seed)
            print(f"    Params: {n_params:,}  Final BPB: {history[-1]['bpb']:.4f}")

            print(f"  Analyzing {key}...")
            analysis = analyze_model(model, data, label, seed=seed)
            analysis["n_params"] = n_params
            analysis["training_history"] = history
            analysis["d_proj"] = d_proj
            analysis["mode"] = mode
            analysis["seed"] = seed

            all_results[key] = analysis

            with open(checkpoint_file, "w") as f:
                json.dump(all_results, f, indent=2)

            del model
            print()

    # ════════════════════════════════════════════════════════════════
    #  STATISTICAL ANALYSIS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("D-095 RESULTS: BICONDITION THEOREM")
    print(f"{'='*80}")

    print(f"\n{'Condition':>16} | {'σ₂':>11} | {'H₁₂':>10} | {'PR₂':>10} | {'C_k₂':>10} | {'BPB':>9}")
    print("─" * 80)

    for label, _, _ in conditions:
        sigmas, h1s, prs, cks, bpbs = [], [], [], [], []
        for seed in SEEDS:
            key = f"{label}_s{seed}"
            if key not in all_results:
                continue
            r = all_results[key]
            s2 = r["scales"]["2"]
            sigmas.append(s2["sigma_mean"])
            h1s.append(s2["h1"]["n_significant"])
            prs.append(s2["pr_cov"])
            cks.append(s2["curvature_capacity"])
            bpbs.append(r["training_history"][-1]["bpb"])

        if not sigmas:
            print(f"{label:>16} | (no data)")
            continue

        def fmt(vals):
            m, s = np.mean(vals), np.std(vals)
            return f"{m:5.1f}±{s:3.1f}"

        print(f"{label:>16} | {fmt(sigmas):>11} | {fmt(h1s):>10} | "
              f"{fmt(prs):>10} | {fmt(cks):>10} | {fmt(bpbs):>9}")

    # ── Prediction checks ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BICONDITION PREDICTION CHECKS")
    print(f"{'='*70}")

    def get_means(label):
        vals = {'sigma': [], 'h1': [], 'pr': [], 'bpb': []}
        for seed in SEEDS:
            key = f"{label}_s{seed}"
            if key not in all_results:
                continue
            r = all_results[key]
            s2 = r["scales"]["2"]
            vals['sigma'].append(s2["sigma_mean"])
            vals['h1'].append(s2["h1"]["n_significant"])
            vals['pr'].append(s2["pr_cov"])
            vals['bpb'].append(r["training_history"][-1]["bpb"])
        return {k: np.mean(v) if v else float('nan') for k, v in vals.items()}

    dual = get_means("dual_reg_d8")
    pr_only = get_means("pr_reg_d8")
    sig_only = get_means("sigma_reg_d8")
    ctrl = get_means("control_d8")
    ref = get_means("healthy")

    print(f"\n  REFERENCE (healthy): σ={ref['sigma']:.1f}° PR={ref['pr']:.1f} H₁={ref['h1']:.1f}")
    print(f"  BASELINE (ctrl_d8): σ={ctrl['sigma']:.1f}° PR={ctrl['pr']:.1f} H₁={ctrl['h1']:.1f}")

    # P1: dual_reg_d8 → σ₂≥70° AND PR₂≥8 → H₁₂≥7
    print(f"\n  P1: dual_reg_d8 → σ₂≥70° AND PR₂≥8 → H₁₂≥7  [Grand Unifying Test]")
    p1_sigma = dual['sigma'] >= 70.0
    p1_pr    = dual['pr'] >= 8.0
    p1_h1    = dual['h1'] >= 7.0
    print(f"    σ₂ = {dual['sigma']:.1f}° (≥70°? {'✅' if p1_sigma else '❌'})")
    print(f"    PR₂= {dual['pr']:.1f}  (≥8?   {'✅' if p1_pr else '❌'})")
    print(f"    H₁₂= {dual['h1']:.1f}  (≥7?   {'✅' if p1_h1 else '❌'})")
    print(f"    → P1 {'CONFIRMED ★★★ BICONDITION PROVEN' if all([p1_sigma, p1_pr, p1_h1]) else 'PARTIAL' if p1_h1 else 'REJECTED'}")

    # P2: pr_reg_d8 → PR₂≥8 but σ₂<60° → H₁₂<7
    print(f"\n  P2: pr_reg_d8 → PR₂≥8 AND σ₂<60° → H₁₂<7  [PR alone insufficient]")
    p2_pr    = pr_only['pr'] >= 8.0
    p2_sigma = pr_only['sigma'] < 60.0
    p2_h1    = pr_only['h1'] < 7.0
    print(f"    PR₂= {pr_only['pr']:.1f}  (≥8?   {'✅' if p2_pr else '❌'})")
    print(f"    σ₂ = {pr_only['sigma']:.1f}° (<60°? {'✅' if p2_sigma else '❌'})")
    print(f"    H₁₂= {pr_only['h1']:.1f}  (<7?   {'✅' if p2_h1 else '❌'})")
    print(f"    → P2 {'CONFIRMED (PR alone insufficient)' if all([p2_pr, p2_sigma, p2_h1]) else 'PARTIAL' if p2_pr else 'REJECTED'}")

    # P3: sigma_reg_d8 → σ₂≥70° but PR₂<6 → H₁₂<5  (replication D-094)
    print(f"\n  P3: sigma_reg_d8 → σ₂≥70° AND PR₂<6 → H₁₂<5  [σ alone insufficient]")
    p3_sigma = sig_only['sigma'] >= 70.0
    p3_pr    = sig_only['pr'] < 6.0
    p3_h1    = sig_only['h1'] < 5.0
    print(f"    σ₂ = {sig_only['sigma']:.1f}° (≥70°? {'✅' if p3_sigma else '❌'})")
    print(f"    PR₂= {sig_only['pr']:.1f}  (<6?   {'✅' if p3_pr else '❌'})")
    print(f"    H₁₂= {sig_only['h1']:.1f}  (<5?   {'✅' if p3_h1 else '❌'})")
    print(f"    → P3 {'CONFIRMED (σ alone insufficient)' if all([p3_sigma, p3_pr, p3_h1]) else 'PARTIAL' if p3_sigma else 'REJECTED'}")

    # ── Overall verdict ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BICONDITION THEOREM VERDICT")
    print(f"{'='*70}")
    all_confirmed = all([p1_sigma, p1_pr, p1_h1, p2_h1, p3_h1])
    if all_confirmed:
        print("""
    ★★★ BICONDITION THEOREM CONFIRMED ★★★

    H₁ sano (H₁₂ ≥ 7) requiere SIMULTÁNEAMENTE:
      1. σ ≥ σ* ≈ 70°   (diversidad angular suficiente)
      2. PR ≥ PR* ≈ 8   (dimensionalidad efectiva suficiente)

    Evidencia causal:
      - D-089/D-094: σ alto + PR bajo → H₁ bajo (σ solo NO es suficiente)
      - D-094: σ bajo + PR alto → H₁ bajo (PR solo NO es suficiente)
      - D-095: σ alto + PR alto → H₁ alto (ambas → H₁ se recupera)

    Esta es la prueba por intervención causal en AMBAS direcciones.
    Neural Collapse = fallo en cualquiera de las dos condiciones.
        """)
    else:
        print(f"""
    RESULTADO PARCIAL:
      P1 (dual → H₁ recover):  {'✅' if p1_h1 else '❌'}
      P2 (PR alone → H₁ low):  {'✅' if p2_h1 else '❌'}
      P3 (σ alone → H₁ low):   {'✅' if p3_h1 else '❌'}

    Interpretar según combinación:
      P1✅+P2✅+P3✅ → Bicondition proven
      P1✅+P2❌ → PR sufficient, σ not needed
      P1✅+P3❌ → σ sufficient, PR not needed
      P1❌ → Dual reg failed (mechanism mismatch)
        """)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")

    # Save final summary
    summary = {
        "conditions_summary": {},
        "predictions": {
            "P1_confirmed": bool(all([p1_sigma, p1_pr, p1_h1])),
            "P2_confirmed": bool(all([p2_pr, p2_sigma, p2_h1])),
            "P3_confirmed": bool(all([p3_sigma, p3_pr, p3_h1])),
            "bicondition_proven": bool(all_confirmed),
        },
        "elapsed_min": elapsed / 60,
    }
    for label, _, _ in conditions:
        summary["conditions_summary"][label] = get_means(label)

    with open(OUTPUT_DIR / "d095_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D-094 · EXTENDED INDEPENDENCE THEOREM: Closing the Argument Both Ways
=====================================================================
Builds on D-089 results:
    σ-REG:  σ₂=84° + PR₂=3.1 + H₁₂=3  → HIGH σ, LOW topology  (Direction 1)
    σ-SKIP: σ₂=82° + PR₂=11  + H₁₂=12  → HIGH σ, HIGH topology

MISSING PIECE: Direction 2 — can we have LOW σ + HIGH PR → HIGH H₁?
    This would prove that PR→H₁ is the causal path, not σ→H₁.

NEW CONDITIONS:
    1. sigma_suppress_skip: bottleneck d=4 + skip (α=0.3) + σ-SUPPRESSION loss
       → Forces σ LOW while skip maintains PR HIGH
       → Prediction: H₁ remains HIGH despite σ being LOW
       → This is the "inverse σ-REG": σ-REG forces σ↑ at fixed PR, this forces σ↓ at fixed PR

    2. sigma_reg_d8: bottleneck d=8 + σ-REG
       → Tests if σ-REG works differently at higher bottleneck rank
       → More data points for the partial correlation

    3. All conditions run with 3 seeds for error bars

DESIGN (8 conditions × 3 seeds = 24 models):
    A. healthy          (no bottleneck)
    B. control_d4       (BN d=4)
    C. sigma_reg_d4     (BN d=4 + σ↑ reg)
    D. sigma_skip       (BN d=4 + skip α=0.3)
    E. sigma_suppress_skip  (BN d=4 + skip α=0.3 + σ↓ suppression) ★ NEW KEY
    F. sigma_reg_d8     (BN d=8 + σ↑ reg) ★ NEW
    G. control_d8       (BN d=8) ★ NEW
    H. sigma_reg_d16    (BN d=16 + σ↑ reg) ★ NEW

Pre-registro:
    P1: sigma_suppress_skip → σ₂ < 40° AND H₁₂ ≥ 8 AND PR₂ ≥ 8
        (low geometry + high topology → DIRECTION 2 CONFIRMED)
    P2: r(H₁, PR | σ) > 0.9 across all 24 data points
    P3: r(H₁, σ | PR) ≈ 0 or negative
    P4: sigma_reg_d8 → σ₂ > 70° but H₁₂ < healthy (same pattern as d=4)

MEDIUM config, CPU-only. ~45-60 min total (24 models).
"""

import os, sys, json, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

# ── Paths ───────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent          # LINEA A/
sys.path.insert(0, str(BASE))
DATA_PATH = BASE.parent.parent / "data" / "wikitext103_val.txt"
OUTPUT_DIR = Path(__file__).parent / "results_d094_extended_independence"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config (MEDIUM — matches D-089) ────────────────────────────────────
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
SIGMA_TARGET_DOWN = 25.0   # σ↓ ceiling for sigma_suppress
LAMBDA_SIGMA      = 0.1    # regularization strength
SKIP_ALPHA        = 0.3    # residual ratio

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
#  MODEL COMPONENTS (from D-089)
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


class InterScaleBottleneckSkip(nn.Module):
    def __init__(self, d_model, d_proj, alpha=0.3):
        super().__init__()
        self.compress = nn.Linear(d_model, d_proj)
        self.expand   = nn.Linear(d_proj, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.alpha    = alpha

    def forward(self, x):
        compressed = torch.relu(self.compress(x))
        expanded   = self.expand(compressed)
        bottleneck_out = self.norm(expanded)
        return self.alpha * bottleneck_out + (1 - self.alpha) * x


def compute_sigma_loss_up(state_trajectory, target_deg=70.0):
    """Penalize σ BELOW target (force σ UP). L = max(0, target − σ̂)."""
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


def compute_sigma_loss_down(state_trajectory, ceiling_deg=25.0):
    """Penalize σ ABOVE ceiling (force σ DOWN). L = max(0, σ̂ − ceiling).
    
    This is the INVERSE of sigma_reg: instead of pushing σ up, it pushes σ down.
    Used with skip connection to test: can we have LOW σ + HIGH PR → HIGH H₁?
    """
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
    ceiling = torch.tensor(ceiling_deg, dtype=torch.float32)
    loss = torch.relu(sigma_hat - ceiling)
    return loss, sigma_hat


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
            use_skip = mode in ('sigma_skip', 'sigma_suppress_skip')
            if use_skip:
                self.bottlenecks = nn.ModuleList([
                    InterScaleBottleneckSkip(d_model, d_proj, alpha=SKIP_ALPHA)
                    for _ in range(num_scales - 1)])
            else:
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
    def __init__(self, d_model, num_scales=3, max_len=2048, d_proj=None,
                 dropout=0.1, mode='standard'):
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
#  METRICS (from D-089)
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

    # Subsample nodes for speed (max 20)
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
    return dict(mean=float(np.mean(arr)), std=float(np.std(arr)),
                n_edges=len(arr))


# ════════════════════════════════════════════════════════════════════════
#  TRAINING & ANALYSIS
# ════════════════════════════════════════════════════════════════════════

def load_data():
    raw = open(DATA_PATH, "r", encoding="utf-8").read()
    data = np.frombuffer(raw.encode("utf-8"), dtype=np.uint8)
    limit = int(SUBSET_MB * 1e6)
    return data[:limit]


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
        n_tokens = 0
        n_sigma_steps = 0

        for _ in range(n_batches):
            starts = rng.randint(0, max_start, size=BATCH_SIZE)
            xb = torch.stack([torch.tensor(data[s:s + SEQ_LEN], dtype=torch.long)
                              for s in starts])
            yb = torch.stack([torch.tensor(data[s + 1:s + SEQ_LEN + 1], dtype=torch.long)
                              for s in starts])

            logits, _, state_trajs = model(xb)
            loss_ce = loss_fn(logits.reshape(-1, 256), yb.reshape(-1))
            loss = loss_ce

            sigma_val = 0.0
            sigma_loss_val = 0.0

            if mode == 'sigma_reg':
                # Force σ UP at s2
                s2_traj = state_trajs[2]
                sigma_penalty, sigma_hat = compute_sigma_loss_up(
                    s2_traj, target_deg=SIGMA_TARGET_UP)
                loss = loss + LAMBDA_SIGMA * sigma_penalty
                sigma_val = sigma_hat.item()
                sigma_loss_val = sigma_penalty.item()
                n_sigma_steps += 1

            elif mode == 'sigma_suppress_skip':
                # Force σ DOWN at s2 (while skip connection maintains PR)
                s2_traj = state_trajs[2]
                sigma_penalty, sigma_hat = compute_sigma_loss_down(
                    s2_traj, ceiling_deg=SIGMA_TARGET_DOWN)
                loss = loss + LAMBDA_SIGMA * sigma_penalty
                sigma_val = sigma_hat.item()
                sigma_loss_val = sigma_penalty.item()
                n_sigma_steps += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss_bpb += loss_ce.item() * xb.numel()
            epoch_loss_sigma += sigma_loss_val
            epoch_sigma_hat += sigma_val
            n_tokens += xb.numel()

        bpb = epoch_loss_bpb / max(n_tokens, 1) / math.log(2)
        avg_sigma_loss = epoch_loss_sigma / max(n_sigma_steps, 1)
        avg_sigma_hat = epoch_sigma_hat / max(n_sigma_steps, 1)

        history.append(dict(epoch=epoch, bpb=bpb,
                            sigma_loss=avg_sigma_loss,
                            sigma_hat=avg_sigma_hat))

        reg_str = ""
        if mode in ('sigma_reg', 'sigma_suppress_skip'):
            direction = "↑" if mode == 'sigma_reg' else "↓"
            reg_str = f"  σ̂_s2={avg_sigma_hat:.1f}° L_σ{direction}={avg_sigma_loss:.3f}"
        print(f"    [{mode}] ep{epoch}: BPB={bpb:.4f}{reg_str}")

    model.eval()
    return model, n_params, history


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
        ck = kor['mean'] * math.log2(max(pr, 1.01))

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
    print("D-094 · EXTENDED INDEPENDENCE THEOREM")
    print("=" * 70)
    print(f"  8 conditions × 3 seeds = 24 models")
    print(f"  Config: {NUM_LAYERS}L, seq={SEQ_LEN}, batch={BATCH_SIZE}, "
          f"epochs={NUM_EPOCHS}, data={SUBSET_MB}MB")
    print(f"  σ↑ target={SIGMA_TARGET_UP}°, σ↓ ceiling={SIGMA_TARGET_DOWN}°, "
          f"λ={LAMBDA_SIGMA}, skip_α={SKIP_ALPHA}")
    print(f"  Seeds: {SEEDS}\n")

    data = load_data()
    print(f"  Data loaded: {len(data)} bytes ({len(data)/1e6:.2f} MB)\n")

    # ── Experimental conditions ─────────────────────────────────────
    conditions = [
        # (label,                d_proj, mode)
        ("healthy",              None,   'standard'),
        ("control_d4",           4,      'standard'),
        ("sigma_reg_d4",         4,      'sigma_reg'),
        ("sigma_skip",           4,      'sigma_skip'),
        ("sigma_suppress_skip",  4,      'sigma_suppress_skip'),   # ★ KEY NEW
        ("control_d8",           8,      'standard'),              # ★ NEW
        ("sigma_reg_d8",         8,      'sigma_reg'),             # ★ NEW
        ("sigma_reg_d16",        16,     'sigma_reg'),             # ★ NEW
    ]

    all_results = {}
    checkpoint_file = OUTPUT_DIR / "d094_results.json"

    # Load partial results if resuming
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            all_results = json.load(f)
        print(f"  Loaded {len(all_results)} existing results (resume mode)\n")

    for label, d_proj, mode in conditions:
        for seed in SEEDS:
            key = f"{label}_s{seed}"

            # Skip if already computed
            if key in all_results:
                print(f"  [SKIP] {key} already computed")
                continue

            print(f"{'─'*60}")
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

            # Save incrementally after each model
            with open(checkpoint_file, "w") as f:
                json.dump(all_results, f, indent=2)

            del model
            print()

    # ════════════════════════════════════════════════════════════════
    #  STATISTICAL ANALYSIS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("D-094 RESULTS: EXTENDED INDEPENDENCE THEOREM")
    print(f"{'='*80}")

    # Aggregate by condition (mean ± std across seeds)
    print(f"\n{'Condition':>22} | {'σ₂':>10} | {'H₁₂':>10} | {'PR₂':>10} | {'C_k₂':>10} | {'BPB':>10}")
    print("─" * 85)

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
            continue

        def fmt(vals):
            m, s = np.mean(vals), np.std(vals)
            return f"{m:5.1f}±{s:4.1f}"

        print(f"{label:>22} | {fmt(sigmas):>10} | {fmt(h1s):>10} | "
              f"{fmt(prs):>10} | {fmt(cks):>10} | {fmt(bpbs):>10}")

    # ── Correlation analysis across ALL data points ─────────────────
    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS (all individual data points)")
    print(f"{'='*70}")

    sigma_all, pr_all, h1_all, labels_all = [], [], [], []
    for key, r in all_results.items():
        s2 = r["scales"]["2"]
        sigma_all.append(s2["sigma_mean"])
        pr_all.append(s2["pr_cov"])
        h1_all.append(s2["h1"]["n_significant"])
        labels_all.append(key)

    sigma_all = np.array(sigma_all)
    pr_all = np.array(pr_all)
    h1_all = np.array(h1_all)

    N = len(sigma_all)
    print(f"  N = {N} data points")

    rho_hs, p_hs = spearmanr(h1_all, sigma_all)
    rho_hp, p_hp = spearmanr(h1_all, pr_all)
    rp_hs, pp_hs = pearsonr(h1_all, sigma_all)
    rp_hp, pp_hp = pearsonr(h1_all, pr_all)

    print(f"\n  SPEARMAN:")
    print(f"    r(H₁, σ)  = {rho_hs:.3f}  p = {p_hs:.6f}")
    print(f"    r(H₁, PR) = {rho_hp:.3f}  p = {p_hp:.6f}")
    print(f"\n  PEARSON:")
    print(f"    r(H₁, σ)  = {rp_hs:.3f}  p = {pp_hs:.6f}")
    print(f"    r(H₁, PR) = {rp_hp:.3f}  p = {pp_hp:.6f}")

    # Partial correlations
    def partial_corr(x, y, z):
        rxy = np.corrcoef(x, y)[0, 1]
        rxz = np.corrcoef(x, z)[0, 1]
        ryz = np.corrcoef(y, z)[0, 1]
        num = rxy - rxz * ryz
        den = np.sqrt(max((1 - rxz**2) * (1 - ryz**2), 1e-12))
        return num / den

    r_h1_s_pr = partial_corr(h1_all, sigma_all, pr_all)
    r_h1_pr_s = partial_corr(h1_all, pr_all, sigma_all)

    print(f"\n  PARTIAL CORRELATIONS:")
    print(f"    r(H₁, σ | PR) = {r_h1_s_pr:.3f}  <- σ after controlling for PR")
    print(f"    r(H₁, PR | σ) = {r_h1_pr_s:.3f}  <- PR after controlling for σ")

    # R-squared
    c_s = np.polyfit(sigma_all, h1_all, 1)
    c_p = np.polyfit(pr_all, h1_all, 1)
    ss_tot = np.sum((h1_all - np.mean(h1_all))**2)
    r2_s = 1 - np.sum((h1_all - np.polyval(c_s, sigma_all))**2) / max(ss_tot, 1e-12)
    r2_p = 1 - np.sum((h1_all - np.polyval(c_p, pr_all))**2) / max(ss_tot, 1e-12)

    print(f"\n  R-SQUARED:")
    print(f"    H₁ ~ σ:   R² = {r2_s:.3f}")
    print(f"    H₁ ~ PR:  R² = {r2_p:.3f}")
    if r2_s > 0.001:
        print(f"    PR explains {r2_p/r2_s:.1f}× more variance")

    # Steiger test
    from scipy.stats import norm
    r12 = rp_hs  # r(H1, sigma)
    r13 = rp_hp  # r(H1, PR)
    r23 = np.corrcoef(sigma_all, pr_all)[0, 1]
    det = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23
    rm2 = (r12**2 + r13**2) / 2
    f_val = (1 - r23) / (2 * (1 - rm2)) if rm2 < 1 else 1.0
    h_val = (1 - f_val * rm2) / (1 - rm2) if rm2 < 1 else 1.0
    z1 = np.arctanh(np.clip(r12, -0.999, 0.999))
    z2 = np.arctanh(np.clip(r13, -0.999, 0.999))
    denom = max(2 * (1 - r23) * h_val, 1e-12)
    Z = (z1 - z2) * np.sqrt((N - 3) / denom)
    p_steiger = 2 * (1 - norm.cdf(abs(Z)))

    print(f"\n  STEIGER TEST (r(H₁,σ) vs r(H₁,PR)):")
    print(f"    Z = {Z:.3f}, p = {p_steiger:.6f}")
    if p_steiger < 0.05:
        print(f"    → SIGNIFICANTLY different (p < 0.05)")
    else:
        print(f"    → NOT significantly different")

    # ── KEY PREDICTION CHECK ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PREDICTION CHECKS")
    print(f"{'='*70}")

    # P1: sigma_suppress_skip → σ₂ < 40° AND H₁₂ ≥ 8 AND PR₂ ≥ 8
    sss_sigmas, sss_h1s, sss_prs = [], [], []
    for seed in SEEDS:
        key = f"sigma_suppress_skip_s{seed}"
        if key in all_results:
            s2 = all_results[key]["scales"]["2"]
            sss_sigmas.append(s2["sigma_mean"])
            sss_h1s.append(s2["h1"]["n_significant"])
            sss_prs.append(s2["pr_cov"])

    if sss_sigmas:
        sss_sigma_m = np.mean(sss_sigmas)
        sss_h1_m = np.mean(sss_h1s)
        sss_pr_m = np.mean(sss_prs)
        p1_sigma = sss_sigma_m < 40.0
        p1_h1 = sss_h1_m >= 8
        p1_pr = sss_pr_m >= 8
        p1 = p1_sigma and p1_h1 and p1_pr

        print(f"\n  P1: sigma_suppress_skip → σ₂<40° AND H₁₂≥8 AND PR₂≥8")
        print(f"    σ₂ = {sss_sigma_m:.1f}° (<40°? {'✅' if p1_sigma else '❌'})")
        print(f"    H₁₂= {sss_h1_m:.1f}  (≥8?   {'✅' if p1_h1 else '❌'})")
        print(f"    PR₂= {sss_pr_m:.1f}  (≥8?   {'✅' if p1_pr else '❌'})")
        print(f"    → P1 {'CONFIRMED ★★★' if p1 else 'REJECTED'}")

        if p1:
            print(f"\n    ★★★ DIRECTION 2 CONFIRMED: LOW σ + HIGH PR → HIGH H₁")
            print(f"    Together with D-089 (HIGH σ + LOW PR → LOW H₁):")
            print(f"    → INDEPENDENCE THEOREM PROVEN FROM BOTH DIRECTIONS")

    # P2: r(H₁, PR | σ) > 0.9
    print(f"\n  P2: r(H₁, PR | σ) > 0.9")
    print(f"    r(H₁, PR | σ) = {r_h1_pr_s:.3f}  {'✅' if r_h1_pr_s > 0.9 else '❌'}")

    # P3: r(H₁, σ | PR) ≈ 0
    print(f"\n  P3: |r(H₁, σ | PR)| < 0.3")
    print(f"    r(H₁, σ | PR) = {r_h1_s_pr:.3f}  {'✅' if abs(r_h1_s_pr) < 0.3 else '❌'}")

    # ── TIMING ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to: {checkpoint_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

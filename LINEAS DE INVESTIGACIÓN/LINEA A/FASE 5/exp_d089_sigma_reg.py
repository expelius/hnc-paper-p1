#!/usr/bin/env python3
"""
D-089 · σ-REGULARIZED BOTTLENECK: Desacoplando Geometría de Topología
======================================================================
Motivación (D-057 + D-051):
    D-057 establece: σ se expande a ~85° ANTES de que H₁ significativo emerja.
    D-051 establece: bottleneck d_proj=4 colapsa σ→16.9° y H₁→4.
    
    Pregunta causal: ¿H₁ es CONSECUENCIA AUTOMÁTICA de σ alto, o H₁ porta 
    información INDEPENDIENTE más allá de la geometría?
    
    Diseño: Forzar σ alto en bottleneck d_proj=4 vía regularización diferenciable
    y observar si H₁ se recupera (→ σ suficiente) o no (→ topología independiente).

Condiciones experimentales:
    1. CONTROL: d_proj=4 sin regularización (replica D-051 para validación)
    2. σ-REG:   d_proj=4 + λ·max(0, σ*−σ̂_s2) como loss adicional
    3. σ-SKIP:  d_proj=4 + skip connection residual que preserva dirección

Predicciones pre-registro:
    P1: σ-REG logra σ_s2 ≥ 70° (vs 16.9° control) — regularización funciona
    P2: H₁ de σ-REG < H₁ de healthy control — topología NO es solo consecuencia de σ
    P3: BPB de σ-REG ≤ BPB control (σ alto no daña, puede ayudar)
    P4: σ-SKIP logra σ parcial pero con BPB similar a control
    
    Escenarios:
    A: σ-REG→σ~80° y H₁~11 (sano) → σ es parámetro de orden único
    B: σ-REG→σ~80° y H₁~4 (colapsado) → topología independiente de geometría
    C: σ-REG→σ~80° y H₁~7 (intermedio) → relación graduada

    Mi predicción: B o C. Razón: linear rank-4 limita dimensiones topológicas
    disponibles independientemente del spread angular.

MEDIUM config. CPU-only. ~15-20 min.
"""

import os, sys, json, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

# ── Paths ───────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent          # LINEA A/
sys.path.insert(0, str(BASE))
DATA_PATH = BASE.parent.parent / "data" / "wikitext103_val.txt"
OUTPUT_DIR = Path(__file__).parent / "results_d089_sigma_reg"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config (MEDIUM — matches D-051 MEDIUM) ─────────────────────────────
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
SEED          = 42
DEVICE        = "cpu"
K_NN          = 12

# σ-REG hyperparameters
SIGMA_TARGET  = 70.0   # target σ in degrees for s2
LAMBDA_SIGMA  = 0.1    # regularization strength
SKIP_ALPHA    = 0.3    # residual ratio for σ-SKIP

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
#  MODEL COMPONENTS (identical to D-051)
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
    """Standard bottleneck d→d_proj→d (same as D-051)."""
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
    """Bottleneck with skip connection: output = α·bottleneck(x) + (1-α)·x.
    The skip preserves angular diversity from pre-bottleneck input."""
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


def compute_sigma_loss(state_trajectory, target_deg=70.0):
    """Differentiable σ penalty: max(0, target − σ̂).
    
    Computes angular spread from the state trajectory at all timesteps
    using a differentiable approximation of pairwise cosine angles.
    """
    # state_trajectory: (B, T, d) — use all positions
    B, T, d = state_trajectory.shape
    # Flatten to (B*T, d) and subsample for efficiency
    flat = state_trajectory.reshape(-1, d)  # (B*T, d)
    N = flat.shape[0]
    
    # Subsample for efficiency (max 200 pairs)
    n_sample = min(N, 64)
    idx = torch.randperm(N)[:n_sample]
    sub = flat[idx]  # (n_sample, d)
    
    # L2 normalize
    norms = sub.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = sub / norms
    
    # Pairwise cosines via gram matrix
    gram = normed @ normed.T  # (n_sample, n_sample)
    
    # Extract upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones_like(gram, dtype=torch.bool), diagonal=1)
    cosines = gram[mask]  # flattened upper triangle
    
    # Differentiable angle: arccos is differentiable in (-1, 1)
    cosines = cosines.clamp(-0.999, 0.999)
    angles_rad = torch.acos(cosines)
    sigma_hat = angles_rad.mean() * (180.0 / math.pi)
    
    # Hinge loss: penalize only if σ < target
    target_rad_deg = torch.tensor(target_deg, dtype=torch.float32)
    loss = torch.relu(target_rad_deg - sigma_hat)
    
    return loss, sigma_hat


class MultiScaleEMABottleneck(nn.Module):
    """Multi-scale GRU with optional inter-scale bottleneck.
    Returns state trajectories and optionally the σ of s2 for regularization."""

    def __init__(self, d_model, num_scales=3, max_len=2048, d_proj=None, mode='standard'):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.d_proj = d_proj
        self.mode = mode  # 'standard', 'sigma_reg', 'sigma_skip'

        self.prestress = PreStress(num_scales, max_len)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])
        self.W_z = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        self.W_r = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        self.W_c = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        self.prestress_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_scales)])

        # Bottlenecks between scales
        if d_proj is not None:
            if mode == 'sigma_skip':
                self.bottlenecks = nn.ModuleList([
                    InterScaleBottleneckSkip(d_model, d_proj, alpha=SKIP_ALPHA)
                    for _ in range(num_scales - 1)
                ])
            else:
                self.bottlenecks = nn.ModuleList([
                    InterScaleBottleneck(d_model, d_proj)
                    for _ in range(num_scales - 1)
                ])
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


class SigmaRegBlock(nn.Module):
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


class SigmaRegHNC(nn.Module):
    """BaselineE with inter-scale bottleneck and optional σ-regularization."""

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
            SigmaRegBlock(d_model, num_scales, max_len, d_proj=d_proj, mode=mode)
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
#  METRICS (identical to D-051)
# ════════════════════════════════════════════════════════════════════════

def compute_sigma(vectors, seed=SEED):
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


def compute_h1(vectors, seed=SEED):
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


def compute_kappa_or(vectors, k=K_NN):
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    n = vectors.shape[0]
    k_use = min(k, n - 1)
    if k_use < 2:
        return dict(mean=float('nan'), std=float('nan'), n_edges=0)

    nn_model = NearestNeighbors(n_neighbors=k_use + 1, algorithm='ball_tree')
    nn_model.fit(vectors)
    dists_knn, indices_knn = nn_model.kneighbors(vectors)

    curvatures = []
    for i in range(n):
        nbrs_i = indices_knn[i, 1:]
        for j_pos in range(min(3, k_use)):
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
#  TRAINING
# ════════════════════════════════════════════════════════════════════════

def load_data():
    raw = open(DATA_PATH, "r", encoding="utf-8").read()
    data = np.frombuffer(raw.encode("utf-8"), dtype=np.uint8)
    limit = int(SUBSET_MB * 1e6)
    return data[:limit]


def train_model(d_proj, mode, data, seed=SEED):
    """Train a SigmaRegHNC with given d_proj and mode.
    
    mode: 'standard' | 'sigma_reg' | 'sigma_skip'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SigmaRegHNC(d_model=D_MODEL, num_layers=NUM_LAYERS,
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
            
            # σ-regularization: penalize low σ on s2 (most vulnerable scale)
            if mode == 'sigma_reg':
                s2_traj = state_trajs[2]  # (B, T, d) — scale 2
                sigma_penalty, sigma_hat = compute_sigma_loss(
                    s2_traj, target_deg=SIGMA_TARGET)
                loss = loss + LAMBDA_SIGMA * sigma_penalty
                sigma_val = sigma_hat.item()
                epoch_loss_sigma += sigma_penalty.item()
                epoch_sigma_hat += sigma_val
                n_sigma_steps += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss_bpb += loss_ce.item() * xb.numel()
            n_tokens += xb.numel()

        bpb = epoch_loss_bpb / max(n_tokens, 1) / math.log(2)
        avg_sigma_loss = epoch_loss_sigma / max(n_sigma_steps, 1)
        avg_sigma_hat = epoch_sigma_hat / max(n_sigma_steps, 1)
        
        history.append(dict(epoch=epoch, bpb=bpb,
                            sigma_loss=avg_sigma_loss,
                            sigma_hat=avg_sigma_hat))
        
        sigma_str = f"  σ̂_s2={avg_sigma_hat:.1f}°  L_σ={avg_sigma_loss:.3f}" if mode == 'sigma_reg' else ""
        print(f"    [{mode}] epoch {epoch}: BPB={bpb:.4f}{sigma_str}")

    model.eval()
    return model, n_params, history


# ════════════════════════════════════════════════════════════════════════
#  EXTRACTION & ANALYSIS
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_scale_vectors(model, data, seed=SEED):
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


def analyze_model(model, data, label):
    scale_vecs = extract_scale_vectors(model, data)
    results = {"condition": label, "scales": {}}

    for k in range(NUM_SCALES):
        vecs = scale_vecs[k]
        sigma_m, sigma_s = compute_sigma(vecs)
        h1 = compute_h1(vecs)
        pr = compute_pr_cov(vecs)
        kor = compute_kappa_or(vecs)
        ck = kor['mean'] * math.log2(max(pr, 1.01))  # curvature capacity

        results["scales"][k] = dict(
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
    print("D-089 · σ-REGULARIZED BOTTLENECK: Geometry vs Topology Decoupling")
    print("=" * 70)
    print(f"  Conditions: CONTROL (d=4), σ-REG (d=4+L_σ), σ-SKIP (d=4+skip)")
    print(f"  Config: {NUM_LAYERS}L, seq={SEQ_LEN}, batch={BATCH_SIZE}, "
          f"epochs={NUM_EPOCHS}, data={SUBSET_MB}MB")
    print(f"  σ-REG: target={SIGMA_TARGET}°, λ={LAMBDA_SIGMA}")
    print(f"  σ-SKIP: α={SKIP_ALPHA}")

    data = load_data()
    print(f"  Data loaded: {len(data)} bytes ({len(data)/1e6:.2f} MB)\n")

    # ── Healthy control (no bottleneck) ─────────────────────────────
    conditions = [
        ("healthy",    None, 'standard'),
        ("control_d4", 4,    'standard'),
        ("sigma_reg",  4,    'sigma_reg'),
        ("sigma_skip", 4,    'sigma_skip'),
    ]
    
    all_results = {}
    
    for label, d_proj, mode in conditions:
        print(f"{'─'*60}")
        dp_str = f"d_proj={d_proj}" if d_proj else "no bottleneck"
        print(f"  Training {label} ({dp_str}, mode={mode})...")
        
        model, n_params, history = train_model(d_proj, mode, data)
        print(f"    Params: {n_params:,}  Final BPB: {history[-1]['bpb']:.4f}")
        
        print(f"  Analyzing {label}...")
        analysis = analyze_model(model, data, label)
        analysis["n_params"] = n_params
        analysis["training_history"] = history
        analysis["d_proj"] = d_proj
        analysis["mode"] = mode
        
        all_results[label] = analysis
        
        # Save incrementally
        out_file = OUTPUT_DIR / "d089_results.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        del model
        print()

    # ── SUMMARY TABLE ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("D-089 SUMMARY: GEOMETRY vs TOPOLOGY DECOUPLING")
    print(f"{'='*80}")
    print(f"{'Condition':>12} | {'BPB':>6} | {'σ₀':>5} {'σ₁':>5} {'σ₂':>5} | "
          f"{'H₁₀':>4} {'H₁₁':>4} {'H₁₂':>4} | "
          f"{'PR₀':>5} {'PR₁':>5} {'PR₂':>5} | "
          f"{'C_k₀':>5} {'C_k₁':>5} {'C_k₂':>5}")
    print("─" * 95)

    for label, _, _ in conditions:
        r = all_results[label]
        s = r["scales"]
        print(f"{label:>12} | "
              f"{r['training_history'][-1]['bpb']:>6.3f} | "
              f"{s[0]['sigma_mean']:>5.1f} {s[1]['sigma_mean']:>5.1f} {s[2]['sigma_mean']:>5.1f} | "
              f"{s[0]['h1']['n_significant']:>4} {s[1]['h1']['n_significant']:>4} {s[2]['h1']['n_significant']:>4} | "
              f"{s[0]['pr_cov']:>5.1f} {s[1]['pr_cov']:>5.1f} {s[2]['pr_cov']:>5.1f} | "
              f"{s[0]['curvature_capacity']:>5.2f} {s[1]['curvature_capacity']:>5.2f} {s[2]['curvature_capacity']:>5.2f}")

    # ── VERDICT ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("D-089 VERDICT: σ vs H₁ INDEPENDENCE")
    print(f"{'='*70}")

    healthy = all_results["healthy"]
    ctrl = all_results["control_d4"]
    sreg = all_results["sigma_reg"]
    sskip = all_results["sigma_skip"]

    # Key metrics for s2 (most affected)
    h_s2_sigma = healthy["scales"][2]["sigma_mean"]
    h_s2_h1 = healthy["scales"][2]["h1"]["n_significant"]
    c_s2_sigma = ctrl["scales"][2]["sigma_mean"]
    c_s2_h1 = ctrl["scales"][2]["h1"]["n_significant"]
    r_s2_sigma = sreg["scales"][2]["sigma_mean"]
    r_s2_h1 = sreg["scales"][2]["h1"]["n_significant"]
    sk_s2_sigma = sskip["scales"][2]["sigma_mean"]
    sk_s2_h1 = sskip["scales"][2]["h1"]["n_significant"]

    print(f"\n  Scale s2 (most vulnerable to bottleneck):")
    print(f"    healthy:    σ={h_s2_sigma:.1f}°  H₁={h_s2_h1}")
    print(f"    control_d4: σ={c_s2_sigma:.1f}°  H₁={c_s2_h1}")
    print(f"    σ-REG:      σ={r_s2_sigma:.1f}°  H₁={r_s2_h1}")
    print(f"    σ-SKIP:     σ={sk_s2_sigma:.1f}°  H₁={sk_s2_h1}")

    # P1: σ-REG achieves σ_s2 ≥ 70°?
    p1 = r_s2_sigma >= 70.0
    print(f"\n  P1 (σ-REG σ_s2 ≥ 70°): {'✅' if p1 else '❌'} σ_s2={r_s2_sigma:.1f}°")

    # P2: H₁ still reduced vs healthy?
    p2 = r_s2_h1 < h_s2_h1 * 0.75  # significantly less
    print(f"  P2 (H₁ still < healthy): {'✅' if p2 else '❌'} "
          f"H₁_reg={r_s2_h1} vs H₁_healthy={h_s2_h1}")

    # Determine scenario
    if p1 and r_s2_h1 >= h_s2_h1 * 0.75:
        scenario = "A (σ sufficient — H₁ is epiphenomenon)"
    elif p1 and r_s2_h1 < h_s2_h1 * 0.5:
        scenario = "B (σ necessary not sufficient — topology is INDEPENDENT)"
    elif p1:
        scenario = "C (graduated — σ enables but doesn't determine H₁)"
    else:
        scenario = "INCONCLUSIVE (σ-REG failed to raise σ)"

    print(f"\n  ══════════════════════════════════════════")
    print(f"  SCENARIO: {scenario}")
    print(f"  ══════════════════════════════════════════")

    # BPB comparison
    print(f"\n  BPB comparison:")
    for label, _, _ in conditions:
        bpb = all_results[label]["training_history"][-1]["bpb"]
        print(f"    {label:>12}: {bpb:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Saved: {OUTPUT_DIR / 'd089_results.json'}")
    print(f"  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()

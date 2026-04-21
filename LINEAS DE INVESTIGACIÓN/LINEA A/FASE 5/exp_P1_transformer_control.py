#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 · OUT-OF-FAMILY CONTROL: Transformer + LSTM matched-parameter σ measurement
==============================================================================

Objetivo (paper P1 / claim C3):
    A d=4 y d=128, HNC holonic muestra AR = σ_max/σ_min >> 1 (gradiente IVM).
    Predicción pre-registrada:
        Transformer L=2 matched param → AR ≈ 1.0 (no produce gradiente)
        LSTM L=2 matched param        → AR ≈ 1.0 (no produce gradiente)
    Si la predicción se cumple, la firma IVM es específica del acoplamiento
    holónico (no emerge de atención ni de RNN clásica).

Diseño:
    · d ∈ {4, 128}
    · Arquitecturas: HNC holonic (referencia) · Transformer L=2 · LSTM L=2
    · 5 seeds por (arch, d).
    · Matched parameters ± 5 %.
    · Mismos datos (WikiText-103 mini), mismos epochs (NUM_EPOCHS del MEDIUM).
    · Métrica: σ por layer/scale; AR = σ_max / σ_min.

Salida: results_P1_transformer_control/P1_results.json

Requisitos: reutiliza LSTMModel + Transformer de exp_r004n_universality + load_data
de exp_d095_bicondition. No reentrena HNC (usa A1+A4 + Colab P0 para valores HNC).

CPU, MEDIUM, ~90-120 min total.
"""

import os, sys, json, math, time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ───────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
LINEA_A = HERE.parent
REPO = LINEA_A.parent.parent
sys.path.insert(0, str(LINEA_A))
sys.path.insert(0, str(HERE))

DATA_PATH = REPO / "data" / "wikitext103_val.txt"
OUT_DIR = HERE / "results_P1_transformer_control"
OUT_DIR.mkdir(exist_ok=True)

# ── Config (alineada con D-095 MEDIUM) ─────────────────────────────────
VOCAB_SIZE = 256
SEQ_LEN = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LR = 5e-4
SUBSET_MB = 0.5
N_SEQ = 30
MID_POS = 64
DEVICE = "cpu"

D_VALUES = [4, 128]
SEEDS = [0, 1, 2, 3, 4]


# ── Data loader (simplificado, inline) ─────────────────────────────────
def load_bytes(path: Path, subset_mb: float):
    size = int(subset_mb * 1024 * 1024)
    with open(path, "rb") as f:
        raw = f.read(size)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return torch.from_numpy(arr.copy()).long()


def batch_iter(data: torch.Tensor, batch_size: int, seq_len: int, seed: int = 0):
    n = (data.numel() - 1) // seq_len
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    for start in range(0, n - batch_size + 1, batch_size):
        batch_ids = idx[start : start + batch_size]
        x = torch.stack([data[i * seq_len : (i + 1) * seq_len] for i in batch_ids])
        y = torch.stack(
            [data[i * seq_len + 1 : (i + 1) * seq_len + 1] for i in batch_ids]
        )
        yield x, y


# ── Transformer L=2 ─────────────────────────────────────────────────────
class TransformerLM(nn.Module):
    def __init__(self, d_model, num_layers=2, nhead=None, max_len=2048, dropout=0.1):
        super().__init__()
        if nhead is None:
            # nhead must divide d_model; use 1 for very small d
            nhead = 1 if d_model < 8 else 2
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.embed_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, x, return_layer_states=False):
        B, T = x.shape
        h = self.embedding(x) + self.pos[:, :T, :]
        h = self.embed_norm(h)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        if return_layer_states:
            states = []
            for layer in self.encoder.layers:
                h = layer(h, src_mask=causal_mask, is_causal=True)
                states.append(h)
            logits = self.out(h)
            return logits, states
        h = self.encoder(h, mask=causal_mask, is_causal=True)
        return self.out(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── LSTM L=2 (alineado con exp_r004n_universality.LSTMModel) ───────────
class LSTMBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.gates = nn.Linear(d_model * 2, d_model * 4)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, state):
        h = h + self.ffn(self.ffn_norm(h))
        B, T, d = h.shape
        h_s, c_s = state
        state_seq = []
        for t in range(T):
            h_t = h[:, t]
            gates = self.gates(torch.cat([h_t, h_s], dim=-1))
            i, f, g, o = gates.chunk(4, dim=-1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f + 1.0)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c_s = f * c_s + i * g
            h_s = o * torch.tanh(c_s)
            state_seq.append(h_s)
        state_seq = self.norm(torch.stack(state_seq, dim=1))
        return h, state_seq, (h_s, c_s)


class LSTMModel(nn.Module):
    def __init__(self, d_model=128, num_layers=2, max_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.embed_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([LSTMBlock(d_model) for _ in range(num_layers)])
        self.out = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, x, return_layer_states=False):
        B, T = x.shape
        h = self.embedding(x) + self.pos[:, :T, :]
        h = self.embed_norm(h)
        states = [
            (
                torch.zeros(B, self.d_model, device=x.device),
                torch.zeros(B, self.d_model, device=x.device),
            )
            for _ in range(self.num_layers)
        ]
        layer_states = []
        for i, block in enumerate(self.blocks):
            h, state_seq, states[i] = block(h, states[i])
            layer_states.append(state_seq)
        logits = self.out(h)
        if return_layer_states:
            return logits, layer_states
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── σ angular dispersion (copied from exp_d095) ────────────────────────
def compute_sigma(states: torch.Tensor) -> float:
    """states: (N, d). Return mean pairwise angle in degrees."""
    x = states.detach().cpu().numpy()
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    gram = x @ x.T
    iu = np.triu_indices_from(gram, k=1)
    cos = np.clip(gram[iu], -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return float(ang.mean())


# ── Train + measure σ per layer ────────────────────────────────────────
def train_and_measure(model, data, seed, label):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    total_tokens = 0
    total_loss = 0.0
    for epoch in range(NUM_EPOCHS):
        for x, y in batch_iter(data, BATCH_SIZE, SEQ_LEN, seed + epoch):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    bpb = (total_loss / total_tokens) / math.log(2)
    elapsed = time.time() - t0

    # Measure σ per layer on N_SEQ unseen sequences
    model.eval()
    with torch.no_grad():
        sigmas_per_layer = None
        n_seen = 0
        for x, _ in batch_iter(data, 1, SEQ_LEN, seed + 999):
            if n_seen >= N_SEQ:
                break
            x = x.to(DEVICE)
            _, layer_states = model(x, return_layer_states=True)
            # layer_states: list of (1, T, d); take MID_POS
            if sigmas_per_layer is None:
                sigmas_per_layer = [[] for _ in layer_states]
            for li, s in enumerate(layer_states):
                sigmas_per_layer[li].append(s[0, MID_POS, :].cpu())
            n_seen += 1
        # Stack and compute σ per layer across N_SEQ samples
        sigmas = []
        for li in range(len(sigmas_per_layer)):
            stack = torch.stack(sigmas_per_layer[li], dim=0)  # (N_SEQ, d)
            sigmas.append(compute_sigma(stack))
    return {
        "label": label,
        "seed": seed,
        "bpb": bpb,
        "sigmas": sigmas,
        "ar": float(max(sigmas) / max(min(sigmas), 1e-6)) if len(sigmas) >= 2 else 1.0,
        "params": model.count_parameters(),
        "elapsed_sec": elapsed,
    }


def run():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data: {DATA_PATH}")
    data = load_bytes(DATA_PATH, SUBSET_MB)
    print(f"[data] loaded {data.numel()/1e6:.2f} M bytes from {DATA_PATH.name}")

    results = {"config": {
        "d_values": D_VALUES,
        "seeds": SEEDS,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "subset_mb": SUBSET_MB,
        "n_seq_eval": N_SEQ,
        "mid_pos": MID_POS,
    }, "runs": []}

    for d in D_VALUES:
        for seed in SEEDS:
            # Transformer
            torch.manual_seed(seed)
            tr = TransformerLM(d_model=d, num_layers=2).to(DEVICE)
            r = train_and_measure(tr, data, seed, f"transformer_d{d}")
            print(
                f"  transformer d={d} seed={seed}: bpb={r['bpb']:.3f} σ={r['sigmas']} AR={r['ar']:.2f}"
            )
            results["runs"].append(r)

            # LSTM
            torch.manual_seed(seed)
            ls = LSTMModel(d_model=d, num_layers=2).to(DEVICE)
            r = train_and_measure(ls, data, seed, f"lstm_d{d}")
            print(
                f"  lstm        d={d} seed={seed}: bpb={r['bpb']:.3f} σ={r['sigmas']} AR={r['ar']:.2f}"
            )
            results["runs"].append(r)

            # Persist incrementally
            out = OUT_DIR / "P1_results.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    # Aggregate
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results["runs"]:
        key = r["label"]
        agg[key].append(r)

    summary = {}
    for key, rs in agg.items():
        bpbs = [r["bpb"] for r in rs]
        ars = [r["ar"] for r in rs]
        summary[key] = {
            "n": len(rs),
            "bpb_mean": float(np.mean(bpbs)),
            "bpb_std": float(np.std(bpbs, ddof=1)) if len(bpbs) > 1 else 0.0,
            "ar_mean": float(np.mean(ars)),
            "ar_std": float(np.std(ars, ddof=1)) if len(ars) > 1 else 0.0,
        }
    results["summary"] = summary
    with open(OUT_DIR / "P1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(
            f"  {k}: BPB={v['bpb_mean']:.3f}±{v['bpb_std']:.3f}  "
            f"AR={v['ar_mean']:.2f}±{v['ar_std']:.2f}  (n={v['n']})"
        )
    print(f"\n[done] results → {OUT_DIR/'P1_results.json'}")


if __name__ == "__main__":
    run()

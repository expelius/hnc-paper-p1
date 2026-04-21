# Paper P1 — Draft (prose)

> **Status:** §1 Introduction and §2 Method complete (2026-04-20).
> §3 Results, §4 Discussion, Abstract: pending experiments (D-095 + A2 + A3 + P1-TRCTRL + Colab P0).
> Complement to [PAPER-P1-HNC-SKELETON.md](PAPER-P1-HNC-SKELETON.md) (structural plan, tables, checklists).
> The prose below is the **authoritative text** to migrate to LaTeX at freeze.

---

## 1. Introduction

### 1.1 The problem

Neural Collapse (NC), introduced by Papyan, Han & Donoho [2020], established that deep classifiers trained past zero training error converge to a highly structured geometry: the penultimate-layer features align with a Simplex Equiangular Tight Frame (ETF), class means collapse to its vertices, and classifier weights align with the features. This phenomenon is remarkable because it is *unpredicted by the loss*: nothing in cross-entropy explicitly asks for equiangularity. The geometry emerges as an attractor of the optimization dynamics under sufficient capacity. A growing literature now characterises NC at initialisation (unconstrained features models, Zhu et al. 2021), under data imbalance (Fang et al. 2021), and as a compositional hierarchy (Hierarchical Neural Collapse, Galanti et al. 2022).

A conspicuous gap remains: **all of this work addresses classification.** The analogous question for *autoregressive* or *generative* models — models whose loss is not a one-hot target but a distribution over next tokens — has received almost no quantitative treatment. This is not a minor oversight. Language models are the single largest deployment of deep learning today, and the shape of their representational space is the object of most interpretability work (Park et al. 2024; Elhage et al. 2022; Templeton et al. 2024). Yet we do not know whether such models converge, in the limit of training, to any canonical geometry at all — let alone what that geometry would be.

### 1.2 Why hierarchical recurrent models

If a generative analogue of NC exists, the natural place to look is not a flat Transformer but a model whose architecture *forces* a hierarchy of representations. Three recent lines of work motivate this: Hierarchical Dilated Language Models (HDLM, NeurIPS 2025), byte-level H-Nets (CMU 2025), and multi-scale state-space models (Mamba, Gu & Dao 2023; S4, Gu 2022). In all of them the hidden state is explicitly partitioned into *scales* that operate at different temporal granularities and that communicate through specific (top-down or bottom-up) pathways. These scales are not latent clusters a reviewer might post-hoc identify: they are architectural primitives the designer chose.

This gives us a well-posed question. **Fix a byte-level model with K explicit scales and a strict hierarchical coupling (scale $k-1 \to k$). Measure the angular geometry of each scale after training. Does anything converge?**

### 1.3 What we find

The answer is yes, and the geometry is unexpected. We introduce HNC (Hierarchical Neural Collapse), a K=3 scale byte-level recurrent model built from EMA-bottlenecked GRUs coupled sequentially s₀ → s₁ → s₂, and study its representational geometry on WikiText-103. Three main results:

- **C1 (IVM signature at low dimension).** At small model dimension $d=4$, the inter-scale angles converge to the Isotropic Vector Matrix configuration (60°/120°) — the same packing that minimises angular variance in $\mathbb{R}^3$ — with negligible perplexity cost relative to a parallel (non-hierarchical) control matched in parameters. We call this configuration the IVM signature.
- **C2 (architectural gradient, amplified by training).** The monotone dispersion gradient $\sigma_0 > \sigma_1 > \sigma_2$ across scales is present already at random initialisation, but with the characteristic topological signature of an *isotropic* point cloud ($H_1^{\text{excess}} < 0$). Training amplifies the gradient by a factor of ≈ 1.5× and simultaneously flips the topology to non-trivial ($H_1^{\text{excess}} > 0$), establishing that HNC does more than expose a passive bottleneck.
- **C3 (specificity to hierarchical coupling).** Matched-parameter controls — a 2-layer Transformer and a 2-layer LSTM at the same $d$ — do not produce the IVM signature. The effect is specific to the combination of recurrent state, explicit K-scale partition, and strict causal coupling between scales.

### 1.4 Why this matters

Three implications:

1. **A canonical geometry for hierarchical generative models.** NC showed that classifiers have a canonical attractor. HNC shows that a class of hierarchical recurrent language models has one too — distinct from ETF, and strictly requiring hierarchical causal asymmetry to emerge.
2. **A measurement-only probe of hierarchy.** The IVM signature is computed from hidden states alone, without supervision or per-sample labels. This makes it a *non-invasive diagnostic* for whether a multi-scale model is actually using its scales (rather than collapsing them via coadaptation).
3. **A falsifiable prediction for the broader design space.** We pre-register (§6) four concrete predictions — including adversarial corpus shuffling (P-A3), a causal permutation test (P-A2), and an out-of-family Transformer control (P-C5) — whose rejection would retract the central claim. The central paper reports only confirmations; refutations, should they occur, are reported separately.

### 1.5 Related work

The closest antecedents are Galanti et al. [2022] on Hierarchical NC in compositional classifiers, Papyan et al. [2020] on the original ETF signature, and recent multi-scale LMs (HDLM, H-Net, Mamba, S4). None of these measures *angular* geometry per scale in an autoregressive model, nor proposes a canonical configuration for hierarchical recurrent representations. The synergetics literature (Fuller 1975) provides the mathematical object — the IVM — but has never been connected to learned representations. We deliberately use the term *neural collapse* for continuity with Papyan's line of work and to signal that HNC is a strict generalisation: classification NC is the K=1, supervised-label special case of what we characterise here at K=3 for byte-level language modelling.

### 1.6 Contributions

1. The HNC architecture: a K=3 sequentially coupled EMA-bottleneck recurrent model that reliably produces the IVM signature at low $d$.
2. Evidence (C1, Table 1 and Figure 1) that this signature emerges at *zero* BPB cost over a matched parallel control.
3. A decomposition (C2) showing that the inter-scale gradient is partly architectural (present at init) and partly learned (amplified and topologically transformed by training).
4. A matched-parameter out-of-family control (C3) showing specificity to hierarchical recurrent coupling.
5. A public pre-registration with four falsifiable predictions, frozen prior to the execution of the corresponding experiments (A2, A3, P1-TRCTRL, A5).
6. Full reproducibility: open code (Apache-2.0), fixed-seed shuffled evaluation corpora (CC-BY-SA 3.0 per upstream), and a Zenodo DOI for the exact commit that produced the reported tables.

### 1.7 Paper structure

§2 defines the HNC architecture, the baselines, and the measurement pipeline. §3 reports main results (C1–C3) and the pre-registered adversarial and causal tests. §4 discusses scope, limitations, and the relationship to synergetics and to broader interpretability work. §5 closes.

---

## 2. Method

### 2.1 Notation and problem setting

We work in the byte-level autoregressive setting. Let $x_{1:T} \in \{0,\ldots,255\}^T$ be a sequence of bytes and $p_\theta(x_t \mid x_{<t})$ the model's next-byte distribution. Training minimises the mean cross-entropy (reported in bits-per-byte, BPB = $-\frac{1}{T\log 2}\sum_t \log p_\theta(x_t \mid x_{<t})$). We take $T = 128$ throughout the main tables; robustness to $T \in \{64, 256\}$ is reported in the appendix.

Given a trained model, we extract per-scale state trajectories from an evaluation set: for scale $k \in \{0, 1, \ldots, K-1\}$ and sequence $i$, $h^{(k)}_{i,t} \in \mathbb{R}^d$ is the hidden state at position $t$. We measure geometry at a fixed anchor position $t = \text{MID\_POS} = 64$ (mid-context) over $N_{\text{seq}}$ independent evaluation sequences.

### 2.2 The HNC architecture

**Backbone.** Each scale $k$ is an EMA-bottlenecked GRU cell of dimension $d$:
$$h^{(k)}_{t} = \text{GRU}_k\!\left(\text{bottleneck}\big(h^{(k-1)}_{t}\big),\; h^{(k)}_{t-1}\right)$$
with the convention $h^{(-1)}_t := \text{emb}(x_t)$ (the byte embedding). The *bottleneck* is a linear projection $\mathbb{R}^d \to \mathbb{R}^{d_{\text{proj}}} \to \mathbb{R}^d$ with an EMA-smoothed low-rank middle layer; when $d_{\text{proj}} = d$ the bottleneck is an identity and the model degrades to a stacked GRU.

**Coupling.** Each scale $k$ sees only the output of scale $k-1$, never its own output downstream (strict bottom-up causality). This is the architectural primitive that distinguishes HNC from a generic stacked RNN: scale $k$ is forced to build its prediction on whatever compressed summary scale $k-1$ has produced, and a periodic PreStress term (see §2.3) penalises scale $k$ for reproducing scale $k-1$'s geometry.

**Output head.** A linear projection from $h^{(K-1)}_t$ to the 256 byte logits. The intermediate states $h^{(0)}, \ldots, h^{(K-2)}$ are *supervised only through their downstream effect on the final logit* — there is no per-scale auxiliary loss in the standard configuration. (Appendix D reports ablations with per-scale losses; the IVM signature is diminished, confirming that the canonical geometry emerges under a *single* terminal supervision signal.)

**Parameter counts.** For $d \in \{4, 64, 128, 256\}$ the HNC model has respectively ≈ $[10^3, 1.3\times10^5, 5.2\times10^5, 2.1\times10^6]$ parameters. All external baselines are matched to within ±1% of these counts by adjusting their layer count and hidden size.

### 2.3 PreStress

The only non-standard term in the training loss. At every epoch we compute, per scale, the angular dispersion $\sigma_k$ (defined in §2.5) of a small set of activations and add a penalty proportional to the minimum pairwise $|\sigma_k - \sigma_j|$. This pushes the model away from *degenerate* solutions where two scales collapse to the same dispersion. It does **not** specify the direction or magnitude of the gradient $\sigma_0 > \sigma_1 > \sigma_2$ — that the gradient forms in one direction rather than the other is an empirical consequence of the coupling.

### 2.4 Controls

**Intra-family control (parallel).** Identical backbone, but every scale reads directly from the byte embedding (`coupling = parallel`). Matched parameters. This isolates the contribution of the *hierarchy* from the contribution of the multi-scale partition alone.

**Out-of-family controls (Transformer, LSTM).**
- *Transformer*: 2-layer causal encoder, $d_{\text{model}} = d$, 4 heads, matched parameters.
- *LSTM*: 2-layer stacked LSTM, $d$ hidden, matched parameters.
Both reuse the `exp_P1_transformer_control.py` protocol: same optimiser, same schedule, same corpus, same seeds (§2.6).

**Untrained control (random init).** The same HNC architecture, evaluated *before* the first gradient step. All geometric metrics are computed on the random-init state with identical $N_{\text{seq}}$, $t$, and subsampling as the trained models. Reported in §3 alongside the trained values to factor out architectural bias.

### 2.5 Metrics

We measure five quantities per scale $k$ on the evaluation set. All are defined operationally in `stats_helpers.py`:

- **Angular dispersion** $\sigma_k$ (degrees). For $N_{\text{seq}}$ state vectors $\{h^{(k)}_i\}$ extracted at $t = 64$, compute the mean pairwise angle
$$\sigma_k = \frac{1}{\binom{N_{\text{seq}}}{2}} \sum_{i<j} \arccos\!\left(\frac{\langle h^{(k)}_i, h^{(k)}_j \rangle}{\|h^{(k)}_i\|\,\|h^{(k)}_j\|}\right) \cdot \frac{180}{\pi}.$$
We also report $\theta_k := \sigma_k$ in the IVM context.

- **Anisotropy ratio** $\text{AR} = \sigma_{\max}/\sigma_{\min}$ across scales. $\text{AR} = 1$ corresponds to perfect scale-equality (all three scales geometrically indistinguishable); $\text{AR} \gg 1$ indicates a strong hierarchy.

- **Participation ratio** $\text{PR}_k = (\text{tr}\,C_k)^2 / \text{tr}(C_k^2)$ where $C_k$ is the covariance of the scale-$k$ activations. PR is a continuous dimensional measure: $\text{PR}_k \in [1, d]$; low PR means the activations concentrate in a few principal directions, high PR means they spread isotropically.

- **Persistent homology** $H_1^{(k)}$. After L2-normalising the $N_{\text{seq}} = 100$ activations and projecting to the first 12 principal components (`PCA_DIM=12`), we compute the rank-1 persistence diagram with ripser. $H_1^{(k)}$ denotes the number of rank-1 features whose persistence exceeds a fixed quantile threshold (reported at the 75th percentile of Gaussian-null persistences; see next).

- **$H_1$ excess.** $H_1^{\text{excess}(k)} = H_1^{(k)}_{\text{trained}} - H_1^{(k)}_{\text{gaussian-null}}$, where the Gaussian null is computed on a cloud with the same mean, covariance, and sample size as the real activations. This isolates the topological complexity that is *not* explainable by the first two moments of the distribution.

### 2.6 Training and evaluation protocol

- **Dataset.** WikiText-103 validation split, first 500 KB, byte-level. Held-out evaluation uses `data/wikitext103_val.txt`; adversarial evaluation (§3.5) uses the three shuffled variants generated by `prep_shuffle_corpus.py` with SEED=20260420. The test split `data/wikitext103_test.txt` is reserved for the final pre-registered BPB table and not inspected during development.
- **Optimisation.** AdamW, $\text{LR} = 5\times 10^{-4}$, weight decay $10^{-2}$, cosine schedule, gradient clip 1.0, batch size 8, $N_{\text{epochs}} = 3$. Configuration HNC-MEDIUM (`NUM_EPOCHS=3, BATCH_SIZE=8, SEQ_LEN=128`).
- **Seeds.** All claims with quantitative effect sizes use $n \geq 5$ seeds: $\{42, 123, 456, 789, 20260420\}$. Smaller pilot runs (e.g. the $K=3 \times 3$-seed A2 permutation test) use $\{42, 123, 456\}$.
- **Evaluation.** $N_{\text{seq}} = 30$ sequences, anchor position $t = 64$, $N_{\text{subsample}} = 100$ for persistent homology.
- **Hardware.** Main tables on a single CPU (Intel, 2.4 GHz class). The scaling table for $d \in \{64, 128, 256\}$ with $n = 5$ was run on Colab (L4/A100) — the script is the same (`COLAB_UNIFIED_P0_R004N_DSWEEP.py`); the hardware only affects runtime.

### 2.7 Statistical analysis pipeline

Every reported effect comes through the same four steps, implemented in `stats_helpers.py`:

1. **Point estimate + percentile bootstrap 95% CI.** 1000 resamples over seeds. For paired comparisons (same seed, different condition), we use the paired bootstrap.
2. **Effect size.** Cohen's $d$ and Hedges' $g$ (bias-corrected for small $n$). Interpretation uses the standard thresholds (0.2 small / 0.5 medium / 0.8 large).
3. **Multiple-comparison control.** Benjamini–Hochberg FDR at $q = 0.05$ over the explicit family of tests of a given section. The family structure is declared in §6 of [OSF-PREREG-P1.md](OSF-PREREG-P1.md) prior to data inspection.
4. **Post-hoc power.** For any claim that fails to reach $p < 0.05$, we report the approximate power at the observed effect size; this distinguishes "null effect" from "underpowered".

No test is run twice with different parameters, no outlier is dropped, and no seed is replaced. The family and its corrections are fixed before the corresponding experiment is executed (see pre-registration timestamp). Deviations from this protocol are permitted only under the rules declared in [OSF-PREREG-P1.md §5](OSF-PREREG-P1.md) and are flagged as *post-hoc exploratory* in the manuscript.

---

## 3. Results

*Pending. Depends on the completion of:*

- *D-095 bicondition CPU run (in progress, ≈ 40 min remaining).*
- *A2 causal permutation test (`exp_A2_causal_permutation.py`, queued, smoke-tested).*
- *A3 adversarial shuffle evaluation (`exp_A3_shuffled_eval.py`, queued, smoke-tested).*
- *P1-TRCTRL Transformer/LSTM controls (`exp_P1_transformer_control.py`, queued, smoke-tested).*
- *Colab P0 scaling sweep $d \in \{64, 128, 256\}$, 5 seeds (≈ 12 h).*

*Template tables are frozen in [PAPER-P1-HNC-SKELETON.md §4](PAPER-P1-HNC-SKELETON.md); only the numerical entries are pending.*

---

## 4. Discussion, 5. Conclusions

*Pending §3 results.*

---

## 6. Pre-registration link

See [OSF-PREREG-P1.md](OSF-PREREG-P1.md). OSF URL and Zenodo DOI will be inserted here after the freeze (see OSF-PREREG-P1.md §9 checklist).

# Paper P1 — Esqueleto operativo

**Título:** *Hierarchical Neural Collapse: θ→60° at Zero Perplexity Cost in Multi-Scale Recurrent Models*
**Longitud objetivo:** 8 pp main + apéndice
**Venue:** ICLR / NeurIPS main track (short)
**Estado:** 90%. Bloqueador único = Colab P0 scaling (corriendo). Stats ya listos.

---

## Abstract (~200 palabras, rellenar con números al cerrar Colab)

- **Contexto:** Neural Collapse (Papyan et al. 2020) describe ETF emergente en clasificación. No existe análogo para modelos de lenguaje recurrentes multi-escala.
- **Gap:** ¿Emerge una geometría canónica cuando un modelo recurrente debe comprimir representaciones a baja dimensión bajo una jerarquía causal?
- **Método:** HNC (Holonic Neural Collapse) model, K=3 escalas EMA-bottleneck, d∈{4,64,128,256}, byte-level WikiText-103. Comparamos holonic (secuencial s₀→s₁→s₂) vs parallel (K escalas independientes) a matched parameters.
- **Resultado principal (C1):** a d=4, los ángulos inter-escala convergen a la configuración IVM (60°/120°) — ΔAR = +51.6° [bootstrap CI: __, __], d_Cohen = __, q_BH < 0.01 — sin coste de perplexity (ΔBPB ≈ 0).
- **Resultado 2 (C2):** el gradiente σ₀ > σ₁ > σ₂ es **arquitectónico** (medible a la inicialización aleatoria, d=__), el entrenamiento lo **amplifica** separando σ de PR y llevando H₁_excess de negativo a positivo.
- **Implicación:** HNC generaliza Neural Collapse al régimen secuencial-jerárquico y provee la primera caracterización geométrica cuantitativa de compresión hierarchica en RNNs.

---

## 1. Introduction (≈1 página)

1. **Hook:** Papyan et al. 2020 mostraron que clasificadores convergen a ETF bajo pérdida balanceada. **¿Y en modelos generativos con jerarquía causal temporal?**
2. **Brecha:** RNNs multi-escala (HNC, H-Net, HDLM) se han evaluado por perplexity, no por geometría. La literatura de interpretabilidad mide features puntuales, no forma del espacio.
3. **Contribución en 3 puntos:**
   - C1: Introducimos HNC y probamos que a baja d el modelo converge a IVM (60°/120°).
   - C2: El gradiente σ es arquitectónico al init; el training lo amplifica (separación σ-PR, H₁_excess cruza cero).
   - C3: **Matched-parameter** Transformer y LSTM NO reproducen la firma IVM → la geometría es específica del acoplamiento holónico.
4. **Mapa del paper.**

---

## 2. Related Work (≈0.75 página)

- Neural Collapse: Papyan 2020, Han et al. 2022 (geometric NC), Zhu et al. 2021 (unconstrained features).
- Hierarchical LMs: HDLM (NeurIPS 2025), H-Net (CMU 2025), Mamba (Gu & Dao 2023), S4 (Gu 2022).
- Synergetics/IVM: Fuller 1975 (contexto); ningún ML paper previo ha medido IVM en activaciones.
- Interpretability geometry: Park et al. 2024 (linear representation hypothesis), Elhage et al. 2022 (superposition). Todos locales, no multi-escala.
- Persistent homology en RNNs: Sun et al. 2023 — complementario.

---

## 3. Method (≈1.25 páginas)

### 3.1 HNC architecture
- K=3 escalas EMA-bottleneck GRU, acoplamiento secuencial s₀→s₁→s₂ vía `InterScaleBottleneck(d_model, d_proj)`.
- Byte-level, WikiText-103, PreStress periódico por escala.
- Fuente: `LINEA A/baseline_e.py`.

### 3.2 Parallel control (intra-familia)
- K=3 escalas idénticas pero **sin acoplamiento secuencial** (cada s_k observa h directamente).
- Matched parameters ± 1%.

### 3.3 External baselines (P1 nuevo)
- **Transformer encoder causal** L=2, d_model=d, matched parameters a HNC holonic.
- **LSTM** L=2, matched. Reutiliza `exp_r004n_universality.py::LSTMModel`.

### 3.4 Métricas
- **σ_k = ángulo medio inter-pares por escala (°)** — dispersión angular.
- **AR = anisotropy ratio** = σ_max/σ_min entre escalas.
- **PR_k = participation ratio** de la covarianza de h por escala.
- **H₁_k, H₁_excess** = persistent homology menos Gaussian null (ver `stats_helpers`).
- **BPB** = bits per byte (compresión).

### 3.5 Rigor estadístico
- n = 5 seeds por configuración.
- Bootstrap CI 95% (1000 resamples), `stats_helpers.bootstrap_ci`.
- Cohen's d / Hedges' g: `cohens_d`, `hedges_g`.
- BH-FDR q=0.05 sobre familia de comparaciones: `bh_fdr`.
- Power analysis: `approx_power_two_sample`.
- Pre-registration OSF con timestamps D-089 (2026-04-07), D-094 (2026-04-11), D-095 (2026-04-20), F10 (fecha commit).

---

## 4. Results

### 4.1 Main claim (C1): θ→60° at zero-BPB cost
**Tabla 1** (llenar):

| Arch | d | BPB | σ₀ | σ₁ | σ₂ | AR | Δ vs parallel |
|---|---|---|---|---|---|---|---|
| HNC holonic | 4 | __ | __ | __ | __ | __ | +51.6° |
| HNC parallel | 4 | __ | __ | __ | __ | __ | — |
| HNC holonic | 64 | __ | __ | __ | __ | __ | __ |
| HNC holonic | 128 | 1.785 | 16.3 | 14.2 | 12.1 | 1.35 | __ |
| HNC holonic | 256 | __ | __ | __ | __ | __ | __ |

**Figura 1** — scatter (BPB, AR) con holonic vs parallel a 4 valores de d. Zero cost line = horizontal ΔBPB=0.

### 4.2 Architectural vs learned (C2): untrained baseline
**Tabla 2** (A1+A4 ya DONE):

| Estado | σ₀ | σ₁ | σ₂ | σ-PR corr | H₁_excess |
|---|---|---|---|---|---|
| Untrained (random init) | __ | __ | __ | alto | < 0 |
| Trained (EXP-13) | 16.3 | 14.2 | 12.1 | bajo | > 0 |

**Figura 2** — barras con bootstrap CI para H₁_excess untrained (negativo) vs trained (positivo). q_BH < 0.01.

Narrativa: el gradiente σ₀>σ₁>σ₂ es un **sesgo inductivo arquitectónico**; el training LO AMPLIFICA separando σ de PR y generando complejidad topológica genuina (H₁_excess > 0).

### 4.2b Causal permutation test (A2, C2-bis)
**Pregunta:** ¿el gradiente σ₀>σ₁>σ₂ depende causalmente del **orden** jerárquico de las escalas, o es sólo un efecto dimensional del bottleneck?

**Diseño:** sobre los checkpoints entrenados de D-095 (5 condiciones × 3 seeds), aplicar una permutación dura de escalas antes de la evaluación: swap s₀↔s₂ y swap s₁↔s₂. Re-medir σ_k, PR_k, H₁_k. Si el fenómeno es causalmente jerárquico, AR debe **colapsar** tras la permutación.

**Tabla 2b** (llenar tras `exp_A2_causal_permutation.py`):

| Condición | AR original | AR tras swap s₀↔s₂ | AR tras swap s₁↔s₂ | ΔH₁_excess |
|---|---|---|---|---|
| healthy      | __ | __ | __ | __ |
| dual_reg_d8  | __ | __ | __ | __ |
| sigma_reg_d8 | __ | __ | __ | __ |

**Predicción pre-registrada (P-A2):** AR_post_swap ≤ 0.5 × AR_original con q_BH < 0.05. Si se rechaza, la geometría NO es causalmente jerárquica y habría que reformular el claim C1 como "dimensional" en vez de "hierarchical".

### 4.3 Out-of-family control (C3): Transformer & LSTM matched
**Tabla 3** (llenar tras `exp_P1_transformer_control.py`):

| Arch | L | d | params | BPB | σ layer-1 | σ layer-2 | AR |
|---|---|---|---|---|---|---|---|
| Transformer | 2 | 4 | __ | __ | __ | __ | ≈ 1.0 (predicción) |
| LSTM | 2 | 4 | __ | __ | __ | __ | ≈ 1.0 (predicción) |
| HNC holonic | — | 4 | __ | __ | __ | __ | **1.5+** |

Predicción pre-registrada: Transformer L2 y LSTM L2 NO producen AR > 1.3 a d=4. Solo HNC holonic lo hace.

### 4.4 Ablations (appendix si falta espacio)
- Bottleneck on/off.
- K=2 vs K=3 vs K=4 (referencia D-099 futuro).
- PreStress on/off.

### 4.5 Adversarial paragraph-shuffle (A3, C4)
**Pregunta:** ¿el gradiente y la firma topológica de s₂ dependen de **estructura discursiva** (ordering supra-oracional), o de estadística local de n-gramas?

**Diseño:** evaluar modelo entrenado (D-095 healthy / dual_reg_d8) sobre tres corpora shuffled ya generados:
- `wikitext103_val_shuf_paragraph.txt` — preserva párrafos, rompe orden discursivo.
- `wikitext103_val_shuf_sentence.txt` — preserva oraciones, rompe cohesión de párrafo.
- `wikitext103_val_shuf_word.txt` — preserva palabras, destruye sintaxis.

Medir σ_k, PR_k, H₁_k por escala y comparar con eval en corpus original.

**Tabla 4** (llenar tras `exp_A3_shuffled_eval.py`):

| Corpus | σ₀ | σ₁ | σ₂ | H₁(s₀) | H₁(s₁) | H₁(s₂) | BPB |
|---|---|---|---|---|---|---|---|
| original         | __ | __ | __ | __ | __ | __ | __ |
| paragraph-shuf   | __ | __ | __ | __ | __ | __ | __ |
| sentence-shuf    | __ | __ | __ | __ | __ | __ | __ |
| word-shuf        | __ | __ | __ | __ | __ | __ | __ |

**Predicción pre-registrada (P-A3):** disociación específica de escala bajo paragraph-shuffle:
- Δσ₀, ΔH₁(s₀), ΔPR(s₀) ≈ 0 (no significativo, q_BH > 0.05).
- Δσ₂, ΔH₁(s₂), ΔPR(s₂) **significativos** con caída ≥15% y q_BH < 0.05.

Si la predicción se cumple, es la prueba causal más limpia de que s₂ codifica **discurso** y no estadística local — valida la interpretación funcional de la jerarquía de escalas.

### 4.6 Tensegrity-Transformer inductive bias (A5, C3-bis)
**Pregunta:** ¿la firma IVM es específica de HNC (GRU holónico con PreStress + EMA bottleneck) o emerge en **cualquier** arquitectura con inductive bias tensegrítico explícito?

**Diseño:** construir un Transformer mínimo L=2 modificado con (i) top-down coupling capa L₂→L₁ (feedback attention residual), (ii) confidence gating por layer (α-gate entrenable que pondera contribución de cada capa al output). Matched parameters a HNC holonic d=4 y d=128. Mismo protocolo de medición que §4.1.

**Tabla 5** (llenar tras `exp_A5_tensegrity_transformer.py`):

| Arch | d | BPB | σ L1 | σ L2 | AR | H₁_excess |
|---|---|---|---|---|---|---|
| Transformer vanilla    | 4 | __ | __ | __ | __ | __ |
| Tensegrity-Transformer | 4 | __ | __ | __ | __ | __ |
| HNC holonic (ref.)     | 4 | __ | __ | __ | __ | __ |

**Predicción pre-registrada (P-A5, doble-ganadora):**
- **Caso A** — Tensegrity-Transformer reproduce AR ≥ 1.3 y H₁_excess > 0 → el hallazgo es **arquitectónico universal** (inductive bias → firma IVM). Claim C1 se generaliza a una familia.
- **Caso B** — no la reproduce → HNC es especial. Hay que identificar *qué* componente del GRU holónico es necesario (candidatos: PreStress periódico, EMA bottleneck, cascada secuencial s_{k-1}→s_k). Nueva subsección de ablations.

Ambos resultados son publicables. Este experimento **no puede perderse**.

---

## 5. Discussion (≈0.75 página)

- **Conexión con IVM/synergetics:** 60° = packing óptimo en R³. Interpretable pero no causal.
- **Generalización de NC:** ETF (d=C−1) vs IVM (d pequeño, K escalas). HNC requiere *hierarchical causal asymmetry* para romper la simetría de permutación entre escalas.
- **Limitaciones:**
  - byte-level solamente (no BPE/word)
  - WikiText-103 único (futuro: C4, Books)
  - hasta d=256 (no GPT-2 scale)
- **Trabajo futuro:** P2 (topología), P3 (bicondition σ∧PR), P4 (liquid tensegrity), P9 (universalidad L-104 sobre H-Net open source).

---

## 6. Reproducibility & Pre-registration (checklist)

- [ ] Código: `baseline_e.py`, `COLAB_UNIFIED_P0_R004N_DSWEEP.py`, `exp_ctrl_A1_A4_untrained_and_null_tda.py`, `stats_helpers.py`, `exp_P1_transformer_control.py` — **todos en repo público antes del submit**.
- [ ] Seeds: {0,1,2,3,4} explícitos.
- [ ] Commits SHA listados para D-089 (`501ff8e` pre-reg), D-094, D-095, F10.
- [ ] OSF DOI pre-reg.
- [ ] Hardware: CPU Intel + Colab L4/A100 (especificado por experimento).

---

## 7. Figuras previstas

| # | Contenido | Archivo fuente |
|---|---|---|
| F1 | (BPB, AR) scatter holonic/parallel × d | Colab P0 CSV + matplotlib |
| F2 | Bootstrap CI H₁_excess untrained vs trained | `results_ctrl_A1_A4/*.json` |
| F3 | σ per scale × d (holonic, 5 seeds, error bars) | Colab P0 CSV |
| F4 | IVM schematic (tetraedro 60°/120°) | figura conceptual |
| F5 (app) | Transformer/LSTM σ per layer control | `results_P1_transformer/` |
| F6 (app) | Ablation K=2/3/4, bottleneck on/off | `results_P1_ablation/` |

---

## 8. Tareas exactas antes de submit

Orden sugerido:

1. **Colab P0** termina → extraer tabla d=64/128/256 (n=5).
2. Correr `exp_P1_transformer_control.py` (CPU, ~2h con d=4 + d=128).
3. Correr `exp_P1_seeds5_d4.py` (pendiente de crear, ~30 min): replicación d=4 holonic/parallel con 5 seeds.
4. `stats_helpers`: bootstrap_ci + hedges_g + bh_fdr → tabla final.
5. Figuras 1-6.
6. Draft 8 pp sobre este esqueleto.
7. OSF pre-reg.
8. Submit arXiv → ICLR/NeurIPS.

---

## 9. Riesgos y mitigaciones

| Riesgo | Prob | Impacto | Mitigación |
|---|---|---|---|
| d=256 no muestra AR > parallel (HNC se diluye) | media | alto | Reformular: "HNC emerge bajo presión de compresión; desaparece cuando d > d_crit" — hallazgo más rico |
| Transformer matched replica AR | baja | alto | Ya sabemos que D-017 mostró atención perjudica; repetir con σ/AR explícito |
| 5 seeds insuficientes | baja | medio | Aumentar a 10 si Colab permite |
| Reviewer pide ImageNet/C4 | alta | bajo | appendix con mini-C4 run |
| "Solo 1 arquitectura" | media | medio | Out-of-family (T, LSTM) + futuro P2 replica en LSTM |

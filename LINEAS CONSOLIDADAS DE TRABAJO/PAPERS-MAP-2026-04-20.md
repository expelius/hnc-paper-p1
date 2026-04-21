# PAPERS MAP — Segmentación de ALPHA monolítico en 8 papers autónomos

**Versión:** v1 · 2026-04-20
**Autor de la decisión:** revisión exhaustiva LINEA A/A2/B/C/D/E/F + SANDBOX (esta sesión)
**Motivo:** ALPHA v1 concentra 5 claims heterogéneas en 8 pp → riesgo de que un reviewer hostil a uno tumbe los otros cuatro. Segmentar → cada paper auto-contenido, venue óptimo, publicación secuencial.

> **Trazabilidad:** cada claim/predicción citada aquí se sustenta en filas de [EXPERIMENTS-LOG.md](EXPERIMENTS-LOG.md) §3 (D-### ↔ P#.§X). Si no hay fila, no hay claim.

---

## 0. Diagnóstico del monolito previo

ALPHA v1 (`PAPER-ALPHA-OUTLINE.md`) mezcla:

| Capa | Claim | Público | Nivel rigor actual |
|---|---|---|---|
| Geometría estática | HNC θ→60° (F10) | ML core | ★★★★ |
| Topología | ρ(σ,H₁)=−1 (R-003) | TDA/math | ★★★ |
| Dinámica (switch) | Liquid Tensegrity ICC=0.21 | DS/RNN | ★★★ |
| Dinámica (input) | Lyapunov ratchet | DS/RNN | ★★ |
| Termodinámica | SADE R²=1, cascada, NESS | stat-phys | ★★ |

Un reviewer ICLR puede aceptar el primero y rechazar el quinto → arrastra todo.

**Decisión:** 8 papers independientes, orden de publicación determinado por (a) madurez de evidencia y (b) dependencias de cómputo aún corriendo.

---

## 1. Los 8 papers

### P1 · HNC — "Hierarchical Neural Collapse"
**Título tentativo:** *"Hierarchical Neural Collapse: θ→60° at Zero Perplexity Cost in Multi-Scale Recurrent Models"*
**Venue objetivo:** ICLR / NeurIPS main (short 9 pp)
**Claim único:** F10 — ΔAR = +51.6° entre holonic(d=4) y parallel(d=4) **a ΔBPB ≈ 0**. La arquitectura holónica colapsa los ángulos inter-escala hacia 60° (geometría IVM) sin coste de compresión.
**Evidencia disponible:**
- F10 d=4: ΔAR=+51.6° (DONE)
- F5b1 bootstrap TI: d=4 holonic TI=47.7° [47.6, 47.8] p<10⁻⁴ (DONE)
- F11 MI↔TI ρ=+0.97 (DONE)
- A1+A4 control untrained (2026-04-20): σ gradient **ya existe al init** → el paper ahora vende "training AMPLIFICA la jerarquía", con H₁_excess cruzando de negativo a positivo (DONE hoy)
**Bloqueadores:** Colab P0 scaling d=64/128/256 (corriendo, ~12h) + 5 seeds replica d=4.
**Próximos pasos:** ver §3.
**Estado:** **90% — PRIORIDAD MÁXIMA. Primer paper en salir.**

---

### P2 · TOPOLOGY-COMPRESSION — "Inverse Scaling σ↔H₁"
**Título tentativo:** *"Geometric Compression-Complexity Tradeoff: Inverse Scaling of Angular Dispersion and Persistent Homology in Hierarchical Recurrent States"*
**Venue objetivo:** TMLR / *Journal of Topology & Analysis* / NeurIPS TDA workshop
**Claim único:** R-003 — ρ(σ, H₁_pers) = −1.00 a través de K=3 escalas. Comprimir angularmente **genera** complejidad topológica.
**Evidencia disponible:**
- EXP-F5 persistent homology: s₀(H₁=7.10) → s₁(14.64) → s₂(18.32) (DONE)
- A4 null TDA Gaussiano (DONE hoy) — H₁_excess bien definido con fallback Cholesky/eigh.
**Bloqueadores:**
- N_SUBSAMPLE=100 es bajo → reviewer TDA pedirá ≥500.
- Necesita ≥5 seeds.
- Necesita replicar en ≥2 arquitecturas (HNC + LSTM) para descartar artefacto de GRU.
**Próximos pasos:** diseñar `exp_P2_topology_replication.py` con N_SUBSAMPLE=500, 5 seeds, 2 arquitecturas.
**Estado:** 70%.

---

### P3 · BICONDITION — "σ AND PR as dual necessary conditions"
**Título tentativo:** *"A Bicondition Theorem: Angular Dispersion and Participation Ratio Are Independently Necessary for Topological Complexity in Hierarchical Representations"*
**Venue objetivo:** NeurIPS workshop (TDA/geometry) / ICML workshop / TMLR
**Claim único:** D-089 + D-094 + D-095. Ni σ↑ solo ni PR↑ solo recuperan H₁; **ambos simultáneamente sí** (dual_reg_d8). Diseño causal con intervenciones bidireccionales.
**Evidencia disponible:**
- D-089 σ alto + PR bajo → H₁ bajo (DONE)
- D-094 σ bajo + PR alto → H₁ bajo (DONE)
- D-095 dual_reg corriendo AHORA (15 modelos, ~40 min)
- A1+A4 recalibra: el gradiente σ existe al init → el hallazgo-training es "separación σ-PR + H₁_excess > 0" (DONE hoy)
**Bloqueadores:** terminar D-095.
**Próximos pasos:** tras D-095 → aplicar `stats_helpers.bh_fdr` a las 5 condiciones × 3 seeds × 3 escalas; bootstrap CI a cada H₁.
**Estado:** 60% (evidencia casi completa; falta análisis estadístico + draft).

---

### P4 · LIQUID TENSEGRITY — "Dynamic phase coexistence in Jacobian"
**Título tentativo:** *"Liquid Tensegrity: Dynamic Crystal-Fluid Phase Coexistence in Recurrent Jacobian Dimensions"*
**Venue objetivo:** *Neural Computation* / ICML dynamical-systems track
**Claim único:** R-004 (a-r). Dimensiones individuales del Jacobiano alternan cristal (PR≈1–5) / fluido (PR≈60–80) con ICC=0.21 y 92% switchers.
**Evidencia disponible:**
- R-004l, R-004m P1 controls (DONE)
- R-004p,q,r constant/shuffled input (DONE)
- R-004n universality (Colab corriendo) — LSTM + Mamba
**Bloqueadores:** Colab R-004n (~12h, corriendo); 5 seeds.
**Próximos pasos:** tras R-004n → paper con C6 universalidad.
**Estado:** 80%.

---

### P5 · INPUT-STABILIZED RECURRENCE — "The music stabilizes the instrument"
**Título tentativo:** *"Input-Stabilized Recurrence: An Asymmetric Lyapunov Ratchet in Language-Driven Neural Dynamics"*
**Venue objetivo:** NeurIPS / ICML DS track
**Claim único:** λ_max 8× más negativo con inglés vs constante; τ_gain≈0.1, τ_loss≈5 (ratchet asimétrico). Refuta edge-of-chaos como requisito universal.
**Evidencia disponible:**
- R-004p constant SR=0.01, R-004q shuffled, R-004r English (DONE)
**Bloqueadores:** replicación 5 seeds; control cross-lingual (H-CROSS).
**Próximos pasos:** experimento mini cross-lingual (ES/ZH/code) sobre checkpoint EXP-13.
**Estado:** 70%.

---

### P6 · SEMANTIC CALORIMETRY — "SADE + NESS"
**Título tentativo:** *"Semantic Calorimetry: A Stochastic Advection-Diffusion Equation with Conservative Potential and Non-Equilibrium Steady State in Recurrent Language Dynamics"*
**Venue objetivo:** *Physical Review E* (stat-phys) / *Entropy* / TMLR
**Claim único:** SADE ∂x/∂t = −∇Ψ + G·∇x + e(t) con R²=1.00, Ψ conservativo CV=14%, cascada ×6035, NESS vía Harada-Sasa.
**Evidencia disponible:**
- Cal-A..I series (DONE, ver `results_cal_*`)
**Bloqueadores:**
- Cero rigor estadístico actual (n=1, sin CI, sin FDR).
- Re-análisis pendiente con `stats_helpers.bootstrap_ci` + BH-FDR.
**Próximos pasos (Carril C del ADDENDUM):** `exp_cal_reanalysis_stats.py` que recompute CI y q-values sobre los JSONs existentes.
**Estado:** 60%.

---

### P7 · PROTEIN FOLDING ISOMORPHISM — "TDA bridge LM ↔ proteins"
**Título tentativo:** *"Computational Protein Folding: Topological Data Analysis Reveals Shared Persistent Homology Between Hierarchical Neural Representations and Protein Structure"*
**Venue objetivo:** **Bioinformatics** / **PLoS Comp Bio** (audiencia cross-disciplinaria)
**Claim único:** 4/6 hipótesis proteicas SUPPORTED: H-ANF (ρ=0.83), H-FUNNEL (F_k monótono), H-MG (intermedios), H-FRUST (loops↔frustration ρ=0.15–0.35).
**Evidencia disponible:**
- `LINEA F/exp_d064_fractal_folding.py`, `exp_d067_bridge1.py`, `exp_d068_theta_dynamics.py`, `bootstrap_L024.py`, `P2_L024_results.json`, `protein_experiment_results/`.
**Bloqueadores:** H-ANF full-data Colab (no mini); H-UNFOLD + H-IDP pendientes.
**Próximos pasos:** ejecutar H-ANF con wikitext103_train (500MB) en Colab GPU; cerrar H-UNFOLD mecánico.
**Estado:** 50%.

---

### P8 · MESH-NEURAL UNIVERSALITY — "62% variance match"
**Título tentativo:** *"Mesh-Neural Universality: Statistical Equivalence Between Physical Tensegrity Meshes and Recurrent Neural Representations (η²≈0.63)"*
**Venue objetivo:** TMLR / *Physical Review E* short / SciPost
**Claim único:** Sandbox cloth rectangular η²_topo=0.98 (alta rigidez) + η²_GRU=0.64 (tensegrity dinámica) emparejan estadísticamente bajo mismas magnitudes de forzamiento.
**Evidencia disponible:**
- `SANDBOX_GEOMETRICO/results_cloth_universality/results.json` (DONE)
- Análogo `results_cloth_figure1/` (DONE)
**Bloqueadores:** un control más (mesh con conectividad escalada a la holónica K=3).
**Próximos pasos:** añadir mesh hex como puente entre rectangular (90°) e IVM (60°).
**Estado:** 70% — paper cortito (4-6 pp).

---

### P9 · UNIVERSALITY — "L-104 as empirical test on H-Net"
**Título tentativo:** *"Testing the Universality of Hierarchical Neural Collapse: Applying the HNC Protocol to Pre-trained H-Net Checkpoints"*
**Venue objetivo:** NeurIPS workshop (interpretability) / TMLR / ICLR Blogpost Track
**Claim único:** L-104 promovida de "hipótesis" a test empírico. Aplicar el mismo protocolo de medición de P1 (σ_k, PR_k, H₁_k, AR, bicondición σ∧PR) sobre checkpoints públicos de **H-Net** (CMU 2025, open source). Si aparecen las mismas firmas → universalidad demostrada; si no → HNC es específico y el Discussion de P1 se reajusta.
**Evidencia disponible:**
- H-Net repo público + checkpoints pre-entrenados.
- `stats_helpers.py`, pipelines de medición ya funcionan en P1.
- `PAPER-P1-HNC-SKELETON.md` define el protocolo de medición.
**Bloqueadores:** cargar H-Net sin bugs de env (tokenizer distinto — vocab HNet != byte-level); decidir cómo mapear sus K etapas a "escalas" comparables con HNC K=3.
**Próximos pasos:** `exp_P9_hnet_universality.py` que (a) clona H-Net repo, (b) carga ckpt, (c) extrae estados por etapa, (d) aplica protocolo P1.
**Estado:** 40% — paper corto (5-7 pp). Post-P1. **Convierte L-104 de hipótesis a resultado publicable, en cualquiera de sus dos direcciones.**

---

## 2. Tabla maestra con dependencias

| Paper | Claim | Madurez | Bloqueador | ETA razonable | Dependencias |
|---|---|---:|---|---|---|
| **P1** | HNC θ→60° | 90% | Colab P0 d=256 + 5 seeds | **más próximo** | stats_helpers ✓, A1+A4 ✓ |
| P3 | Bicondition σ∧PR | 60% | D-095 corriendo | corto plazo | D-095 → FDR → draft |
| P4 | Liquid Tensegrity | 80% | Colab R-004n | corto plazo | R-004n → universality table |
| P2 | σ↔H₁ inverse | 70% | n≥5, N_sub≥500, 2 archs | medio plazo | compute libre post-P1 |
| P8 | Mesh universality | 70% | hex mesh control | medio plazo | un script sandbox |
| P5 | Input ratchet | 70% | cross-lingual | medio plazo | checkpoint EXP-13 |
| P9 | Universalidad H-Net | 40% | cargar ckpt H-Net | medio plazo | post-P1 (reusa protocolo) |
| P6 | Calorimetry | 60% | re-análisis FDR | medio-largo | Carril C del ADDENDUM |
| P7 | Protein isomorphism | 50% | H-ANF full-data + UNFOLD | largo | Colab libre post-P0 |

---

## 3. Qué HAY que hacer para cada paper (checklist ejecutable)

### P1 (PRIORIDAD MÁXIMA)
- [ ] Colab P0 devuelve d=64/128/256 F10 5 seeds  ← **corriendo**
- [ ] Tabla final con bootstrap CI + Cohen's d + BH-FDR (usar `stats_helpers`)
- [ ] Figura F10 main + figura A1+A4 control + figura scaling d
- [ ] Matched-param Transformer L=2 control a d=4 y d=256 (ver `exp_P1_transformer_control.py`)
- [ ] A2 causal permutation test (`exp_A2_causal_permutation.py`) — post-D-095
- [ ] A3 shuffled corpora evaluation (`exp_A3_shuffled_eval.py`) — post-D-095
- [ ] A5 Tensegrity-Transformer inductive bias (`exp_A5_tensegrity_transformer.py`)
- [ ] Draft 8 pp (esqueleto en `PAPER-P1-HNC-SKELETON.md`)
- [ ] OSF pre-registration timestamps D-089/D-094/D-095/F10

### P3
- [ ] D-095 termina ← **corriendo**
- [ ] `stats_helpers.bh_fdr` sobre 5 condiciones × 3 seeds × 3 escalas
- [ ] Bootstrap CI para H₁ por condición
- [ ] Draft 8 pp (reutilizar §Intro y §Methods de P1)

### P4
- [ ] Colab R-004n termina ← **corriendo**
- [ ] Tabla universalidad HNC / LSTM / Mamba
- [ ] Draft 9 pp

### P2
- [ ] `exp_P2_topology_replication.py` — N_SUBSAMPLE=500, 5 seeds, HNC+LSTM
- [ ] Bootstrap CI sobre H₁ y ρ(σ,H₁)
- [ ] Draft 7 pp

### P5
- [ ] `exp_P5_cross_lingual_ratchet.py` — inglés/español/chino/código sobre EXP-13
- [ ] Replica n=5 seeds para R-004p,q,r
- [ ] Draft 7 pp

### P6
- [ ] `exp_P6_cal_reanalysis_stats.py` — lee todos `results_cal_*/cal_*_summary.json`, aplica bootstrap_ci + bh_fdr
- [ ] Re-formular R²=1 con CI (ahora mismo es n=1 y por eso R²=1 exacto — delata overfit al fit lineal)
- [ ] Draft 10 pp

### P7
- [ ] Re-ejecutar H-ANF con wikitext103_train completo (Colab GPU)
- [ ] Cerrar H-UNFOLD (mecánico) y H-IDP (disorder index)
- [ ] Draft 8 pp

### P8
- [ ] `exp_P8_mesh_hex.py` — añadir mesh hexagonal como puente 90°↔60°
- [ ] Draft 5 pp

---

## 4. Descartes explícitos (qué NO se publica solo)

| Hallazgo | Destino |
|---|---|
| ALPHA v1 monolítico | **DEPRECADO**. Material se reutiliza en P1+P4+P6. |
| SADE-net prescriptiva | Post-tesis (idea de arquitectura sin evidencia) |
| Oscillatory/neurociencia (§6) | Capítulo tesis OMEGA. 1/13 predicciones no publica |
| Jitterbug / IVM-como-meta | Archivado (R-001 cerrado) |
| LINEA D sola (GPT-2) | Sección related-work + appendix en P1 y P2 |
| LINEA C/E puramente teóricas | Capítulos fundamentales de OMEGA (tesis) |
| LINEA A2 CFI | Cerrada. Subsumida por V8 |

---

## 5. Cadencia de publicación sugerida

```
 Ahora ──► Colab P0 + D-095 terminan
           │
           ▼
        [P1 arXiv] ──► feedback comunidad
           │
           ▼
        [P3 + P4 arXiv en paralelo]
           │
           ▼
        [P2 + P5 + P8]  ← mismos datos, 3 papers cortos
           │
           ▼
        [P6]  ← requiere re-análisis pesado
           │
           ▼
        [P7]  ← paper cross-disciplinario, venue distinto
           │
           ▼
        OMEGA = tesis doctoral (integra los 8)
```

Total: 8 papers en ~12 meses a ritmo 1.5/mes es realista con GPU Colab + CPU local + rigor `stats_helpers` ya implementado.

---

## 6. Qué se gana con la segmentación

1. **Inmunidad entre papers:** si un reviewer tumba P6 (calorimetría), P1-P4 siguen vivos.
2. **Venue óptimo por paper:** P7 a Bioinformatics, P6 a PRE, P1 a ICLR — cada uno en su ecosistema.
3. **Velocidad:** P1 puede salir a arXiv en cuanto termine Colab. No espera a SADE ni a proteínas.
4. **Rigor enfocable:** cada paper cumple `stats_helpers` (n≥5, bootstrap CI, BH-FDR) sin diluir.
5. **Citabilidad cruzada:** cada paper cita al anterior → serie coherente que refuerza credibilidad.

---

## 7. Documentos asociados

- `PAPER-P1-HNC-SKELETON.md` — esqueleto operativo de P1 (secciones, figs, tablas)
- `LINEAS CONSOLIDADAS DE TRABAJO/ADDENDUM-PRUNING-2026-04-20.md` — decisiones de carriles paralelos
- `LINEAS DE INVESTIGACIÓN/LINEA A/FASE 5/stats_helpers.py` — rigor estadístico compartido
- `LINEAS DE INVESTIGACIÓN/LINEA A/FASE 5/exp_ctrl_A1_A4_untrained_and_null_tda.py` — recalibración L-102
- `LINEAS DE INVESTIGACIÓN/LINEA A/FASE 5/exp_P1_transformer_control.py` — **NUEVO** baseline matched para P1

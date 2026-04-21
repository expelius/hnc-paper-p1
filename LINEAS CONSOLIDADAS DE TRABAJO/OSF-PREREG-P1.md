# A7 · OSF PRE-REGISTRATION — Paper P1 (HNC θ→60°)

> **Estado:** BORRADOR listo para copiar a OSF. 🔴 BLOQUEANTE submit arXiv.
> **Fecha de redacción:** 2026-04-20
> **Subir a OSF ANTES de enviar P1 a arXiv.** El timestamp OSF es la evidencia pública de pre-registro.

---

## 0. Título

*Hierarchical Neural Collapse: An Intrinsic 60° Angular Signature in Multi-Scale Byte-Level Language Models*

**Autores:**

1. **Juan David Zuluaga-Monroy, MD** — Interventional Cardiologist, Independent Researcher.
   ORCID: [0000-0002-9865-471X](https://orcid.org/0000-0002-9865-471X)
2. **Diego Fernando Zuluaga-Monroy** — Electronic Engineer, Independent Researcher.
   ORCID: [0009-0007-8935-169X](https://orcid.org/0009-0007-8935-169X)

**Afiliación común:** Independent Researchers (Colombia).
**Contacto (corresponsal):** Juan David Zuluaga-Monroy — juanda.zuluaga@urosario.edu.co.

## 1. Hipótesis principal (claim C1 del paper)

En modelos byte-level con arquitectura HNC (Hierarchical Neural Collapse) de K=3 escalas acopladas top-down, el ángulo promedio inter-token $\theta$ del estado oculto converge a $60° \pm 5°$ durante el entrenamiento, produciendo una firma IVM (Isotropic Vector Matrix) estable bajo re-entrenamiento con distintas semillas.

## 2. Predicciones pre-registradas

### 2.1 Claim C1 — HNC θ→60° (sección §4 del paper)

- **P-C1-1.** Tras $N_{\text{epochs}}=3$ sobre WikiText-103 byte-level (0.5 MB), con $d\in\{64,128,256\}$ y $K=3$, $n\geq 5$ seeds: $\mathbb{E}[\theta_{\text{HNC}}] \in [55°, 65°]$ con CI95 bootstrap ⊂ [50°, 70°].
- **P-C1-2.** Cohen's $d$ entre $\theta_{\text{HNC}}$ y $\theta_{\text{random-init}}$ ≥ 1.5 (large effect).
- **P-C1-3.** La firma θ→60° es **estable** al cambiar $d$: $|\mathbb{E}[\theta|d=64] - \mathbb{E}[\theta|d=256]| \leq 7°$.

### 2.2 Claim C2 — gradiente σ₀ > σ₁ > σ₂ (§4.2)

- **P-C2-1.** En modelos entrenados, $\sigma_0 > \sigma_1 > \sigma_2$ con Cohen's $d$(σ₀, σ₂) ≥ 1.0.
- **P-C2-2.** Diferencia entrenado vs. untrained: $\Delta\text{AR}_{\text{trained-init}} \geq 0.5$ (training amplifica el gradiente; ver CTRL-A1A4).
- **P-C2-3.** El gradiente arquitectónico (at init) NO basta para explicar el rango observado en entrenados: $|\text{AR}_{\text{trained}}| \geq 1.5 \times |\text{AR}_{\text{init}}|$.

### 2.3 Claim C3 — identidad causal de escala (§4.2b, test A2)

- **P-A2.** Para modelos $\{$healthy, dual_reg_d8$\}$ × seeds $\{42, 123, 456\}$:
  - $\text{AR}_{\text{null}}^{\text{perm}} \leq 0.5 \times \text{AR}_{\text{orig}}$ (permutation null con N_PERM=500)
  - p-value permutacional $< 0.05$
  - Tras BH-FDR sobre las 6 pruebas, todas significativas a $q=0.05$.
  - **Secundario:** swap funcional BPB con scales invertidos degrada BPB ≥ 1.2×.

### 2.4 Claim C4 — disociación s₂ vs s₀ bajo shuffle (§4.5, test A3)

Sobre corpora INFRA-02 (shuffle con SEED=20260420, tres granularidades):

- **P-A3-1.** $|\Delta\sigma(s_2)|_{\text{paragraph−val}}$ significativo, Cohen's $d\geq 0.8$.
- **P-A3-2.** $|\Delta H_1(s_2)|_{\text{paragraph−val}}$ significativo, Cohen's $d\geq 0.8$.
- **P-A3-3.** $|\Delta\text{PR}(s_2)|_{\text{paragraph−val}}$ significativo, Cohen's $d\geq 0.8$.
- **P-A3-4.** $|\Delta\sigma(s_0)|_{\text{paragraph−val}}$ NO significativo, Cohen's $|d| < 0.3$.
- **P-A3-5.** Gradiente en s₂: $|\Delta(\text{paragraph})| > |\Delta(\text{sentence})| > |\Delta(\text{word})|$.
- **P-A3-6.** Gradiente inverso en s₀: $|\Delta(\text{word})| > |\Delta(\text{sentence})| > |\Delta(\text{paragraph})|$.

Todas con BH-FDR sobre la familia de 27 tests, $q=0.05$.

### 2.5 Claim C5 — sesgo inductivo específico de HNC (§4.6, tests P1-TRCTRL + A5)

- **P-C5-1 (P1-TRCTRL).** Transformer L=2 matched-param a $d\in\{4,128\}$ → $\text{AR} \leq 1.3$ (vs. $\text{AR}_{\text{HNC}} \geq 2.0$).
- **P-C5-2 (P1-TRCTRL).** LSTM L=2 matched-param a $d\in\{4,128\}$ → $\text{AR} \leq 1.3$.
- **P-A5-1.** Transformer + top-down coupling + confidence gating (Tensegrity-Transformer) sí produce $\text{AR} \geq 1.8$ ("doble-ganadora": la firma IVM depende del acoplamiento jerárquico, no del backbone).
- **P-A5-2.** Si Tensegrity-Transformer NO reproduce AR≥1.8, la firma es específica del recurrent-state HNC y P1 Discussion debe reajustarse.

## 3. Diseño experimental

### 3.1 Datos

- **Entrenamiento:** `data/wikitext103_val.txt`, primeros 500 KB (byte-level).
- **Evaluación adversarial (A3):** `data/wikitext103_val_shuf_paragraph.txt`, `..._shuf_sentence.txt`, `..._shuf_word.txt`. Generados con `prep_shuffle_corpus.py`, SEED=20260420, 1,134,789 bytes cada uno.
- **Test:** `data/wikitext103_test.txt` (no usado para selección de hiperparámetros).

### 3.2 Arquitectura HNC (parámetros del preregistro)

| Parámetro | Valor |
|---|---|
| D_MODEL | 128 (reportado también 64, 256 en sweep) |
| K (num_scales) | 3 |
| NUM_LAYERS | 1 |
| SEQ_LEN | 128 |
| BATCH_SIZE | 8 |
| NUM_EPOCHS | 3 |
| LR | 5e-4 (AdamW, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Grad clip | 1.0 |
| Activation vocab | byte-level (256) |

### 3.3 Configuraciones de medición

| Métrica | Definición operativa | Script |
|---|---|---|
| $\theta$ / $\sigma_k$ | ángulo medio inter-token en `state_trajs[k][:, MID_POS, :]` sobre $N_{\text{seq}}=30$ secuencias | `compute_sigma()` |
| PR | participation ratio sobre covarianza de los N=30 vectores | `compute_pr_cov()` |
| $H_1$ | persistent homology (ripser, maxdim=1) sobre PCA(12) tras L2-normalize, N_SUBSAMPLE=100 | `compute_h1()` |
| $H_1^{\text{null}}$ | mismo pipeline sobre cloud Gaussiano con covarianza emparejada | `exp_ctrl_A1_A4` |
| AR | $\sigma_{\max}/\sigma_{\min}$ sobre las 3 escalas | derivada |
| BPB | cross-entropy/log(2) sobre batches de eval | `evaluate_bpb()` |

### 3.4 Seeds

- `SEEDS = [42, 123, 456]` (mínimo para n≥3).
- Para claim C1 y P1 main result: **n≥5** (seeds 42, 123, 456, 789, 20260420).
- Colab P0 corrida como referencia scaling d=64/128/256 (~12 h, en vuelo).

### 3.5 Pipeline estadístico

- Bootstrap percentil CI95 con $n_{\text{boot}}=1000$ (`bootstrap_ci`).
- Efectos: Cohen's $d$ + Hedges' $g$ (`stats_helpers.cohens_d`, `hedges_g`).
- Familia de tests: Benjamini–Hochberg FDR, $q=0.05$ (`bh_fdr`).
- Paired bootstrap para diferencias cross-corpus (`paired_bootstrap_diff`).

## 4. Criterios de refutación (condición de falsación explícita)

El paper P1 NO sale a arXiv si cualquiera de las siguientes ocurre tras ejecutar los 5 scripts pre-registrados (ver §6):

- **F1.** $|\mathbb{E}[\theta_{\text{HNC}}] - 60°| > 10°$ con CI95 disjunto de [50°, 70°].
- **F2.** $\text{AR}_{\text{null}}^{\text{perm}} \geq \text{AR}_{\text{orig}}$ en cualquier condición del test A2 (BH-FDR no retiene).
- **F3.** Transformer L=2 matched-param produce AR ≥ 2.0 (P-C5-1 falla → la firma NO es específica del acoplamiento HNC; reescritura major requerida).
- **F4.** A3 no distingue paragraph-shuffle de word-shuffle en s₂ (Cohen's d<0.3 en ambos) → el claim de disociación jerárquica semántica colapsa.

Si F3 se cumple, P1 pasa a **análisis alternativo** (o se retira) antes de submit.

## 5. Desviaciones permitidas del protocolo

Se permite:
- Aumentar $n_{\text{seeds}}$ si el CI bootstrap no cumple el ancho pre-declarado.
- Cambiar `SUBSET_MB` si el tiempo de compute se vuelve prohibitivo — reportando la desviación.
- Añadir tests secundarios a §4.5/§4.6 *después de* haber corrido los primarios.

NO se permite:
- Cambiar criterios de refutación después de ver resultados.
- Dropear seeds "outliers".
- Añadir o quitar condiciones experimentales al test A2 / A3 / P1-TRCTRL después de ejecutar.

## 6. Scripts y artefactos depositados

Repositorio público (GitHub / Zenodo DOI al freeze):

```
LINEAS DE INVESTIGACIÓN/LINEA A/FASE 5/
├── stats_helpers.py                           # INFRA-01
├── prep_shuffle_corpus.py                     # INFRA-03 (SEED=20260420)
├── exp_ctrl_A1_A4_untrained_and_null_tda.py   # CTRL-A1A4 — DONE (results/)
├── exp_d089_sigma_reg.py                      # D-089 — DONE
├── exp_d094_extended_independence.py          # D-094 — DONE
├── exp_d095_bicondition.py                    # D-095 — RUN
├── exp_A2_causal_permutation.py               # A2 — QUEUED, smoke ✓
├── exp_A3_shuffled_eval.py                    # A3 — QUEUED, smoke ✓
├── exp_P1_transformer_control.py              # P1-TRCTRL — QUEUED, smoke ✓
data/
├── wikitext103_val.txt
├── wikitext103_val_shuf_paragraph.txt         # INFRA-02
├── wikitext103_val_shuf_sentence.txt
└── wikitext103_val_shuf_word.txt
LINEAS CONSOLIDADAS DE TRABAJO/
├── EXPERIMENTS-LOG.md                         # bitácora trazabilidad
├── PAPERS-MAP-2026-04-20.md                   # eje epistémico
├── PAPER-P1-HNC-SKELETON.md                   # esqueleto P1 con §4.2b/§4.5/§4.6
├── ADDENDUM-PRUNING-2026-04-20.md             # carriles A/B/C
└── OSF-PREREG-P1.md                           # ESTE documento
```

## 7. Timestamps / historial

| Fecha | Hito |
|---|---|
| 2026-03-16 | Segunda ola R-004 + F-series (D-057, D-064, D-068) |
| 2026-04-14 | `STRATEGIC-REVIEW-2026-04-14.md` |
| 2026-04-16 | `PREREGISTRO_R004n_2026-04-16.md` (precursor técnico) |
| 2026-04-16 | `D-065-LAGRANGIANO-SADE-2026-04-16.md` |
| 2026-04-20 | Critical review, segmentación ALPHA→P1..P9 |
| 2026-04-20 | CTRL-A1A4 ejecutado (5 seeds), corpora shuffle generados |
| 2026-04-20 | A2, A3, P1-TRCTRL código listo + smoke tests pasados |
| 2026-04-20 | **Este documento** (OSF pre-reg borrador) |
| [TBD]      | D-095 termina → ejecutar A2 + A3 + P1-TRCTRL |
| [TBD]      | Colab P0 termina → scaling d table |
| [TBD]      | **Freeze + subir a OSF** → DOI |
| [TBD]      | Submit arXiv P1 |

## 8. Conflictos de interés / financiamiento

- **Financiamiento:** ninguno. Investigación realizada de forma independiente y auto-financiada por los autores.
- **Conflictos de interés:** los autores declaran no tener conflictos de interés.
- **Afiliaciones comerciales:** ninguno de los autores tiene afiliación comercial o contractual con entidades que puedan beneficiarse de los resultados.
- **Licencia código:** Apache License 2.0 (ver [LICENSE](../LICENSE)).
- **Licencia documentos:** CC BY 4.0 (ver [LICENSE-docs.md](../LICENSE-docs.md)).

## 9. Checklist pre-submit OSF

- [ ] Completar autoría (§0) y contacto.
- [ ] Completar conflictos/financiamiento (§8).
- [ ] Verificar que `EXPERIMENTS-LOG.md` tiene filas OK para: CTRL-A1A4, D-089, D-094, D-095, A2, A3, P1-TRCTRL.
- [ ] Freeze de repositorio (tag git `p1-preregister-v1`).
- [ ] Zenodo snapshot del tag → DOI.
- [ ] Subir este documento + manifiesto de archivos §6 a proyecto OSF nuevo "HNC-P1-PREREG".
- [ ] Registrar timestamp OSF.
- [ ] Añadir DOI OSF a `PAPER-P1-HNC-SKELETON.md` §Reproducibility.
- [ ] Entonces (y sólo entonces) enviar a arXiv.

---

## Instrucciones para el autor (cómo ejecutar A7)

1. Completar autoría y financiamiento.
2. `git tag p1-preregister-v1` en el estado actual del repo.
3. Push del tag a GitHub.
4. Zenodo → release → obtener DOI.
5. Crear proyecto OSF nuevo: *"HNC — Hierarchical Neural Collapse P1 pre-registration"*.
6. Subir este documento como *Pre-Registration* en OSF (formato: Open-Ended Registration).
7. Enlazar DOI Zenodo como *Materials*.
8. Registrar (timestamp).
9. Copiar URL OSF + DOI a [PAPER-P1-HNC-SKELETON.md](PAPER-P1-HNC-SKELETON.md) §Reproducibility.
10. Actualizar [EXPERIMENTS-LOG.md](EXPERIMENTS-LOG.md) §3.4 fila A7 → estado `OK` con la URL.

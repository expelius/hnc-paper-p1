# EXPERIMENTS LOG — Bitácora central append-only

**Creada:** 2026-04-20
**Propósito:** única fuente de verdad que mapea `D-###` (roadmap temporal) ↔ `P#.§X` (claim epistémico en paper) ↔ script ↔ resultado.

## Reglas de uso

1. **Append-only.** Nunca borrar filas. Si un experimento se invalida, añadir una fila nueva con estado `SUPERSEDED by D-###` y justificación 1-línea.
2. **Un D-ID por experimento.** Si es re-run con misma config pero más seeds, usar el mismo D-ID con sufijo `b`/`c` (`D-095b`).
3. **Dos ejes obligatorios por fila:**
   - **Roadmap:** D-ID + fecha + carril (A/B/C del ADDENDUM §5 / Colab).
   - **Paper:** qué paper `P#` y qué sección `§X` alimenta (primary + secondary).
4. **Resultado en una línea** — detalles van al artefacto (`results_*.json` / figura / sección del paper).
5. **Pre-registro:** si el experimento tiene predicción pre-registrada, citar el ID (`P-A2`, `P-A3`, `P-D095`…). Si no, dejar `ad-hoc`.

## Plantilla de header para scripts NUEVOS

Pegar como primer bloque después de shebang/encoding, antes del docstring:

```python
# ═══════════════════════════════════════════════════════════════
# TRACE
#   D-ID     : D-###
#   FEEDS    : P#.§X (primary), P#.§Y (secondary)     # papers alimentados
#   PRE-REG  : P-XXX (ver PAPER-P#-SKELETON §Z) | ad-hoc
#   CARRIL   : A (local CPU) | B (Carril B paper-ready) | C (análisis) | Colab
#   SEEDS    : n=#
#   DATE     : YYYY-MM-DD
#   DEPENDS  : D-### (si reutiliza ckpts/datos)
#   LOG      : EXPERIMENTS-LOG.md#D-###
# ═══════════════════════════════════════════════════════════════
```

Y **cada nuevo experimento añade una fila en §3 abajo antes de ejecutar** (pre-registro mínimo).

---

## 1. Convenciones

- **Estado:** `RUN` (corriendo) · `OK` (terminó, resultado registrado) · `FAIL` (no convergió / bug) · `SUPERSEDED` (reemplazado) · `QUEUED` (pre-registrado, no ejecutado).
- **Carril:**
  - `A` = CPU local, experimentos del ADDENDUM §5 carril A.
  - `B` = paper-ready scripts (esqueletos P1…P9).
  - `C` = re-análisis estadístico (stats_helpers, FDR, re-CI sobre experimentos viejos).
  - `Colab` = GPU, scaling / R-004n / D-sweep.
- **Feeds** usa notación `P#.§X`: `P1.§4.2b` = Paper 1 sección 4.2b del esqueleto.

---

## 2. Artefactos de referencia (no-experimentos, pero citables)

| ID | Artefacto | Ubicación | Rol |
|---|---|---|---|
| INFRA-01 | `stats_helpers.py` | `LINEAS DE INVESTIGACIÓN/LINEA A/FASE 5/` | bootstrap CI + Cohen's d + BH-FDR para TODO experimento nuevo |
| INFRA-02 | Corpus shuffled paragraph/sentence/word | `data/wikitext103_val_shuf_*.txt` | insumo de A3 (P1.§4.5) |
| INFRA-03 | `prep_shuffle_corpus.py` | FASE 5/ | generador de INFRA-02, SEED=20260420 |
| DOC-01 | `PAPERS-MAP-2026-04-20.md` | LINEAS CONSOLIDADAS/ | eje epistémico (9 papers) |
| DOC-02 | `PAPER-P1-HNC-SKELETON.md` | LINEAS CONSOLIDADAS/ | predicciones P-A2, P-A3, P-A5 + §4.2b, §4.5, §4.6 |
| DOC-03 | `ADDENDUM-PRUNING-2026-04-20.md` | LINEAS CONSOLIDADAS/ | eje temporal (carriles A/B/C) |
| DOC-04 | `ROADMAP-GENERAL.md` | LINEAS CONSOLIDADAS/ | eje temporal macro |

---

## 3. Bitácora (append-only)

### 3.1 Retroactivo — experimentos previos relevantes para papers actuales

| D-ID | Fecha | Script | Carril | Seeds | Feeds | Pre-reg | Estado | Resultado 1-línea |
|---|---|---|---|---|---|---|---|---|
| D-057 | 2026-03 | `exp_d057_h2phase.py` | A | n=? | P1.§4 (σ expansion), P3.§1 (motivación) | ad-hoc | OK | σ se expande a ~85° antes de que H₁ significativo emerja |
| D-064 | 2026-03 | `exp_d064_fractal_folding.py` | A | n=? | P2.§1, P4.§1 | ad-hoc | OK | Fractal folding signature |
| D-067 | 2026-03 | `exp_d067_bridge1.py` / `exp_d067_bridge_gpt2.py` | A | n=? | P5.§1 (input ratchet) | ad-hoc | OK | Bridge hypothesis bases |
| D-068 | 2026-03 | `exp_d068_theta_dynamics.py` | A | n=? | P1.§3 (θ→60°) | ad-hoc | OK | θ dynamics trayectorias |
| D-070 | 2026-03 | `exp_d070_berry_phase.py` | A | n=? | P4.§2 | ad-hoc | OK | Berry phase signature |
| D-085 | 2026-03 | `exp_d085_topo2_sweep.py` | A | n=? | P2.§1 (σ↔H₁ inverse) | ad-hoc | OK | Sweep topológico base de P2 |
| D-089 | 2026-04 | `exp_d089_sigma_reg.py` | A | n=3 | P3.§1 (Direction 1 bicondition) | ad-hoc (pre-P3) | OK | σ-REG: σ₂=84°+PR₂=3.1+H₁₂=3 → σ HIGH no basta para H₁ alto |
| D-094 | 2026-04 | `exp_d094_extended_independence.py` | A | n=3 | P3.§1 (Direction 2 bicondition) | ad-hoc (pre-P3) | OK | σ-SUPPRESS: σ₂=13°+PR₂=10→H₁₂=4.7 → PR HIGH tampoco basta sola |
| CTRL-A1A4 | 2026-04-20 | `exp_ctrl_A1_A4_untrained_and_null_tda.py` | A | n=5 | P1.§4.3 (untrained control) | P-A1, P-A4 | OK | Gradiente σ₀>σ₁>σ₂ existe at init (sesgo arq); H₁_excess<0 at init; training amplifica jerarquía |
| INFRA-02/03 | 2026-04-20 | `prep_shuffle_corpus.py` | A | SEED=20260420 | P1.§4.5 (A3 data) | N/A | OK | 3 corpora shuffled 1,134,789 B c/u |

### 3.2 En vuelo (carril A local + Colab)

| D-ID | Fecha | Script | Carril | Seeds | Feeds | Pre-reg | Estado | Resultado 1-línea |
|---|---|---|---|---|---|---|---|---|
| D-095 | 2026-04-20 | `exp_d095_bicondition.py` | A (local CPU) | n=3 × 5 cond = 15 models | P3.§Results (primary), P1.§4.2b checkpoints | P-D095 | **OK — REINTERPRETA P3** | P1✅ (dual: σ=82° PR=10.3 H₁=13.3, por encima de healthy 11.7); P2❌ (pr_reg σ=50° PR=16.6 H₁=11.3, **σ NO es necesaria si PR sube**); P3❌ (sigma_reg σ=84° PR=3.3 H₁=5.0, σ NO es suficiente). Seed-level revela **dos atractores**: (A) σ≈86°+PR≈9→H₁≈9, (B) σ≈33°+PR≈20→H₁≈12. Conclusión causal: **PR es el driver fundamental de H₁; σ es consecuencia correlacionada**. Bicondición estricta σ∧PR → PR-monótona con σ como correlato. Ver §6 abajo. |
| P0-SCALE | 2026-04-20 | Colab UNIFIED notebook | Colab | n=5 | P1.§4.4 (scaling d), P4.§R-004n | P-P0 | RUN | d=64/128/256 × F10 × 5 seeds + R-004n + D-sweep, ~12 h |

### 3.3 Pre-registrados / QUEUED (aún no ejecutados)

| D-ID | Fecha plan | Script | Carril | Seeds plan | Feeds | Pre-reg | Estado | Trigger |
|---|---|---|---|---|---|---|---|---|
| A2 | post-D-095 | `exp_A2_causal_permutation.py` | A | n=3 × 2 cond (healthy + dual_reg_d8) | P1.§4.2b (primary) | **P-A2** (PAPER-P1 §4.2b) | QUEUED (code ready, smoke ✓ 2026-04-20) | Permutation null AR vía shuffle de scale-labels por ejemplo + functional swap BPB secundario. Predicción: AR_null ≤ 0.5×AR_orig, p<0.05, BH-FDR pasa. Reusa clases de `exp_d095_bicondition.py` vía import. Training interno (~20 min), no depende de ckpts externos. |
| A3 | post-D-095 | `exp_A3_shuffled_eval.py` | A | n=3 × 1 cond × 4 corpora = 12 evals | P1.§4.5 (primary) | **P-A3** (PAPER-P1 §4.5) | QUEUED (code ready, smoke ✓ 2026-04-20) | Entrena HNC-healthy sobre val, evalúa sobre {val, shuf_paragraph, shuf_sentence, shuf_word} (INFRA-02). 27 tests (3 scales × 3 metrics × 3 shufflings) + BH-FDR q=0.05. 6 predicciones P-A3-{1..6}. Reusa clases D-095. ~20 min CPU. |
| P1-TRCTRL | post-Colab | `exp_P1_transformer_control.py` | B | n=5 × d∈{4,128} | P1.§4.6 (primary) | P-P1-C3 (AR ≈ 1.0 no-HNC) | QUEUED | Smoke-test OK; esperar CPU libre |
| A5 | post-P1-TRCTRL | `exp_A5_tensegrity_transformer.py` | B | n=5 × d∈{4,128} | P1.§4.6 (primary) | **P-A5** (PAPER-P1 §4.6) | QUEUED | Transformer + top-down coupling + confidence gating |
| P9-HNET | post-P1 | `exp_P9_hnet_universality.py` | B | pendiente | P9 (primary), P1.§5 future-work | P-P9 a definir | QUEUED | Clonar H-Net repo + ckpt → aplicar protocolo P1 |
| D-096 | 2026-04-20 | `exp_d096_pr_monotone.py` | A | n=8 × 4 cond = 32 | P3.§Results (rewrite primary), P1.§4.2b (AR secondary) | **P-D096** (1..4, ver script docstring) | QUEUED | Replica pr_reg con n=8 + λ_PR=0.5 + anti-sigma_forcing + **AR cross-scale measurement**. Discrimina H_PR-MONOTONE vs H_TWO-CHANNEL. Si AR colapsa con σ→30° entonces bicondición correcta pero en ejes ortogonales (σ→AR, PR→H₁). ~105 min CPU. |

### 3.4 Tareas de gobernanza (no-código pero bloqueantes)

| ID | Fecha plan | Acción | Feeds | Estado | Trigger |
|---|---|---|---|---|---|
| A7 | pre-submit P1 | OSF pre-registration + código público | P1 submission | � BORRADOR listo ([OSF-PREREG-P1.md](OSF-PREREG-P1.md), 2026-04-20) | Falta: autoría, financiamiento, git tag, Zenodo DOI, subir a OSF. Ver §9 checklist en el doc. |
| C-1 | continuo | Re-análisis Cal-* con `stats_helpers` | P6 | QUEUED | carril C cuando CPU libre |

---

## 4. Cómo actualizar la bitácora

### Al iniciar un experimento nuevo:

1. Elegir próximo `D-ID` libre (o letra `A#`/`P#-XXX` si es A2-like o paper-specific).
2. **Antes de lanzar**, añadir fila en §3.3 (QUEUED) con feeds y pre-reg.
3. Añadir TRACE header al script (plantilla arriba).
4. Mover fila a §3.2 (RUN) al ejecutar.
5. Mover fila a §3.1 (OK/FAIL) al terminar, con resultado 1-línea.

### Al cerrar un paper:

1. En `PAPER-P#-SKELETON.md` §Results, citar los D-IDs que sustentan cada claim.
2. Marcar en esta bitácora `feeds` con `P#.§X [FROZEN]` cuando el paper se envíe.

### Cuando un experimento se invalida:

- **No borrar la fila.** Añadir fila nueva con `SUPERSEDED by D-###` y la razón. El experimento original queda como evidencia de que se consideró.

---

## 5. Relación con los demás documentos

```
ROADMAP-GENERAL.md          ←── orden macro de fases
       │
ADDENDUM-PRUNING §5         ←── carriles A/B/C paralelos (qué hago esta semana)
       │
EXPERIMENTS-LOG.md (este)   ←── bitácora append-only D-### ↔ P#.§X
       │
       ├──→ exp_D###_*.py   (código + TRACE header)
       │
       └──→ PAPERS-MAP-2026-04-20.md  ←── 9 papers, madurez, bloqueadores
                 │
                 └──→ PAPER-P#-SKELETON.md  ←── claims + predicciones pre-registradas
```

**Regla de oro:** cualquier afirmación en un paper debe ser trazable a una fila de §3 de este log. Si no hay fila, no hay claim.

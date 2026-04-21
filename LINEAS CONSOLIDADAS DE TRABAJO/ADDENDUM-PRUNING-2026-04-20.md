# ADDENDUM-PRUNING-2026-04-20.md

> Addendum crítico a STRATEGIC-REVIEW-2026-04-14.md
> Post-review honesta 2026-04-20 + hallazgo control A1+A4
> **Objetivo**: recortar scope, blindar rigor, enfocar ALPHA.
>
> **Trazabilidad por experimento:** [EXPERIMENTS-LOG.md](EXPERIMENTS-LOG.md) (bitácora append-only D-### ↔ P#.§X). Este ADDENDUM define *carriles*; el log define *evidencia*.

---

## 0. Contexto

El proyecto tiene ~99 experimentos (D-001..D-099), ~104 "L-lessons",
7 líneas consolidadas (§0..§6), ~41 "Parts" en Strategic Review. La
relación ambición/evidencia se ha desalineado. Este documento establece:

- Qué se **congela** (MOONSHOT, post-ALPHA).
- Qué se **continúa** (núcleo ALPHA).
- Cómo se **recalibra L-102** a la luz del hallazgo A1+A4.
- Qué **rigor mínimo** aplicamos desde hoy.

---

## 1. Recalibración de L-102 (obligatoria)

### Evidencia nueva (A1+A4, 2026-04-20)
HNC sin entrenar (random init, 5 seeds):

| scale | σ (°, CI95)         | PR (CI95)            | H1_excess vs Gauss  |
|-------|---------------------|----------------------|---------------------|
| s0    | 54.7 [52.5, 57.0]   | 10.69 [10.34, 11.02] | −2.4 [−4.0, −0.9] ★ |
| s1    | 42.6 [40.2, 45.1]   | 10.42 [9.76, 10.87]  | −2.1 [−3.6, 0.0]  ★ |
| s2    | 35.5 [33.3, 38.0]   | 9.43 [8.38, 10.25]   | −3.0 [−4.3, −1.8] ★ |

(★ = FDR-significativo)

### Implicaciones
1. **El gradiente σ₀ > σ₁ > σ₂ NO es emergente del training**. Es sesgo
   inductivo del stack `embedding + positional + EMA bottleneck`. Existe
   al init.
2. **PR es ~uniforme a ~10 en todas las escalas al init**. PR↑ en s₂
   observado en modelos entrenados (cuando se observa) sí es efecto de
   training — el PR se *desarma* del valor init.
3. **H1_excess negativo al init** significa que las trayectorias random
   son más *isotrópicas* que un Gaussian matched. Cualquier H1_excess > 0
   en modelos entrenados (p.ej. F5, D-094) es un efecto **real y fuerte**
   porque vence un baseline negativo.

### Nueva formulación de L-102 (reemplaza versión anterior)
> **L-102 (revisada)**: Entre las firmas angulares-dimensionales de un
> modelo jerárquico, el gradiente σ es en gran medida sesgo inductivo de
> la arquitectura. Lo que el training *produce* es (a) la disociación
> entre σ y PR (PR ya no uniforme), y (b) el incremento de H1 por encima
> del baseline Gaussian matched. La "Bicondición" se redefine como:
> **H1_excess > 0 requiere simultáneamente σ comprimido Y PR elevado, los
> dos más allá de lo que la arquitectura sola produce.**

Esto es más fuerte y más defensible que la versión original.

### Acción
- Actualizar el manuscrito ALPHA con tabla de init vs trained.
- L-103, L-104 (universalidad) requieren replicar el control A1+A4 en
  cada arquitectura que se compare; no aceptar afirmación de
  universalidad sin ese control.

---

## 2. CARRILES PARALELOS — sin freeze, con prioridades (revisado 2026-04-20)

Decisión del usuario: mantener todo activo, trabajar en paralelo. El
freeze original se reemplaza por tres carriles con prioridad explícita.
La regla no es "no tocar" sino "no canibalizar tiempo del carril A".

### Carril A — ALPHA-crítico (prioridad máxima)
Cualquier bloque de tiempo largo se asigna aquí primero.

| Tarea                                 | Estado   | Tiempo   |
|---------------------------------------|----------|----------|
| Colab P0 scaling d=64/128/256         | corriendo| ~12 h GPU|
| R-004n GRU/LSTM/Mamba matched         | corriendo| ~4 h GPU |
| D-095 bicondition (5 conds × 3 seeds) | corriendo| ~40 min  |
| A1+A4 (untrained + null TDA)          | ✅       | —        |
| A1+A4 sobre modelo TRAINED            | pendiente| ~1 h CPU |
| A2 permutación causal escalas         | pendiente| ~30 min  |
| A3 eval sobre 3 shuffled corpora      | pendiente| ~1 h CPU |
| Esqueleto ALPHA (8 pp, C1..C6)        | pendiente| ~2 h     |
| OSF pre-registration                  | pendiente| ~1 h     |

### Carril B — Extensión arquitectónica (ejecución asíncrona)
Se lanza cada D-09x *después* de que Colab confirme el resultado ALPHA
correspondiente. No se apilan ni compiten por CPU mientras Colab corre.

| D-09x     | Inspiración     | Lanza cuando                                  |
|-----------|-----------------|-----------------------------------------------|
| D-096 HDLM loss        | HDLM NeurIPS25 | Colab confirma σ gradient a d=256           |
| D-097 Dynamic α        | H-Net          | Colab confirma L-102 a d=256                 |
| D-098 Geom smoothing   | H-Net          | D-095 + A2 consolidados                      |
| D-099 K=4              | Arquitectura   | C4 (ordinal H1) confirmado a d=256           |

Regla: solo **uno** de B corriendo a la vez en CPU local, solapado con
análisis de Colab (sin conflicto de recursos).

### Carril C — MOONSHOT (asíncrono, low-cost, regla de rigor)
Activo pero con regla obligatoria: cada metáfora física debe acompañarse
de (a) cantidad calculable, (b) predicción numérica, (c) test.
Si no cumple → queda en appendix teórico de OMEGA (tesis), no en paper.

| Área                           | Acción asíncrona permitida                   |
|--------------------------------|----------------------------------------------|
| §4 Calorimetría / NESS         | Re-análisis Cal-* con stats_helpers (CI/FDR) |
| §5 Analogía proteica           | H-ANF Colab (cuando GPU libre)               |
| §6 Neurociencia computacional  | Documentar 13 predicciones con falsadores    |
| SADE-net                       | Sólo teoría, sin entrenamiento               |
| Berry phase, Bi₂Se₃            | Derivar cantidad calculable o archivar       |

**Regla operativa**: los 3 carriles avanzan; el único blindaje es que
C1..C6 de ALPHA no pueden quedar sin seeds/CIs/baselines por trabajar
en B o C. Auditoría semanal del ratio de tiempo por carril.

---

## 3. NÚCLEO ALPHA — qué se ejecuta y publica

### 3.1 Claims que entran a ALPHA (numerados C1–C6)

| # | Claim                                                                       | Evidencia actual                     | Falsador declarado                   |
|---|-----------------------------------------------------------------------------|--------------------------------------|--------------------------------------|
| C1| Arquitectura determina gradiente σ (sesgo inductivo)                         | A1+A4 control                        | σ gradient ausente en ≥3 seeds init  |
| C2| Training amplifica la separación σ-PR y produce H1_excess > 0                | D-094 + A1 comparación               | H1_excess ≤ 0 en modelos entrenados  |
| C3| Intervención σ↑ sola (D-089) no basta para H1: PR↑ es necesario              | D-089, D-094                         | PR-solo recupera H1 sin σ            |
| C4| Topology ordinal H1(s₀)<H1(s₁)<H1(s₂) post-training con FDR                  | D-094, replicar a d=256              | Orden se invierte a d=256            |
| C5| F10 ΔAR=+51.6° a ΔBPB≈0 bajo presión de empaquetamiento                      | F10 replicado a d∈{128, 256}, 5 seeds| ΔAR < 20° a d=256                    |
| C6| Efecto presente también en LSTM/Mamba matched (universalidad arquitectónica) | R-004n Colab en curso                | Efecto ausente en ≥1 arq. matched    |

**Scope mínimo del paper: C1, C2, C3, C5** (los más defendibles).
C4 y C6 entran solo si resultados de Colab scaling lo soportan.

### 3.2 Rigor mínimo obligatorio (aplica desde hoy)
1. **n ≥ 5 seeds** para cualquier tabla del paper (ideal 10).
2. **Bootstrap CI 95%** en toda media (1000 resamples, `stats_helpers.py`).
3. **Cohen's d** (o Hedges' g si n<10) en toda comparación binaria.
4. **BH-FDR** (q=0.05) sobre familia C1..C6 completa.
5. **Power calculada ex-ante**: `approx_power_two_sample(d, n)`.
6. **Baseline control A1+A4** reportado junto a cada tabla de métricas.

### 3.3 Controles externos obligatorios en ALPHA
- Transformer vanilla matched params: σ/PR/H1/AR reportados.
- Mamba/S4 matched params: ídem.
- HNC al init (A1): ídem.
- Gaussian null (A4) por celda con H1.

### 3.4 Orden de ejecución inmediato

| # | Tarea                                              | Tiempo   | Bloquea     |
|---|----------------------------------------------------|----------|-------------|
| 1 | Colab P0 d=64/128/256 scaling                      | ~12 h    | C4, C5, C6  |
| 2 | Colab R-004n GRU/LSTM/Mamba matched                | ~4 h     | C6          |
| 3 | D-095 bicondition 5 conds × 3 seeds (local CPU)    | en curso | C3          |
| 4 | A1+A4 control DONE                                 | ✅       | C1, C2      |
| 5 | A3 corpus paragraph-shuffle + eval                 | ~2 h     | C4 disoc.   |
| 6 | A2 permutación escalas en checkpoint trained       | ~30 min  | C4 causalidad|
| 7 | Expandir A1+A4 a condición TRAINED para C2         | ~1 h     | C2          |
| 8 | Esqueleto manuscrito ALPHA (markdown, 8 pp)        | ~2 h     | —           |

---

## 4. Decisiones congeladas

- **Sin HARKing**: cualquier L-lesson nueva requiere pre-registro
  datado antes del experimento. L-103/L-104 re-etiquetadas como
  "hipótesis" hasta evidencia con rigor §3.2.
- **Nomenclatura**: en externos (paper, tesis, charlas) usar C1..C6,
  F1..Fn, E1..En. La numeración D-xxx y §CCCLXV se queda en
  cuaderno de laboratorio.
- **Metáforas físicas**: Berry phase, Bi₂Se₃, Navier-Stokes semántico
  no entran a ALPHA. Se reservan para OMEGA (tesis) o papers
  posteriores, y *solo* si cada una lleva una cantidad calculable
  + predicción numérica + test.
- **OSF pre-registration**: antes de enviar ALPHA, subir código +
  checkpoints con DOI. Resuelve HARKing de una vez.

---

## 5. Propuestas creativas — estado + asignación operativa

| # | Propuesta                            | Estado                     | Script / Ventana ejecución |
|---|---------------------------------------|----------------------------|----------------------------|
| A1| Untrained baseline                    | ✅ DONE — ver §1           | `exp_ctrl_A1_A4_untrained_and_null_tda.py` |
| A2| Permutación causal de escalas         | 🟡 PROGRAMADO              | `exp_A2_causal_permutation.py` — ejecutar cuando D-095 termine (carga ckpts healthy/dual_reg_d8, swap s₀↔s₂ y s₁↔s₂, re-mide AR + H₁). **Sección §4.2b del P1.** |
| A3| Paragraph-shuffle corpus adversarial  | 🟡 PROGRAMADO              | Corpora DONE (`prep_shuffle_corpus.py`). Falta `exp_A3_shuffled_eval.py` sobre ckpts D-095. **Sección §4.5 del P1.** Predicción: Δσ₂, ΔH₁(s₂), ΔPR(s₂) significativos bajo paragraph-shuffle; Δs₀ ≈ 0 |
| A4| Null TDA (Gaussian null)              | ✅ DONE — ver §1           | integrado en A1+A4 |
| A5| Tensegrity Transformer                | 🟡 PROGRAMADO (no post)    | `exp_A5_tensegrity_transformer.py` — Transformer L=2 + top-down coupling + confidence gating. **Sección §4.6 del P1.** Carril B, después de `exp_P1_transformer_control.py`. Resultado publicable en ambas direcciones |
| A6| Dataset discurso-shuffle (= A3)       | Fusionado en A3            | — |
| A7| OSF + código público pre-ALPHA        | 🔴 BLOQUEANTE SUBMIT P1    | 1h de trabajo manual. **No se envía P1 a arXiv sin DOI OSF.** Subir: código (baseline_e, stats_helpers, exp_ctrl_A1_A4, exp_d089/094/095, exp_P1_transformer_control), corpora shuffled, predicciones pre-registradas §4.2b/§4.5/§4.6 |
| A8| Evaluar L-104 sobre H-Net open source | 🟡 PROMOVIDO A PAPER P9    | `exp_P9_hnet_universality.py`. Ver `PAPERS-MAP-2026-04-20.md` §P9. Post-P1 |

---

## 6. Métrica de salud del proyecto (revisión mensual)

| Métrica                                     | Umbral saludable | Ahora (2026-04-20) |
|---------------------------------------------|------------------|--------------------|
| Ratio experimentos-con-n≥5 / total          | ≥ 0.5            | <0.1 🔴            |
| Ratio claims con falsador declarado / total | ≥ 0.9            | parcial 🟡         |
| Tiempo desde último scaling run             | ≤ 30 días        | corriendo ahora 🟢 |
| # metáforas físicas sin isomorfismo formal  | ≤ 2              | ≥ 6 🔴             |
| # papers con draft estructural              | ≥ 1              | 0 🔴                |

**Acción correctiva inmediata**: reducir a ≤ 2 metáforas retenidas,
pasar ≥ 50% de D-xxx a n≥5 (re-ejecutando en Colab lo crítico),
escribir esqueleto ALPHA esta semana.

---

**Firma metodológica**: este addendum tiene fecha. Cualquier claim
posterior que contradiga §1 o §3.2 requiere modificación formal de
este documento, no sobrescribir silencioso.

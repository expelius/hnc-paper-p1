# Nota de desviación de protocolo — corrección de corpora A3

**Fecha:** 2026-04-21
**Estudio:** Paper P1 — HNC Alignment Rule
**Pre-registro original:** OSF [osf.io/r36ha](https://osf.io/r36ha/overview) — tag `v1.0.0-preregister`, commit `1300b878`, Zenodo DOI [10.5281/zenodo.19673375](https://doi.org/10.5281/zenodo.19673375), publicado 2026-04-20.
**Tipo de desviación:** corrección de artefactos de datos antes de la recolección. NO se modifican predicciones, hipótesis, ni criterios de análisis.

---

## 1. Descripción del problema

El script `prep_shuffle_corpus.py` incluido en el tag `v1.0.0-preregister` contenía un bug en la lógica de detección de párrafos: asumía que los párrafos estaban separados por líneas en blanco, pero el archivo `wikitext103_val.txt` usa un único `\n` entre párrafos. En consecuencia:

- `wikitext103_val_shuf_paragraph.txt` resultó **byte-idéntico** a `wikitext103_val.txt` (ningún párrafo fue permutado).
- `wikitext103_val_shuf_word.txt` tenía un tamaño ligeramente distinto por un bug secundario en el manejo de tokens de fin de línea.
- `wikitext103_val_shuf_sentence.txt` NO estaba afectado.

### Verificación (SHA256 de los archivos en el tag `v1.0.0-preregister`)

| Archivo | SHA256 (en tag) | Tamaño |
|---|---|---|
| `wikitext103_val.txt` | `5ed1a5b2a7fa2a0210b7091720743ff3252392ffff448dd1ba83795f332b403e` | 1 137 250 B |
| `wikitext103_val_shuf_paragraph.txt` | `5ed1a5b2...` ← **idéntico a val.txt (BUG)** | 1 137 250 B |
| `wikitext103_val_shuf_sentence.txt` | `5b69ad85db23d6bdbc488355dc27a391487f086ee872809f5a8476dafd3428c1` | 1 135 686 B |
| `wikitext103_val_shuf_word.txt` | `a51a0b36cfeb1d3eb5d9efa376842a83c5e78484630bbfa8eb6fe02d0e500a01` | 1 134 790 B |

## 2. Corrección

Se corrigió `prep_shuffle_corpus.py` (commit `6686966` en el repositorio principal, SEED=20260420 sin cambios) y se regeneraron los dos corpora afectados **antes** de ejecutar el experimento A3.

### SHA256 de los corpora efectivamente usados en el experimento A3 (tag `v1.0.1-data-correction`)

| Archivo | SHA256 (corregido) | Tamaño |
|---|---|---|
| `wikitext103_val.txt` | `5ed1a5b2a7fa2a0210b7091720743ff3252392ffff448dd1ba83795f332b403e` | 1 137 250 B (sin cambios) |
| `wikitext103_val_shuf_paragraph.txt` | `162d1941e2bab2d066e3ebfa3883748e452db8648b0d9115e993a6125004bb2e` | 1 137 250 B |
| `wikitext103_val_shuf_sentence.txt` | `5b69ad85db23d6bdbc488355dc27a391487f086ee872809f5a8476dafd3428c1` | 1 135 686 B (sin cambios) |
| `wikitext103_val_shuf_word.txt` | `0c5473aa43e89f31968b92f3af230247d4cd8f2b476c864fbd60732618513083` | 1 137 250 B |

## 3. Alcance de la desviación

**Lo que NO cambió:**
- Predicciones P-A3-1 hasta P-A3-6 (§2.4 del pre-registro).
- Criterios de refutación F4.
- Método de análisis (BH-FDR, q=0.05, 27 tests).
- Seeds {42, 123, 456}, arquitectura, hiperparámetros.
- SEED de permutación del corpus (20260420).

**Lo que cambió:**
- Contenido binario de dos archivos `.txt` de evaluación adversarial.
- Script `prep_shuffle_corpus.py` (bug fix en detección de separador de párrafos).

## 4. Momento de la corrección

La corrección se realizó **antes** de que `exp_A3_shuffled_eval.py` se ejecutara sobre los corpora. No se analizaron ni se vieron resultados con los corpora buggy (que, por definición, habrían dado Δ=0 para paragraph-shuffle al ser idénticos a `val`, lo cual es un fallo detectable pero vacío).

## 5. Resultados del experimento A3 (con corpora corregidos)

Ejecutado 2026-04-21. Archivo: `LINEAS DE INVESTIGACIÓN/LINEA A/FASE 5/results_A3_shuffled_eval/A3_results.json`.

| Predicción | Resultado | Notas |
|---|---|---|
| P-A3-1: Δσ₂ paragraph−val significativo, d≥0.8 | ✅ PASS (d=−0.96) | |
| P-A3-2: ΔH₁ paragraph−val significativo, d≥0.8 | ✗ FAIL | |
| P-A3-3: ΔPR paragraph−val significativo, d≥0.8 | ✗ FAIL | |
| P-A3-4: Δσ₀ paragraph−val NO significativo, \|d\|<0.3 | ✗ FAIL (d=−5.31) | **Homeostasis compensatoria: σ₀↑ cuando estructura párrafo se destruye.** Dirección opuesta a la hipótesis nula del pre-reg. Hallazgo exploratorio no-predicho. |
| P-A3-5: gradiente s₂ paragraph>sentence>word | ✅ PASS (Δσ=[2.30°, 1.32°, 0.16°]) | |
| P-A3-6: gradiente inverso s₀ word>sentence>paragraph | ✗ FAIL | |

Total: 6/27 tests significativos bajo BH-FDR (q=0.05). 2/6 predicciones primarias confirmadas.

**El criterio de refutación F4** (A3 no distingue paragraph-shuffle de word-shuffle en s₂, \|d\|<0.3 en ambos) NO se activó: el gradiente P-A3-5 es monotónico y significativo.

## 6. Conformidad con OSF Preregistration Standard

Esta desviación cumple con §5 del pre-registro ("Desviaciones permitidas del protocolo"). La cláusula "Aumentar $n_{\text{seeds}}$... reportando la desviación" establece el precedente de reporte; aplicamos el mismo principio a esta corrección de artefactos de datos. La cláusula prohibitiva "Añadir o quitar condiciones experimentales al test A2 / A3 / P1-TRCTRL después de ejecutar" NO aplica: las tres condiciones (paragraph, sentence, word) se mantienen; sólo se corrige el contenido de dos archivos antes de ejecutar.

## 7. Artefactos publicados con esta corrección

- Nuevo tag de release: `v1.0.1-data-correction` (commit `f29e895`).
- Zenodo DOI nuevo: [10.5281/zenodo.19687612](https://doi.org/10.5281/zenodo.19687612).
- Archivos actualizados:
  - `LINEAS DE INVESTIGACIÓN/LINEA A/FASE 5/prep_shuffle_corpus.py`
  - `data/wikitext103_val_shuf_paragraph.txt`
  - `data/wikitext103_val_shuf_word.txt`
  - Este documento (`PROTOCOL-DEVIATION-2026-04-21.md`).

El pre-registro original `v1.0.0-preregister` **no se modifica** — permanece como registro congelado de las predicciones pre-datos.

---

**Firma:** registro público de desviación conforme a buenas prácticas de reproducibilidad.
**Contacto:** (el autor del pre-registro original).

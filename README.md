# Hierarchical Neural Collapse — Paper P1 reproducibility bundle

This repository is the frozen reproducibility snapshot for the paper
**"Hierarchical Neural Collapse: An Intrinsic 60° Angular Signature in
Multi-Scale Byte-Level Language Models"** (pre-registration, v1).

It is generated from a larger private research monorepo via
`scripts/prep_osf_bundle.ps1`. Only the files strictly needed to
reproduce the paper P1 tables and to verify the pre-registered
predictions are included here.

## Authors

- Juan David Zuluaga-Monroy, MD (Interventional Cardiology) — Independent Researcher, Colombia. ORCID [0000-0002-9865-471X](https://orcid.org/0000-0002-9865-471X).
- Diego Fernando Zuluaga-Monroy, EE — Independent Researcher, Colombia. ORCID `[PENDING]`.

## Contents

| Path | Purpose |
|---|---|
| `LICENSE` | Apache-2.0 — all code |
| `LICENSE-docs.md` | CC-BY-4.0 — all documentation, figures, data |
| `CITATION.cff` | Citation metadata (GitHub renders this) |
| `.zenodo.json` | Zenodo release metadata |
| `requirements.txt` | Python dependencies |
| `data/` | WikiText-103 evaluation corpora + shuffled controls (fixed seed) |
| `LINEAS DE INVESTIGACIÓN/LINEA A/FASE 5/` | Experimental scripts (stats_helpers, exp_ctrl_A1_A4, exp_A2, exp_A3, exp_P1_transformer_control, ...) |
| `LINEAS CONSOLIDADAS DE TRABAJO/OSF-PREREG-P1.md` | Pre-registration document (frozen 2026-04-20) |
| `LINEAS CONSOLIDADAS DE TRABAJO/PAPER-P1-HNC-SKELETON.md` | Paper skeleton with result tables |
| `LINEAS CONSOLIDADAS DE TRABAJO/PAPER-P1-DRAFT.md` | §1 Introduction + §2 Method prose |

## Reproducing

1. Create a Python 3.11+ environment and install:
   ```
   pip install -r requirements.txt
   ```
2. Run the untrained + null-TDA controls (A1, A4):
   ```
   cd "LINEAS DE INVESTIGACIÓN/LINEA A/FASE 5"
   python exp_ctrl_A1_A4_untrained_and_null_tda.py
   ```
3. Run the pre-registered predictions (after trained checkpoints are
   available; see `OSF-PREREG-P1.md §6`):
   ```
   python exp_A2_causal_permutation.py
   python exp_A3_shuffled_eval.py
   python exp_P1_transformer_control.py
   ```
4. Apply the frozen statistical pipeline from `stats_helpers.py`
   (bootstrap CI, Cohen's d / Hedges' g, BH-FDR q=0.05).

## Pre-registration

OSF project: <https://osf.io/hkt6c/>.
Zenodo DOI: `10.5281/zenodo.[PENDING]` (to be filled after the first release).

The four pre-registered predictions (P-C1, P-A2, P-A3, P-A5) were frozen
**prior** to the execution of experiments A2, A3, and P1-TRCTRL. See
`OSF-PREREG-P1.md §5` for the rejection rules.

## Licenses

- **Code** — Apache License 2.0 (patent grant, commercial use allowed,
  attribution required). See `LICENSE`.
- **Documentation, figures, paper text, data derivatives** — Creative
  Commons Attribution 4.0 (CC-BY-4.0). See `LICENSE-docs.md`.
- **Upstream data** — WikiText-103 is redistributed under its original
  CC-BY-SA 3.0 license (Merity et al. 2016). See `data/README.md`.

## Citation

If you use this code or cite the pre-registration, please use the
metadata in `CITATION.cff` (GitHub renders a "Cite this repository"
button). Once the Zenodo DOI is minted, cite that in preference.

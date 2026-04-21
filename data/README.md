# Data directory

This directory contains the **WikiText-103 byte-level corpora** used for the HNC paper P1 pre-registered experiments. Only the files required for reproducibility are tracked in git; everything else is ignored.

## Tracked files

| File | Purpose | Generator |
|---|---|---|
| `wikitext103_val.txt`            | Original validation split (byte-level, UTF-8). Used for training and as control condition in A3. | external (WikiText-103 release) |
| `wikitext103_test.txt`           | Test split, reserved — never touched for hyperparameter selection. | external |
| `wikitext103_val_shuf_paragraph.txt` | Validation shuffled at paragraph level. Preserves within-paragraph word order; destroys inter-paragraph semantics. | `../LINEAS DE INVESTIGACIÓN/LINEA A/FASE 5/prep_shuffle_corpus.py`, SEED=20260420 |
| `wikitext103_val_shuf_sentence.txt`  | Validation shuffled at sentence level. Preserves within-sentence word order. | same script, SEED=20260420 |
| `wikitext103_val_shuf_word.txt`  | Validation shuffled at word level. Destroys local syntax too. | same script, SEED=20260420 |

All files are **1,134,789 bytes** (the shuffled variants are generated from the first ≈1.1 MB of the validation set, same byte count across granularities).

## Regenerating the shuffled corpora

```powershell
cd "LINEAS DE INVESTIGACIÓN/LINEA A/FASE 5"
python prep_shuffle_corpus.py
```

The script is deterministic given `SEED=20260420`; re-running produces byte-identical files.

## Provenance

The WikiText-103 dataset is distributed under the CC BY-SA 3.0 license by Salesforce Research (Merity et al., 2016). Our shuffled derivatives are also released under CC BY-SA 3.0 for consistency with the upstream license (note: this is different from the CC BY 4.0 used for our own documentation; see [../LICENSE-docs.md](../LICENSE-docs.md)).

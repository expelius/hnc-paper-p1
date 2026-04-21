"""
prep_shuffle_corpus.py
======================
A3 — Adversarial corpus generation for the 2026-04-20 plan.

Generates three variants of the WikiText-103 val corpus that preserve
local structure at varying scales but destroy structure above that
scale. Lets us test whether s2 (discourse) geometry specifically
depends on cross-paragraph structure.

Variants produced (all same total byte count as source):
  - wikitext103_val_shuf_paragraph.txt  : paragraphs shuffled
                                          → destroys discourse, keeps sentences
  - wikitext103_val_shuf_sentence.txt   : sentences shuffled
                                          → destroys paragraphs, keeps syntax
  - wikitext103_val_shuf_word.txt       : words shuffled within paragraphs
                                          → destroys syntax, keeps n-grams

Prediction (paper ALPHA C4 disociation):
  - On paragraph-shuffled: σ_s2 ↑, PR_s2 ↓, H1_s2 ↓. s0/s1 largely preserved.
  - On sentence-shuffled:  s1 and s2 degrade. s0 preserved.
  - On word-shuffled:      all scales degrade, but s0 relatively most.
If confirmed → clean dissociation between scales and linguistic layers.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
SRC = DATA_DIR / "wikitext103_val.txt"
SEED = 20260420


def load_src() -> str:
    return SRC.read_text(encoding="utf-8")


def shuffle_paragraphs(text: str, rng: random.Random) -> str:
    paras = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in paras if p.strip()]
    rng.shuffle(paras)
    return "\n\n".join(paras) + "\n"


def shuffle_sentences(text: str, rng: random.Random) -> str:
    # Simple sentence splitter: . ! ? followed by space / newline.
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    rng.shuffle(sentences)
    return " ".join(sentences) + "\n"


def shuffle_words_in_paragraphs(text: str, rng: random.Random) -> str:
    out_paras = []
    for p in re.split(r"\n\s*\n", text):
        if not p.strip():
            continue
        words = p.split()
        rng.shuffle(words)
        out_paras.append(" ".join(words))
    return "\n\n".join(out_paras) + "\n"


def main():
    assert SRC.exists(), f"Source corpus not found: {SRC}"
    text = load_src()
    n_bytes = len(text.encode("utf-8"))
    print(f"Source: {SRC.name}  bytes={n_bytes:,}")

    rng = random.Random(SEED)
    variants = {
        "wikitext103_val_shuf_paragraph.txt": shuffle_paragraphs(text, random.Random(SEED + 1)),
        "wikitext103_val_shuf_sentence.txt": shuffle_sentences(text, random.Random(SEED + 2)),
        "wikitext103_val_shuf_word.txt":      shuffle_words_in_paragraphs(text, random.Random(SEED + 3)),
    }

    for name, content in variants.items():
        out = DATA_DIR / name
        out.write_text(content, encoding="utf-8")
        nb = len(content.encode("utf-8"))
        ratio = nb / n_bytes
        print(f"  wrote {name:48s}  bytes={nb:,}  ratio={ratio:.3f}")

    print("\nDone. Use these as drop-in replacements for DATA_PATH in any")
    print("analysis script to compare trained-model metrics across shuffles.")


if __name__ == "__main__":
    main()

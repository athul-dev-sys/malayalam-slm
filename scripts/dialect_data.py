"""Stage 6: build dialect/register adaptation datasets.

1. Film-register corpus: OpenSubtitles v2024 ml side -> clean (reuse Stage-2
   filters) -> train/val text.
2. Kasaragod eval probes: from a dialect->standard lexicon CSV (columns:
   dialect,standard[,english]) -> synthetic probe sentences by substitution
   (flagged synthetic; validation-gated before *training* use).

Usage: PYTHONIOENCODING=utf-8 python scripts/dialect_data.py [--lexicon path.csv]
"""
import argparse
import io
import os
import re
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from build_corpus import keep_para  # reuse measured filter thresholds

OUT = os.path.join(ROOT, "data", "dialect")
OS_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-ml.txt.zip"


def subtitle_lines(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        ml_name = [n for n in z.namelist() if n.endswith(".ml")][0]
        with z.open(ml_name) as f:
            for raw in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                s = raw.strip()
                if s:
                    yield s


def clean_dialogue(lines):
    """Subtitles need lighter filtering than web text: short utterances are the point."""
    kept = []
    stats = {"in": 0, "kept": 0, "drop": {}}
    for s in lines:
        stats["in"] += 1
        s = re.sub(r"[ \t]{2,}", " ", s)
        s = "".join(ch for ch in s if ch >= " " or ch in "\u200c\u200d")
        if not (12 <= len(s) <= 500):
            stats["drop"]["len"] = stats["drop"].get("len", 0) + 1
            continue
        ml = sum(1 for ch in s if "\u0d00" <= ch <= "\u0d7f")
        non_space = sum(1 for ch in s if not ch.isspace())
        if non_space == 0 or ml / non_space < 0.70:
            stats["drop"]["lid"] = stats["drop"].get("lid", 0) + 1
            continue
        if len(set(s)) / len(s) < 0.035:
            stats["drop"]["char_rep"] = stats["drop"].get("char_rep", 0) + 1
            continue
        kept.append(s)
        stats["kept"] += 1
    return kept, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="", help="CSV with dialect,standard[,english] columns")
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    zip_path = os.path.join(OUT, "opensubtitles_en-ml.zip")
    if not os.path.exists(zip_path):
        print("downloading OpenSubtitles en-ml ...")
        urllib.request.urlretrieve(OS_URL, zip_path)
    lines = subtitle_lines(zip_path)
    kept, stats = clean_dialogue(lines)
    print("film-register stats:", stats)

    import unicodedata

    kept = [unicodedata.normalize("NFC", s) for s in kept]
    # exact dedup
    seen, ded = set(), []
    for s in kept:
        if s in seen:
            continue
        seen.add(s)
        ded.append(s)
    print(f"after dedup: {len(ded)} lines, {sum(len(s) for s in ded)/1e6:.1f}M chars")

    n_val = max(500, int(len(ded) * args.val_frac))
    with io.open(os.path.join(OUT, "film_train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ded[n_val:]))
    with io.open(os.path.join(OUT, "film_val.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ded[:n_val]))

    if args.lexicon and os.path.exists(args.lexicon):
        import csv

        pairs = []
        with io.open(args.lexicon, "r", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    pairs.append((row[0].strip(), row[1].strip()))
        print(f"lexicon pairs: {len(pairs)}")
        with io.open(os.path.join(OUT, "kasaragod_lexicon.tsv"), "w", encoding="utf-8") as f:
            for d, s in pairs:
                f.write(f"{d}\t{s}\n")
    print("done ->", OUT)


if __name__ == "__main__":
    main()

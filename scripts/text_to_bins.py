"""Tokenize arbitrary train/val text files into uint16 memmap bins (for dialect/FT data).

Usage: PYTHONIOENCODING=utf-8 python scripts/text_to_bins.py <train.txt> <val.txt> <out_dir> [lines_per_doc=N]
Lines are documents (dialogue utterances); BOS+EOS wrapped.
"""
import io
import os
import re
import sys
import unicodedata

import numpy as np
import sentencepiece as spm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.path.join(ROOT, "artifacts", "tokenizer", "malayalam_spm.model")
ws = re.compile(r"[ \t]{2,}")


def main():
    train_txt, val_txt, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    tok_path = sys.argv[4] if len(sys.argv) > 4 else TOK
    os.makedirs(out_dir, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=tok_path)

    def encode_file(path, out_name):
        docs = []
        with io.open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    docs.append(ws.sub(" ", unicodedata.normalize("NFC", s)))
        encs = sp.encode(docs, out_type=int, add_bos=True, add_eos=True)
        ids = [i for e in encs for i in e]
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(os.path.join(out_dir, out_name))
        return {"docs": len(docs), "tokens": len(arr), "chars": sum(len(d) for d in docs)}

    stats = {
        "train": encode_file(train_txt, "train.bin"),
        "val": encode_file(val_txt, "val.bin"),
    }
    stats["tokenizer"] = tok_path
    import json

    with io.open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

"""Stage 2 final phase: global re-dedup across sources, doc-level shuffle, 98/1/1 split,
tokenization to uint16 memmaps. Also saves raw test text for baseline evals.

Usage: PYTHONIOENCODING=utf-8 python scripts/split_tokenize.py <tokenizer.model>
"""
import hashlib
import io
import json
import os
import random
import sys
import time

import sentencepiece as spm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "clean")
OUT = os.path.join(ROOT, "data", "tokenized")
DOC_SEP = "\x1e"


def norm_key(s):
    key = "".join(ch for ch in s if "\u0d00" <= ch <= "\u0d7f")
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()


def iter_docs(path):
    """Yield docs as lists of paragraph strings; \x1e marks boundary, else 1-para docs."""
    import gzip

    op = gzip.open if path.endswith(".gz") else io.open
    paras = []
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip() == DOC_SEP:
                if paras:
                    yield paras
                    paras = []
            elif line.strip():
                paras.append(line.strip())
    if paras:
        yield paras


def main():
    tok_path = sys.argv[1]
    sp = spm.SentencePieceProcessor(model_file=tok_path)
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()
    sources = sorted(
        f for f in os.listdir(CLEAN) if f.endswith((".txt", ".txt.gz"))
    )
    print("sources:", sources)

    # ---- collect + global re-dedup ----
    seen = set()
    docs = []  # (source, [paras])
    stats = {"per_source": {}, "global_dup_cross_source": 0}
    for fname in sources:
        name = fname[:-4] if fname.endswith(".txt") else fname[:-7]
        n_docs = n_paras = n_dup = 0
        for paras in iter_docs(os.path.join(CLEAN, fname)):
            kept = []
            for p in paras:
                k = norm_key(p)
                if k in seen:
                    n_dup += 1
                    continue
                seen.add(k)
                kept.append(p)
            if kept:
                docs.append((name, kept))
                n_docs += 1
                n_paras += len(kept)
        stats["per_source"][name] = {"docs": n_docs, "paras": n_paras, "cross_dup": n_dup}
        stats["global_dup_cross_source"] += n_dup
        print(f"{name}: docs={n_docs} paras={n_paras} cross-dup={n_dup}")

    # ---- shuffle + split ----
    random.Random(1337).shuffle(docs)
    n = len(docs)
    n_test = max(1000, n // 100)
    n_val = max(1000, n // 100)
    test_docs = docs[:n_test]
    val_docs = docs[n_test : n_test + n_val]
    train_docs = docs[n_test + n_val :]
    print(f"split: train={len(train_docs)} val={len(val_docs)} test={len(test_docs)} docs")

    # ---- tokenize each split ----
    import re

    ws = re.compile(r"[ \t]{2,}")

    def doc_text(paras):
        return "\n\n".join(ws.sub(" ", p) for p in paras)

    def encode_split(split_docs, out_name, also_text=False):
        ids_all = []
        n_chars = 0
        text_out = io.open(os.path.join(OUT, out_name + ".txt"), "w", encoding="utf-8") if also_text else None
        texts = [doc_text(paras) for _, paras in split_docs]
        encs = sp.encode(texts, out_type=int, add_bos=True, add_eos=True)
        for t, enc in zip(texts, encs):
            ids_all.extend(enc)
            n_chars += len(t)
            if text_out:
                text_out.write(t + "\n\n" + DOC_SEP + "\n\n")
        if text_out:
            text_out.close()
        arr_path = os.path.join(OUT, out_name + ".bin")
        import numpy as np

        arr = np.array(ids_all, dtype=np.uint16)
        arr.tofile(arr_path)
        return {
            "tokens": len(arr), "chars": n_chars,
            "tokens_per_char": len(arr) / max(n_chars, 1),
            "bin_mb": os.path.getsize(arr_path) / 1e6,
        }

    res = {}
    res["train"] = encode_split(train_docs, "train")
    res["val"] = encode_split(val_docs, "val")
    res["test"] = encode_split(test_docs, "test", also_text=True)  # raw text for baseline evals
    res["total_tokens"] = res["train"]["tokens"] + res["val"]["tokens"] + res["test"]["tokens"]
    stats["split"] = res
    stats["tokenizer"] = tok_path
    stats["vocab_size"] = sp.get_piece_size()
    stats["elapsed_min"] = round((time.time() - t0) / 60, 1)

    with io.open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(json.dumps(stats["split"], indent=1))
    print(f"TOTAL TOKENS: {res['total_tokens']:,}")
    print(f"elapsed {stats['elapsed_min']} min")


if __name__ == "__main__":
    main()

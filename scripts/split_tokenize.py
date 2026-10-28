"""Stage 2 final phase: global re-dedup across sources, doc-level shuffle, 98/1/1 split,
tokenization to uint16 memmaps. Also saves raw test text for baseline evals.

Streams the cleaned corpus twice so peak RAM is O(unique paragraphs + n_docs) rather
than O(corpus). Pass 1 globally dedups paragraphs and writes the surviving documents
to a length-prefixed temp file. Pass 2 assigns the shuffled split labels and tokenizes
in batches, appending directly to the three uint16 bins.

Files without explicit \\x1e document separators are treated as one paragraph per
document (IndicCorpV2 has no doc metadata; the original implementation wrongly held
the whole file as a single document).

Usage: PYTHONIOENCODING=utf-8 python scripts/split_tokenize.py <tokenizer.model>
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import struct
import time

import numpy as np
import sentencepiece as spm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "clean")
OUT = os.path.join(ROOT, "data", "tokenized")
WORK = os.path.join(ROOT, "data", "work")
DOC_SEP = "\x1e"
WS_RE = re.compile(r"[ \t]{2,}")
NON_ML_RE = re.compile(r"[^\u0d00-\u0d7f]")
SPLIT_NAMES = ("train", "val", "test")
TEST_SPLIT = 2
SPLIT_SEED = 1337
ENCODE_BATCH_CHARS = 2_000_000
ENCODE_BATCH_DOCS = 1024


def norm_key(s):
    # identical semantics to the original per-character join, but C-speed
    key = NON_ML_RE.sub("", s)
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()


def _iter_lines(path, max_bytes=0):
    """Yield decoded lines, optionally stopping after max_bytes of raw input."""
    mode = "rb"
    with (gzip.open(path, mode) if path.endswith(".gz") else io.open(path, mode)) as f:
        if max_bytes <= 0:
            for line in f:
                yield line.decode("utf-8", errors="replace")
            return
        remaining = max_bytes
        tail = b""
        while remaining > 0:
            chunk = f.read(min(1 << 20, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            parts = (tail + chunk).split(b"\n")
            tail = parts.pop()
            for p in parts:
                yield p.decode("utf-8", errors="replace") + "\n"
        if tail:
            yield tail.decode("utf-8", errors="replace")


def source_has_doc_sep(path, max_bytes=0, sample_chars=4_000_000):
    seen = 0
    for line in _iter_lines(path, max_bytes):
        if line.rstrip("\r\n") == DOC_SEP:
            return True
        seen += len(line)
        if seen >= sample_chars:
            return False
    return False


def iter_docs(path, para_docs, max_bytes=0):
    """Yield documents as lists of paragraph strings.

    para_docs=True: every blank-line paragraph block is its own document.
    para_docs=False: paragraphs accumulate until an explicit \\x1e boundary.
    """
    paras = []
    buf = []

    def flush_para():
        if buf:
            paras.append(" ".join(buf))
            buf.clear()

    for line in _iter_lines(path, max_bytes):
        # \x1e is Unicode whitespace, so strip() would erase the marker before comparison
        raw = line.rstrip("\r\n")
        if raw == DOC_SEP:
            flush_para()
            if paras:
                yield paras
                paras = []
            continue
        s = raw.strip()
        if not s:
            flush_para()
            if para_docs and paras:
                yield paras
                paras = []
        else:
            buf.append(s)
    flush_para()
    if paras:
        yield paras


def reduce_sources(sources, tmp_docs, limits=None):
    """Pass 1: global paragraph dedup, write surviving docs to temp file. Returns (n_docs, stats)."""
    seen = set()
    n_docs = 0
    limits = limits or {}
    stats = {"per_source": {}, "global_dup_cross_source": 0, "documents_total": 0, "chars_total": 0}

    with io.open(tmp_docs, "wb") as out:
        for fname in sources:
            path = os.path.join(CLEAN, fname)
            name = fname[:-4] if fname.endswith(".txt") else fname[:-7]
            max_bytes = limits.get(fname, 0)
            para_docs = not source_has_doc_sep(path, max_bytes)
            src = {
                "docs": 0, "paras": 0, "paras_kept": 0, "cross_dup": 0,
                "chars": 0, "doc_boundaries": "paragraph" if para_docs else "separator",
                "max_bytes": max_bytes,
            }
            for paras in iter_docs(path, para_docs, max_bytes):
                kept = []
                for p in paras:
                    src["paras"] += 1
                    k = norm_key(p)
                    if k in seen:
                        src["cross_dup"] += 1
                        continue
                    seen.add(k)
                    kept.append(p)
                if not kept:
                    continue
                text = "\n\n".join(WS_RE.sub(" ", p) for p in kept)
                raw = text.encode("utf-8")
                out.write(struct.pack("<I", len(raw)))
                out.write(raw)
                n_docs += 1
                src["docs"] += 1
                src["paras_kept"] += len(kept)
                src["chars"] += len(text)
            stats["per_source"][name] = src
            stats["global_dup_cross_source"] += src["cross_dup"]
            print(f"{name}: docs={src['docs']} paras_kept={src['paras_kept']} "
                  f"cross-dup={src['cross_dup']} mode={src['doc_boundaries']}", flush=True)

    stats["documents_total"] = n_docs
    stats["chars_total"] = sum(s["chars"] for s in stats["per_source"].values())
    return n_docs, stats


def make_split_labels(n_docs):
    """Doc-level shuffle + 98/1/1 slice, matching the original max(1000, 1%) rule."""
    if n_docs < 3:
        raise SystemExit(f"need at least 3 documents for a train/val/test split, got {n_docs}")
    n_test = max(1000, n_docs // 100)
    n_val = max(1000, n_docs // 100)
    if n_test + n_val >= n_docs:
        # tiny-corpus fallback; the Stage 2 corpus is far above this threshold
        n_test = max(1, n_docs // 100)
        n_val = max(1, n_docs // 100)
    if n_test + n_val >= n_docs:
        raise SystemExit(f"too few documents for three non-empty splits: {n_docs}")
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n_docs)
    labels = np.zeros(n_docs, dtype=np.uint8)
    labels[perm[:n_test]] = 2
    labels[perm[n_test:n_test + n_val]] = 1
    return labels, n_test, n_val


def tokenize_split(sp, tmp_docs, labels, n_docs, stats):
    """Pass 2: tokenize to uint16 bins in bounded batches; test text saved verbatim."""
    tmp_bins = {sid: os.path.join(OUT, name + ".bin.tmp") for sid, name in enumerate(SPLIT_NAMES)}
    tmp_test_txt = os.path.join(OUT, "test.txt.tmp")
    fh = {sid: io.open(path, "wb") for sid, path in tmp_bins.items()}
    test_out = io.open(tmp_test_txt, "w", encoding="utf-8", newline="\n")
    per_split = {
        name: {"tokens": 0, "chars": 0, "docs": 0} for name in SPLIT_NAMES
    }
    batch = []
    batch_chars = 0
    idx = 0
    vocab_size = sp.get_piece_size()
    if vocab_size > 65536:
        raise SystemExit(f"vocab {vocab_size} does not fit uint16 bins")

    def flush():
        nonlocal batch, batch_chars
        if not batch:
            return
        encs = sp.encode([t for _, t in batch], out_type=int, add_bos=True, add_eos=True)
        grouped = {0: [], 1: [], 2: []}
        for (sid, text), enc in zip(batch, encs):
            arr = np.asarray(enc, dtype=np.uint16)
            if arr.size and int(arr.max()) >= vocab_size:
                raise SystemExit("token id out of vocabulary range")
            grouped[sid].append(arr)
            meta = per_split[SPLIT_NAMES[sid]]
            meta["tokens"] += len(arr)
            meta["chars"] += len(text)
            meta["docs"] += 1
            if sid == TEST_SPLIT:
                test_out.write(text + "\n\n" + DOC_SEP + "\n\n")
        for sid, arrays in grouped.items():
            if arrays:
                np.concatenate(arrays).tofile(fh[sid])
        batch = []
        batch_chars = 0

    with io.open(tmp_docs, "rb") as f:
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            if len(hdr) != 4:
                raise SystemExit("truncated document record header")
            (n_bytes,) = struct.unpack("<I", hdr)
            raw = f.read(n_bytes)
            if len(raw) != n_bytes:
                raise SystemExit("truncated document record body")
            text = raw.decode("utf-8")
            batch.append((int(labels[idx]), text))
            batch_chars += len(text)
            idx += 1
            if batch_chars >= ENCODE_BATCH_CHARS or len(batch) >= ENCODE_BATCH_DOCS:
                flush()
                print(f"tokenized {idx:,}/{n_docs:,} docs", flush=True)
        flush()

    if idx != n_docs:
        raise SystemExit(f"document count mismatch: read {idx}, expected {n_docs}")
    for handle in fh.values():
        handle.close()
    test_out.close()

    for sid, name in enumerate(SPLIT_NAMES):
        final = os.path.join(OUT, name + ".bin")
        os.replace(tmp_bins[sid], final)
        per_split[name]["bin_mb"] = os.path.getsize(final) / 1e6
        per_split[name]["tokens_per_char"] = (
            per_split[name]["tokens"] / per_split[name]["chars"] if per_split[name]["chars"] else 0.0
        )
    os.replace(tmp_test_txt, os.path.join(OUT, "test.txt"))
    stats["split"] = per_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer", nargs="?", default=os.path.join(
        ROOT, "artifacts", "tokenizer", "malayalam_spm.model"
    ))
    ap.add_argument("--sources", default="", help="comma-separated file names under data/clean")
    ap.add_argument("--max-mb", action="append", default=[], metavar="FILE=MB",
                    help="bound one source's raw bytes (repeatable); unbounded otherwise")
    args = ap.parse_args()
    tok_path = args.tokenizer

    limits = {}
    for spec in args.max_mb:
        name, _, mb = spec.partition("=")
        name, mb = name.strip(), mb.strip()
        if not name or not mb:
            raise SystemExit(f"bad --max-mb spec: {spec!r} (expected FILE=MB)")
        limits[name] = int(float(mb) * 1024 * 1024)

    sp = spm.SentencePieceProcessor(model_file=tok_path)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    t0 = time.time()

    if args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        sources = sorted(f for f in os.listdir(CLEAN) if f.endswith((".txt", ".txt.gz")))
    if not sources:
        raise SystemExit(f"no cleaned corpus files in {CLEAN}")
    for fname in sources:
        if not os.path.exists(os.path.join(CLEAN, fname)):
            raise SystemExit(f"missing source: {os.path.join(CLEAN, fname)}")
        if fname in limits:
            size = os.path.getsize(os.path.join(CLEAN, fname))
            if limits[fname] > size:
                limits[fname] = size
    print("sources:", sources, flush=True)
    print("limits:", {k: f"{v/1e6:.0f}MB" for k, v in limits.items()} or "none", flush=True)

    tmp_docs = os.path.join(WORK, "stage2_docs.tmp")
    n_docs, stats = reduce_sources(sources, tmp_docs, limits)
    print(f"global dedup: {n_docs:,} docs, {stats['chars_total']:,} chars", flush=True)

    labels, n_test, n_val = make_split_labels(n_docs)
    print(f"split: train={n_docs - n_test - n_val:,} val={n_val:,} test={n_test:,} docs", flush=True)
    tokenize_split(sp, tmp_docs, labels, n_docs, stats)

    stats["total_tokens"] = sum(s["tokens"] for s in stats["split"].values())
    stats["source_scope"] = {
        "sources": sources,
        "max_bytes": limits,
        "note": "bounded slice; rerun without --max-mb for full-corpus stats" if limits else "full cleaned sources",
    }
    stats["tokenizer"] = tok_path
    stats["vocab_size"] = sp.get_piece_size()
    stats["split_seed"] = SPLIT_SEED
    stats["split_method"] = "numpy_pcg64_permutation (doc-level shuffle then 98/1/1 slice)"
    stats["elapsed_min"] = round((time.time() - t0) / 60, 1)

    stats_path = os.path.join(OUT, "stats.json")
    tmp_stats = stats_path + ".tmp"
    with io.open(tmp_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    os.replace(tmp_stats, stats_path)

    os.remove(tmp_docs)
    print(json.dumps(stats["split"], indent=1))
    print(f"TOTAL TOKENS: {stats['total_tokens']:,}")
    print(f"elapsed {stats['elapsed_min']} min")


if __name__ == "__main__":
    main()

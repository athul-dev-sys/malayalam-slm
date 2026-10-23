"""Stage 2 corpus cleaning: ingest -> NFC -> script-LID -> junk filter -> dedup.

Reads a source (file path or '-' for stdin; '--xz' to decompress stdin),
writes clean paragraphs (doc-separated by \x1e when available) + per-step stats.

Usage:
  python scripts/build_corpus.py clean data/raw/mlwiki_prose.txt data/clean/wiki
  curl -sL <url> | python scripts/build_corpus.py clean - data/clean/indiccorp
  curl -sL <url> | python scripts/build_corpus.py clean - data/clean/cc100 --xz
"""
import hashlib
import io
import json
import lzma
import re
import sys
import time
import unicodedata

ML_LO, ML_HI = "\u0d00", "\u0d7f"
DOC_SEP = "\x1e"
HTTP_RE = re.compile(r"https?://", re.IGNORECASE)
CTRL_OK = {"\u200c", "\u200d", "\n", "\t"}
CTRL_STRIP = {c for c in map(chr, range(32)) if c not in CTRL_OK} | {"\u00a0"}


def para_stats(s):
    ml = latin = digits = other = 0
    for ch in s:
        if ML_LO <= ch <= ML_HI:
            ml += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
        elif ch.isdigit():
            digits += 1
        elif not ch.isspace():
            other += 1
    non_space = ml + latin + digits + other
    return ml, latin, digits, other, non_space


def keep_para(s, stats):
    """Return (kept: bool, reason: str). Updates stats dict."""
    s = "".join(ch for ch in s if ch not in CTRL_STRIP)
    if not (60 <= len(s) <= 8192):
        return False, "len"
    ml, latin, digits, other, non_space = para_stats(s)
    if non_space == 0 or ml / non_space < 0.70:
        return False, "lid"
    if latin / non_space > 0.25:
        return False, "manglish"
    if digits / non_space > 0.30:
        return False, "digits"
    if len(HTTP_RE.findall(s)) > 2:
        return False, "urls"
    # Malayalam prose has a small char inventory (median unique/len ~ 0.12);
    # 0.035 is below measured glyph-spam (0.034) and above only ~1% of legit wiki text
    if len(set(s)) / len(s) < 0.035:
        return False, "char_rep"
    words = s.split()
    if len(words) >= 8 and len(set(words)) / len(words) < 0.35:
        return False, "word_rep"
    return True, ""


def norm_key(s):
    key = "".join(ch for ch in s if ML_LO <= ch <= ML_HI)
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()


def iter_paras(text_stream, doc_blocks=False):
    """Yield ('para'|'doc', string).

    doc_blocks=False: blank-line block = paragraph (IndicCorpV2 style).
    doc_blocks=True:  blank-line block = document; lines inside = paragraphs (CC100 style).
    \x1e line = explicit doc boundary (wiki extractor style).
    """
    buf = []
    for line in text_stream:
        line = line.rstrip("\n")
        if line == DOC_SEP:
            if buf:
                yield "para", " ".join(buf)
                buf = []
            yield "doc", ""
            continue
        if line.strip() == "":
            if buf:
                if doc_blocks:
                    for b in buf:
                        yield "para", b
                    yield "doc", ""
                else:
                    yield "para", " ".join(buf)
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        if doc_blocks:
            for b in buf:
                yield "para", b
        else:
            yield "para", " ".join(buf)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "clean"
    if mode != "clean":
        raise SystemExit("only mode 'clean' supported here")
    src = sys.argv[2]
    out_prefix = sys.argv[3]
    xz = "--xz" in sys.argv
    doc_blocks = "--doc-blocks" in sys.argv

    if src == "-":
        raw = sys.stdin.buffer
        if xz:
            raw = lzma.open(raw, "rt", encoding="utf-8", errors="replace")
        else:
            raw = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    else:
        raw = io.open(src, "r", encoding="utf-8", errors="replace")

    stats = {
        "source": src,
        "paras_in": 0, "docs_in": 0, "paras_kept": 0, "bytes_out": 0,
        "drop": {}, "dup": 0, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    seen = set()
    out_path = out_prefix + ".txt"
    t0 = time.time()
    with io.open(out_path, "w", encoding="utf-8") as out:
        for kind, s in iter_paras(raw, doc_blocks=doc_blocks):
            if kind == "doc":
                stats["docs_in"] += 1
                out.write(DOC_SEP + "\n\n")
                continue
            stats["paras_in"] += 1
            s = unicodedata.normalize("NFC", s)
            kept, reason = keep_para(s, stats)
            if not kept:
                stats["drop"][reason] = stats["drop"].get(reason, 0) + 1
                continue
            k = norm_key(s)
            if k in seen:
                stats["dup"] += 1
                continue
            seen.add(k)
            out.write(s + "\n\n")
            stats["paras_kept"] += 1
            stats["bytes_out"] += len(s.encode("utf-8")) + 2
            if stats["paras_in"] % 1_000_000 == 0:
                el = time.time() - t0
                print(f"... {stats['paras_in']/1e6:.0f}M in, kept {stats['paras_kept']/1e6:.1f}M, "
                      f"dup {stats['dup']/1e6:.1f}M, {el/60:.1f} min", flush=True)

    stats["elapsed_min"] = round((time.time() - t0) / 60, 1)
    stats["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with io.open(out_prefix + "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

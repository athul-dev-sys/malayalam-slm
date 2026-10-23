"""Build the tokenizer-training corpus and the held-out fertility eval set.

Mix: IndicCorpV2 head sample (web text) + mlwiki prose (encyclopedic).
Eval: IndicCorpV2 mid-file sample (unseen region) + held-out wiki slice.

Usage: python scripts/prepare_tokenizer_corpus.py
"""
import io
import os
import random
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
WORK = os.path.join(ROOT, "data", "tokenizer_work")
os.makedirs(WORK, exist_ok=True)

ML_RANGE = ("\u0d00", "\u0d7f")
ALLOWED_EXTRA = set(" \n\t.,;:!?()[]{}\"'-–—…0123456789/%+*=<>@#$&_|~^`")


def script_ok(s: str, thresh=0.85) -> bool:
    if not s:
        return False
    ok = 0
    for ch in s:
        if ML_RANGE[0] <= ch <= ML_RANGE[1] or ch in ALLOWED_EXTRA or ch.isascii() and ch.isalpha():
            ok += 1
    return ok / len(s) >= thresh


def stream_paragraphs(path, limit_bytes=None):
    """Yield paragraph strings from an IndicCorpV2-style file (blank-line separated)."""
    buf = []
    read = 0
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            read += len(line.encode("utf-8", errors="replace"))
            if limit_bytes and read > limit_bytes:
                break
            if line.strip() == "":
                if buf:
                    yield " ".join(buf)
                    buf = []
            else:
                buf.append(line.strip())
    if buf:
        yield " ".join(buf)


def main():
    rng = random.Random(13)

    # -- train corpus --
    train_path = os.path.join(WORK, "tokenizer_train.txt")
    n_web = n_wiki = 0
    with io.open(train_path, "w", encoding="utf-8") as out:
        # web: use up to ~250MB of the head sample
        for p in stream_paragraphs(os.path.join(RAW, "indiccorp_ml_head400mb.txt"), limit_bytes=250 * 1024 * 1024):
            p = unicodedata.normalize("NFC", p)
            if len(p) >= 60 and script_ok(p):
                out.write(p + "\n")
                n_web += 1
        # wiki: all prose, but hold out every 50th paragraph for eval
        wiki_path = os.path.join(RAW, "mlwiki_prose.txt")
        if os.path.exists(wiki_path):
            heldout = io.open(os.path.join(WORK, "eval_wiki.txt"), "w", encoding="utf-8")
            for i, p in enumerate(stream_paragraphs(wiki_path)):
                p = unicodedata.normalize("NFC", p)
                if len(p) < 60 or not script_ok(p):
                    continue
                if i % 50 == 0:
                    heldout.write(p + "\n")
                else:
                    out.write(p + "\n")
                    n_wiki += 1
            heldout.close()
    print(f"train: web_paras={n_web} wiki_paras={n_wiki}")

    # -- eval corpus (mid-file sample, unseen) --
    eval_path = os.path.join(WORK, "eval_web.txt")
    n_ev = 0
    with io.open(eval_path, "w", encoding="utf-8") as out:
        for p in stream_paragraphs(os.path.join(RAW, "indiccorp_ml_mid50mb.txt"), limit_bytes=50 * 1024 * 1024):
            p = unicodedata.normalize("NFC", p)
            if len(p) >= 60 and script_ok(p):
                out.write(p + "\n")
                n_ev += 1
    print(f"eval: web_paras={n_ev}")

    # combined eval set (web + wiki halves)
    combo = os.path.join(WORK, "eval_combined.txt")
    with io.open(combo, "w", encoding="utf-8") as out:
        for src in (os.path.join(WORK, "eval_web.txt"), os.path.join(WORK, "eval_wiki.txt")):
            if os.path.exists(src):
                with io.open(src, "r", encoding="utf-8") as f:
                    out.write(f.read())
    for name in ("tokenizer_train.txt", "eval_combined.txt"):
        sz = os.path.getsize(os.path.join(WORK, name))
        print(f"{name}: {sz/1e6:.1f} MB")


if __name__ == "__main__":
    main()

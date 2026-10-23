"""Measure all candidate + baseline tokenizers on held-out Malayalam text.

Usage: PYTHONIOENCODING=utf-8 python scripts/eval_tokenizers.py
Writes: eval/tokenizer_comparison.json + markdown table to stdout.
"""
import io
import json
import os
import re

import sentencepiece as spm
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(ROOT, "artifacts", "tokenizer", "candidates")
WORK = os.path.join(ROOT, "data", "tokenizer_work")
EVALDIR = os.path.join(ROOT, "eval")
os.makedirs(EVALDIR, exist_ok=True)

EVAL_TEXT = os.path.join(WORK, "eval_combined.txt")
MAX_PARAS = 20000
MAX_EVAL_BYTES = 30 * 1024 * 1024

CHILLU_WORDS = [
    "കേരളം", "നമ്മൾ", "ഇന്ത്യയിൽ", "സ്ത്രീ", "ജ്ഞാനം", "ക്ഷേത്രം", "ശ്രീകൃഷ്ണൻ",
    "വിദ്യാഭ്യാസം", "സംസ്ഥാനം", "പ്രധാനമന്ത്രി", "വിമർശനം", "സങ്കടം", "ദൃഷ്ടി",
    "ഋതു", "ൠയം", "നമ്പർ", "അമ്മയ്\u200d", "കഴിഞ്ഞു", "താഴെ", "മുകളിൽ",
]


def load_paras():
    paras = []
    size = 0
    with io.open(EVAL_TEXT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paras.append(line)
            size += len(line.encode("utf-8"))
            if len(paras) >= MAX_PARAS or size >= MAX_EVAL_BYTES:
                break
    return paras


class SpmTok:
    kind = "spm"

    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.name = os.path.basename(path).replace(".model", "")

    def encode(self, text):
        return self.sp.encode(text, out_type=int, add_bos=False, add_eos=False)

    def decode(self, ids):
        return self.sp.decode(ids)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def piece_of(self, i):
        return self.sp.id_to_piece(i)


class HfJsonTok:
    kind = "hf"

    def __init__(self, path):
        self.tok = Tokenizer.from_file(path)
        self.name = os.path.basename(path).replace(".json", "")

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode(ids)

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    def piece_of(self, i):
        return self.tok.id_to_token(i)


class TransformersTok:
    kind = "hf-hub"

    def __init__(self, hub_name, alias=None):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(hub_name)
        self.name = alias or hub_name.replace("/", "_")

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self):
        return self.tok.vocab_size

    def piece_of(self, i):
        return self.tok.convert_ids_to_tokens(i)


def measure(tok, paras):
    n_tok = n_word = n_char = n_unk = n_exact = n_zwj_paras = n_zwj_kept = 0
    word_re = re.compile(r"\S+")
    for p in paras:
        ids = tok.encode(p)
        n_tok += len(ids)
        n_word += len(word_re.findall(p))
        n_char += len(p)
        n_unk += sum(1 for i in ids if tok.piece_of(i) in ("<unk>", "[UNK]", "<|unk|>"))
        if tok.decode(ids) == p:
            n_exact += 1
        if "\u200d" in p or "\u200c" in p:
            n_zwj_paras += 1
            if tok.decode(tok.encode(p)) == p:
                n_zwj_kept += 1
    return {
        "vocab_size": tok.vocab_size,
        "tokens_per_word": n_tok / max(n_word, 1),
        "tokens_per_char": n_tok / max(n_char, 1),
        "unk_rate": n_unk / max(n_tok, 1),
        "roundtrip_exact_rate": n_exact / len(paras),
        "zwj_paras": n_zwj_paras,
        "zwj_kept": n_zwj_kept,
    }


def chillu_test(tok):
    out = {}
    for w in CHILLU_WORDS:
        ids = tok.encode(w)
        pieces = [tok.piece_of(i) for i in ids]
        out[w] = {"n_tokens": len(ids), "pieces": pieces[:12], "roundtrip": tok.decode(ids) == w}
    return out


def main():
    paras = load_paras()
    total_chars = sum(len(p) for p in paras)
    print(f"eval paras={len(paras)} chars={total_chars}")

    toks = []
    for f in sorted(os.listdir(CAND)):
        if f.endswith(".model"):
            toks.append(SpmTok(os.path.join(CAND, f)))
        elif f.endswith(".json"):
            toks.append(HfJsonTok(os.path.join(CAND, f)))

    baselines = [
        ("google/muril-base-cased", "muril"),
        ("xlm-roberta-base", "xlmr"),
        ("ai4bharat/IndicBERTv2-MLM-only", "indicbertv2"),
        ("openai-community/gpt2", "gpt2"),
    ]
    for hub, alias in baselines:
        try:
            toks.append(TransformersTok(hub, alias))
        except Exception as e:
            print(f"SKIP baseline {hub}: {type(e).__name__}: {e}")

    results = {}
    for tok in toks:
        m = measure(tok, paras)
        m["chillu"] = chillu_test(tok)
        results[tok.name] = m
        print(f"{tok.name:34s} v={m['vocab_size']:7d} tok/word={m['tokens_per_word']:.3f} "
              f"tok/char={m['tokens_per_char']:.4f} unk={m['unk_rate']:.2e} rt={m['roundtrip_exact_rate']:.3f}")

    with open(os.path.join(EVALDIR, "tokenizer_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # markdown table
    rows = sorted(results.items(), key=lambda kv: kv[1]["tokens_per_word"])
    print("\n| tokenizer | vocab | tok/word | tok/char | unk rate | roundtrip |")
    print("|---|---|---|---|---|---|")
    for name, m in rows:
        print(f"| {name} | {m['vocab_size']} | {m['tokens_per_word']:.3f} | {m['tokens_per_char']:.4f} | "
              f"{m['unk_rate']:.1e} | {m['roundtrip_exact_rate']:.3f} |")


if __name__ == "__main__":
    main()

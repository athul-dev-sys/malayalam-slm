"""Train candidate tokenizers on the prepared corpus (Stage 1 assessment).

Candidates:
  A) SentencePiece Unigram, vocab 32k, char_coverage 1.0, identity normalization
  B) SentencePiece BPE,    vocab 32k, char_coverage 1.0, identity normalization
  C) SentencePiece Unigram, vocab 32k, char_coverage 0.9995 (conjunct-drop test)
  D) Byte-level BPE (HF tokenizers), vocab 32k (GPT-2 style pre-tokenizer)
  E/F/G) winner-family variants at 16k / 50k vocab (trained for the vocab-size decision)

Usage: python scripts/train_tokenizers.py
"""
import os

import sentencepiece as spm
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "data", "tokenizer_work")
OUT = os.path.join(ROOT, "artifacts", "tokenizer", "candidates")
os.makedirs(OUT, exist_ok=True)
CORPUS = os.path.join(WORK, "tokenizer_train.txt")

SPM_COMMON = dict(
    input=CORPUS,
    model_prefix=None,  # set per model
    character_coverage=1.0,
    normalization_rule_name="identity",  # we pre-NFC; do not let SPM rewrite chillu
    input_sentence_size=2_000_000,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=False,
    num_threads=os.cpu_count(),
)


def train_spm(name, model_type, vocab, coverage=1.0, byte_fallback=True):
    prefix = os.path.join(OUT, name)
    spm.SentencePieceTrainer.train(
        input=CORPUS,
        model_prefix=prefix,
        model_type=model_type,
        vocab_size=vocab,
        character_coverage=coverage,
        normalization_rule_name="identity",
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        byte_fallback=byte_fallback,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )
    print(f"trained {name}")


def train_bbpe(name, vocab):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train([CORPUS], trainer)
    path = os.path.join(OUT, f"{name}.json")
    tok.save(path)
    print(f"trained {name}")


if __name__ == "__main__":
    import sys

    phase = sys.argv[1] if len(sys.argv) > 1 else "core"
    if phase == "core":
        train_spm("spm_unigram_32k", "unigram", 32000, 1.0)
        train_spm("spm_bpe_32k", "bpe", 32000, 1.0)
        train_spm("spm_unigram_32k_cov9995", "unigram", 32000, 0.9995, byte_fallback=False)
        train_bbpe("bbpe_32k", 32000)
    elif phase == "vocab":
        train_spm("spm_unigram_16k", "unigram", 16000, 1.0)
        train_spm("spm_unigram_50k", "unigram", 50000, 1.0)
        train_bbpe("bbpe_16k", 16000)
        train_bbpe("bbpe_50k", 50000)
    else:
        raise SystemExit("phase must be 'core' or 'vocab'")

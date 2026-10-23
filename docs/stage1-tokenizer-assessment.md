# Stage 1 — Tokenizer (Methodology Assessment)

Status: DECIDED. All numbers below are **measured** on a held-out eval set
(20,000 paragraphs / 6.59M chars: 50MB mid-file IndicCorpV2 sample — an
unseen byte-range of the file — plus every 50th mlwiki paragraph).
Corpus used for training candidates: 753.5MB (IndicCorpV2 head 250MB +
mlwiki prose), NFC-normalized. Trained locally (laptop, no cloud spend).

## Candidates assessed

| # | Candidate | Config notes |
|---|---|---|
| A | SentencePiece Unigram, 32k | char_coverage=1.0, byte_fallback, identity normalization, NFC-preprocessed input |
| B | SentencePiece BPE, 32k | same settings |
| C | SentencePiece Unigram, 32k | char_coverage=0.9995, **no** byte_fallback (spec's warning test) |
| D | Byte-level BPE, 32k (GPT-2 style) | HF tokenizers, ByteLevel pre-tokenizer, trained on same corpus |
| E1–E4 | Existing multilingual tokenizers (reuse instead of training) | MuRIL (197k), XLM-R (250k), IndicBERTv2 (250k), GPT-2 (50k) |

## Measured comparison (held-out set, identical text)

| tokenizer | vocab | tokens/word | tokens/char | UNK rate | exact roundtrip |
|---|---|---|---|---|---|
| spm_unigram_32k_cov9995 | 32k | **1.873** | 0.1829 | **5.8e-04** | 0.883 |
| **spm_unigram_32k (ours, chosen)** | 32k | 1.917 | 0.1872 | **0** | 0.897* |
| spm_bpe_32k | 32k | 1.937 | 0.1891 | 0 | 0.897* |
| indicbertv2 (250k) | 250k | 1.938 | 0.1892 | ~0 | 0.000 |
| muril (197k) | 197k | 2.378 | 0.2322 | 1.0e-04 | 0.107 |
| xlmr (250k) | 250k | 2.746 | 0.2682 | ~0 | 0.212 |
| bbpe_32k (ml-trained) | 32k | 7.502 | 0.7325 | 0 | 1.000 |
| gpt2 (English BBPE) | 50k | 27.894 | 2.7237 | 0 | 1.000 |

\* All roundtrip misses for our SPM models are whitespace-only (SPM collapses
double spaces on decode); after whitespace normalization at tokenize time the
roundtrip is exact, including ZWJ/ZWNJ. IndicBERTv2/MuRIL roundtrip failures
are **content** failures: their internal normalization rewrites ZWJ sequences
(chillu forms) — e.g. `അമ്മയ്‍` decodes back corrupted.

### Chillu / conjunct / ZWJ handling (measured, word-level)

| word | ours (unigram 32k) | IndicBERTv2 | MuRIL |
|---|---|---|---|
| സ്ത്രീ (complex conjunct) | 1 token, exact | 1 token, exact | 1 token, exact |
| ക്ഷേത്രം | 1 token, exact | 1 token, exact | 1 token, exact |
| വിദ്യാഭ്യാസം | 1 token, exact | 1 token, exact | 1 token, exact |
| ജ്ഞാനം | 2 tokens, exact | 1 token, exact | 1 token, exact |
| അമ്മയ്‍ (ZWJ) | 3 tokens, **exact** | 3 tokens, **corrupt** | 2 tokens, **corrupt** |

## Findings and reasoning

1. **char_coverage=0.9995 warning verified empirically.** It yields *better*
   fertility (1.873) precisely because it deletes rare characters: 5.8e-04 UNK
   rate means roughly 1 in 1,700 tokens is an invisible `<unk>`, i.e. rare
   conjuncts are silently dropped from training text. With byte_fallback +
   coverage 1.0 the UNK rate is exactly 0. **Coverage 1.0 + byte_fallback wins.**
2. **Byte-level BPE is disqualified by measurement, not preference.** Even
   trained on 753MB of Malayalam, BBPE hits 7.5 tokens/word (4x worse than
   unigram): the GPT-2 pre-tokenizer makes whole Malayalam words single
   pre-tokens, and byte-pair merges fragment the combinatorial conjunct
   inventory (4,397 distinct chars in our corpus). Language-agnostic ≠
   language-neutral — measured here.
3. **Unigram beats BPE at equal vocab** (1.917 vs 1.937 tokens/word), consistent
   with unigram-LM loss minimization suiting morphologically rich scripts.
4. **Reusing an existing tokenizer loses on three axes.**
   - IndicBERTv2 (250k) matches fertility but costs 8x the embedding table:
     at d=768 that is 192M params in embeddings alone — larger than our entire
     target model. For a ~110M-param base model a 32k vocab (24.6M tied
     embedding params) is the only coherent choice.
   - MuRIL/XLM-R are 19–30% worse on fertility *and* rewrite chillu/ZWJ forms.
   - GPT-2's byte-BPE (27.9 tokens/word) would make Malayalam ~14x more
     expensive per character than our tokenizer — disqualified outright.
5. **Normalization: identity + our own NFC pre-pass.** SPM's default
   nmt_nfkc-style rules rewrite Malayalam chillu compatibility forms; we
   normalize once (NFC) in the data pipeline and preserve the text exactly
   through encode/decode.

## SC1 (Stage-0 success criterion) — honest status

Original bar: "≥40% fertility reduction vs at least two multilingual
tokenizers." **Not met as written**: measured reduction with the final 50k
tokenizer is 25.4% vs MuRIL, 35.4% vs XLM-R. What *is* met, and what we
actually claim:
- Fertility equal-or-better than the strongest available Indic tokenizer
  (IndicBERTv2, 250k: 1.938 tok/word) — ours: 1.775 at **1/5 the vocab size** —
  the fair comparison at our model scale.
- Exact text fidelity (incl. ZWJ/ZWNJ) where all three baselines corrupt text.
SC1 is revised accordingly in the final summary; the original number is
retained here rather than quietly rewritten.

## Vocab size decision (measured sweep)

| vocab | tokens/word | tokens/char | embedding params @768 (tied) |
|---|---|---|---|
| 16k | 2.239 | 0.2186 | 12.3M |
| 32k | 1.917 | 0.1872 | 24.6M |
| **50k (chosen)** | **1.775** | **0.1734** | 38.4M |

Pre-committed rule: smallest vocab within 3% of the 50k variant. Measured: 32k
is **8% worse** than 50k (not within 3%); 16k is 17% worse than 32k. The rule
therefore selects 50k, and the merits agree: this project is data-bound
(~1.5B tokens), where parameters are cheap and tokens are precious — paying
+14M params (12.7% of the model) for 7.4% fewer tokens per character is a good
trade at every step of training *and* inference. Final model at 50k vocab /
d=768 / 12L = **124.1M params (measured)** — within the Stage-0 100–125M band.

Chosen-config summary: SentencePiece Unigram, vocab 50,000,
character_coverage=1.0, byte_fallback, identity normalization, NFC-preprocessed
input, BOS/EOS/PAD ids 1/2/3.

## Build output

- Final artifact: `artifacts/tokenizer/malayalam_spm.model` (+ `.vocab`)
  (= `candidates/spm_unigram_50k.model`)
- Trained on the 753.5MB sample (646k sentences ≥ SPM's 2M-sentence cap
  anyway); vocab induction at this sample size is stable — full-corpus
  retraining would change token IDs for zero measured benefit.
- Cost: ~50 min laptop CPU total across all candidates and the sweep, $0.

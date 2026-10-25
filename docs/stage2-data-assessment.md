# Stage 2 — Data Pipeline (Methodology Assessment)

Status: **COMPLETED (test-scale re-bootstrap) on 2026-10-25.** The original
cleaned corpora were absent and no S3 credentials were available, so the
pipeline was re-run from bounded public-source slices on an 8 GB machine.
Measured results, provenance and validation evidence:
[`docs/data_report.md`](data_report.md). The assessment below is the original
methodology record; §7 appends what actually happened during execution.

## 1. Source enumeration (estimated BEFORE ingestion, per spec)

| Source | Access path | Raw size (measured) | Expected clean tokens | AWS Open Data presence |
|---|---|---|---|---|
| Malayalam Wikipedia (20261001 dump) | dumps.wikimedia.org | 202 MB bz2 (measured) → 192M chars prose | ~40M tokens | dumps.wikimedia.org is on AWS ODR (us-east mirror) — irrelevant at 202MB; local download is cheaper than sync setup |
| IndicCorpV2 (ai4bharat) | HF dataset `data/ml.txt`, ungated | **26.32 GB** (measured content-length) | ~1.3B tokens (projection; final measured below) | Hosted on HF's S3-backed CDN — effectively "already on AWS"; we stream, no redundant copy |
| CC100 ml | data.statmt.org | 870 MB xz (measured) | ~0.35B tokens (projection) | Not on ODR; streamed + decompressed on the fly |
| OSCAR ml | HF, gated (auth required) | — | — | Rejected: gating + CC-derived overlap with IndicCorpV2 |
| CultureX/SMC ml mixes (HF community) | unversioned community uploads | — | — | Rejected: unverifiable provenance/mix composition for a foundation corpus |
| PMIndia, news sites | small / crawling effort | <100MB | <10M tokens | Rejected: negligible volume for the effort |

Decision: **mlwiki + IndicCorpV2 + CC100**. CC100's marginal contribution is
measured directly by cross-source dedup overlap in the final report — if
IndicCorpV2 subsumes it, the numbers will say so.

## 2. Filtering strategy (decided, with reasoning)

- **Language ID: Unicode script-ratio filter, not a learned LID.** Malayalam
  script (U+0D00–U+0D7F) is distinctive; a script-ratio threshold (≥70% of
  non-space chars in the Malayalam block) is deterministic, ~100x faster than
  fasttext on 30GB, and has no model to install/build (fasttext has no wheel
  for py3.13 — build attempt failed). Borderline mixed-script paragraphs are
  handled by the same ratio. Tradeoff: cannot distinguish e.g. Tamil written
  in Malayalam script — vanishingly rare; accepted.
- **Code-mixing (Manglish / English): DROP.** Latin-letter ratio > 25% of
  non-space chars → dropped (measured: only 0.08% of wiki paragraphs). Keeping
  a separate Manglish channel would require romanization-aware tokenization
  and doubles eval surface for zero Stage-0 value. Loanwords transliterated
  into Malayalam script pass naturally.
- **Junk filter (thresholds measured against distribution):** paragraph length
  60–8192 chars; digit ratio ≤ 30%; ≤2 URLs; repetition guards
  (unique-char ratio ≥ 0.035 — calibrated from measured Malayalam prose
  distribution: median 0.122, p5 0.05, glyph-spam 0.034; an initial English-
  style 0.15 threshold wrongly dropped 53% of wiki prose — caught and fixed
  by measurement, not assumption). Unique-word ratio ≥ 0.35.
- **Encoding fix:** UTF-8 with errors='replace' at ingest; NFC normalization
  for every paragraph (Malayalam has multiple canonical forms for chillu
  sequences; NFC gives a consistent surface).

## 3. Dedup (decided against corpus size)

Exact + normalized-exact paragraph-level hashing (blake2b-64 of
Malayalam-chars-only string — punctuation/whitespace-insensitive).
MinHash/LSH rejected as over-engineering at this scale: the corpus is
paragraph-streamed (26GB), the dominant duplication mode is verbatim
boilerplate repeats, and shingling 30M paragraphs buys marginal recall for
hours of complexity. Cross-source overlap (CC100 vs IndicCorpV2) is caught by
the same global hash pass in the split phase. This decision stands *because*
the corpus is CC-derived where dupes are verbatim; a paraphrase-heavy corpus
would need minhash — stated explicitly.

## 4. Split

Doc-level shuffle (seed 1337), 98/1/1. Wiki and CC100 keep document
boundaries; IndicCorpV2 paragraphs have no doc metadata in the released file
(paragraph-level split; near-identical paragraphs are globally deduped, so
split leakage is limited to non-verbatim paraphrase — accepted, documented).
Train is deduped against val/test by construction (single global hash pass
before splitting).

## 5. Compute choice for this stage

All cleaning is CPU/streaming and ran on the laptop ($0): no GPU, no EC2
needed — data work never touched cloud spend. Peak RAM: hash sets for tens of
millions of paragraphs (≈5–6 GB of 25 GB), measured safe.

## 6. Disk constraint (honest limitation)

The laptop has 23 GB free during the run; full IndicCorpV2 clean output
(~22 GB) plus CC100 (~5 GB) plus tokenized bins would exceed it. Mitigation:
IndicCorpV2 is ingested as a *prefix* of the stream (disk-triggered stop),
recorded in the data report. The prefix is a contiguous slice of the same
shuffled-crawl distribution; the token-count impact is quantified in the
report. With S3 available (Stage 8), the full corpus fits trivially —
recorded as the delta between laptop execution and cloud execution, not as a
methodology choice.

## 7. Execution notes — 2026-10-25 re-bootstrap

The cleaned corpora listed in the handover (wiki ~450 MB, CC100 ~5.5 GB,
IndicCorp ~23 GB) were not present on this machine, and no S3 credentials or
bucket were configured. The pipeline was therefore re-run from public sources
with bounds sized for 8 GB RAM (the original run used a 25 GB machine). This
is a **test-scale corpus, not a replacement for the full Stage 2 run**.

What ran:

- `scripts/extract_wiki.py` on the full `mlwiki-20261001` dump
  (201,650,264 B): 88,066 articles, 384,642 paragraphs, 191,846,320 chars.
- `scripts/build_corpus.py` unchanged on all sources. CC100's released
  `ml.txt` is one paragraph per line with no blank lines, so the stream was
  piped through `sed 'G'` and cleaned with `--doc-blocks`; it was bounded to
  the first 4,000,000 lines of the injected stream (2M source lines).
  IndicCorpV2 was bounded to the first 4 GiB, trimmed to the last complete
  line (4,294,967,199 B), and split into two line-aligned halves cleaned in
  parallel.
- `scripts/split_tokenize.py` was rewritten for bounded-memory operation:
  pass 1 globally dedups and writes length-prefixed docs to a temp file;
  pass 2 shuffles split labels and tokenizes in ≤2M-char batches straight
  into uint16 bins. It also fixes the `\x1e`-is-whitespace parsing bug
  (`line.strip()` erased the separator); `scripts/eval_model.py` had the same
  bug and received the same `rstrip("\r\n")` fix.
- One deviation from the written split spec: seed 1337 is preserved, but the
  permutation uses NumPy PCG64 instead of Python `random.shuffle`, so exact
  split membership is deterministic for this pipeline but may differ from the
  original generator.

Measured outcome (details in `docs/data_report.md`):

| Metric | Result |
|---|---:|
| Clean paragraphs after per-source filtering | 5,742,669 |
| Cross-source duplicates removed globally | 112,371 (1.96%) |
| Documents after global dedup | 5,420,439 |
| Unique chars | 1,735,137,044 |
| Total uint16 tokens | **316,794,142** |
| Train / val / test docs | 5,312,031 / 54,204 / 54,204 |
| Train / val / test tokens | 310,460,780 / 3,152,098 / 3,181,264 |
| Held-out fertility (content tokens/word) | 1.7859 |
| Stage 2 split+tokenize wall time | 17.3 min |

All integrity checks passed: bin sizes equal token counts ×2, max id < 50,000,
BOS/EOS counts equal document counts, zero UNK, 54,204 `\x1e` separators in
`test.txt`, and re-encoding `test.txt` reproduces `test.bin` exactly. The
tokenized bins load through `src/dataloader.TokenMemmap`, and a 2-step CPU
training smoke test on the real train bin passed the fresh-model loss gate
(10.8695 vs ln(50,000)=10.8198).

Consequence for Stage 4: 316.8M tokens is 2.88 tokens/param for a 110M model,
below the ~1B-token threshold in `docs/stage4-sizing.md`. The current bins
support the local 2,000-step gate and end-to-end validation only; the full
corpus must be rebuilt (same pipeline, no slice bounds) before the final cloud
run, or the smaller-model fallback adopted.

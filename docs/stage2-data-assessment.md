# Stage 2 — Data Pipeline (Methodology Assessment)

Status: pipeline BUILT and running; final counts appended to
`docs/data_report.md` when streams complete. Everything below marked
"measured" comes from the live run; projections are labeled as such.

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

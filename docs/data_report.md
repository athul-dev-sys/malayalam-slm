# Stage 2 Data Report — Measured Results

Status: **COMPLETED (test-scale re-bootstrap)** on 2026-10-25. The Stage 2
command ran end-to-end and its outputs passed every integrity check below.
This report records exact measured numbers; no projections are mixed in.

Important scope caveat: **this run produced 316.8M tokens, not the ~1.7B-token
full-corpus target.** The original raw/cleaned corpora were absent from this
machine, so the pipeline was re-bootstrapped from bounded public-source slices
(the mission brief permits a smaller IndicCorp slice for testing). The
pipeline, splitter and bins are final and validated; the *volume* is
test-scale and must be expanded before the full Stage 4 pretraining run (see
§9).

## 1. What was produced

| Artifact | Size | Contents |
|---|---|---|
| `data/tokenized/train.bin` | 620,921,560 B | 310,460,780 uint16 token ids |
| `data/tokenized/val.bin` | 6,304,196 B | 3,152,098 uint16 token ids |
| `data/tokenized/test.bin` | 6,362,528 B | 3,181,264 uint16 token ids |
| `data/tokenized/test.txt` | 48,285,507 B | 54,204 held-out documents, `\x1e`-separated |
| `data/tokenized/stats.json` | 1,532 B | per-source, dedup, split and token stats |

SHA-256:

| File | SHA-256 |
|---|---|
| `train.bin` | `98d5aa24d45170bfdd78f1afcc3a8b08698af9e593d3d8f2eb55ec5b5190547a` |
| `val.bin` | `09c1d471197643e347a6910980db0abe25e123e54e0845fa939651bc25b2a229` |
| `test.bin` | `c8942449b0c68b515ef8ff76751f69b149dff7b4097496364dc44b0beec962a9` |
| `test.txt` | `72053656e25d6e36d8deca6d14669678ab33fec168f3f4a992dceddb406eace3` |
| `stats.json` | `dc139db01759f246d44737e771f4f7a7a5c29d8f150325c68a9a2d1deebb13b8` |

Each document is encoded independently with BOS prepended and EOS appended.
Bins are plain concatenations of documents; `src/dataloader.py` samples random
fixed-length windows from them.

## 2. Source provenance and scope actually used

| Source | Snapshot / URL | Raw download | Scope used in this run |
|---|---|---|---|
| Malayalam Wikipedia | `mlwiki-20261001-pages-articles.xml.bz2` from `dumps.wikimedia.org` | 201,650,264 B | **Full dump** (88,066 articles extracted) |
| CC100 ml | `https://data.statmt.org/cc-100/ml.txt.xz` | 870,518,796 B (≈8.05 GB decompressed) | **Bounded prefix**: first 4,000,000 lines of the blank-line-injected stream (2,000,000 source lines); 1,837,516 non-empty paragraph blocks observed |
| IndicCorpV2 ml | `https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/ml.txt` | 26,319,980,640 B total | **First 4 GiB** (`bytes=0-4294967295`), trimmed to the last complete line → 4,294,967,199 B, split into two line-aligned ~2 GiB halves (16.3% of the file) |

Raw intermediate sizes on disk:

| File | Bytes |
|---|---|
| `mlwiki_prose.txt` (extracted prose) | 508,601,938 |
| `indiccorp_4gib.txt` (trimmed) | 4,294,967,199 |
| `indiccorp_part1.txt` | 2,147,484,388 |
| `indiccorp_part2.txt` | 2,147,482,811 |
| `cc100_ml.txt.xz` (compressed) | 870,518,796 |

Wiki extraction was performed with the existing `scripts/extract_wiki.py`:
`articles=88066 paragraphs=384642 chars=191846320`.

## 3. Cleaning (existing `scripts/build_corpus.py`, unchanged)

Applied to every source: NFC normalization, paragraph length 60–8192 chars,
Malayalam script ratio ≥ 70%, Latin/Manglish ratio ≤ 25%, digit ratio ≤ 30%,
≤ 2 URLs, unique-char ratio ≥ 0.035, unique-word ratio ≥ 0.35, and
per-source exact dedup on the Malayalam-only paragraph text (blake2b-64).

### Per-source cleaning stats

| Source | Paragraphs in | Kept | Dup (within source) | Length | LID | Char-rep | Word-rep | URL | Manglish |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CC100 prefix | 1,837,516 | 845,561 | 145,007 | 781,243 | 65,003 | 1 | 1 | 1 | 699 |
| IndicCorp part1 | 2,692,011 | 2,267,326 | 7,180 | 400,535 | 3,963 | 12,928 | 10 | 32 | 37 |
| IndicCorp part2 | 2,696,308 | 2,269,616 | 7,094 | 402,668 | 3,990 | 12,876 | 15 | 19 | 30 |
| Wiki | 384,642 | 360,166 | 2,741 | 159 | 18,289 | 2,550 | 108 | 313 | 316 |
| **Total** | **7,610,477** | **5,742,669** | **162,022** | **1,584,605** | **91,245** | **28,355** | **134** | **365** | **1,082** |

On-disk clean output: 4,920,825,373 B total. `stats.json` `bytes_out` values
exclude the extra byte per newline added by Windows CRLF text-mode writes;
the tokenizer reads with universal newlines, so no `\r` reached any document.

Clean-output composition after cross-source dedup:

| Source | Docs | Share of docs | Chars | Share of chars |
|---|---:|---:|---:|---:|
| IndicCorp halves combined | 4,495,298 | 82.9% | 1,405,366,979 | 81.0% |
| CC100 prefix | 845,561 | 15.6% | 191,402,314 | 11.0% |
| Wiki | 79,580 | 1.5% | 138,367,751 | 8.0% |
| **Total** | **5,420,439** | **100%** | **1,735,137,044** | **100%** |

## 4. Global dedup and 98/1/1 split

- Single global pass over all four cleaned sources in sorted order
  (`cc100`, `indiccorp_part1`, `indiccorp_part2`, `wiki`) using the same
  normalized hash as cleaning (blake2b-64 over Malayalam-only characters).
- **112,371 cross-source duplicate paragraphs removed** (1.96% of the
  5,742,669 cleaned paragraphs). Because CC100 runs first, it loses none;
  IndicCorp part1 loses 14,336 (0.63%), part2 27,308 (1.20%), and wiki
  70,727 (**19.6% of its cleaned paragraphs**, reflecting heavy wiki/web
  overlap already present in IndicCorpV2).
- 5,420,439 documents remained (1,735,137,044 chars). Wiki documents that
  lost all their paragraphs to cleaning plus global dedup were dropped
  entirely (88,066 → 79,580 documents).
- Doc-level shuffle with seed `1337`, then a 98/1/1 slice:
  `n_val = n_test = max(1000, n_docs // 100) = 54,204` documents each.

| Split | Docs | Tokens | Tokens share | Chars | tok/char (incl. BOS/EOS) |
|---|---:|---:|---:|---:|---:|
| Train | 5,312,031 | 310,460,780 | 98.00% | 1,700,440,937 | 0.18258 |
| Val | 54,204 | 3,152,098 | 0.995% | 17,247,190 | 0.18276 |
| Test | 54,204 | 3,181,264 | 1.004% | 17,448,917 | 0.18232 |
| **Total** | **5,420,439** | **316,794,142** | **100%** | **1,735,137,044** | **0.18258** |

Document-boundary semantics: wiki articles and CC100 paragraphs retain true
document boundaries; IndicCorpV2 has no document metadata in the released
file, so each blank-line paragraph is treated as one document (4,495,298 of
the 5,420,439 documents). Dedup is global before splitting, so the test split
has no verbatim paragraph overlap with train.

## 5. Tokenizer and measured fertility

Tokenizer: `artifacts/tokenizer/malayalam_spm.model` — 50k SentencePiece
Unigram, byte fallback, NFC. Special ids: unk=0, bos=1, eos=2, pad=3.

Held-out test split (54,204 docs, 17,448,917 chars):

- Content tokens (excluding BOS/EOS): 3,072,856
- Whitespace words: 1,720,601
- **1.7859 content tokens/word** and **0.1761 content tokens/char**

This independently reproduces the tokenizer decision's 1.775 tokens/word on
held-out corpus text (Stage 1 benchmark), confirming native fertility on real
training data rather than only the benchmark sample.

## 6. Validation evidence

All checks were run against the final files; all passed.

| Check | Result |
|---|---|
| Bin file size = tokens × 2 | train 620,921,560 B, val 6,304,196 B, test 6,362,528 B — exact |
| Max token id < 50000 | train 49,999; val 49,856; test 49,926 |
| BOS count = EOS count = doc count | train 5,312,031; val 54,204; test 54,204 |
| Unknown tokens | 0 in all splits |
| `test.txt` separators | 54,204 (`\x1e`), docs 54,204, chars 17,448,917 (matches stats) |
| Re-encode `test.txt` → compare to `test.bin` | exact match, 3,181,264 tokens |
| `src.dataloader.TokenMemmap.get_batch` | (4,128) int64 windows from all three bins, ids in range |
| Fresh-model loss on real train bin (tiny preset, CPU) | 10.8695 vs ln(50000)=10.8198, delta +0.0497 (gate 0.25) |
| 2-step CPU training smoke test | completed; sampled text decoded |

The train bin was scanned in 50M-token chunks, so validation did not require
loading the 621 MB memmap fully into RAM.

## 7. Deviations from the original Stage 2 assumptions

1. **Streaming two-pass splitter instead of in-memory.** `split_tokenize.py`
   was rewritten: pass 1 globally dedups and writes length-prefixed documents
   to a temp file; pass 2 assigns shuffled split labels and tokenizes in
   bounded batches. Peak RAM is O(unique paragraphs + one batch), not
   O(corpus). Hash semantics are unchanged.
2. **Split RNG.** Seed 1337 is preserved, but the permutation uses NumPy
   PCG64 (`np.random.default_rng(1337).permutation`) rather than Python
   `random.shuffle`. Consequently split membership is deterministic for this
   pipeline but may differ from the original generator's assignment.
3. **Duplicate retention order.** Global dedup keeps the first occurrence in
   source order (CC100 → IndicCorp part1 → part2 → wiki). The original
   one-pass loader's retention order was not documented; this only changes
   which copy of a duplicate survives, not the unique-text set.
4. **CC100 line format.** The released `ml.txt` is one paragraph per line with
   no blank lines, which the cleaner would have merged into a single document.
   The stream was piped through `sed 'G'` to inject blank lines and cleaned
   with `--doc-blocks`, making each line a paragraph/document.
5. **`\x1e` parser bug fixed.** Python treats U+001E as whitespace, so
   `line.strip() == "\x1e"` never matched and all doc-separated text collapsed
   into one document. Both `scripts/split_tokenize.py` and
   `scripts/eval_model.py` now compare `line.rstrip("\r\n")` before stripping.
6. **IndicCorp slice, not full file.** 4 GiB of 26.32 GB (16.3%) were used
   because this machine has 8 GB RAM and the full clean outputs would also
   exceed the free disk budget. The slice is contiguous.
7. **CC100 prefix, not full file.** 2,000,000 source lines were cleaned out of
   the full 8.05 GB decompressed stream.

## 8. Environment

- Windows 10.0.26200, Git Bash, Python 3.13.2
- torch 2.7.1+cu118, NumPy 2.2.6, sentencepiece 0.2.2
- 8 GB RAM (27.3 GB commit limit with pagefile), ~12 CPU threads, SSD
- Stage 2 wall time: **17.3 min** for pass 1 + pass 2 (excluding downloads and
  cleaning, which took ~20 min for the IndicCorp halves and ~6 min for CC100)
- Data root: `C:\mal_slm_data` (junctioned to `malayalam-slm/data/` to avoid
  OneDrive syncing multi-GB files)

## 9. Limitations and implication for Stage 4

- **316.8M train+val+test tokens is test-scale.** The Stage 0/4 plan targets
  ~1.7B unique tokens for a 110M-parameter model (~20 tokens/param). At 110M
  params this corpus provides only **2.88 tokens/param**, and Stage 4's own
  decision table says <1B tokens triggers the "smaller model + single epoch"
  fallback rather than the 110M primary strategy.
- The full-corpus path is unchanged and now proven: stream/download full
  IndicCorpV2 (26.32 GB) and full CC100 (8.05 GB decompressed) plus wiki,
  clean with `build_corpus.py`, then rerun `split_tokenize.py`. Extrapolating
  the measured IndicCorp yield over its 6.13x larger raw slice gives roughly
  1.6B IndicCorp tokens; adding full CC100 (Stage 2 projection ~0.35B tokens)
  and wiki (~40M) suggests ~2B total, around the ~1.7B Stage 0 projection.
  The final number must be measured, not assumed.
- Source skew: 81% of characters come from the IndicCorp slice, 11% from a
  CC100 prefix, 8% from wiki. Domain/time coverage is therefore narrower than
  the intended three-source mix.
- Dedup is exact/normalized only (no near-duplicate or template detection);
  this matches the Stage 2 decision and its stated constraint.
- No checksums were computed for the 4.9 GB of clean text; the tokenized
  artifacts above are checksummed. Clean files can be regenerated from the
  documented sources and commands.

## 10. Reproduction commands

```bash
# 0) Data root (external, then junction/link into the repo)
#    C:\mal_slm_data/{raw,clean,tokenized,work}

# 1) Wiki (full dump)
curl -L -o data/raw/mlwiki-20261001-pages-articles.xml.bz2 \
  https://dumps.wikimedia.org/mlwiki/20261001/mlwiki-20261001-pages-articles.xml.bz2
python scripts/extract_wiki.py data/raw/mlwiki-20261001-pages-articles.xml.bz2 \
  data/raw/mlwiki_prose.txt
python scripts/build_corpus.py clean data/raw/mlwiki_prose.txt data/clean/wiki

# 2) CC100 (bounded prefix: 2M source lines, blank line injected per line)
curl -L -o data/raw/cc100_ml.txt.xz https://data.statmt.org/cc-100/ml.txt.xz
xz -dc data/raw/cc100_ml.txt.xz | sed 'G' | head -n 4000000 | \
  python scripts/build_corpus.py clean - data/clean/cc100 --doc-blocks

# 3) IndicCorpV2 (first 4 GiB, trimmed to last complete line, split in half)
curl -L -r 0-4294967295 -o data/raw/indiccorp_4gib.txt \
  https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/ml.txt
python - <<'PY'
import os
p = "data/raw/indiccorp_4gib.txt"
with open(p, "r+b") as f:
    pos = os.path.getsize(p)
    while pos > 0:
        pos -= 1
        f.seek(pos)
        if f.read(1) == b"\n":
            f.truncate(pos + 1)
            break
PY
split -n l/2 -d data/raw/indiccorp_4gib.txt data/raw/indiccorp_part
mv data/raw/indiccorp_part00 data/raw/indiccorp_part1.txt
mv data/raw/indiccorp_part01 data/raw/indiccorp_part2.txt
python scripts/build_corpus.py clean data/raw/indiccorp_part1.txt data/clean/indiccorp_part1
python scripts/build_corpus.py clean data/raw/indiccorp_part2.txt data/clean/indiccorp_part2

# 4) Final Stage 2 phase (global dedup + 98/1/1 split + uint16 memmaps)
PYTHONIOENCODING=utf-8 python scripts/split_tokenize.py \
  artifacts/tokenizer/malayalam_spm.model
```

The full-corpus run uses exactly the same steps without the `head`/`-r` bounds
and without splitting IndicCorp.

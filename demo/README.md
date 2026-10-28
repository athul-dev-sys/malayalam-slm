# Malayalam SLM — local demo

A self-contained FastAPI + HTML demo of the current artifacts. No cloud, no CDN,
no external JS. Everything shown is read from measured local files or computed
live in the request.

## Run

Double-click `demo/run_demo.bat` (starts the server and opens the browser), or:

```bash
PYTHONIOENCODING=utf-8 python -m uvicorn demo.app:app --host 127.0.0.1 --port 8010
# open http://127.0.0.1:8010
```

Port 8000 is occupied by another app on this machine, so the demo uses 8010.
Dependencies: `fastapi`, `uvicorn`, `sentencepiece`, `torch`, `transformers`
(tokenizer files only for the baselines).

## What is inside

| Tab | What it does |
|---|---|
| **Tokenizer** | Live comparison of the 50k SPM Unigram tokenizer against MuRIL, XLM-R and IndicBERTv2 tokenizers on pasted Malayalam. Shows token chips, fertility, UNK rate and exact-decode roundtrip. Also pulls random held-out paragraphs from `data/tokenized/test.txt`. |
| **Generate** | Prompt + temperature/top-k/max-token controls against the best-validation demo checkpoint, reloaded automatically as training checkpoints advance. |
| **Training & Data** | Loss curve and validation points parsed from `checkpoints/demo/train.log`, corpus/source/dedup stats from `data/tokenized/stats.json`, latest training samples, and the full tokenizer benchmark from `eval/tokenizer_comparison.json`. |

## Current artifacts (measured, 2026-10-26)

Tokenized demo corpus — `scripts/split_tokenize.py` with a bound on IndicCorp:

| Split | Tokens | Docs | Chars |
|---|---:|---:|---:|
| train | 578,343,997 | 4,284,937 | 3,007,630,974 |
| val | 5,956,063 | 43,723 | 30,954,864 |
| test | 5,893,359 | 43,723 | 30,589,343 |
| **total** | **590,193,419** | **4,372,383** | **3,069,175,181** |

Sources used: full cleaned CC100 (2.02B chars), full cleaned wiki (0.13B chars),
and the first 2.5 GiB of the 23.3 GB cleaned IndicCorp file (0.92B chars).
265,719 cross-source duplicate paragraphs were removed globally. Split is
doc-level, NumPy PCG64 seed 1337, 98/1/1. Tokenization took 6.6 min.

This is a **bounded demo slice**, not the full cleaned corpus (the complete
clean files are ~29 GB, an estimated ~1.8B tokens — not yet tokenized). The
earlier 316.8M-token test-scale run is documented in `docs/data_report.md`.

Tokenizer: `artifacts/tokenizer/malayalam_spm.model` — 50,000 SPM Unigram,
character_coverage=1.0, byte fallback, NFC input. On the benchmark corpus it
measures 1.775 content tokens/word (MuRIL 2.378, IndicBERTv2 1.938, XLM-R 2.746),
0 UNK and exact Malayalam roundtrip (SPM only collapses repeated spaces).

Demo model — `checkpoints/demo/`:

```text
preset      tiny (4 layers, 4 heads, 256 d_model)
params      16.1M (tied embeddings, 50k vocab)
context     512
batch       8 micro-batch × 12 grad-accum = 49,152 tokens/step
steps       8,000  (~393M tokens, 0.68 epoch of train.bin)
optimizer   AdamW, lr 6e-4, warmup 200, cosine to 10%, clip 1.0, bf16
hardware    local RTX 4060 Laptop 8 GB (3.75 GB VRAM)
throughput  ~92k tokens/s early, ~66k by the end (laptop thermal throttling)
wall time   98.4 min measured; ~0 cloud cost
result      final val loss 4.7567 (best, step 8000)
```

Sanity gates: initial loss 10.8230 vs ln(50000)=10.8198 (delta +0.0033); the
overfit-batch gate was passed in `docs/sanity-logs.md`. Final validation loss
4.7567 (best, step 8000). The UI loads the best-validation checkpoint (checked
by mtime); inference runs on CPU to keep the GPU free for training.

## Held-out model evaluation (Stage 5)

`scripts/eval_model.py --n-docs 300` on `data/tokenized/test.txt` (300 docs,
330,082 chars), full result in `eval/results.json`:

| model | params | tokens/char | loss (nats/token) | bits/char |
|---|---:|---:|---:|---:|
| Malayalam SLM (ours, causal) | 16.1M | 0.195 | 4.6685 | **1.3101** |
| XLM-R base (MLM pseudo) | 278.3M | 0.272 | 1.6414 | 0.6433 |
| IndicBERTv2 (MLM pseudo) | 278.3M | 0.197 | 3.4420 | 0.9776 |
| MuRIL base (MLM pseudo) | 237.8M | 0.237 | 4.3282 | 1.4818 |

Ours is exact causal log-loss converted to bits per character. The baselines are
**MLM pseudo-log-loss** (15% random token masking, bidirectional context), a
different estimator that scores only masked tokens — report side by side, never
as a like-for-like win. The honest read: a 16.1M causal model trained for
98 minutes already beats MuRIL's pseudo-BPC on this test set, but trails the
278M-parameter XLM-R and IndicBERTv2, which is expected at this budget.

Ten sampled completions are saved in `eval/human_eval_samples.txt`.

## Suggested 3-minute demo flow

1. **Tokenizer tab** — leave the default clean paragraph, click *Analyze*: 50k
   vocab beats 197k–250k baselines on tokens/word, with 0 UNK and no lost
   Malayalam characters. Then click *Random corpus sample* to show it runs on
   real held-out text.
2. **Training & Data tab** — point out the 590.2M measured tokens, dedup counts,
   the loss/val curve, the tokenizer benchmark, and the checkpoint-evaluation
   table (ours vs MLM baselines, with the method caveat printed). Note the
   bounded-slice caveat (the full clean corpus is ~29 GB and not yet tokenized).
3. **Generate tab** — generate from the final step-8000 checkpoint: coherent
   Malayalam news-style text at ~0.9 s per 96 tokens on CPU. All local, $0.

Honest caveat to say out loud: this checkpoint is a 16.1M-parameter demo model
trained for 98 minutes on a slice, not the 110M-param Stage-4 model.

## Rebuild / retrain

```bash
# tokenize the bounded demo slice (indiccorp.txt capped at 2.5 GiB)
PYTHONIOENCODING=utf-8 python scripts/split_tokenize.py \
  artifacts/tokenizer/malayalam_spm.model \
  --sources cc100.txt.gz,indiccorp.txt,wiki.txt.gz \
  --max-mb indiccorp.txt=2560

# train the demo checkpoint
PYTHONIOENCODING=utf-8 python scripts/train.py \
  --preset tiny --block-size 512 --batch-size 8 --grad-accum 12 \
  --steps 8000 --lr 6e-4 --warmup-steps 200 \
  --data-dir data/tokenized --out-dir checkpoints/demo \
  --val-interval 250 --ckpt-interval 250 --sample-interval 250
```

`data/` and `checkpoints/` are gitignored; the demo needs the local artifacts
above to show live numbers.

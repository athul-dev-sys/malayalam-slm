# Stage 3 — Training Pipeline (Methodology Assessment)

Status: implemented + sanity-checked locally (logs below). The spec's
"CURRENT STATE" claimed an existing transformer — the repo was empty, so the
model was implemented as part of this stage (flagged in Stage 0).

## 1. Architecture decisions

| Decision | Options considered | Choice + reasoning |
|---|---|---|
| Attention | manual softmax vs `F.scaled_dot_product_attention` | SDPA (flash kernel on CUDA when available) — same math, faster, less code to get wrong |
| Norm placement | post-LN vs pre-LN | Pre-LN (GPT-2 style): stable at small scale without warmup tricks, better grad flow at 12 layers |
| Embeddings | tied vs untied lm_head | **Tied**: at 32k vocab × 768, tying saves 24.6M params (22% of the model) — significant at our scale; GPT-2 precedent; untied reserved for the (unlikely) 50k-vocab path |
| Positions | learned absolute vs RoPE | Learned absolute: fine at ctx ≤ 1024, one less failure mode; RoPE's extrapolation benefit is out of scope (no long-context target in Stage 0) |
| Activation | GELU vs SwiGLU | GELU 4x (GPT-2): SwiGLU buys ~equivalent quality at 8/3 d_ff with more plumbing; simplicity wins at 110M |
| Dropout | 0.0 vs 0.05 | 0.0 default: modern short-run practice; single-epoch-ish data makes regularization less critical; configurable |
| Init | N(0, 0.02) + residual scaling | GPT-2 scheme (0.02/sqrt(2·n_layer) on residual projections) — verified by the ln(vocab) sanity check |

## 2. Context length (512 vs 1024 vs longer)

Data reality: wiki paragraphs average ~500 chars (~95 tokens); IndicCorpV2
paragraphs similar; documents (wiki articles, CC100 pages) are the natural
units. Tradeoff as required:

- VRAM is NOT the binding constraint (Stage 0 principle: don't size from the
  8GB laptop — A10G 24GB on cloud). Attention cost scales ~T² but at T=1024,
  d=768 it is still small for a 110M model.
- Use case: general ml next-token modeling on prose; no long-range task
  (dialect adaptation data is dialogue lines / short paragraphs).
- Data support: documents longer than 1024 tokens exist (wiki articles) but
  are a minority; a 1024 window covers >90% of docs whole.

**Decision: block_size 1024 for the full pretrain** (cost is modest at our
scale, covers nearly all documents). **Sanity/validation runs use 512** to fit
the 8GB laptop at usable batch sizes. The tokenizer data was trained on
paragraph-level text either way; window choice doesn't affect tokenization.

## 3. Training-loop methodology (per spec)

AdamW (β1 0.9, β2 0.95, wd 0.1 on ≥2-D params only, fused on CUDA),
warmup 2% of steps + cosine decay to 10% of peak, grad clip 1.0, bf16 autocast
(no GradScaler needed for bf16), gradient accumulation for effective batch,
resumable checkpoints (model+optimizer+step, atomic rename) with optional
S3-sync hook, periodic val + fixed-seed sample generation.

## 4. Mandatory sanity checks (measured results)

Protocol per spec — cheapest viable instance: the laptop GPU itself.

1. **Overfit one tiny batch:** `train.py --preset tiny --overfit-batch 200`
   — a 11.5M-param model on one fixed 16×512 batch must fall below 0.1 loss.
2. **Fresh-model loss ≈ ln(vocab):** printed at step 0 on every run; the
   script hard-fails if |loss − ln(vocab)| > 0.25.

Results: see `docs/sanity-logs.md` (appended after execution — no numbers
claimed before they exist).

## 5. Validation-before-commitment

A small config (`--preset small`, 23M params) trains end-to-end locally for
2k steps on the real tokenized data before any cloud run is requested. The
full pretrain config differs only in size/steps — the loop code is identical,
which is the point of validating at small scale.

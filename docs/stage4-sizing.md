# Stage 4 — Model Sizing and Pretraining (Methodology Assessment)

Status: framework decided; the sizing table gets its final numbers from
`data/tokenized/stats.json` (Stage 2 completion). Any number below marked
"measured" is final; "estimate" items are resolved by the first 100 cloud
steps before a full run commits (per Stage 8 compute rules).

## 1. Data-bound sizing (tokens-per-parameter)

Applied heuristic: ~20 tokens/param (Chinchilla-style reference, not gospel).

| Clean unique tokens (measured: see data report) | 20:1 params | Model preset | Actual params @32k vocab |
|---|---|---|---|
| ~0.7B (laptop-prefix minimum) | 35M | medium (8L/512d) | 41.8M |
| ~1.2B (expected prefix) | 60M | — between — | — |
| ~1.7B (full-corpus projection) | 85M | base (12L/768d) | **109.9M (measured)** |

The three data-conditioned strategies from the spec:

| Strategy | Verdict | Reasoning |
|---|---|---|
| Smaller model + single epoch | **Fallback** if final tokens < 1B | cleanest, zero multi-epoch risk |
| ~110M + ~1.5–2 epochs | **PRIMARY** if tokens land 1.2–1.7B | mild repetition is well-tolerated at this scale (Muennighoff et al. scaling-data-constrained: up to ~4 epochs ≈ near-linear gains); we stay ≤2 |
| Continued pretraining from an existing multilingual checkpoint | Rejected | IndicBERTv2 is MLM/BERT-style (incompatible objective), no small causal ml checkpoint exists; XLM-R's ml fertility penalty (30%) contradicts the native-tokenization thesis |
| 300M–1B model | Rejected (Stage 0) | no data to feed it; measured sources cap ~1.8B raw tokens even ignoring disk |

## 2. Instance selection (cost-per-result, not raw speed)

FLOPs for full pretrain ≈ 6 · N · D. For N=110M, D=1.7B: 1.1e18 FLOPs.

| Option | Peak bf16 | Realistic MFU | Est. tok/s | Est. hours (1.7B) | $/hr (Spot) | Est. cost |
|---|---|---|---|---|---|---|
| g5.xlarge (A10G 24GB) | 125 TF | ~40% | ~75k | ~6.3 | $0.40 | **~$2.5** |
| g6e.xlarge (L40S 48GB) | 362 TF | ~40% | ~220k | ~2.1 | $0.85 | ~$1.8 |
| p4d.24xlarge (8×A100) | huge | — | ~600k+ | ~0.8 | $10+ | $8+ and mostly idle |

**Decision: g5.xlarge Spot.** The job is small enough that the A100-class
options are dominated on cost-per-result (they idle on a 110M model; 8-GPU
nodes waste 7 GPUs). L40S is the runner-up if Spot capacity for g5 is poor.
VRAM check: 110M bf16 + AdamW states + activations at batch 24×1024 fits
in 24GB with headroom (measured on the 4060 at batch 16×512 → extrapolated;
verified on-instance before launch).

## 3. Spot vs on-demand

Spot, because: checkpoints every 500 steps (~4 min of work at 75k tok/s),
auto-resume from S3-synced `last.pt` (train.py --resume), and the run is short
(≈6h). Interruption costs ≤ minutes of redo; the 60–80% Spot discount dwarfs
it. On-demand only if measured interruption frequency makes > 2 restarts/hour.

## 4. Validation-before-commitment sequence

1. Local (4060): `--preset tiny --overfit-batch` sanity (mandatory gates).
2. Local: `--preset small` 2k steps on real bins — validates loop + data E2E.
3. Cloud (g5.xlarge, Spot, 100 steps): measure actual tok/s and cost/hr;
   if measured cost extrapolates > 2x estimate, re-decide instance before the
   full run. (This replaces assumption with measurement per the spec.)
4. Full run: `aws/launch_train_spot.sh` — syncs bins from S3 to NVMe first,
   trains resumably, syncs checkpoints on interval.

## 5. Tracking during the run

Train/val loss every 500 steps; bits-per-character reported by
`eval_model.py` (val loss × tokens/char / ln2) — comparable across tokenizer
choices as required; fixed-seed sample generations every 1000 steps to
`train.log` for qualitative drift checks.

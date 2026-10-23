# Stage 8 — AWS infrastructure

Not yet provisioned: no AWS credentials are configured in this environment, so
everything below is ready-to-run templates. Bucket name and region must be
filled in by the operator (marked TODO). Nothing here is launched automatically.

## S3 layout (per spec)

```
s3://TODO-BUCKET/
  raw/          # scraped/downloaded corpora, untouched (we stream-clean, so only dump files land here)
  clean/        # normalized, deduped text (per source)
  tokenized/    # uint16 memmaps + tokenizer artifact + stats.json (versioned)
  checkpoints/  # model + optimizer state, per run
  adapters/     # LoRA / fine-tuned weights per dialect
  eval/         # test sets, generations, logs
```

## Setup (run once, with your own credentials)

```bash
aws s3api create-bucket --bucket TODO-BUCKET --region TODO-REGION
aws s3api put-public-access-block --bucket TODO-BUCKET \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
# versioning for traceable data/eval artifacts
aws s3api put-bucket-versioning --bucket TODO-BUCKET --versioning-configuration Status=Enabled
# expire noncurrent objects after 30 days (covers superseded checkpoints kept as separate keys)
aws s3api put-bucket-lifecycle-configuration --bucket TODO-BUCKET --lifecycle-configuration '{
  "Rules": [
    {"ID": "expire-old-checkpoints", "Filter": {"Prefix": "checkpoints/"}, "Status": "Enabled",
     "NoncurrentVersionExpiration": {"NoncurrentDays": 30}},
    {"ID": "expire-old-adapters", "Filter": {"Prefix": "adapters/"}, "Status": "Enabled",
     "NoncurrentVersionExpiration": {"NoncurrentDays": 60}}
  ]}'
```

IAM: instance role with `s3:GetObject/PutObject/ListBucket` on the bucket only,
no static keys on the instance (see `iam-policy.json`).

## Training rules enforced by our scripts

- Never train directly off S3: `launch_train_spot.sh` syncs `tokenized/` from
  S3 to the instance's local NVMe (`/home/ubuntu/data`) before training.
- Checkpoints go to local disk, then sync to S3 on an interval
  (`train.py --sync-cmd "aws s3 sync ... --exclude '*.tmp'"`). Mandatory on Spot.
- Keep best + last only: `last.pt`, `best.pt` are overwritten in place
  (versioning on the bucket preserves history).
- Dialect data with licensing concerns goes under `raw/dialect-restricted/`
  with a separate, tighter policy (documented in Stage 6 assessment).

## Instance selection per stage (applied at every training run)

| Stage | Instance | Est. hours | Est. cost (Spot) | Notes |
|---|---|---|---|---|
| Tokenizer training | local laptop | ~1 | $0 | done locally |
| Data cleaning/dedup | local laptop (25GB RAM) | ~2 | $0 | CPU-bound, streams, no S3 needed until upload |
| Sanity checks + small validation | local RTX 4060 | ~1 | $0 | overfit-batch + 2k-step run |
| Full pretrain (~110M, ~2B tokens) | g5.xlarge (A10G 24GB) Spot | 7-9 | $3-6 | resumable; see Stage 4 math |
| Dialect LoRA/full-FT ablation | g5.xlarge Spot | 1-2 per arm | $1-2 | 4 arms (2 methods x 2 dialects) |
| Eval | local 4060 | <1 | $0 | baselines are small |

Numbers are *estimates* (see docs/stage4-sizing.md for derivation and the
first-100-steps cloud validation that replaces assumptions with measurement).
g5.xlarge Spot pricing assumed $0.40/hr (region-dependent; check before launch).
On-demand only justified if Spot interruption frequency makes checkpoint churn
dominate — for an 8h run with 500-step checkpointing, it does not.

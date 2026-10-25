# Malayalam Small Language Model (SLM)

A base language model engineered specifically for Malayalam, featuring a native-fertility subword tokenizer, a causal decoder-only transformer architecture, and dialect adaptation infrastructure.

---

## 📌 Project Status & Handover Overview

This project was built through a structured 8-stage methodology. The repository is ready for the next phase.

| Stage | Focus Area | Status | Deliverables / Notes |
|---|---|---|---|
| **Stage 0** | Target & Scope Definition | ✅ Completed | [docs/stage0-target.md](docs/stage0-target.md) — 110M params, base model + dialect target. |
| **Stage 1** | Tokenizer Suite | ✅ Completed | [artifacts/tokenizer/malayalam_spm.model](artifacts/tokenizer/malayalam_spm.model) — 50k Unigram, 1.77 tok/word, 0 UNKs. |
| **Stage 2** | Data Pipeline | ✅ Completed (test-scale) | [docs/data_report.md](docs/data_report.md) — clean, dedup, 98/1/1 split, uint16 bins: **316.8M tokens** (train 310.5M / val 3.15M / test 3.18M), 0 UNK. Rebuilt from bounded public-source slices; full ~1.7B-token corpus still required before the final pretraining run. |
| **Stage 3** | Transformer Model | ✅ Completed | [src/model.py](src/model.py), [scripts/train.py](scripts/train.py) — Passed $\ln(\text{vocab})$ & overfit sanity gates. |
| **Stage 4** | Sizing & Pretraining Plan | ✅ Completed | [docs/stage4-sizing.md](docs/stage4-sizing.md) — 110M params, 1024 context, Spot `g5.xlarge` plan (~$2.50). |
| **Stage 5** | Evaluation Framework | ✅ Completed | [scripts/eval_model.py](scripts/eval_model.py) — Cross-tokenizer bits-per-character (BPC) & perplexity. |
| **Stage 6** | Dialect Adaptation | ✅ Completed | [scripts/train_lora.py](scripts/train_lora.py), [scripts/run_ablation.sh](scripts/run_ablation.sh) — LoRA vs Full-FT protocol. |
| **Stage 7** | Instruction Tuning Scope | ✅ Completed | [docs/stage7-instruction-data.md](docs/stage7-instruction-data.md) — Relegated to post-Stage 6 demo due to ToS. |
| **Stage 8** | AWS Infrastructure | ✅ Completed | [aws/](aws/) — S3 lifecycle, IAM policy, Spot instance launch & sync scripts. |

---

## 🚀 Quickstart for Collaborators

### 1. Environment Setup
```bash
git clone <your-repo-url>
cd malayalam-slm
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Data Pipeline (Stage 2 completed on this machine)
Stage 2 has been run end-to-end. The current `data/tokenized/` contains
**316,794,142 tokens** (train 310.46M / val 3.15M / test 3.18M) as uint16
memmaps, plus `test.txt` and `stats.json`, produced by a global cross-source
dedup and a doc-level 98/1/1 split (seed 1337). Exact source ranges, cleaning
stats and validation evidence are in [docs/data_report.md](docs/data_report.md).

`data/` is gitignored; the artifacts live on the execution machine. To rebuild
them from scratch, follow the reproduction commands in the data report
(`scripts/extract_wiki.py`, `scripts/build_corpus.py`, then):
```bash
python scripts/split_tokenize.py artifacts/tokenizer/malayalam_spm.model
```

**Scale caveat:** the current run used bounded public-source slices (full wiki,
a 2M-line CC100 prefix, and the first 4 GiB of IndicCorpV2) because the
original corpora were absent and this machine has 8 GB RAM. That is enough to
validate the pipeline and run the local Stage 4 gate, but the full ~1.7B-token
corpus must be rebuilt before the final cloud pretraining run.

### 3. Local Small Validation Run (Stage 4 gate)
Before spending cloud budget, validate the end-to-end training loop on the tokenized data for 2,000 steps locally:
```bash
python scripts/train.py --preset small --steps 2000 --data-dir data/tokenized --out-dir checkpoints/local_small
```

### 4. Full Cloud Pretraining (Stage 8)
> The current Stage 2 bins are test-scale (310M train tokens, 2.9 tokens/param
> at 110M params). Rebuild the full ~1.7B-token corpus (same pipeline, no
> slice bounds) before committing the full pretraining budget. The local
> 2,000-step gate below can run on the current bins.

1. Create your private S3 bucket and attach the IAM role described in [aws/README.md](aws/README.md).
2. Sync `data/tokenized/` and `artifacts/tokenizer/` to `s3://<YOUR-BUCKET>/tokenized/`.
3. Launch an EC2 Spot `g5.xlarge` instance using [aws/launch_train_spot.sh](aws/launch_train_spot.sh) (~6 hours, ~$2.50 total).

### 5. Dialect Adaptation (Stage 6)
Once the base checkpoint is trained, run the 4-arm ablation (LoRA vs Full-FT on Colloquial Subtitles vs Kasaragod dialect):
```bash
bash scripts/run_ablation.sh checkpoints/base_model/best.pt
```

---

## 📂 Repository Layout

```
├── artifacts/
│   └── tokenizer/
│       ├── malayalam_spm.model        # Final 50k SentencePiece Unigram model
│       ├── malayalam_spm.vocab        # Vocab file
│       └── candidates/                # 8 candidate models trained & evaluated
├── aws/
│   ├── README.md                      # Complete S3 & Spot setup instructions
│   ├── iam-policy.json                # S3 least-privilege instance policy
│   ├── launch_train_spot.sh           # Spot launch template
│   └── userdata_train.sh              # Cloud-init provisioning script
├── docs/                              # Formal methodology & decision assessments
│   ├── data_report.md                 # Stage 2 measured results & provenance
│   ├── sanity-logs.md
│   ├── stage0-target.md
│   ├── stage1-tokenizer-assessment.md
│   ├── stage2-data-assessment.md
│   ├── stage3-training-assessment.md
│   ├── stage4-sizing.md
│   ├── stage6-dialect-assessment.md
│   └── stage7-instruction-data.md
├── eval/
│   └── tokenizer_comparison.json      # Empirical benchmark results
├── scripts/
│   ├── build_corpus.py                # Data cleaning, LID & dedup streaming
│   ├── dialect_data.py                # OpenSubtitles & Kasaragod data prep
│   ├── eval_model.py                  # Loss, PPL, and BPC evaluation
│   ├── generate.py                    # Inference script
│   ├── split_tokenize.py              # Memmap bin generation
│   ├── train.py                       # Pretraining loop
│   ├── train_lora.py                  # Dialect LoRA fine-tuning
│   └── run_ablation.sh                # Dialect ablation harness
└── src/
    ├── dataloader.py                  # Memory-mapped uint16 binary token loader
    └── model.py                       # Pre-LN causal Transformer architecture
```

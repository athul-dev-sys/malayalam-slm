# Malayalam Small Language Model (SLM)

A base language model engineered specifically for Malayalam, featuring a native-fertility subword tokenizer, a causal decoder-only transformer architecture, and dialect adaptation infrastructure.

---

## 📌 Project Status & Handover Overview

This project was built through a structured 8-stage methodology. The repository is ready for the next phase.

| Stage | Focus Area | Status | Deliverables / Notes |
|---|---|---|---|
| **Stage 0** | Target & Scope Definition | ✅ Completed | [docs/stage0-target.md](docs/stage0-target.md) — 110M params, base model + dialect target. |
| **Stage 1** | Tokenizer Suite | ✅ Completed | [artifacts/tokenizer/malayalam_spm.model](artifacts/tokenizer/malayalam_spm.model) — 50k Unigram, 1.77 tok/word, 0 UNKs. |
| **Stage 2** | Data Pipeline | ✅ Completed | [docs/data_report.md](docs/data_report.md) — clean, dedup, 98/1/1 split, uint16 bins. Earlier test-scale run: 316.8M tokens. Current local bins: **590.2M tokens** (train 578.3M / val 6.0M / test 5.9M) from a bounded 8.5 GB clean slice — see [demo/README.md](demo/README.md). Full ~1.8B-token corpus remains to be tokenized. |
| **Stage 3** | Transformer Model | ✅ Completed | [src/model.py](src/model.py), [scripts/train.py](scripts/train.py) — Passed $\ln(\text{vocab})$ & overfit sanity gates. |
| **Stage 4** | Sizing & Pretraining Plan | ✅ Completed | [docs/stage4-sizing.md](docs/stage4-sizing.md) — 110M params, 1024 context, Spot `g5.xlarge` plan (~$2.50). |
| **Stage 5** | Evaluation Framework | ✅ Completed | [scripts/eval_model.py](scripts/eval_model.py) — Cross-tokenizer bits-per-character (BPC) & perplexity. Demo checkpoint: **1.310 BPC** vs MuRIL 1.482 (MLM pseudo-BPC; not directly comparable). See [demo/README.md](demo/README.md). |
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
The current `data/tokenized/` contains **590,193,419 tokens** (train 578.34M /
val 5.96M / test 5.89M) as uint16 memmaps, plus `test.txt` and `stats.json`,
produced by a global cross-source dedup and a doc-level 98/1/1 split (seed
1337). This run used full cleaned CC100 + wiki and the first 2.5 GiB of cleaned
IndicCorp (a bounded demo slice); the earlier 316.8M-token test-scale run and
its provenance are in [docs/data_report.md](docs/data_report.md), and the
current run's measured numbers are in [demo/README.md](demo/README.md).

`data/` is gitignored; the artifacts live on the execution machine. To rebuild
them from scratch, follow the reproduction commands in the data report
(`scripts/extract_wiki.py`, `scripts/build_corpus.py`, then):
```bash
python scripts/split_tokenize.py artifacts/tokenizer/malayalam_spm.model
```

**Scale caveat:** the current run tokenized full cleaned CC100 and wiki plus the
first 2.5 GiB of cleaned IndicCorpV2 to keep the local demo slice bounded
(~8.5 GB input, 590.2M tokens). That is enough to validate the pipeline and run
the local Stage 4 gate, but the full ~29 GB clean corpus (~1.8B tokens,
estimated) must be tokenized before the final cloud pretraining run.

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

### 5. Local Demo
A self-contained demo server (tokenizer playground, generation, training/data
dashboard) runs entirely locally. On Windows, double-click `demo/run_demo.bat`,
or run:
```bash
PYTHONIOENCODING=utf-8 python -m uvicorn demo.app:app --host 127.0.0.1 --port 8010
# open http://127.0.0.1:8010
```
See [demo/README.md](demo/README.md) for measured corpus/checkpoint details and
the training command. No cloud spend.

### 6. Dialect Adaptation (Stage 6)
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
├── demo/                              # Local FastAPI + HTML demo (see demo/README.md)
│   ├── app.py                         # Tokenizer / generation / stats endpoints
│   ├── static/                        # Self-contained UI (no CDN)
│   └── README.md                      # Runbook + measured demo artifacts
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

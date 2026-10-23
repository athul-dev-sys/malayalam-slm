#!/usr/bin/env bash
# Stage 6: dialect adaptation ablation — {lora, full-ft} x {film} + base control.
# Kasaragod arm activates only after the lexicon is placed and the native-validation
# gate (docs/stage6-dialect-assessment.md #4) passes.
# Each run: modest GPU (local 4060 for smoke, g5.xlarge Spot for real), 1-2h, ~$1-2.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${BASE:-checkpoints/best.pt}

# build dialect bins if missing (uses final tokenizer)
if [ ! -f data/dialect/bins/film/train.bin ]; then
  PYTHONIOENCODING=utf-8 python scripts/dialect_data.py
  PYTHONIOENCODING=utf-8 python scripts/text_to_bins.py \
    data/dialect/film_train.txt data/dialect/film_val.txt data/dialect/bins/film
fi

# arm 1: LoRA on film-register
PYTHONIOENCODING=utf-8 python scripts/train_lora.py \
  --base "$BASE" --data-dir data/dialect/bins/film \
  --out artifacts/adapters/film_lora --steps 1000

# arm 2: full fine-tune (fresh optimizer, low LR) on film-register
PYTHONIOENCODING=utf-8 python scripts/train.py \
  --preset base --resume "$BASE" --reset-optimizer \
  --data-dir data/dialect/bins/film \
  --lr 1e-5 --steps 1000 --val-interval 200 --ckpt-interval 500 \
  --out-dir artifacts/adapters/film_fullft

# control numbers come from $BASE evaluated on both val sets (logged by both arms)
echo "ablation complete -> artifacts/adapters/"

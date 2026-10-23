#!/usr/bin/env bash
# Runs on the EC2 instance at boot (passed as UserData).
# Syncs tokenized data from S3 to local disk, then runs resumable training.
set -euo pipefail
B=TODO-BUCKET
HOME_D=/home/ubuntu
REPO=$HOME_D/malayalam-slm

cd $HOME_D
# repo: replace with your actual source (private git, or tarball in S3)
aws s3 cp s3://$B/checkpoints/repo.tgz repo.tgz || true
if [ -f repo.tgz ]; then tar xzf repo.tgz; else git clone TODO-REPO-URL malayalam-slm; fi

mkdir -p $HOME_D/data/tokenized
aws s3 sync s3://$B/tokenized/ $HOME_D/data/tokenized/      # NEVER train directly off S3
aws s3 cp s3://$B/checkpoints/last.pt $REPO/checkpoints/last.pt || echo "no prior checkpoint, fresh run"

cd $REPO
PYTHONIOENCODING=utf-8 nohup python scripts/train.py \
  --preset base \
  --steps 20000 \
  --batch-size 24 \
  --grad-accum 4 \
  --lr 6e-4 \
  --warmup-steps 400 \
  --val-interval 500 \
  --ckpt-interval 500 \
  --sample-interval 1000 \
  --out-dir checkpoints \
  --resume checkpoints/last.pt \
  --sync-cmd "aws s3 sync checkpoints/ s3://$B/checkpoints/ --exclude *.tmp" \
  > train.log 2>&1 &
echo "training started; tail -f $REPO/train.log"

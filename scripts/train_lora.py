"""Stage 6: LoRA fine-tuning of the base model on dialect/register data,
with dual evaluation (dialect val + standard val = catastrophic-forgetting check).

Usage: PYTHONIOENCODING=utf-8 python scripts/train_lora.py \
  --base checkpoints/best.pt --data-dir data/dialect/bins/film \
  --standard-val-bin data/tokenized/val.bin --out artifacts/adapters/film_lora
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataloader import TokenMemmap
from model import GPT, ModelConfig


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(model: GPT, r: int):
    n = 0
    for block in model.transformer.h:
        block.attn.qkv = LoRALinear(block.attn.qkv, r)
        block.attn.proj = LoRALinear(block.attn.proj, r)
        block.mlp.c_fc = LoRALinear(block.mlp.c_fc, r)
        block.mlp.c_proj = LoRALinear(block.mlp.c_proj, r)
        n += 4
    return n


@torch.no_grad()
def val_loss(model, data, block_size, device, iters=20):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = data.get_batch(block_size, 8, device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--standard-val-bin", default=os.path.join(ROOT, "data", "tokenized", "val.bin"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--val-interval", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev_type = "cuda" if args.device.startswith("cuda") else "cpu"

    ck = torch.load(args.base, map_location=args.device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = GPT(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    base_std = ck.get("val_loss")

    train_data = TokenMemmap(os.path.join(args.data_dir, "train.bin"))
    val_data = TokenMemmap(os.path.join(args.data_dir, "val.bin"))
    std_val = TokenMemmap(args.standard_val_bin) if os.path.exists(args.standard_val_bin) else None

    v0 = val_loss(model, val_data, args.block_size, args.device)
    s0 = val_loss(model, std_val, args.block_size, args.device) if std_val else None
    print(f"base model on dialect val: {v0:.4f}" + (f" | standard val: {s0:.4f}" if s0 else ""))

    n_lora = add_lora(model, args.r)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA r={args.r} on {n_lora} linears; trainable params: {sum(p.numel() for p in lora_params)/1e6:.2f}M")

    opt = torch.optim.AdamW(lora_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
                            fused=args.device.startswith("cuda"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)

    log = {"base_dialect_val": v0, "base_standard_val": s0, "r": args.r, "steps": []}
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = train_data.get_batch(args.block_size, args.batch_size, args.device)
            with torch.autocast(device_type="cuda" if device_type(args) else "cpu",
                                dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0:
            print(f"step {step} loss={loss.item()*args.grad_accum:.4f}")
        if step % args.val_interval == 0 or step == args.steps:
            v = val_loss(model, val_data, args.block_size, args.device)
            s = val_loss(model, std_val, args.block_size, args.device) if std_val else None
            print(f"step {step} dialect_val={v:.4f} ({(v0-v)/v0*100:+.1f}%)" +
                  (f" standard_val={s:.4f} ({(s-s0)/s0*100:+.1f}%)" if s else ""))
            log["steps"].append({"step": step, "dialect_val": v, "standard_val": s})

    adapter_sd = {k: v for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}
    torch.save({"adapter": adapter_sd, "r": args.r, "base_config": cfg.__dict__,
                "base_ckpt": args.base, "log": log}, os.path.join(args.out, "lora.pt"))
    with io_open(os.path.join(args.out, "log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print("saved adapter ->", os.path.join(args.out, "lora.pt"))


def io_open(path, mode="w"):
    import io

    return io.open(path, mode, encoding="utf-8")


if __name__ == "__main__":
    main()

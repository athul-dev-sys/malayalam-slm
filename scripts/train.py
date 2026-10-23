"""Training loop: AdamW, warmup+cosine LR, grad clip, bf16 autocast, resumable checkpoints.

Sanity modes (Stage 3, mandatory before real runs):
  --overfit-batch N : train one fixed batch; exit non-zero if loss doesn't fall below 0.1
  initial-loss check is always printed (expected ~= ln(vocab_size))

Usage examples:
  PYTHONIOENCODING=utf-8 python scripts/train.py --preset tiny --overfit-batch 200
  PYTHONIOENCODING=utf-8 python scripts/train.py --preset small --steps 2000
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from dataloader import TokenMemmap
from model import GPT, ModelConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRESETS = {
    # vocab 32k assumed; adjusted automatically from tokenizer
    "tiny": dict(n_layer=4, n_head=4, n_embd=256),        # ~11M  @32k vocab
    "small": dict(n_layer=6, n_head=6, n_embd=384),       # ~26M  @32k vocab
    "medium": dict(n_layer=8, n_head=8, n_embd=512),      # ~47M  @32k vocab
    "base": dict(n_layer=12, n_head=12, n_embd=768),      # ~109M @32k vocab
}


def get_tokenizer(vocab_hint_path):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=vocab_hint_path)
    return sp


def build_config(args, vocab_size):
    preset = PRESETS[args.preset]
    return ModelConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=preset["n_layer"],
        n_head=preset["n_head"],
        n_embd=preset["n_embd"],
        dropout=args.dropout,
    )


def lr_at(step, total_steps, peak_lr, warmup_steps, min_frac=0.1):
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


@torch.no_grad()
def estimate_loss(model, data, args, eval_iters=25):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = data.get_batch(args.block_size, args.batch_size, args.device)
        with torch.autocast(device_type=args.device_type, dtype=torch.bfloat16, enabled=args.bf16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_generate(model, sp, args, n=64, prompt=""):
    model.eval()
    ids = sp.encode(prompt, out_type=int, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=args.device)
    out = model.generate(x, max_new_tokens=n, temperature=0.8, top_k=40)[0].tolist()
    model.train()
    return sp.decode(out)


def save_ckpt(path, model, opt, step, cfg, val_loss, args):
    tmp = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "config": cfg.__dict__,
            "val_loss": val_loss,
            "args": vars(args),
        },
        tmp,
    )
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small", choices=list(PRESETS))
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "tokenized"))
    p.add_argument("--tokenizer", default=os.path.join(ROOT, "artifacts", "tokenizer", "malayalam_spm.model"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--val-interval", type=int, default=250)
    p.add_argument("--ckpt-interval", type=int, default=500)
    p.add_argument("--sample-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--overfit-batch", type=int, default=0, help="N>0: sanity mode, overfit one fixed batch")
    p.add_argument("--resume", default="")
    p.add_argument("--reset-optimizer", action="store_true", help="resume weights but start fresh optimizer (fine-tuning)")
    p.add_argument("--sync-cmd", default="", help="shell command run after each checkpoint (e.g. aws s3 sync ...)")
    args = p.parse_args()
    args.device_type = "cuda" if args.device.startswith("cuda") else "cpu"
    args.bf16 = (not args.no_bf16) and args.device_type == "cuda"

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    sp = get_tokenizer(args.tokenizer)
    vocab_size = sp.get_piece_size()
    cfg = build_config(args, vocab_size)
    model = GPT(cfg).to(args.device)
    nparams = model.num_params()
    print(f"model: preset={args.preset} params={nparams/1e6:.1f}M vocab={vocab_size} ctx={cfg.block_size}")

    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        if p_.dim() >= 2:
            decay.append(p_)
        else:
            no_decay.append(p_)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=(args.device_type == "cuda"),
    )

    step0 = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        if args.reset_optimizer:
            step0 = 0
            print(f"resumed weights from {args.resume} (fresh optimizer, fine-tuning)")
        else:
            opt.load_state_dict(ck["optimizer"])
            step0 = ck["step"]
            print(f"resumed from {args.resume} at step {step0}")

    train_data = TokenMemmap(os.path.join(args.data_dir, "train.bin"))
    val_path = os.path.join(args.data_dir, "val.bin")
    val_data = TokenMemmap(val_path) if os.path.exists(val_path) else None

    # ---- sanity check 1: initial loss ~ ln(vocab) — fresh models only ----
    if not args.resume:
        model.eval()
        xb, yb = train_data.get_batch(args.block_size, args.batch_size, args.device)
        with torch.autocast(device_type=args.device_type, dtype=torch.bfloat16, enabled=args.bf16):
            _, loss = model(xb, yb)
        expected = math.log(vocab_size)
        print(f"initial loss={loss.item():.4f} ln(vocab)={expected:.4f} (delta {loss.item()-expected:+.4f})")
        if abs(loss.item() - expected) > 0.25:
            print("FAIL: initial loss not close to ln(vocab_size) — check init or tokenizer ids")
            sys.exit(1)
        model.train()

    # ---- sanity check 2 (optional): overfit one fixed batch ----
    if args.overfit_batch:
        xb, yb = train_data.get_batch(args.block_size, args.batch_size, args.device)
        for it in range(args.overfit_batch):
            with torch.autocast(device_type=args.device_type, dtype=torch.bfloat16, enabled=args.bf16):
                _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if it % 10 == 0 or it == args.overfit_batch - 1:
                print(f"overfit it={it} loss={loss.item():.4f}")
        if loss.item() > 0.1:
            print(f"FAIL: overfit-batch loss stuck at {loss.item():.4f}")
            sys.exit(1)
        print(f"overfit-batch PASS (final loss {loss.item():.4f})")
        return

    # ---- training loop ----
    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    print(f"tokens/step={tokens_per_step} total_tokens={tokens_per_step*args.steps/1e6:.1f}M")
    t_start = time.time()
    best_val = float("inf")

    step = step0
    while step < args.steps:
        lr = lr_at(step, args.steps, args.lr, args.warmup_steps, args.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro in range(args.grad_accum):
            xb, yb = train_data.get_batch(args.block_size, args.batch_size, args.device)
            with torch.autocast(device_type=args.device_type, dtype=torch.bfloat16, enabled=args.bf16):
                _, loss = model(xb, yb)
                loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1

        if step % 50 == 0 or step == 1:
            dt = time.time() - t_start
            tps = (step - step0) * tokens_per_step / max(dt, 1e-6)
            print(f"step {step}/{args.steps} loss={accum_loss:.4f} lr={lr:.2e} tok/s={tps:,.0f} elapsed={dt/60:.1f}m")

        if val_data is not None and step % args.val_interval == 0:
            v = estimate_loss(model, val_data, args)
            print(f"step {step} VAL loss={v:.4f}")
            if v < best_val:
                best_val = v
                save_ckpt(os.path.join(args.out_dir, "best.pt"), model, opt, step, cfg, v, args)

        if step % args.ckpt_interval == 0 or step == args.steps:
            save_ckpt(os.path.join(args.out_dir, "last.pt"), model, opt, step, cfg, None, args)
            if args.sync_cmd:
                os.system(args.sync_cmd)

        if step % args.sample_interval == 0 or step == args.steps:
            print("--- sample ---")
            print(sample_generate(model, sp, args, n=48, prompt=""))
            print("--------------")

    print(f"done. best_val={best_val:.4f}")
    with open(os.path.join(args.out_dir, "run_summary.json"), "w") as f:
        json.dump({"preset": args.preset, "params": nparams, "steps": args.steps,
                   "best_val": best_val, "tokens_seen": tokens_per_step * args.steps}, f, indent=1)


if __name__ == "__main__":
    main()

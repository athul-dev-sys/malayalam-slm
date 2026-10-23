"""Generate samples for evaluation (SC3 human-fluency protocol + demo).

Usage: PYTHONIOENCODING=utf-8 python scripts/generate.py --ckpt checkpoints/best.pt \
  [--n 50] [--prompt "..."] [--out eval/human_eval_samples.txt]
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import GPT, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "artifacts", "tokenizer", "malayalam_spm.model"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = GPT(cfg).to(args.device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {model.num_params()/1e6:.1f}M params, ctx={cfg.block_size}")

    torch.manual_seed(args.seed)
    prompt_ids = sp.encode(args.prompt, out_type=int, add_bos=True) if args.prompt else [sp.bos_id()]
    samples = []
    for i in range(args.n):
        x = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model.generate(x, args.max_tokens, temperature=args.temperature, top_k=args.top_k)
        text = sp.decode(out[0].tolist())
        if not args.prompt:  # strip leading BOS artifact
            text = sp.decode(out[0][1:].tolist())
        samples.append(f"### sample {i+1}\n{text}\n")
        print(f"--- {i+1}/{args.n} ---\n{text}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        import io

        with io.open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(samples))
        print("wrote", args.out)


if __name__ == "__main__":
    main()

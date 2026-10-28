"""Stage 5 evaluation: exact bpc for our causal model + pseudo-perplexity bpc for MLM baselines.

Our model: sequential pass over test docs (chunked to ctx), exact nats & chars -> bpc.
Baselines (XLM-R / MuRIL / IndicBERTv2): 15% random token masking (fixed seed),
mean NLL per masked token -> bpc via that tokenizer's tokens/char. Method caveat:
MLM pseudo-LL is not strictly comparable to causal LL (bidirectional context);
reported side-by-side, never aggregated.

Usage: PYTHONIOENCODING=utf-8 python scripts/eval_model.py --ckpt checkpoints/last.pt \
         [--baselines xlm-roberta-base google/muril-base-cased ai4bharat/IndicBERTv2-MLM-only] \
         [--n-docs 300]
"""
import argparse
import io
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import GPT, ModelConfig


def load_docs(path, n_docs):
    docs, buf = [], []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\r\n")
            # \x1e is Unicode whitespace, so strip() would erase the marker before comparison
            if raw == "\x1e":
                if buf:
                    docs.append("\n\n".join(buf))
                    buf = []
            elif raw.strip():
                buf.append(raw.strip())
    if buf:
        docs.append("\n\n".join(buf))
    return docs[:n_docs]


@torch.no_grad()
def eval_causal(ckpt_path, docs, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(ck["model"])

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=ck["args"]["tokenizer"])
    ctx = cfg.block_size
    total_nats = 0.0
    total_tokens = 0
    total_chars = sum(len(d) for d in docs)
    for d in docs:
        ids = sp.encode(d, out_type=int, add_bos=True, add_eos=True)
        for i in range(0, len(ids) - 1, ctx):
            chunk = ids[i : i + ctx]
            if len(chunk) < 2:
                continue
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                _, loss = model(x, y)
            # model returns the mean CE loss; multiply back to get the exact sum of nats
            nll = loss * y.numel()
            total_nats += nll.item()
            total_tokens += len(chunk) - 1
    nats_per_tok = total_nats / total_tokens
    bpc = total_nats / (total_chars * math.log(2))
    return {
        "params": model.num_params(),
        "n_docs": len(docs), "n_tokens": total_tokens, "n_chars": total_chars,
        "loss_nats_per_token": nats_per_tok,
        "ppl": math.exp(nats_per_tok),
        "tokens_per_char": total_tokens / total_chars,
        "bpc": bpc,
    }


@torch.no_grad()
def eval_mlm_baseline(hub_name, docs, device, seed=13, mask_frac=0.15, max_tokens=400):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hub_name)
    model = AutoModelForMaskedLM.from_pretrained(hub_name).to(device).eval()
    g = torch.Generator(device="cpu").manual_seed(seed)

    total_nats = 0.0
    total_masked = 0
    total_chars = 0
    total_tokens = 0
    for d in docs:
        enc = tok(d, return_tensors="pt", truncation=True, max_length=max_tokens,
                  return_offsets_mapping=True)
        ids = enc["input_ids"][0]
        # count only the prefix actually scored, so truncated docs don't dilute tok/char
        total_chars += int(enc["offset_mapping"][0][:, 1].max())
        total_tokens += len(ids)
        labels = ids.clone()
        prob = torch.full((len(ids),), mask_frac)
        mask = torch.rand(len(ids), generator=g) < prob
        special = torch.tensor([tok.cls_token_id, tok.sep_token_id, tok.pad_token_id])
        mask[torch.isin(ids, special)] = False
        if mask.sum() == 0:
            continue
        labels[~mask] = -100
        inp = ids.clone()
        inp[mask] = tok.mask_token_id
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            out = model(input_ids=inp[None].to(device))
        nll = F.cross_entropy(out.logits[0].float(), labels.to(device), reduction="sum", ignore_index=-100)
        total_nats += nll.item()
        total_masked += int(mask.sum())
    nats_per_masked = total_nats / max(total_masked, 1)
    tok_per_char = total_tokens / total_chars
    bpc = nats_per_masked * tok_per_char / math.log(2)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "hub": hub_name, "params": n_params,
        "n_docs": len(docs), "n_tokens": total_tokens, "n_chars": total_chars,
        "max_tokens_per_doc": max_tokens,
        "nats_per_masked_token": nats_per_masked,
        "pseudo_ppl": math.exp(nats_per_masked),
        "tokens_per_char": tok_per_char,
        "bpc": bpc,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="")
    p.add_argument("--test-text", default=os.path.join(ROOT, "data", "tokenized", "test.txt"))
    p.add_argument("--n-docs", type=int, default=300)
    p.add_argument("--baselines", nargs="*", default=[])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    docs = load_docs(args.test_text, args.n_docs)
    print(f"eval docs={len(docs)} chars={sum(len(d) for d in docs):,}")

    results = {}
    if args.ckpt:
        results["ours"] = eval_causal(args.ckpt, docs, args.device)
        print("ours:", json.dumps(results["ours"], indent=1))
    for hub in args.baselines:
        try:
            results[hub] = eval_mlm_baseline(hub, docs, args.device)
            print(hub, json.dumps(results[hub], indent=1))
        except Exception as e:
            print(f"baseline {hub} FAILED: {type(e).__name__}: {e}")

    out = os.path.join(ROOT, "eval", "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    if results:
        print("\n| model | params | tok/char | loss (nats/tok) | bpc |")
        print("|---|---|---|---|---|")
        for name, r in sorted(results.items(), key=lambda kv: kv[1]["bpc"]):
            loss = r.get("loss_nats_per_token") or r.get("nats_per_masked_token")
            print(f"| {name} | {r['params']/1e6:.0f}M | {r['tokens_per_char']:.3f} | {loss:.4f} | {r['bpc']:.4f} |")


if __name__ == "__main__":
    main()

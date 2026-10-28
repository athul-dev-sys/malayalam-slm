"""Local demo server for the Malayalam SLM.

Serves a self-contained web UI plus:
  /api/tokenize  live 50k-SPM vs MuRIL / XLM-R / IndicBERTv2 tokenizer comparison
  /api/generate  text generation from the current demo checkpoint
  /api/training  loss history parsed from the training log
  /api/stats     tokenized-corpus, cleaning and tokenizer-benchmark stats
  /api/sample    a random held-out corpus paragraph

Run: python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
"""
import json
import os
import random
import re
import threading
import time
import unicodedata
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
STATIC = BASE / "static"
DATA_DIR = ROOT / "data" / "tokenized"
CLEAN_DIR = ROOT / "data" / "clean"
EVAL_DIR = ROOT / "eval"
CKPT_DIR = ROOT / "checkpoints" / "demo"
LOG_PATH = CKPT_DIR / "train.log"
TOKENIZER_PATH = ROOT / "artifacts" / "tokenizer" / "malayalam_spm.model"

import sys

sys.path.insert(0, str(ROOT / "src"))
from model import GPT, ModelConfig  # noqa: E402

import sentencepiece as spm  # noqa: E402

sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))

app = FastAPI(title="Malayalam SLM demo")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

BASELINES = [
    ("google/muril-base-cased", "MuRIL"),
    ("xlm-roberta-base", "XLM-R"),
    ("ai4bharat/IndicBERTv2-MLM-only", "IndicBERTv2"),
]
UNK_PIECES = {"<unk>", "[UNK]", "<|unk|>"}
WORD_RE = re.compile(r"\S+")
ML_CHAR_RE = re.compile(r"[\u0d00-\u0d7f]")

_baselines = {}
_baselines_lock = threading.Lock()
_model_lock = threading.Lock()
_gen_lock = threading.Lock()
_model_cache = {"key": None, "model": None, "cfg": None, "step": None, "val_loss": None}


def warm_baselines():
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        print(f"transformers unavailable: {e}")
        return
    for hub, name in BASELINES:
        try:
            tok = AutoTokenizer.from_pretrained(hub)
            with _baselines_lock:
                _baselines[name] = tok
            print(f"baseline ready: {name}")
        except Exception as e:
            with _baselines_lock:
                _baselines[name] = e
            print(f"baseline failed: {name}: {type(e).__name__}: {e}")


threading.Thread(target=warm_baselines, daemon=True).start()


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def ml_ratio(text):
    dense = re.sub(r"\s+", "", text)
    if not dense:
        return 0.0
    return len(ML_CHAR_RE.findall(dense)) / len(dense)


class TokenizeIn(BaseModel):
    text: str


class GenerateIn(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 96
    temperature: float = 0.8
    top_k: int = 40
    seed: int | None = None


def basic_metrics(text, ids, pieces, decoded, vocab_size, extra=None):
    n_word = len(WORD_RE.findall(text))
    n_char = len(text)
    n_unk = sum(1 for p in pieces if p in UNK_PIECES)
    out = {
        "vocab_size": vocab_size,
        "token_count": len(ids),
        "tokens_per_word": len(ids) / max(n_word, 1),
        "tokens_per_char": len(ids) / max(n_char, 1),
        "unk_rate": n_unk / max(len(ids), 1),
        "roundtrip": decoded == text,
    }
    if extra:
        out.update(extra)
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    train = training_state()
    ckpt_files = [p for p in (CKPT_DIR / "best.pt", CKPT_DIR / "last.pt") if p.exists()]
    stats = read_json(DATA_DIR / "stats.json")
    return {
        "model_ready": bool(ckpt_files),
        "training": train,
        "total_tokens": (stats or {}).get("total_tokens"),
        "vocab_size": sp.get_piece_size(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


@app.post("/api/tokenize")
def tokenize(body: TokenizeIn):
    raw = body.text[:3000]
    text = unicodedata.normalize("NFC", raw)
    if not text.strip():
        raise HTTPException(400, "empty text")

    ids = sp.encode(text, out_type=int, add_bos=False, add_eos=False)
    pieces = [sp.id_to_piece(i) for i in ids]
    decoded = sp.decode(ids)
    ours = basic_metrics(text, ids, pieces, decoded, sp.get_piece_size())
    ours["name"] = "Malayalam-SPM-50k"
    ours["byte_fallback"] = sum(1 for p in pieces if p.startswith("<0x"))
    ours["tokens"] = pieces[:240]
    ours["ids"] = ids[:240]
    ours["truncated"] = len(pieces) > 240

    baselines = []
    errors = []
    with _baselines_lock:
        loaded = dict(_baselines)
    for hub, name in BASELINES:
        tok = loaded.get(name)
        if tok is None:
            errors.append(f"{name}: still loading")
            continue
        if isinstance(tok, Exception):
            errors.append(f"{name}: unavailable ({type(tok).__name__})")
            continue
        try:
            bids = tok.encode(text, add_special_tokens=False)
            bpieces = [str(p) for p in tok.convert_ids_to_tokens(bids)]
            bdecoded = tok.decode(bids, skip_special_tokens=True)
            m = basic_metrics(text, bids, bpieces, bdecoded, tok.vocab_size)
            m["name"] = name
            m["hub"] = hub
            baselines.append(m)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    baselines.sort(key=lambda m: m["tokens_per_word"])
    return {
        "chars": len(text),
        "words": len(WORD_RE.findall(text)),
        "nfc_changed": text != raw,
        "ours": ours,
        "baselines": baselines,
        "errors": errors,
    }


def pick_checkpoint():
    for name in ("best.pt", "last.pt"):
        p = CKPT_DIR / name
        if p.exists():
            return p
    return None


def get_model():
    path = pick_checkpoint()
    if path is None:
        raise HTTPException(409, "No checkpoint yet — the demo model is still training.")
    key = (str(path), path.stat().st_mtime)
    if _model_cache["key"] == key:
        return _model_cache
    with _model_lock:
        if _model_cache["key"] != key:
            ck = torch.load(str(path), map_location="cpu", weights_only=False)
            cfg = ModelConfig(**ck["config"])
            model = GPT(cfg)
            model.load_state_dict(ck["model"])
            model.eval()
            _model_cache.update(
                key=key, model=model, cfg=cfg,
                step=ck.get("step"), val_loss=ck.get("val_loss"),
            )
            print(f"loaded checkpoint {path.name} step={ck.get('step')} "
                  f"params={model.num_params()/1e6:.1f}M")
    return _model_cache


@app.post("/api/generate")
def generate(body: GenerateIn):
    prompt = body.prompt[:1200]
    max_new = max(8, min(int(body.max_new_tokens), 200))
    temperature = max(0.05, min(float(body.temperature), 2.0))
    top_k = None if int(body.top_k) <= 0 else min(int(body.top_k), 500)

    cache = get_model()
    model, cfg = cache["model"], cache["cfg"]
    with _gen_lock:
        if body.seed is not None:
            torch.manual_seed(int(body.seed))
        ids = sp.encode(prompt, out_type=int, add_bos=True) if prompt.strip() else [sp.bos_id()]
        x = torch.tensor([ids], dtype=torch.long)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(x, max_new, temperature=temperature, top_k=top_k)
        seconds = time.time() - t0
    new_ids = out[0].tolist()[len(ids):]
    completion = sp.decode(new_ids)
    return {
        "prompt": prompt,
        "completion": completion,
        "new_tokens": len(new_ids),
        "seconds": round(seconds, 2),
        "temperature": temperature,
        "top_k": top_k or 0,
        "checkpoint": {"step": cache["step"], "val_loss": cache["val_loss"]},
    }


STEP_RE = re.compile(r"^step (\d+)/(\d+) loss=([\d.]+) lr=(\S+) tok/s=([\d,]+) elapsed=([\d.]+)m")
VAL_RE = re.compile(r"^step (\d+) VAL loss=([\d.]+)")
MODEL_RE = re.compile(r"^model: preset=(\w+) params=([\d.]+)M vocab=(\d+) ctx=(\d+)")
TOKSTEP_RE = re.compile(r"^tokens/step=(\d+) total_tokens=([\d.]+)M")


def training_state():
    state = {
        "running": False, "step": None, "total_steps": None, "loss": None,
        "val_loss": None, "best_val": None, "tok_s": None, "elapsed_min": None,
        "preset": None, "params_m": None, "vocab": None, "ctx": None,
        "tokens_per_step": None, "planned_tokens_m": None,
        "samples": [], "points": [], "vals": [], "log_exists": False,
    }
    if not LOG_PATH.exists():
        return state
    state["log_exists"] = True
    points, vals, samples = [], [], []
    current_step = None
    in_sample = False
    buf = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                m = STEP_RE.match(line)
                if m:
                    current_step = int(m.group(1))
                    state["step"] = current_step
                    state["total_steps"] = int(m.group(2))
                    state["loss"] = float(m.group(3))
                    state["tok_s"] = int(m.group(5).replace(",", ""))
                    state["elapsed_min"] = float(m.group(6))
                    points.append([current_step, state["loss"]])
                    continue
                m = VAL_RE.match(line)
                if m:
                    v = float(m.group(2))
                    vals.append([int(m.group(1)), v])
                    state["val_loss"] = v
                    state["best_val"] = v if state["best_val"] is None else min(state["best_val"], v)
                    continue
                m = MODEL_RE.match(line)
                if m:
                    state["preset"] = m.group(1)
                    state["params_m"] = float(m.group(2))
                    state["vocab"] = int(m.group(3))
                    state["ctx"] = int(m.group(4))
                    continue
                m = TOKSTEP_RE.match(line)
                if m:
                    state["tokens_per_step"] = int(m.group(1))
                    state["planned_tokens_m"] = float(m.group(2))
                    continue
                if line == "--- sample ---":
                    in_sample, buf = True, []
                    continue
                if line == "--------------":
                    if buf:
                        samples.append({"step": current_step, "text": "\n".join(buf).strip()})
                    in_sample = False
                    continue
                if in_sample:
                    buf.append(line)
                if line.startswith("done."):
                    state["running"] = False
    except OSError:
        return state
    state["samples"] = samples[-6:]
    state["points"] = points
    state["vals"] = vals
    state["running"] = state["total_steps"] is not None and state["step"] is not None and state["step"] < state["total_steps"]
    return state


@app.get("/api/training")
def training():
    return training_state()


@app.get("/api/stats")
def stats():
    tokenized = read_json(DATA_DIR / "stats.json")
    comparison = read_json(EVAL_DIR / "tokenizer_comparison.json") or {}
    compact = {}
    for name, m in comparison.items():
        if not isinstance(m, dict):
            continue
        compact[name] = {k: v for k, v in m.items() if k != "chillu"}
    clean = {}
    for p in sorted(CLEAN_DIR.glob("*_stats.json")):
        payload = read_json(p)
        if payload:
            clean[p.name.replace("_stats.json", "")] = payload
    return {
        "tokenized": tokenized,
        "tokenizer_eval": compact,
        "model_eval": read_json(EVAL_DIR / "results.json"),
        "clean": clean,
        "training": training_state(),
        "run_summary": read_json(CKPT_DIR / "run_summary.json"),
        "model_ready": pick_checkpoint() is not None,
    }


@app.get("/api/sample")
def sample():
    path = DATA_DIR / "test.txt"
    if not path.exists():
        return {"text": None}
    try:
        size = path.stat().st_size
        if size < 8192:
            return {"text": None}
        with open(path, "rb") as f:
            f.seek(random.randrange(0, size - 8192))
            chunk = f.read(8192).decode("utf-8", errors="ignore")
    except OSError:
        return {"text": None}
    parts = re.split(r"\n\s*\x1e\s*\n|\n\s*\n", chunk)
    cands = []
    for part in parts:
        text = re.sub(r"\s+", " ", part).strip()
        if 120 <= len(text) <= 700 and ml_ratio(text) >= 0.7:
            cands.append(text)
    return {"text": random.choice(cands) if cands else None}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8010")))

# Sanity-check logs (Stage 3 mandatory gates)

Environment: local RTX 4060 Laptop (8GB), torch 2.6.0+cu124, bf16 autocast.
Tokenizer under test: `spm_unigram_32k` candidate (final vocab decision
pending sweep; sanity checks are tokenizer-independent).

## Gate 1 — fresh-model loss ≈ ln(vocab_size)

```
$ python scripts/train.py --preset tiny --steps 5 --data-dir data/tokenizer_work/sanity_bins
model: preset=tiny params=11.5M vocab=32000 ctx=512
initial loss=10.4065 ln(vocab)=10.3735 (delta +0.0330)
```

PASS (|delta| = 0.033, gate 0.25). Model init produces near-uniform
next-token distribution as required.

## Gate 2 — overfit one tiny batch to ~0 loss

```
$ python scripts/train.py --preset tiny --overfit-batch 200 --data-dir data/tokenizer_work/sanity_bins
overfit it=70 loss=3.7293
overfit it=80 loss=2.9172
overfit it=90 loss=2.1283
overfit it=100 loss=1.4330
overfit it=110 loss=0.8737
overfit it=120 loss=0.4715
overfit it=130 loss=0.2245
overfit it=140 loss=0.0982
overfit it=150 loss=0.0474
overfit it=160 loss=0.0265
overfit it=170 loss=0.0168
overfit it=180 loss=0.0117
overfit it=190 loss=0.0086
overfit it=199 loss=0.0067
overfit-batch PASS (final loss 0.0067)
```

PASS (0.0067 < 0.1 gate). Gradients, optimizer, bf16 autocast, clipping all
function; the model can memorize — capacity + connectivity confirmed.

## Cost

~4 minutes local GPU, $0. No cloud spend for sanity gates (spec: cheapest
viable instance — the laptop is it).

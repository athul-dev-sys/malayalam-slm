# Stage 0 — Target Definition (Methodology Assessment)

Status: DECIDED (2026-10-23). Evidence gathered before writing: mlwiki 20261001
articles dump = 202 MB bz2 (measured, dumps.wikimedia.org); IndicCorpV2
Malayalam raw file `ml.txt` = 26.32 GB (measured, HF content-length); CC100
ml = ~1 GB class (not yet verified — URL moved; to resolve in Stage 2).
Local environment measured: RTX 4060 8 GB, torch 2.6.0+cu124, 39 GB free disk.

## Reality flag (deviation from spec)

The spec's CURRENT STATE claims a decoder-only transformer is already
implemented and points to "existing code/repo here." The working directory is
**empty** — no model code exists. Decision: Stage 3 includes implementing the
transformer from scratch rather than assuming one. This is flagged rather than
silently papered over.

## 1. Target use case

Options considered:

| Option | Verdict | Reasoning |
|---|---|---|
| (a) General Malayalam text generator | Rejected as primary | "Generate fluent ml text" is unverifiable without a use case; bpc bar is weak evidence alone; demo value low |
| (b) Base model for downstream fine-tuning | **CHOSEN (primary)** | Small-team/timeline reality: we can control pretraining quality end-to-end; every later stage (dialect, optional SFT) consumes this artifact; measurable via held-out bpc vs existing models |
| (c) Dialect-research artifact | **CHOSEN (secondary)** | Stage 6 is already in the project goal; dialect adaptation *on top of* the base model is the differentiating research contribution |
| (d) Other (chatbot/assistant) | Rejected | Requires instruction data + RLHF-adjacent work with licensing risk (Stage 7); timeline-infeasible; no eval capacity for dialogue quality |

**Decision: (b) + (c).** A Malayalam-native base model whose first downstream
application is dialect adaptation. The base model is the deliverable; dialect
adapters demonstrate its value.

## 2. Success criteria (falsifiable, measurable)

- **SC1 — Tokenization:** native tokenizer cuts tokens/word (fertility) by
  ≥ 40% vs. at least two widely-used multilingual tokenizers (MuRIL, XLM-R)
  on held-out Malayalam text. Measured in Stage 1.
- **SC2 — Base model quality:** held-out Malayalam bits-per-character (bpc,
  cross-tokenizer comparable) beats at least one established baseline in the
  same-or-larger size class — candidates: IndicBERTv2 (278M, trained on
  IndicCorpV2), MuRIL (238M), XLM-R-base (270M). Measured in Stage 5 on one
  identical held-out set. Stretch: beat all three.
- **SC3 — Human fluency:** native speakers rate ≥ 70% of 50 unconditional
  samples fluent (binary per sample). Protocol: 1–2 raters (project member +
  one recruited native speaker), fixed prompt set, blind to model identity.
- **SC4 — Dialect adaptation:** ≥ 15% perplexity reduction on dialect held-out
  text vs. base model, with ≤ 2% relative regression on standard Malayalam
  held-out text. Measured in Stage 6 ablation.

Failure of SC2 alone does not kill the project if SC1 + SC4 hold strongly
(native tokenization + dialect adaptation are contributions independent of
raw loss parity); this tradeoff is stated up front.

## 3. Non-goals

- **Instruction tuning / RLHF: OUT of core scope.** Justification: (i)
  distillation-based instruct data carries provider-ToS risk (Stage 7 check
  pending anyway), (ii) no capacity for dialogue-quality eval, (iii) timeline.
  A tiny supervised demo fine-tune is allowed *only* as a post-Stage-6 stretch
  using openly-licensed ml instruction data, decided by Stage 7's assessment.
- **Multimodal: OUT.** No data, no compute case, no eval path.
- **Beating large commercial models: OUT.** Not a criterion anywhere above.

## 4. Model scale (revisited — is 300M–1B justified?)

**Data is the ceiling; more AWS compute does not change that.**

Evidence now: IndicCorpV2 ml = 26.3 GB raw (CommonCrawl-heavy), mlwiki ≈ 200 MB
compressed. Expected *clean, deduped, LID-passed* corpus ≈ 1–2.5 B native
tokens (IndicCorpV2 raw is 10× its clean core; exact number measured in Stage
2 — this is an estimate, flagged as such).

Applying ~20 tokens/param (Chinchilla-style reference, not gospel):

| Clean tokens | Optimal-ish params | Verdict |
|---|---|---|
| 1 B | ~50 M | under-uses available compute/data |
| 2 B | ~100 M | **target zone** |
| 5 B | ~250 M | not reachable with measured sources |

- **300M–1B rejected:** would require 6–20 B clean tokens that do not exist in
  measured sources; multi-epoch training at that scale on a CC-derived corpus
  trades overfitting risk for negligible gains, and an 8 GB laptop can't even
  sanity-run such configs cheaply.
- **Embedding cost at ml-native vocab (32k–50k, Stage-1 decision):** small
  either way; not a scaling bottleneck.
- **Decision: target ~100–125 M params** (roughly d_model 768, 12 layers,
  12 heads), revisited only if Stage-2 measured clean tokens exceed ~3 B.

## 5. Where evidence is insufficient

- Exact clean token count (drives final param count) → measured by Stage 2
  pipeline itself; no separate experiment needed.
- CC100 ml accessibility (URL moved) → resolved during Stage 2 source
  enumeration; CC-derived overlap with IndicCorpV2 means it's additive, not
  critical.
- Whether IndicCorpV2's ml.txt quality survives LID/junk filtering → Stage 2
  filter statistics will answer; if > 70% is dropped, scale drops accordingly
  and Stage 4 math re-runs.

# Stage 6 — Dialect Adaptation (Methodology Assessment)

Status: ASSESSED + scripts built. Runs execute after the base model exists
(Stage 4). No dialect training has run yet — all numbers below are
source-availability facts, not results.

## 1. What "Malayalam dialect" means for data sourcing (assessment)

Written web Malayalam (news, Wikipedia) is standardized register. Dialectal
variation (North Malabar vs Central vs South Travancore, Kasaragod, Lakshadweep
Jeseri) lives in speech, film dialogue, and social media. This constrains every
data option below: there is no large, publicly available, dialect-labeled
Malayalam prose corpus.

## 2. Data sourcing options

| Source | Realistic volume | Authenticity | Licensing/consent | Verdict |
|---|---|---|---|---|
| Film dialogue subtitles (OpenSubtitles v2024, ml side) — **verified available** (58.8MB zip, HTTP 200) | ~250–400MB raw ml text after unzip, less after cleaning | High for *colloquial register*; **not geographically dialect-labeled** | Subtitles are copyrighted derivative works: research-use tolerance, no redistribution → keep in restricted prefix per Stage 8 | **Chosen as primary adaptation corpus** — honest framing: register adaptation (colloquial/film), not labeled-dialect adaptation |
| Kasaragod dialect lexicon (IEEE DataPort, 1,577 dialect→standard entries) | ~1.6k word pairs | High (human-collected) | Published dataset | **Chosen as eval seed + lexical augmentation source** |
| Transcribed speech corpora (e.g., Lekshmi et al. 2020 narrational/interview) | Unknown, likely small; scripted speech is standard register | High where dialectal | Access/licensing unverified | Rejected for now (insufficient information to decide — would need a data-use request; not blocking) |
| Social media scraping | Potentially large | Mixed, Manglish-heavy | Site ToS + consent problems | Rejected |
| Synthetic teacher-rewriting (strong LLM rewrites standard→dialect) | Unlimited | **Low**: current strong models do not produce authentic regional Malayalam dialect reliably | Provider ToS restrictions (see Stage 7) | Rejected as primary; allowed only as Kasaragod *augmentation* via the lexicon, flagged lower-confidence, gated on native validation (below) |

**Per-dialect realistic corpus sizes, stated before committing:**
- Colloquial/film register: ~50–150MB clean text (measured after cleaning; the
  pipeline reports exact numbers).
- Kasaragod: ~1.6k lexical pairs + lexicon-substituted synthetic prose
  (lower-confidence, validation-gated). No authentic Kasaragod prose corpus of
  training scale exists publicly — stated plainly rather than assumed.

## 3. Adaptation method: LoRA vs full fine-tuning vs other PEFT

With cloud compute, full-FT cost is not the constraint — *data volume* is.
Decision is therefore deferred to a measured head-to-head ablation (the spec's
requirement), not chosen by default:

- **Arms:** {LoRA (r=16, attn+mlp), full-FT (low LR)} × {film-register data,
  Kasaragod-augmented data} = 4 arms + base model as control.
- **Metrics:** dialect-set val loss (target ≥15% ppl reduction vs base — SC4)
  AND standard-Malayalam val loss (catastrophic-forgetting check: ≤2% relative
  regression — SC4).
- **Prediction (to be falsified by measurement):** with ~100MB-class data,
  full-FT on a 110M model overfits and forgets; LoRA wins on the forgetting
  metric. If measurement says otherwise, full-FT is adopted.
- Prefix-tuning/(IA)^3 noted as future work; two arms are what the ablation
  budget supports.

## 4. Synthetic-data validation gate (per spec)

Any lexicon-substituted synthetic Kasaragod text must pass: 50 samples, 2
native raters, ≥80% judged "natural Kasaragod phrasing" before entering
training. If the gate fails, synthetic data is dropped and Kasaragod scope
shrinks to eval probes only. Not yet run — no raters engaged yet.

## 5. Compute plan (per Stage 8 rules)

Each adaptation arm: g5.xlarge Spot, 1–2h, ~$1–2. Checkpoints/adapters sync to
`s3://…/adapters/<dialect>/<method>/`. Total ablation ≤ $8. Runs locally on the
4060 for a smoke pass first (same sanity pattern as Stage 3).

## 6. Build artifacts (ready, unexecuted)

- `scripts/dialect_data.py` — download/clean OpenSubtitles ml; build
  Kasaragod eval probes from the lexicon; emit train/val text per target.
- `scripts/train_lora.py` — LoRA wrapper + training loop on dialect bins;
  evaluates BOTH dialect and standard val each interval.
- `scripts/run_ablation.sh` — runs all 4 arms + control, writes
  `eval/dialect_ablation.json`.

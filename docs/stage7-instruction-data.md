# Stage 7 — Instruction Data (scope decision)

Status: DECIDED — OUT of core scope. Optional post-Stage-6 stretch, tightly
constrained by licensing. Written 2026-10-23; no instruction data has been
generated or collected.

## Options considered

| Option | Volume achievable | Cost | Licensing risk | QC feasibility | Verdict |
|---|---|---|---|---|---|
| Distillation from a strong teacher (GPT/Claude/Gemini outputs) | 10k–100k samples easily | API spend | **High**: OpenAI/Anthropic/Google terms restrict using outputs to train competing models; "competing" is arguably any LM | Medium (needs native review) | Rejected |
| Translate existing English instruction set (Alpaca/OpenAssistant) + native review | 50k+ | Cheap (or free via IndicTrans2) | Low (OSA is Apache-2.0, Alpaca is for research) but translation-quality risk | Medium: needs native review of mt quality | Stretch-only |
| Scraped ml Q&A/forum (e.g., public forums) | Small, noisy | Scraping effort | Medium–high (site ToS, personal data) | Low (no raters for volume) | Rejected |
| Hand-written seed + self-instruct | ~1k seeds, expanded | Team time | Low | High per-item | Rejected for core (timeline) |
| Do nothing (base model + dialect only) | — | — | — | — | **Core scope** |

## Decision

1. **Core scope excludes instruction tuning** — justified by (i) ToS risk on the
   only high-volume path (teacher distillation), (ii) no eval capacity for
   dialogue/instruction quality (Stage-0 SC3 covers fluency, not task-following),
   (iii) timeline.
2. **Stretch (only after Stage 6 lands):** a small LoRA SFT for demo purposes
   using ONLY datasets carrying explicit open licenses (Apache-2.0/CC-BY/ODC-BY
   or CC0), machine-translated to Malayalam via an openly-licensed translation
   model, with a documented quality filter:
   - language-ID (script ratio ≥ 0.8), length filter, exact dedup
   - manual spot-check: 100 samples, 1 native rater, ≥ 80% pass or the set is
     dropped
   - marked demo-quality; NOT a project deliverable or success criterion.
3. If any option would use an external model's outputs, the provider's current
   ToS must be re-checked at execution time (they change; the table above
   reflects the assessment date, not a permanent clearance).

# Next Tasks — Tokenizer and GPT Refinement

Handover notes for the three workstreams after the current demo checkpoint.
Each section is a self-contained prompt: copy it into the member's coding agent
and have them work on a branch, not on `main`.

Shared rules:

- All numbers must be measured from committed scripts or logs. Label estimates
  as estimates. Never present an unmeasured number as a result.
- Do not launch AWS resources. Cloud work is limited to scripts, measurements
  and cost estimates.
- Do not commit `data/`, `checkpoints/`, `*.pt`, `*.log` or secrets.
- Use `PYTHONIOENCODING=utf-8` on Windows.
- Keep the current 50k tokenizer artifact and 590.2M-token bins intact; any
  replacement artifact must keep the old one and state the re-tokenization cost.
- Open a PR with: what changed, commands run, measured before/after numbers,
  and what is still unmeasured.

---

## 1. Anjana M — Tokenizer refinement and tokenizer-side evaluation

You are working in the Malayalam SLM repository. Your job is to audit and
refine the native tokenizer with measurements, not assumptions. Read
`README.md`, `docs/stage1-tokenizer-assessment.md`, `docs/data_report.md`,
`scripts/eval_tokenizers.py`, `scripts/train_tokenizers.py`,
`scripts/prepare_tokenizer_corpus.py`, `scripts/split_tokenize.py` and
`scripts/text_to_bins.py` first.

Current state: final artifact is a 50,000-piece SentencePiece Unigram model with
`character_coverage=1.0`, byte fallback, identity normalization and NFC input.
On a held-out set it measured 1.775 tokens/word, 0.1734 tokens/char, 0 UNK and
0.8975 exact roundtrip (all misses are SentencePiece collapsing double spaces).
The 590.2M-token local bins were produced with this tokenizer.

Work items:

1. Reproduce the Stage-1 benchmark from scratch. If
   `data/tokenizer_work/eval_combined.txt` is absent, rebuild the tokenizer
   corpus with `python scripts/prepare_tokenizer_corpus.py`, then run
   `python scripts/eval_tokenizers.py`. Confirm the published numbers before
   changing anything; if they differ, explain why.
2. Add `scripts/audit_tokenizer.py` that measures, on the final 50k model:
   - tokens/word, tokens/char, UNK rate and exact/whitespace-normalized
     roundtrip on at least 20,000 held-out paragraphs, broken down by source
     (CC100, Wikipedia, IndicCorpV2).
   - byte-fallback usage: count `<0xXX>` pieces over a large sample, list the
     most frequent fallback bytes and the words/characters that trigger them.
   - vocabulary utilization: distinct pieces used, never-used pieces, top-100
     pieces by frequency, and the effective vocabulary size on real text.
   - word-level fragmentation: distribution of tokens per word and the worst
     100 frequent words by token count.
3. Add `tests/test_tokenizer.py` covering: NFC roundtrip, ZWJ and ZWNJ
   preservation, chillu forms, complex conjuncts (`സ്ത്രീ`, `ക്ഷേത്രം`,
   `ജ്ഞാനം`, `വിദ്യാഭ്യാസം`, `ശ്രീകൃഷ്ണൻ`), `അമ്മയ്‍`, `ൠ`, `ഋ`, numerals,
   punctuation, ASCII words, URLs and emoji. Tests must assert exact decode,
   not just "no exception".
4. Check tokenizer corpus stability: retrain a 50k Unigram candidate on two
   disjoint 250MB samples of the cleaned corpus and compare fertility and
   token-ID overlap. Conclude whether the current artifact is stable enough to
   freeze. Do not replace the final artifact unless the measured gain is real
   and you also state the cost of re-tokenizing 590.2M tokens (or the full
   corpus).
5. Fix only real defects you can demonstrate (for example incorrect BOS/EOS
   handling, document-separator handling, uint16 overflow, or decode
   normalization), and add a regression test for each fix. Do not change token
   IDs silently.
6. Write `docs/stage1-tokenizer-refinement.md` with method, measured tables,
   failures found, decision and exact reproduction commands. Update
   `eval/tokenizer_comparison.json` only from a fresh run, and update the
   Stage-1 row in `README.md` only if the measured numbers change.

Acceptance: the audit is reproducible on this machine, all tests pass, every
number is traceable to a command, and the existing bins still match the final
tokenizer (or the required re-tokenization is explicitly documented).

Branch: `tokenizer/refinement-anjana`.

---

## 2. Paul Mathew — GPT architecture and training-loop refinement

You are working in the Malayalam SLM repository. Your job is to harden and
measure the decoder-only GPT and its training loop, then produce evidence for
the 110M-parameter configuration. Read `src/model.py`, `src/dataloader.py`,
`scripts/train.py`, `scripts/eval_model.py`, `docs/stage3-training-assessment.md`,
`docs/stage4-sizing.md`, `docs/sanity-logs.md` and `demo/README.md` first.

Current state: `src/model.py` is a pre-LN GPT with causal SDPA, tied embeddings,
learned absolute positions and GELU. The local demo is 4 layers / 4 heads /
256 d (16.1M params, 50k vocab), trained 8,000 steps on the 590.2M-token bins;
best validation loss 4.7567, held-out BPC 1.3101. The target is 12/12/768,
which measures 124.1M params at a 50k vocab. There is no test suite yet.

Work items:

1. Fix and test correctness issues before tuning:
   - `GPT.num_params(non_embedding=True)` currently subtracts only the position
     table and only when embeddings are untied. Define "total" and
     "non-embedding" counts correctly for tied and untied models, and test both.
   - Validate config (head dimension, block size, vocab size against the
     tokenizer) and save a machine-readable config with every checkpoint.
   - Add a causal-mask test: changing tokens at positions `t+1..T` must not
     change logits at position `t`.
   - Add a resume test: save, resume with the same seed, and verify the next
     optimizer step and LR match an uninterrupted run within tolerance.
   - Check gradient accumulation: loss must be scaled correctly, clipping must
     happen once per optimizer step, and the result should match a single large
     batch within tolerance.
2. Review `src/dataloader.py`. Random windows currently can start inside a
   document and cross document boundaries; bins concatenate documents with EOS.
   Measure whether this matters (document-boundary-aware packing or masking
   versus the current sampler) and document the decision. Add deterministic
   train/val sampling and a fixed validation batch.
3. Build `tests/` (pytest) for model shape, causal masking, tied weights, init
   statistics, the `ln(V)` gate, overfit-one-batch, checkpoint roundtrip and
   dataloader bounds. Keep tests fast enough to run on CPU.
4. Run controlled ablations at the 16M scale on the existing bins, one change
   at a time, fixed seed and fixed token budget (state steps and tokens):
   - baseline: learned positions, GELU, tied embeddings, dropout 0.0;
   - RoPE instead of learned positions;
   - SwiGLU instead of GELU (adjust d_ff to keep params comparable);
   - untied embeddings;
   - dropout 0.1;
   - block size 512 vs 1024, if the 8GB GPU allows.
   Report validation loss/BPC, tokens/s, peak VRAM and wall time for each.
   Reject variants that do not beat the baseline at equal tokens.
5. 110M readiness on the RTX 4060 (no cloud launch): verify the parameter count
   at 50k vocab, measure forward/backward memory and tokens/s at context 1024
   for the largest batch that fits, test activation checkpointing if needed,
   and test `torch.compile` only if it gives a measured speedup without
   breaking CPU or checkpoint loading. Update the Stage-4 time and cost
   estimate with measured throughput; clearly label the cloud cost as an
   estimate.
6. Write `docs/stage3-model-refinement.md` with the bug fixes, ablation table,
   recommended configuration and rejected options. Update `README.md` Stage 3/4
   only with measured numbers.

Acceptance: all sanity gates still pass (fresh loss ≈ ln(50000), overfit below
0.1), the new tests pass, each ablation has a reproducible command, and no
result is claimed without a log.

Branch: `model/refinement-paul`.

---

## 3. Arshad V A — Evaluation, scale-up and demo refinement

You are working in the Malayalam SLM repository. Your job is to make the
evaluation defensible, prepare the full-corpus scale-up, and harden the demo and
handover. Read `scripts/eval_model.py`, `eval/results.json`, `demo/app.py`,
`demo/README.md`, `aws/README.md`, `docs/stage4-sizing.md`,
`docs/stage6-dialect-assessment.md` and `docs/stage7-instruction-data.md` first.

Current state: the demo checkpoint (16.1M, step 8000) scored 1.3101 BPC on 300
held-out documents. Baselines scored with MLM pseudo-log-loss over the first 400
tokens (MuRIL 1.4818, IndicBERTv2 0.9776, XLM-R 0.6433); our model scores full
documents, so the comparison has a documented but unresolved asymmetry. The
local bins contain 590.2M tokens from a bounded 8.5GB slice; the full ~29GB
clean corpus (~1.8B tokens, estimated) is not tokenized. No AWS resource has
been launched.

Work items:

1. Make the BPC comparison apples-to-apples. Score our model and each baseline
   on the same character/token prefix (for example the first 400 tokens for
   every model) and also report full-document numbers separately. Add per-document
   bootstrap 95% confidence intervals. Keep the method caveat in the output.
2. Increase the held-out evaluation set if runtime allows (target 1,000
   documents), record evaluation wall time, and add a synthetic test with known
   character length to verify the char denominator and offset mapping.
3. Evaluate the step-8000 checkpoint on the held-out test split and refresh
   `eval/results.json`, `eval/human_eval_samples.txt` and the demo evaluation
   panel. Add zero-shot completion samples with fixed prompts and seeds; state
   that they are samples, not a fluency score.
4. Measure the full-corpus tokenization plan on this machine: input size,
   expected tokens, free disk, RAM, tokenizer throughput and wall time. If it
   fits, run it without deleting the current bins and write a manifest with
   checksums. If it does not fit, produce the exact command and a measured
   partial run, and state the remaining requirement. Do not commit the data.
5. Prepare the 1.7B-token training run for handoff: verify bin integrity,
   checksums and S3 sync manifest, update the Spot time/cost estimate using
   Paul's measured tokens/s, and confirm the IAM policy is least-privilege.
   Do not launch anything.
6. Harden the demo: auto-load the newest/best checkpoint, show confidence
   intervals and the method caveat, add a `/health` endpoint, handle a missing
   checkpoint without crashing, and add a smoke test that starts the app and
   checks all API endpoints. Verify `demo/run_demo.bat` from a clean shell.
7. Write `docs/stage5-evaluation.md`, `docs/aws-cost-summary.md` and a final
   report skeleton that maps each Stage-0 success criterion to measured
   evidence, explicitly listing what remains unmeasured.

Acceptance: all evaluation numbers are reproducible and the asymmetry is
removed or clearly quantified; the demo runs from a clean clone; no cloud
resource is launched; the cost summary is labeled as an estimate.

Branch: `eval/scale-arshad`.

---

## Coordination

- One branch and one PR per workstream; do not push directly to `main`.
- Anjana owns tokenizer files and tokenizer tests; Paul owns model, dataloader,
  training loop and model tests; Arshad owns evaluation, demo, AWS scripts and
  docs. Cross-cutting changes need a note in the PR.
- Rebase on `main` before opening the PR. Report measured results in the PR
  description, not only in commit messages.
- If a change would invalidate the current bins or checkpoint, stop and raise
  it before merging.

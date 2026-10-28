# Malayalam SLM — Project Diary

> Dates are left blank to be filled from the team's meeting records. Team: Athul Sunil Kumar, Anjana M, Paul Mathew, Arshad V A.

## Problem Definition and Objectives

Malayalam is spoken by about 35 million people, yet open language models still treat it as a low-resource language. Multilingual tokenizers fragment Malayalam words into roughly 2.0–2.7 subword tokens per word, which lengthens sequences, raises training and inference cost, and weakens morphological fidelity. There is no small, reproducible, Malayalam-native base model with measured tokenizer, data, training and evaluation evidence that a student team can build and audit end to end.

This project builds a Malayalam Small Language Model from scratch: a native 50k SentencePiece tokenizer, a cleaned and deduplicated corpus with document-level splits, a decoder-only transformer (110M-parameter target, 16.1M local demo), a cross-tokenizer bits-per-character evaluation against public baselines, and the infrastructure for dialect adaptation and a budgeted Spot pretraining run.

**Objectives**

- Train a native 50k Malayalam tokenizer with 0 UNK and lower fertility than MuRIL, XLM-R and IndicBERTv2.
- Build a reproducible corpus pipeline with measured cleaning, deduplication and 98/1/1 document-level splits.
- Implement and sanity-check a decoder-only transformer and a resumable training loop.
- Evaluate with exact causal bits-per-character and MLM pseudo-BPC baselines, stating method limits.
- Prepare a LoRA vs full-fine-tuning dialect ablation and a budgeted AWS Spot execution plan.
- Record every methodology decision with measured evidence; never fabricate numbers.

## Project Workflow

Each stage follows the same cycle: assess the methodology and record the decision, implement it, measure the result, and review before moving on. A failed gate sends the team back to the assessment instead of forward.

| Stage | Main work | Output / gate |
|---|---|---|
| 0 | Target and scope | Stage-0 decision document |
| 1 | Tokenizer candidates, benchmark, final artifact | 50k Unigram model + fertility table |
| 2 | Cleaning, LID, dedup, split, tokenization | uint16 bins + data report |
| 3 | Transformer and training loop | ln(vocab) and overfit sanity gates |
| 4 | Sizing, budget, local validation | Sizing document + local run |
| 5 | BPC evaluation harness and baselines | results.json + method caveats |
| 6 | Dialect adaptation protocol | LoRA/full-FT assessment + scripts |
| 7 | Instruction-tuning scope | Scope decision document |
| 8 | AWS S3/IAM/Spot infrastructure | Launch scripts + cost plan |

Stages are iterative rather than strictly linear: the tokenizer feeds the data pipeline, the data feeds training, and evaluation can send the team back to data or model design. Work is integrated through Git, and every measured number in the diary comes from a script or log in the repository.

## Project Design

**Tokenizer:** SentencePiece Unigram, 50,000 pieces, character coverage 1.0, NFC input, byte fallback; ZWJ/ZWNJ and chillu forms preserved. Measured 1.775 tokens/word and 0 UNK.

**Data:** Streaming cleaning (NFC, length, Malayalam script-ratio LID, Manglish/URL/repetition filters) → per-source and global dedup → document-level 98/1/1 split (seed 1337) → uint16 bins with BOS/EOS.

**Model:** Decoder-only pre-LN transformer with causal scaled dot-product attention, tied embeddings, learned positions and GELU. 110M target (12 layers / 12 heads / 768 d); 16.1M local demo (4/4/256); context 512 demo, 1024 planned.

**Training:** AdamW (β 0.9/0.95, weight decay 0.1), warmup + cosine decay to 10%, gradient clipping 1.0, bf16, gradient accumulation, atomic checkpoints and resume.

**Evaluation:** Exact causal nats-per-character converted to bits-per-character; MLM pseudo-perplexity for public baselines with fixed masking seed; results reported side by side, never aggregated.

**Dialect:** Four-arm ablation: LoRA vs full fine-tuning on Colloquial Subtitles and Kasaragod dialect data, evaluated by held-out BPC and sampled output review.

**Infrastructure:** Local RTX 4060 for development and demo; AWS S3 + Spot g5.xlarge for the 110M full-corpus run.

## Platform / Tools Used

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Deep learning | PyTorch 2.7 with CUDA 11.8, bf16 autocast, scaled dot-product attention |
| Tokenization | SentencePiece, HuggingFace tokenizers/transformers |
| Data | NumPy memmaps, streaming gzip/text readers, blake2b dedup |
| Evaluation | Custom causal BPC harness, MLM pseudo-perplexity baselines |
| Demo | FastAPI + Uvicorn, offline HTML/CSS/JavaScript |
| Cloud | AWS S3, EC2 Spot (g5.xlarge), IAM least privilege |
| Collaboration | Git, GitHub, Markdown assessment documents |
| Hardware | RTX 4060 Laptop GPU (8 GB), Windows 11 |

## Team and Roles

| Member | Role |
|---|---|
| Athul Sunil Kumar | Model & Training |
| Anjana M | Tokenizer & Evaluation |
| Paul Mathew | Data Pipeline |
| Arshad V A | Infrastructure, Demo & Documentation |

---

## Weekly Assignment and Progress Assessment

**Week No:** 1  
**Date:** ____________________  
**Progress (E / G):** ________

**Works Assigned to the Project Team**

- Review available Malayalam corpora and multilingual baselines; define the target model, success criteria and team roles.
- Set up the repository, Python/PyTorch environment and stage-wise workflow.

**Works completed**

- Surveyed Malayalam resources (CC100, IndicCorpV2, Malayalam Wikipedia) and public baselines (MuRIL, XLM-R, IndicBERTv2).
- Wrote the Stage-0 target document: 110M-parameter decoder-only base model, native tokenizer, dialect adaptation, and measurable acceptance criteria.
- Initialized the repository and the assess → implement → measure → review workflow for all eight stages.

**Works pending if any**

None.

**Individual Contribution**

| Member | Contribution |
|---|---|
| Athul Sunil Kumar | Drafted the target/scope document, model-size rationale and training environment. |
| Anjana M | Surveyed baseline tokenizers and defined fertility/UNK comparison criteria. |
| Paul Mathew | Inventoried corpus sources, sizes, licences and availability. |
| Arshad V A | Set up Git/GitHub, folder structure and documentation templates. |

**Any other matters discussed during the meeting**

Decision: write a methodology assessment before coding each stage, and defer all cloud spending until the local validation gate passes.

**Student Name and Signature:** ________________________________  
**Signature of the Guide:** ________________________________


---

## Weekly Assignment and Progress Assessment

**Week No:** 2  
**Date:** ____________________  
**Progress (E / G):** ________

**Works Assigned to the Project Team**

- Assess tokenizer algorithms, vocabulary sizes, normalization and character coverage for Malayalam.
- Prepare tokenizer training/evaluation text and define the candidate grid.

**Works completed**

- Compared SentencePiece Unigram, SentencePiece BPE and byte-level BPE; selected Unigram for Malayalam morphology.
- Defined eight candidates across 16k/32k/50k vocabularies and character coverage 1.0 vs the 0.9995 warning test.
- Assembled a held-out tokenizer evaluation corpus with a chillu/ZWJ word list.

**Works pending if any**

Candidate training and the full benchmark (next week).

**Individual Contribution**

| Member | Contribution |
|---|---|
| Athul Sunil Kumar | Set selection criteria (fertility, UNK, roundtrip, cost) and the candidate grid. |
| Anjana M | Implemented candidate training and evaluation scripts; sampled tokenizer training text. |
| Paul Mathew | Prepared cleaned tokenizer-training text and the held-out evaluation set. |
| Arshad V A | Set up experiment tracking and reproducibility notes. |

**Any other matters discussed during the meeting**

fastText has no working build in the Python 3.13 environment, so a Unicode script-ratio language filter was evaluated as the fallback candidate.

**Student Name and Signature:** ________________________________  
**Signature of the Guide:** ________________________________


---

## Weekly Assignment and Progress Assessment

**Week No:** 3  
**Date:** ____________________  
**Progress (E / G):** ________

**Works Assigned to the Project Team**

- Train all tokenizer candidates; measure fertility, UNK rate, roundtrip and chillu/ZWJ handling.
- Select and package the final tokenizer artifact.

**Works completed**

- Trained eight candidates and benchmarked them on held-out Malayalam text.
- Selected the 50k SentencePiece Unigram model: 1.775 tokens/word vs MuRIL 2.378, IndicBERTv2 1.938 and XLM-R 2.746, with 0 UNK.
- Verified exact Malayalam roundtrip (SentencePiece only collapses repeated spaces) and correct chillu/ZWJ segmentation; packaged the model, vocab and comparison JSON.

**Works pending if any**

None.

**Individual Contribution**

| Member | Contribution |
|---|---|
| Athul Sunil Kumar | Reviewed fertility/UNK trade-offs and approved the 50k Unigram configuration. |
| Anjana M | Ran the eight-candidate benchmark and chillu/ZWJ tests; produced the results table. |
| Paul Mathew | Verified normalization and roundtrip behaviour on corpus samples. |
| Arshad V A | Packaged artifacts and documented the tokenizer decision. |

**Any other matters discussed during the meeting**

The 0.9995 coverage warning was verified empirically, not assumed: it produced UNKs. Coverage 1.0 with byte fallback won; byte-level BPE was about 4× worse on fertility.

**Student Name and Signature:** ________________________________  
**Signature of the Guide:** ________________________________


---

## Weekly Assignment and Progress Assessment

**Week No:** 4  
**Date:** ____________________  
**Progress (E / G):** ________

**Works Assigned to the Project Team**

- Design the corpus cleaning, language filtering, deduplication and split strategy.
- Implement a streaming pipeline suitable for a memory-constrained machine.

**Works completed**

- Fixed the source set: CC100 Malayalam, IndicCorpV2 Malayalam and Malayalam Wikipedia.
- Specified filters: NFC, 60–8192-character paragraphs, ≥70% Malayalam-script ratio, Manglish/URL/repetition guards, and per-source blake2b-64 exact dedup.
- Designed the two-pass streaming splitter: document-level 98/1/1 split with PCG64 seed 1337, uint16 bins and a \x1e document separator.

**Works pending if any**

End-to-end pipeline execution (next week).

**Individual Contribution**

| Member | Contribution |
|---|---|
| Athul Sunil Kumar | Defined cleaning thresholds and split invariants. |
| Paul Mathew | Implemented cleaning, script-ratio LID and the Manglish/URL/repetition filters. |
| Anjana M | Implemented tokenizer integration and UNK/roundtrip checks on cleaned text. |
| Arshad V A | Estimated disk/RAM needs and documented data provenance. |

**Any other matters discussed during the meeting**

Limited local RAM and disk ruled out in-memory processing, so the splitter reads sources twice and writes only compact uint16 bins.

**Student Name and Signature:** ________________________________  
**Signature of the Guide:** ________________________________


---

## Weekly Assignment and Progress Assessment

**Week No:** 5  
**Date:** ____________________  
**Progress (E / G):** ________

**Works Assigned to the Project Team**

- Run the pipeline on available sources; generate train/val/test bins, statistics and held-out test text.
- Validate bin integrity and document the data report.

**Works completed**

- Produced a test-scale corpus of 316.8M tokens (train 310.5M / val 3.15M / test 3.18M) with 0 UNK.
- Verified exact bin sizes, token-id ranges and a re-encode match against test.txt; confirmed document-level split isolation.
- Wrote the data report with per-source cleaning, dedup and split statistics and reproduction commands.

**Works pending if any**

Full-corpus tokenization; the run used bounded public-source slices, recorded as a scale caveat.

**Individual Contribution**

| Member | Contribution |
|---|---|
| Athul Sunil Kumar | Reviewed split leakage and validation evidence. |
| Paul Mathew | Ran cleaning and dedup; recorded per-source statistics. |
| Anjana M | Ran tokenization/split and verified bin sizes and re-encoding. |
| Arshad V A | Wrote the data report and reproduction instructions. |

**Any other matters discussed during the meeting**

Bounded slices were used because the original corpora were absent and local storage was limited; the report states this explicitly rather than presenting test-scale numbers as full-corpus results.

**Student Name and Signature:** ________________________________  
**Signature of the Guide:** ________________________________


---

## Weekly Assignment and Progress Assessment

**Week No:** 6  
**Date:** ____________________  
**Progress (E / G):** ________

**Works Assigned to the Project Team**

- Implement the decoder-only transformer and resumable training loop.
- Run the mandatory sanity gates.

**Works completed**

- Implemented the pre-LN causal transformer with tied embeddings, learned positions, GELU, scaled dot-product attention, and both tiny (16.1M) and 110M configurations.
- Added AdamW, warmup + cosine schedule, gradient clipping, bf16, gradient accumulation, atomic checkpointing and resume.
- Passed both sanity gates: initial loss 10.8230 vs ln(50000)=10.8198 (Δ +0.0033), and a tiny-batch overfit run reaching near-zero loss.

**Works pending if any**

Full-scale pretraining; the cloud run has not been launched.

**Individual Contribution**

| Member | Contribution |
|---|---|
| Athul Sunil Kumar | Implemented the model, initialization scheme and optimizer settings. |
| Paul Mathew | Implemented the memmap dataloader and validation batching. |
| Anjana M | Added loss/evaluation instrumentation and the sanity-gate scripts. |
| Arshad V A | Implemented checkpoint save/resume and run scripts. |

**Any other matters discussed during the meeting**

Sanity gates were treated as blocking: no training run was accepted until the fresh-model loss matched ln(vocab) and the overfit test converged.

**Student Name and Signature:** ________________________________  
**Signature of the Guide:** ________________________________


---

## Weekly Assignment and Progress Assessment

**Week No:** 7  
**Date:** ____________________  
**Progress (E / G):** ________

**Works Assigned to the Project Team**

- Compute the compute/data budget and pretraining plan.
- Build the cross-tokenizer evaluation harness and run local validation.

**Works completed**

- Completed Stage-4 sizing: 110M parameters, 1024 context, ~1.7B-token target, Spot g5.xlarge for about 6 hours at an estimated ~$2.50.
- Ran the local 2,000-step validation gate on the tokenized data.
- Built the evaluation harness: exact causal nats/char → BPC, plus MLM pseudo-perplexity for MuRIL/XLM-R/IndicBERTv2 with a fixed masking seed and documented caveats.

**Works pending if any**

Baseline evaluation on a final model and human-fluency scoring.

**Individual Contribution**

| Member | Contribution |
|---|---|
| Athul Sunil Kumar | Produced the sizing math and cloud budget. |
| Paul Mathew | Validated data throughput and token counts. |
| Anjana M | Implemented the BPC harness and fixed a logits bug that scored only the last position. |
| Arshad V A | Profiled GPU memory/throughput and prepared the cost sheet. |

**Any other matters discussed during the meeting**

BPC was chosen over perplexity so tokenizers can be compared on equal footing; MLM pseudo-log-loss is reported only side by side with causal BPC because it uses bidirectional context.

**Student Name and Signature:** ________________________________  
**Signature of the Guide:** ________________________________


---

## Weekly Assignment and Progress Assessment

**Week No:** 8  
**Date:** ____________________  
**Progress (E / G):** ________

**Works Assigned to the Project Team**

- Build a larger bounded corpus, train a demo checkpoint, run baseline evaluation and build the demo application.

**Works completed**

- Re-tokenized full CC100 + wiki + the first 2.5 GiB of IndicCorpV2: 590.2M tokens, 4.37M documents, 3.07B characters, 265,719 cross-source duplicates removed, 0 UNK.
- Trained the 16.1M demo model for 8,000 steps (393.2M tokens); final validation loss 4.7567 in 98.4 minutes on the RTX 4060 at zero cloud cost.
- Measured held-out BPC 1.3101 vs MuRIL 1.4818 (MLM pseudo), IndicBERTv2 0.9776 and XLM-R 0.6433, with method caveats.
- Built and browser-verified a FastAPI demo: live tokenizer comparison, generation, and a training/data dashboard.

**Works pending if any**

The 110M full-corpus cloud run and larger-scale evaluation.

**Individual Contribution**

| Member | Contribution |
|---|---|
| Athul Sunil Kumar | Trained the demo model and tuned batch/accumulation for 8 GB VRAM. |
| Paul Mathew | Produced the 590.2M-token slice and updated data statistics. |
| Anjana M | Ran baseline evaluation, BPC analysis and sample generation. |
| Arshad V A | Built and tested the demo app and documented the measured results. |

**Any other matters discussed during the meeting**

Laptop thermal throttling reduced throughput from ~92k to ~66k tokens/s; atomic checkpoints allowed the demo to keep serving the best model during training.

**Student Name and Signature:** ________________________________  
**Signature of the Guide:** ________________________________


---

## Weekly Assignment and Progress Assessment

**Week No:** 9  
**Date:** ____________________  
**Progress (E / G):** ________

**Works Assigned to the Project Team**

- Assess dialect adaptation and prepare the LoRA vs full-FT ablation.
- Prepare AWS infrastructure and the cost plan; decide the instruction-tuning scope.

**Works completed**

- Completed the dialect assessment: four-arm ablation (LoRA vs full fine-tuning on Colloquial Subtitles and Kasaragod data) with adapter-training and evaluation scripts.
- Prepared AWS S3 lifecycle rules, least-privilege IAM, Spot g5.xlarge launch and data-sync scripts, with a ~$2.50 estimated compute cost.
- Wrote the instruction-tuning scope decision, deferring implementation for time and terms-of-service reasons.

**Works pending if any**

Ablation execution (requires the base checkpoint and dialect data); the cloud run is not launched.

**Individual Contribution**

| Member | Contribution |
|---|---|
| Athul Sunil Kumar | Defined the LoRA/full-FT protocol and hyperparameters. |
| Paul Mathew | Prepared dialect data scripts and reviewed licensing. |
| Anjana M | Designed the dialect evaluation plan (held-out BPC plus samples). |
| Arshad V A | Wrote the IAM/S3/Spot scripts and cost estimate. |

**Any other matters discussed during the meeting**

Spot was chosen over on-demand for the interruptible training job; no cloud resource was launched without explicit budget approval.

**Student Name and Signature:** ________________________________  
**Signature of the Guide:** ________________________________


---

## Weekly Assignment and Progress Assessment

**Week No:** 10  
**Date:** ____________________  
**Progress (E / G):** ________

**Works Assigned to the Project Team**

- Finalize the demo and documentation; verify artifacts and list remaining work and risks.

**Works completed**

- Pulled the team repository, fixed the evaluation and tokenizer-bound bugs, rebuilt the demo corpus, trained and evaluated the demo checkpoint, and added the evaluation panel to the dashboard.
- Wrote the demo runbook, updated the project README/data notes, and saved ten human-evaluation samples for review.
- Re-verified every demo tab and API, and recorded remaining work with measured status rather than estimates.

**Works pending if any**

Tokenize the full ~29 GB corpus (~1.8B tokens estimated); launch the 110M Spot run; execute the dialect ablation; write the AWS cost summary and final Stage-0 comparison; score human fluency.

**Individual Contribution**

| Member | Contribution |
|---|---|
| Athul Sunil Kumar | Final checkpoint evaluation and results review. |
| Paul Mathew | Full-corpus scale-up plan: disk, RAM and runtime estimates. |
| Anjana M | Reproducibility checks and human-evaluation sample generation. |
| Arshad V A | Demo QA, documentation and diary compilation. |

**Any other matters discussed during the meeting**

Core pipeline, demo and measured evaluation are complete; the remaining gap is compute-bound (full corpus and 110M run). No 110M result will be claimed until it is measured.

**Student Name and Signature:** ________________________________  
**Signature of the Guide:** ________________________________

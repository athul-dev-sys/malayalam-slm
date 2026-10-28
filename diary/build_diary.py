"""Build the Malayalam SLM project diary (DOCX + Markdown).

Content is generated from the measured project artifacts in this repository.
Dates are intentionally left blank for the team to fill from meeting records.
"""
import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCX_PATH = os.path.join(OUT_DIR, "Malayalam_SLM_Project_Diary.docx")
MD_PATH = os.path.join(OUT_DIR, "Malayalam_SLM_Project_Diary.md")

TEAM = [
    ("Athul Sunil Kumar", "Model & Training"),
    ("Anjana M", "Tokenizer & Evaluation"),
    ("Paul Mathew", "Data Pipeline"),
    ("Arshad V A", "Infrastructure, Demo & Documentation"),
]

OBJECTIVES = [
    "Train a native 50k Malayalam tokenizer with 0 UNK and lower fertility than MuRIL, XLM-R and IndicBERTv2.",
    "Build a reproducible corpus pipeline with measured cleaning, deduplication and 98/1/1 document-level splits.",
    "Implement and sanity-check a decoder-only transformer and a resumable training loop.",
    "Evaluate with exact causal bits-per-character and MLM pseudo-BPC baselines, stating method limits.",
    "Prepare a LoRA vs full-fine-tuning dialect ablation and a budgeted AWS Spot execution plan.",
    "Record every methodology decision with measured evidence; never fabricate numbers.",
]

WORKFLOW = [
    ("0", "Target and scope", "Stage-0 decision document"),
    ("1", "Tokenizer candidates, benchmark, final artifact", "50k Unigram model + fertility table"),
    ("2", "Cleaning, LID, dedup, split, tokenization", "uint16 bins + data report"),
    ("3", "Transformer and training loop", "ln(vocab) and overfit sanity gates"),
    ("4", "Sizing, budget, local validation", "Sizing document + local run"),
    ("5", "BPC evaluation harness and baselines", "results.json + method caveats"),
    ("6", "Dialect adaptation protocol", "LoRA/full-FT assessment + scripts"),
    ("7", "Instruction-tuning scope", "Scope decision document"),
    ("8", "AWS S3/IAM/Spot infrastructure", "Launch scripts + cost plan"),
]

DESIGN = [
    ("Tokenizer", "SentencePiece Unigram, 50,000 pieces, character coverage 1.0, NFC input, byte fallback; ZWJ/ZWNJ and chillu forms preserved. Measured 1.775 tokens/word and 0 UNK."),
    ("Data", "Streaming cleaning (NFC, length, Malayalam script-ratio LID, Manglish/URL/repetition filters) → per-source and global dedup → document-level 98/1/1 split (seed 1337) → uint16 bins with BOS/EOS."),
    ("Model", "Decoder-only pre-LN transformer with causal scaled dot-product attention, tied embeddings, learned positions and GELU. 110M target (12 layers / 12 heads / 768 d); 16.1M local demo (4/4/256); context 512 demo, 1024 planned."),
    ("Training", "AdamW (β 0.9/0.95, weight decay 0.1), warmup + cosine decay to 10%, gradient clipping 1.0, bf16, gradient accumulation, atomic checkpoints and resume."),
    ("Evaluation", "Exact causal nats-per-character converted to bits-per-character; MLM pseudo-perplexity for public baselines with fixed masking seed; results reported side by side, never aggregated."),
    ("Dialect", "Four-arm ablation: LoRA vs full fine-tuning on Colloquial Subtitles and Kasaragod dialect data, evaluated by held-out BPC and sampled output review."),
    ("Infrastructure", "Local RTX 4060 for development and demo; AWS S3 + Spot g5.xlarge for the 110M full-corpus run."),
]

TOOLS = [
    ("Language", "Python 3.13"),
    ("Deep learning", "PyTorch 2.7 with CUDA 11.8, bf16 autocast, scaled dot-product attention"),
    ("Tokenization", "SentencePiece, HuggingFace tokenizers/transformers"),
    ("Data", "NumPy memmaps, streaming gzip/text readers, blake2b dedup"),
    ("Evaluation", "Custom causal BPC harness, MLM pseudo-perplexity baselines"),
    ("Demo", "FastAPI + Uvicorn, offline HTML/CSS/JavaScript"),
    ("Cloud", "AWS S3, EC2 Spot (g5.xlarge), IAM least privilege"),
    ("Collaboration", "Git, GitHub, Markdown assessment documents"),
    ("Hardware", "RTX 4060 Laptop GPU (8 GB), Windows 11"),
]

WEEKS = [
    {
        "n": 1,
        "assigned": [
            "Review available Malayalam corpora and multilingual baselines; define the target model, success criteria and team roles.",
            "Set up the repository, Python/PyTorch environment and stage-wise workflow.",
        ],
        "completed": [
            "Surveyed Malayalam resources (CC100, IndicCorpV2, Malayalam Wikipedia) and public baselines (MuRIL, XLM-R, IndicBERTv2).",
            "Wrote the Stage-0 target document: 110M-parameter decoder-only base model, native tokenizer, dialect adaptation, and measurable acceptance criteria.",
            "Initialized the repository and the assess → implement → measure → review workflow for all eight stages.",
        ],
        "pending": "None.",
        "who": [
            ("Athul Sunil Kumar", "Drafted the target/scope document, model-size rationale and training environment."),
            ("Anjana M", "Surveyed baseline tokenizers and defined fertility/UNK comparison criteria."),
            ("Paul Mathew", "Inventoried corpus sources, sizes, licences and availability."),
            ("Arshad V A", "Set up Git/GitHub, folder structure and documentation templates."),
        ],
        "other": "Decision: write a methodology assessment before coding each stage, and defer all cloud spending until the local validation gate passes.",
    },
    {
        "n": 2,
        "assigned": [
            "Assess tokenizer algorithms, vocabulary sizes, normalization and character coverage for Malayalam.",
            "Prepare tokenizer training/evaluation text and define the candidate grid.",
        ],
        "completed": [
            "Compared SentencePiece Unigram, SentencePiece BPE and byte-level BPE; selected Unigram for Malayalam morphology.",
            "Defined eight candidates across 16k/32k/50k vocabularies and character coverage 1.0 vs the 0.9995 warning test.",
            "Assembled a held-out tokenizer evaluation corpus with a chillu/ZWJ word list.",
        ],
        "pending": "Candidate training and the full benchmark (next week).",
        "who": [
            ("Athul Sunil Kumar", "Set selection criteria (fertility, UNK, roundtrip, cost) and the candidate grid."),
            ("Anjana M", "Implemented candidate training and evaluation scripts; sampled tokenizer training text."),
            ("Paul Mathew", "Prepared cleaned tokenizer-training text and the held-out evaluation set."),
            ("Arshad V A", "Set up experiment tracking and reproducibility notes."),
        ],
        "other": "fastText has no working build in the Python 3.13 environment, so a Unicode script-ratio language filter was evaluated as the fallback candidate.",
    },
    {
        "n": 3,
        "assigned": [
            "Train all tokenizer candidates; measure fertility, UNK rate, roundtrip and chillu/ZWJ handling.",
            "Select and package the final tokenizer artifact.",
        ],
        "completed": [
            "Trained eight candidates and benchmarked them on held-out Malayalam text.",
            "Selected the 50k SentencePiece Unigram model: 1.775 tokens/word vs MuRIL 2.378, IndicBERTv2 1.938 and XLM-R 2.746, with 0 UNK.",
            "Verified exact Malayalam roundtrip (SentencePiece only collapses repeated spaces) and correct chillu/ZWJ segmentation; packaged the model, vocab and comparison JSON.",
        ],
        "pending": "None.",
        "who": [
            ("Athul Sunil Kumar", "Reviewed fertility/UNK trade-offs and approved the 50k Unigram configuration."),
            ("Anjana M", "Ran the eight-candidate benchmark and chillu/ZWJ tests; produced the results table."),
            ("Paul Mathew", "Verified normalization and roundtrip behaviour on corpus samples."),
            ("Arshad V A", "Packaged artifacts and documented the tokenizer decision."),
        ],
        "other": "The 0.9995 coverage warning was verified empirically, not assumed: it produced UNKs. Coverage 1.0 with byte fallback won; byte-level BPE was about 4× worse on fertility.",
    },
    {
        "n": 4,
        "assigned": [
            "Design the corpus cleaning, language filtering, deduplication and split strategy.",
            "Implement a streaming pipeline suitable for a memory-constrained machine.",
        ],
        "completed": [
            "Fixed the source set: CC100 Malayalam, IndicCorpV2 Malayalam and Malayalam Wikipedia.",
            "Specified filters: NFC, 60–8192-character paragraphs, ≥70% Malayalam-script ratio, Manglish/URL/repetition guards, and per-source blake2b-64 exact dedup.",
            "Designed the two-pass streaming splitter: document-level 98/1/1 split with PCG64 seed 1337, uint16 bins and a \\x1e document separator.",
        ],
        "pending": "End-to-end pipeline execution (next week).",
        "who": [
            ("Athul Sunil Kumar", "Defined cleaning thresholds and split invariants."),
            ("Paul Mathew", "Implemented cleaning, script-ratio LID and the Manglish/URL/repetition filters."),
            ("Anjana M", "Implemented tokenizer integration and UNK/roundtrip checks on cleaned text."),
            ("Arshad V A", "Estimated disk/RAM needs and documented data provenance."),
        ],
        "other": "Limited local RAM and disk ruled out in-memory processing, so the splitter reads sources twice and writes only compact uint16 bins.",
    },
    {
        "n": 5,
        "assigned": [
            "Run the pipeline on available sources; generate train/val/test bins, statistics and held-out test text.",
            "Validate bin integrity and document the data report.",
        ],
        "completed": [
            "Produced a test-scale corpus of 316.8M tokens (train 310.5M / val 3.15M / test 3.18M) with 0 UNK.",
            "Verified exact bin sizes, token-id ranges and a re-encode match against test.txt; confirmed document-level split isolation.",
            "Wrote the data report with per-source cleaning, dedup and split statistics and reproduction commands.",
        ],
        "pending": "Full-corpus tokenization; the run used bounded public-source slices, recorded as a scale caveat.",
        "who": [
            ("Athul Sunil Kumar", "Reviewed split leakage and validation evidence."),
            ("Paul Mathew", "Ran cleaning and dedup; recorded per-source statistics."),
            ("Anjana M", "Ran tokenization/split and verified bin sizes and re-encoding."),
            ("Arshad V A", "Wrote the data report and reproduction instructions."),
        ],
        "other": "Bounded slices were used because the original corpora were absent and local storage was limited; the report states this explicitly rather than presenting test-scale numbers as full-corpus results.",
    },
    {
        "n": 6,
        "assigned": [
            "Implement the decoder-only transformer and resumable training loop.",
            "Run the mandatory sanity gates.",
        ],
        "completed": [
            "Implemented the pre-LN causal transformer with tied embeddings, learned positions, GELU, scaled dot-product attention, and both tiny (16.1M) and 110M configurations.",
            "Added AdamW, warmup + cosine schedule, gradient clipping, bf16, gradient accumulation, atomic checkpointing and resume.",
            "Passed both sanity gates: initial loss 10.8230 vs ln(50000)=10.8198 (Δ +0.0033), and a tiny-batch overfit run reaching near-zero loss.",
        ],
        "pending": "Full-scale pretraining; the cloud run has not been launched.",
        "who": [
            ("Athul Sunil Kumar", "Implemented the model, initialization scheme and optimizer settings."),
            ("Paul Mathew", "Implemented the memmap dataloader and validation batching."),
            ("Anjana M", "Added loss/evaluation instrumentation and the sanity-gate scripts."),
            ("Arshad V A", "Implemented checkpoint save/resume and run scripts."),
        ],
        "other": "Sanity gates were treated as blocking: no training run was accepted until the fresh-model loss matched ln(vocab) and the overfit test converged.",
    },
    {
        "n": 7,
        "assigned": [
            "Compute the compute/data budget and pretraining plan.",
            "Build the cross-tokenizer evaluation harness and run local validation.",
        ],
        "completed": [
            "Completed Stage-4 sizing: 110M parameters, 1024 context, ~1.7B-token target, Spot g5.xlarge for about 6 hours at an estimated ~$2.50.",
            "Ran the local 2,000-step validation gate on the tokenized data.",
            "Built the evaluation harness: exact causal nats/char → BPC, plus MLM pseudo-perplexity for MuRIL/XLM-R/IndicBERTv2 with a fixed masking seed and documented caveats.",
        ],
        "pending": "Baseline evaluation on a final model and human-fluency scoring.",
        "who": [
            ("Athul Sunil Kumar", "Produced the sizing math and cloud budget."),
            ("Paul Mathew", "Validated data throughput and token counts."),
            ("Anjana M", "Implemented the BPC harness and fixed a logits bug that scored only the last position."),
            ("Arshad V A", "Profiled GPU memory/throughput and prepared the cost sheet."),
        ],
        "other": "BPC was chosen over perplexity so tokenizers can be compared on equal footing; MLM pseudo-log-loss is reported only side by side with causal BPC because it uses bidirectional context.",
    },
    {
        "n": 8,
        "assigned": [
            "Build a larger bounded corpus, train a demo checkpoint, run baseline evaluation and build the demo application.",
        ],
        "completed": [
            "Re-tokenized full CC100 + wiki + the first 2.5 GiB of IndicCorpV2: 590.2M tokens, 4.37M documents, 3.07B characters, 265,719 cross-source duplicates removed, 0 UNK.",
            "Trained the 16.1M demo model for 8,000 steps (393.2M tokens); final validation loss 4.7567 in 98.4 minutes on the RTX 4060 at zero cloud cost.",
            "Measured held-out BPC 1.3101 vs MuRIL 1.4818 (MLM pseudo), IndicBERTv2 0.9776 and XLM-R 0.6433, with method caveats.",
            "Built and browser-verified a FastAPI demo: live tokenizer comparison, generation, and a training/data dashboard.",
        ],
        "pending": "The 110M full-corpus cloud run and larger-scale evaluation.",
        "who": [
            ("Athul Sunil Kumar", "Trained the demo model and tuned batch/accumulation for 8 GB VRAM."),
            ("Paul Mathew", "Produced the 590.2M-token slice and updated data statistics."),
            ("Anjana M", "Ran baseline evaluation, BPC analysis and sample generation."),
            ("Arshad V A", "Built and tested the demo app and documented the measured results."),
        ],
        "other": "Laptop thermal throttling reduced throughput from ~92k to ~66k tokens/s; atomic checkpoints allowed the demo to keep serving the best model during training.",
    },
    {
        "n": 9,
        "assigned": [
            "Assess dialect adaptation and prepare the LoRA vs full-FT ablation.",
            "Prepare AWS infrastructure and the cost plan; decide the instruction-tuning scope.",
        ],
        "completed": [
            "Completed the dialect assessment: four-arm ablation (LoRA vs full fine-tuning on Colloquial Subtitles and Kasaragod data) with adapter-training and evaluation scripts.",
            "Prepared AWS S3 lifecycle rules, least-privilege IAM, Spot g5.xlarge launch and data-sync scripts, with a ~$2.50 estimated compute cost.",
            "Wrote the instruction-tuning scope decision, deferring implementation for time and terms-of-service reasons.",
        ],
        "pending": "Ablation execution (requires the base checkpoint and dialect data); the cloud run is not launched.",
        "who": [
            ("Athul Sunil Kumar", "Defined the LoRA/full-FT protocol and hyperparameters."),
            ("Paul Mathew", "Prepared dialect data scripts and reviewed licensing."),
            ("Anjana M", "Designed the dialect evaluation plan (held-out BPC plus samples)."),
            ("Arshad V A", "Wrote the IAM/S3/Spot scripts and cost estimate."),
        ],
        "other": "Spot was chosen over on-demand for the interruptible training job; no cloud resource was launched without explicit budget approval.",
    },
    {
        "n": 10,
        "assigned": [
            "Finalize the demo and documentation; verify artifacts and list remaining work and risks.",
        ],
        "completed": [
            "Pulled the team repository, fixed the evaluation and tokenizer-bound bugs, rebuilt the demo corpus, trained and evaluated the demo checkpoint, and added the evaluation panel to the dashboard.",
            "Wrote the demo runbook, updated the project README/data notes, and saved ten human-evaluation samples for review.",
            "Re-verified every demo tab and API, and recorded remaining work with measured status rather than estimates.",
        ],
        "pending": "Tokenize the full ~29 GB corpus (~1.8B tokens estimated); launch the 110M Spot run; execute the dialect ablation; write the AWS cost summary and final Stage-0 comparison; score human fluency.",
        "who": [
            ("Athul Sunil Kumar", "Final checkpoint evaluation and results review."),
            ("Paul Mathew", "Full-corpus scale-up plan: disk, RAM and runtime estimates."),
            ("Anjana M", "Reproducibility checks and human-evaluation sample generation."),
            ("Arshad V A", "Demo QA, documentation and diary compilation."),
        ],
        "other": "Core pipeline, demo and measured evaluation are complete; the remaining gap is compute-bound (full corpus and 110M run). No 110M result will be claimed until it is measured.",
    },
]


def set_base_styles(doc):
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.9)
    section.bottom_margin = Cm(1.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.space_after = Pt(3)
    normal.paragraph_format.line_spacing = 1.05


def heading(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text.upper())
    r.bold = True
    r.font.size = Pt(12)
    p.paragraph_format.space_after = Pt(10)


def label(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(10.5)
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(2)


def body(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def bullet(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.55)
    p.paragraph_format.first_line_indent = Cm(-0.4)
    p.add_run("•  " + text)
    return p


def table(doc, headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(9.5)
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)
    return t


def signature_block(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.add_run("Student Name and Signature: ________________________________")
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p2.add_run("Signature of the Guide: ________________________________")


def add_front_matter(doc):
    heading(doc, "Problem Definition and Objectives")
    body(doc,
         "Malayalam is spoken by about 35 million people, yet open language models still treat it as a "
         "low-resource language. Multilingual tokenizers fragment Malayalam words into roughly 2.0–2.7 "
         "subword tokens per word, which lengthens sequences, raises training and inference cost, and "
         "weakens morphological fidelity. There is no small, reproducible, Malayalam-native base model "
         "with measured tokenizer, data, training and evaluation evidence that a student team can build "
         "and audit end to end.")
    body(doc,
         "This project builds a Malayalam Small Language Model from scratch: a native 50k SentencePiece "
         "tokenizer, a cleaned and deduplicated corpus with document-level splits, a decoder-only "
         "transformer (110M-parameter target, 16.1M local demo), a cross-tokenizer bits-per-character "
         "evaluation against public baselines, and the infrastructure for dialect adaptation and a "
         "budgeted Spot pretraining run.")
    label(doc, "Objectives")
    for item in OBJECTIVES:
        bullet(doc, item)

    doc.add_page_break()
    heading(doc, "Project Workflow")
    body(doc,
         "Each stage follows the same cycle: assess the methodology and record the decision, implement it, "
         "measure the result, and review before moving on. A failed gate sends the team back to the "
         "assessment instead of forward.")
    table(doc, ["Stage", "Main work", "Output / gate"],
          [(s, w, o) for s, w, o in WORKFLOW], [1.5, 9.0, 6.5])
    body(doc,
         "Stages are iterative rather than strictly linear: the tokenizer feeds the data pipeline, the "
         "data feeds training, and evaluation can send the team back to data or model design. Work is "
         "integrated through Git, and every measured number in the diary comes from a script or log in "
         "the repository.")

    doc.add_page_break()
    heading(doc, "Project Design")
    for name, text in DESIGN:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(name + ": ")
        r.bold = True
        p.add_run(text)

    doc.add_page_break()
    heading(doc, "Platform / Tools Used")
    table(doc, ["Category", "Tools"], TOOLS, [4.2, 12.8])

    heading(doc, "Team and Roles")
    table(doc, ["Member", "Role"], TEAM, [6.0, 11.0])


def add_week(doc, week):
    doc.add_page_break()
    heading(doc, "Weekly Assignment and Progress Assessment")
    p = doc.add_paragraph()
    p.paragraph_format.tab_stops.add_tab_stop(Cm(16.6), WD_TAB_ALIGNMENT.RIGHT)
    r = p.add_run("Week No: ")
    r.bold = True
    p.add_run(str(week["n"]))
    p.add_run("\tDate: ____________________")
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p2.add_run("Progress (E / G): ________")
    p2.paragraph_format.space_after = Pt(4)

    label(doc, "Works Assigned to the Project Team")
    for item in week["assigned"]:
        bullet(doc, item)

    label(doc, "Works completed")
    for item in week["completed"]:
        bullet(doc, item)

    label(doc, "Works pending if any")
    body(doc, week["pending"])

    label(doc, "Individual Contribution")
    table(doc, ["Member", "Contribution"], week["who"], [5.2, 11.8])

    label(doc, "Any other matters discussed during the meeting")
    body(doc, week["other"])

    signature_block(doc)


def build_docx():
    doc = Document()
    set_base_styles(doc)
    add_front_matter(doc)
    for week in WEEKS:
        add_week(doc, week)
    doc.save(DOCX_PATH)


def build_markdown():
    lines = []
    lines.append("# Malayalam SLM — Project Diary\n")
    lines.append("> Dates are left blank to be filled from the team's meeting records. "
                 "Team: Athul Sunil Kumar, Anjana M, Paul Mathew, Arshad V A.\n")
    lines.append("## Problem Definition and Objectives\n")
    lines.append("Malayalam is spoken by about 35 million people, yet open language models still treat it "
                 "as a low-resource language. Multilingual tokenizers fragment Malayalam words into roughly "
                 "2.0–2.7 subword tokens per word, which lengthens sequences, raises training and inference "
                 "cost, and weakens morphological fidelity. There is no small, reproducible, "
                 "Malayalam-native base model with measured tokenizer, data, training and evaluation "
                 "evidence that a student team can build and audit end to end.\n")
    lines.append("This project builds a Malayalam Small Language Model from scratch: a native 50k "
                 "SentencePiece tokenizer, a cleaned and deduplicated corpus with document-level splits, a "
                 "decoder-only transformer (110M-parameter target, 16.1M local demo), a cross-tokenizer "
                 "bits-per-character evaluation against public baselines, and the infrastructure for "
                 "dialect adaptation and a budgeted Spot pretraining run.\n")
    lines.append("**Objectives**\n")
    for item in OBJECTIVES:
        lines.append(f"- {item}")
    lines.append("\n## Project Workflow\n")
    lines.append("Each stage follows the same cycle: assess the methodology and record the decision, "
                 "implement it, measure the result, and review before moving on. A failed gate sends the "
                 "team back to the assessment instead of forward.\n")
    lines.append("| Stage | Main work | Output / gate |")
    lines.append("|---|---|---|")
    for s, w, o in WORKFLOW:
        lines.append(f"| {s} | {w} | {o} |")
    lines.append("\nStages are iterative rather than strictly linear: the tokenizer feeds the data "
                 "pipeline, the data feeds training, and evaluation can send the team back to data or "
                 "model design. Work is integrated through Git, and every measured number in the diary "
                 "comes from a script or log in the repository.\n")
    lines.append("## Project Design\n")
    for name, text in DESIGN:
        lines.append(f"**{name}:** {text}\n")
    lines.append("## Platform / Tools Used\n")
    lines.append("| Category | Tools |")
    lines.append("|---|---|")
    for c, t in TOOLS:
        lines.append(f"| {c} | {t} |")
    lines.append("\n## Team and Roles\n")
    lines.append("| Member | Role |")
    lines.append("|---|---|")
    for m, r in TEAM:
        lines.append(f"| {m} | {r} |")
    for week in WEEKS:
        lines.append(f"\n---\n\n## Weekly Assignment and Progress Assessment\n")
        lines.append(f"**Week No:** {week['n']}  ")
        lines.append("**Date:** ____________________  ")
        lines.append("**Progress (E / G):** ________\n")
        lines.append("**Works Assigned to the Project Team**\n")
        for item in week["assigned"]:
            lines.append(f"- {item}")
        lines.append("\n**Works completed**\n")
        for item in week["completed"]:
            lines.append(f"- {item}")
        lines.append(f"\n**Works pending if any**\n\n{week['pending']}\n")
        lines.append("**Individual Contribution**\n")
        lines.append("| Member | Contribution |")
        lines.append("|---|---|")
        for m, c in week["who"]:
            lines.append(f"| {m} | {c} |")
        lines.append(f"\n**Any other matters discussed during the meeting**\n\n{week['other']}\n")
        lines.append("**Student Name and Signature:** ________________________________  ")
        lines.append("**Signature of the Guide:** ________________________________\n")
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    build_docx()
    build_markdown()
    print("wrote", DOCX_PATH)
    print("wrote", MD_PATH)

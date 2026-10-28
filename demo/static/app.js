const DEFAULT_SAMPLES = [
  "മൂന്നു സഹോദരിമാരുണ്ടായിരുന്നു ഗുരുദേവന്. തേവിയമ്മ, കൊച്ചു, മാത എന്നിവരായിരുന്നു അവർ. നാണു എന്നാണ് കുട്ടിക്കാലത്ത് വിളിച്ചിരുന്നത്. അദ്ദേഹത്തിന്റെ അമ്മാവൻ കൃഷ്ണൻ വൈദ്യൻ അറിയപ്പെടുന്ന ഒരു ആയുർവേദവൈദ്യനും സംസ്കൃതപണ്ഡിതനുമായിരുന്നു. ജനിച്ചത് വയൽവാരം വീട്ടിൽ ആയിരുന്നെങ്കിലും മാതൃകുടുംബം മണയ്ക്കൽ ക്ഷേത്രത്തിനു പടിഞ്ഞാറുള്ള ഇലഞ്ഞിക്കൽ വീടാണ്.",
  "നാണുവിന്റെ പതിനഞ്ചാമത്തെ വയസ്സിൽ അമ്മ മരിച്ചു, തന്റെ കൗമാരകാലം അച്ഛനെയും അമ്മാവനെയും സഹായിച്ചും, പഠനത്തിലും, അടുത്തുള്ള മണയ്ക്കൽ ക്ഷേത്രത്തിൽ ആരാധനയിൽ മുഴുകിയും കഴിഞ്ഞു. ചെറുപ്പത്തിലേ കാർഷികവൃത്തിയിൽ നാണു തൽപരനായിരുന്നു.",
  "1913-ൽ ആലുവയിൽ നാരായണഗുരു ഒരു ആശ്രമം സ്ഥാപിച്ചു. അദ്വൈത ആശ്രമം എന്നായിരുന്നു അതിന്റെ പേര്. “ഓം സാഹോദര്യം സർവത്ര” എന്ന തത്ത്വത്തിൽ അധിഷ്ഠിതമായിരുന്നു അദ്വൈത ആശ്രമം. ദൈവത്തിന്റെ കണ്ണിൽ എല്ലാവരും തുല്യരാണെന്ന് മനുഷ്യരെ ബോധവാന്മാരാക്കുക എന്നതായിരുന്നു അദ്ദേഹത്തിന്റെ ലക്ഷ്യം."
];

const $ = (sel) => document.querySelector(sel);

const tokInput = $("#tokInput");
const charCount = $("#charCount");
const tokBtn = $("#tokBtn");
const tokLoading = $("#tokLoading");
const tokError = $("#tokError");
const tokResults = $("#tokResults");
const ourMetrics = $("#ourMetrics");
const tokenChips = $("#tokenChips");
const tokenNote = $("#tokenNote");
const cmpBars = $("#cmpBars");
const statusPill = $("#statusPill");
const statusText = $("#statusText");
const genStatus = $("#genStatus");
const genBtn = $("#genBtn");
const genPrompt = $("#genPrompt");
const genOutput = $("#genOutput");
const genMeta = $("#genMeta");
const genError = $("#genError");
const temperature = $("#temperature");
const maxTokens = $("#maxTokens");
const topK = $("#topK");
const tempVal = $("#tempVal");
const maxVal = $("#maxVal");
const topkVal = $("#topkVal");
const lossChart = $("#lossChart");
const trainSamples = $("#trainSamples");
const sourceTable = $("#sourceTable");
const tokTable = $("#tokTable");
const evalTable = $("#evalTable");
const scopeNote = $("#scopeNote");
const dataMetrics = $("#dataMetrics");
const dataStatus = $("#dataStatus");
const dataConfig = $("#dataConfig");
const footerGpu = $("#footerGpu");

let sampleIdx = 0;
let lastStatus = null;
let lastStats = null;
let dataTimer = null;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function compact(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

function pct(x, digits = 2) {
  if (!x) return "0";
  const v = x * 100;
  if (v < 0.0001) return v.toExponential(1) + "%";
  return v.toFixed(digits) + "%";
}

function metric(value, label, cls) {
  const box = el("div", "metric" + (cls ? " " + cls : ""));
  box.appendChild(el("div", "value", value));
  box.appendChild(el("div", "label", label));
  return box;
}

function showError(node, message) {
  node.textContent = message;
  node.classList.remove("hidden");
}

function hideError(node) {
  node.classList.add("hidden");
  node.textContent = "";
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = {};
  try { data = await res.json(); } catch (_) { /* ignore */ }
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

/* ---------------- tabs ---------------- */

function setTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "data") {
    loadData();
    if (!dataTimer) dataTimer = setInterval(loadData, 15000);
  } else if (dataTimer) {
    clearInterval(dataTimer);
    dataTimer = null;
  }
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => setTab(btn.dataset.tab));
});

/* ---------------- tokenizer ---------------- */

function updateCharCount() {
  const text = tokInput.value;
  const words = (text.match(/\S+/g) || []).length;
  charCount.textContent = `${text.length} chars · ${words} words`;
}

tokInput.addEventListener("input", updateCharCount);

$("#sampleBtn").addEventListener("click", async () => {
  try {
    const res = await fetch("/api/sample");
    const data = await res.json();
    if (data.text) {
      tokInput.value = data.text;
      updateCharCount();
      analyze();
      return;
    }
  } catch (_) { /* fall through */ }
  sampleIdx = (sampleIdx + 1) % DEFAULT_SAMPLES.length;
  tokInput.value = DEFAULT_SAMPLES[sampleIdx];
  updateCharCount();
  analyze();
});

function displayPiece(piece) {
  return piece.replace(/▁/g, "␣");
}

function renderTokenize(r) {
  clear(ourMetrics);
  ourMetrics.appendChild(metric(String(r.ours.token_count), "tokens (ours)", "accent"));
  ourMetrics.appendChild(metric(r.ours.tokens_per_word.toFixed(3), "tokens / word", "accent"));
  ourMetrics.appendChild(metric(r.ours.tokens_per_char.toFixed(4), "tokens / char"));
  ourMetrics.appendChild(metric(r.ours.unk_rate === 0 ? "0" : pct(r.ours.unk_rate), "unknown rate",
    r.ours.unk_rate === 0 ? "" : "warn"));
  ourMetrics.appendChild(metric(r.ours.roundtrip ? "exact" : "lossy", "decode roundtrip",
    r.ours.roundtrip ? "" : "warn"));
  ourMetrics.appendChild(metric(String(r.ours.byte_fallback), "byte-fallback pieces"));

  clear(tokenChips);
  r.ours.ids.forEach((id, i) => {
    const piece = r.ours.tokens[i];
    const chip = el("span", "chip" + (piece.startsWith("<0x") ? " byte" : ""));
    chip.textContent = displayPiece(piece);
    chip.title = `id ${id}`;
    tokenChips.appendChild(chip);
  });
  tokenNote.textContent = (r.ours.truncated
    ? `Showing the first ${r.ours.tokens.length} of ${r.ours.token_count} tokens. ␣ marks a word boundary; red chips are byte-fallback.`
    : "␣ marks a word boundary; red chips are byte-fallback. Hover a chip for its token id.") +
    (r.nfc_changed ? " Input was NFC-normalized before tokenization." : "");

  clear(cmpBars);
  const rows = [r.ours].concat(r.baselines);
  const maxFert = Math.max(...rows.map((m) => m.tokens_per_word), 0.001);
  rows.forEach((m) => {
    const row = el("div", "cmp-row" + (m === r.ours ? " ours" : ""));
    const name = el("div", "cmp-name");
    name.appendChild(el("span", null, m.name));
    if (m === r.ours) name.appendChild(el("span", "badge ours", "OURS"));
    name.appendChild(el("span", "badge", `vocab ${compact(m.vocab_size)}`));
    if (m.unk_rate > 0) name.appendChild(el("span", "badge amber", `${pct(m.unk_rate)} UNK`));
    if (!m.roundtrip) name.appendChild(el("span", "badge amber", "roundtrip ✗"));
    row.appendChild(name);

    const track = el("div", "cmp-track");
    const fill = el("div", "cmp-fill");
    fill.style.width = Math.max(3, (m.tokens_per_word / maxFert) * 100).toFixed(1) + "%";
    track.appendChild(fill);
    row.appendChild(track);

    const value = el("div", "cmp-value");
    const strong = el("b", null, m.tokens_per_word.toFixed(3));
    value.appendChild(strong);
    value.appendChild(document.createTextNode(` tok/word · ${m.token_count} tokens`));
    row.appendChild(value);
    cmpBars.appendChild(row);
  });

  if (r.errors && r.errors.length) {
    const note = el("p", "hint", r.errors.join(" · "));
    cmpBars.appendChild(note);
  }
  tokResults.classList.remove("hidden");
}

async function analyze() {
  const text = tokInput.value.trim();
  if (!text) return;
  hideError(tokError);
  tokLoading.classList.remove("hidden");
  tokBtn.disabled = true;
  try {
    const data = await postJSON("/api/tokenize", { text });
    renderTokenize(data);
  } catch (e) {
    showError(tokError, e.message);
  } finally {
    tokLoading.classList.add("hidden");
    tokBtn.disabled = false;
  }
}

tokBtn.addEventListener("click", analyze);
tokInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
});

/* ---------------- generate ---------------- */

function syncSlider(input, output, digits) {
  const update = () => { output.textContent = digits ? Number(input.value).toFixed(digits) : input.value; };
  input.addEventListener("input", update);
  update();
}

syncSlider(temperature, tempVal, 2);
syncSlider(maxTokens, maxVal, 0);
syncSlider(topK, topkVal, 0);

document.querySelectorAll(".chip-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    genPrompt.value = btn.dataset.prompt || "";
    genPrompt.focus();
  });
});

async function doGenerate() {
  hideError(genError);
  genBtn.disabled = true;
  genBtn.textContent = "Generating…";
  genMeta.textContent = "running model";
  try {
    const data = await postJSON("/api/generate", {
      prompt: genPrompt.value,
      max_new_tokens: Number(maxTokens.value),
      temperature: Number(temperature.value),
      top_k: Number(topK.value),
    });
    clear(genOutput);
    if (data.prompt && data.prompt.trim()) {
      genOutput.appendChild(el("span", "gen-prompt", data.prompt));
    }
    genOutput.appendChild(el("span", "gen-completion",
      data.completion || "(model produced no further tokens)"));
    genMeta.textContent =
      `${data.new_tokens} tokens · ${data.seconds}s · T=${data.temperature} · ` +
      `top-k=${data.top_k || "off"} · step ${data.checkpoint.step ?? "?"}`;
  } catch (e) {
    showError(genError, e.message);
    genMeta.textContent = "";
  } finally {
    genBtn.disabled = false;
    genBtn.textContent = "Generate";
  }
}

genBtn.addEventListener("click", doGenerate);
genPrompt.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") doGenerate();
});

/* ---------------- status ---------------- */

function renderStatus(s) {
  lastStatus = s;
  const t = s.training || {};
  if (t.running) {
    statusPill.classList.add("live");
    statusText.textContent = `training · step ${t.step ?? 0}/${t.total_steps ?? "?"}`;
  } else if (s.model_ready) {
    statusPill.classList.remove("live");
    statusText.textContent = "checkpoint ready";
  } else {
    statusPill.classList.remove("live");
    statusText.textContent = t.log_exists ? "model training" : "preparing data";
  }
  footerGpu.textContent = s.gpu || "";

  if (s.model_ready) {
    const parts = ["Checkpoint ready"];
    if (t.step) parts.push(`training step ${t.step}`);
    if (t.val_loss) parts.push(`val loss ${t.val_loss.toFixed(3)}`);
    genStatus.textContent = parts.join(" · ") +
      ". Generation uses the best-validation checkpoint so far (saved every 250 steps).";
  } else if (t.running) {
    genStatus.textContent = `Training in progress — step ${t.step ?? 0}/${t.total_steps ?? "?"}` +
      (t.loss ? `, loss ${t.loss.toFixed(3)}` : "") +
      ". Generation unlocks at the first checkpoint.";
  } else {
    genStatus.textContent = "No checkpoint yet. Start the demo training run, then reload this page.";
  }
}

async function refreshStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    renderStatus(s);
    if (lastStats) dataStatus.textContent = dataStatusLine(s, lastStats);
  } catch (_) { /* server may be starting */ }
}

/* ---------------- data ---------------- */

function dataStatusLine(s, stats) {
  const t = s.training || {};
  if (t.running) {
    return `Training live · step ${t.step ?? 0}/${t.total_steps ?? "?"}` +
      (t.loss ? ` · loss ${t.loss.toFixed(3)}` : "") +
      (t.val_loss ? ` · val ${t.val_loss.toFixed(3)}` : "") +
      (t.tok_s ? ` · ${compact(t.tok_s)} tok/s` : "") +
      (t.elapsed_min ? ` · ${t.elapsed_min.toFixed(1)} min` : "");
  }
  if (stats.tokenized) {
    return "Training run finished or idle. All figures below are read from the measured logs.";
  }
  return "Waiting for training data to finish tokenizing.";
}

function fillSourceTable(tokenized) {
  clear(sourceTable);
  if (!tokenized || !tokenized.per_source) {
    sourceTable.appendChild(el("tr", null, "No tokenized data yet."));
    return;
  }
  const head = el("tr");
  ["source", "docs", "paragraphs kept", "chars", "cross-dup", "boundaries"].forEach((h, i) => {
    head.appendChild(el("th", i ? "num" : null, h));
  });
  sourceTable.appendChild(head);
  Object.entries(tokenized.per_source).forEach(([name, src]) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, name));
    tr.appendChild(el("td", "num", compact(src.docs)));
    tr.appendChild(el("td", "num", compact(src.paras_kept)));
    tr.appendChild(el("td", "num", compact(src.chars)));
    tr.appendChild(el("td", "num", compact(src.cross_dup)));
    tr.appendChild(el("td", null, src.doc_boundaries));
    sourceTable.appendChild(tr);
  });
}

function fillTokTable(evaluation) {
  clear(tokTable);
  const rows = Object.entries(evaluation || {})
    .filter(([, m]) => m && m.tokens_per_word !== undefined)
    .sort((a, b) => a[1].tokens_per_word - b[1].tokens_per_word);
  if (!rows.length) {
    tokTable.appendChild(el("tr", null, "Tokenizer benchmark not found."));
    return;
  }
  const head = el("tr");
  ["tokenizer", "vocab", "tokens/word", "tokens/char", "UNK", "exact roundtrip"].forEach((h, i) => {
    head.appendChild(el("th", i > 1 && i < 5 ? "num" : null, h));
  });
  tokTable.appendChild(head);
  rows.forEach(([name, m]) => {
    const ours = /(^|_)50k(_|$)/.test(name) && !/bpe/i.test(name);
    const tr = el("tr", ours ? "ours" : null);
    const nameCell = el("td", null, prettyTokName(name));
    if (ours) nameCell.appendChild(el("span", "badge ours", "OURS")).style.marginLeft = "8px";
    tr.appendChild(nameCell);
    tr.appendChild(el("td", "num", compact(m.vocab_size)));
    tr.appendChild(el("td", "num", m.tokens_per_word.toFixed(3)));
    tr.appendChild(el("td", "num", m.tokens_per_char.toFixed(4)));
    tr.appendChild(el("td", "num", pct(m.unk_rate, 3)));
    tr.appendChild(el("td", null, m.roundtrip_exact_rate !== undefined
      ? (m.roundtrip_exact_rate * 100).toFixed(1) + "%" : "—"));
    tokTable.appendChild(tr);
  });
}

function prettyTokName(name) {
  return name
    .replace(/^spm_unigram_/, "SPM Unigram · ")
    .replace(/^spm_bpe_/, "SPM BPE · ")
    .replace(/^bbpe_/, "Byte-level BPE · ")
    .replace(/_cov9995/, " · cov 0.9995")
    .replace(/^muril$/, "MuRIL (baseline)")
    .replace(/^xlmr$/, "XLM-R (baseline)")
    .replace(/^indicbertv2$/, "IndicBERTv2 (baseline)")
    .replace(/^gpt2$/, "GPT-2 (baseline)")
    .replace(/^google\/muril-base-cased$/, "MuRIL (baseline)")
    .replace(/^xlm-roberta-base$/, "XLM-R (baseline)")
    .replace(/^ai4bharat\/IndicBERTv2-MLM-only$/, "IndicBERTv2 (baseline)");
}

function fillEvalTable(evaluation) {
  clear(evalTable);
  const entries = Object.entries(evaluation || {}).filter(([, r]) => r && r.bpc !== undefined);
  if (!entries.length) {
    evalTable.appendChild(el("tr", null,
      "Evaluation not run yet. Run scripts/eval_model.py to populate this table."));
    return;
  }
  const head = el("tr");
  ["model", "params", "tokens/char", "loss (nats/token)", "bits/char"].forEach((h, i) => {
    head.appendChild(el("th", i ? "num" : null, h));
  });
  evalTable.appendChild(head);
  entries.sort((a, b) => {
    if (a[0] === "ours") return -1;
    if (b[0] === "ours") return 1;
    return a[1].bpc - b[1].bpc;
  });
  entries.forEach(([key, r]) => {
    const tr = el("tr", key === "ours" ? "ours" : null);
    const nameCell = el("td", null, key === "ours"
      ? "Malayalam SLM (causal, this checkpoint)" : prettyTokName(key));
    if (key === "ours") nameCell.appendChild(el("span", "badge ours", "OURS")).style.marginLeft = "8px";
    tr.appendChild(nameCell);
    tr.appendChild(el("td", "num", compact(r.params)));
    tr.appendChild(el("td", "num", r.tokens_per_char.toFixed(3)));
    const loss = r.loss_nats_per_token !== undefined
      ? r.loss_nats_per_token : r.nats_per_masked_token;
    tr.appendChild(el("td", "num", loss !== undefined ? loss.toFixed(4) : "—"));
    tr.appendChild(el("td", "num", r.bpc.toFixed(4)));
    evalTable.appendChild(tr);
  });
}

function renderData(stats) {
  lastStats = stats;
  const s = lastStatus || { training: stats.training };
  dataStatus.textContent = dataStatusLine(s, stats);
  const tr = stats.training || {};
  const cfg = [];
  if (tr.preset) cfg.push(`${tr.preset} preset`);
  if (tr.params_m) cfg.push(`${tr.params_m}M params`);
  if (tr.vocab) cfg.push(`vocab ${compact(tr.vocab)}`);
  if (tr.ctx) cfg.push(`ctx ${tr.ctx}`);
  if (tr.tokens_per_step) cfg.push(`${compact(tr.tokens_per_step)} tokens/step`);
  if (tr.planned_tokens_m) cfg.push(`${tr.planned_tokens_m}M planned tokens`);
  dataConfig.textContent = cfg.join(" · ");
  const t = stats.tokenized;

  clear(dataMetrics);
  if (t) {
    dataMetrics.appendChild(metric(compact(t.total_tokens), "total tokens", "accent"));
    dataMetrics.appendChild(metric(compact(t.split?.train?.tokens), "train tokens"));
    dataMetrics.appendChild(metric(compact((t.split?.val?.tokens || 0) + (t.split?.test?.tokens || 0)), "held-out tokens"));
    dataMetrics.appendChild(metric(compact(t.documents_total), "documents"));
    dataMetrics.appendChild(metric(compact(t.chars_total), "characters"));
    dataMetrics.appendChild(metric(compact(t.global_dup_cross_source), "cross-source dups removed"));
  } else {
    dataMetrics.appendChild(metric("…", "tokenized data still being built"));
  }

  fillSourceTable(t);
  fillTokTable(stats.tokenizer_eval);
  fillEvalTable(stats.model_eval);

  if (t && t.source_scope) {
    const limits = t.source_scope.max_bytes || {};
    const bits = Object.entries(limits).map(([f, b]) => `${f} capped at ${(b / 2 ** 30).toFixed(1)} GiB`);
    scopeNote.textContent = (bits.length
      ? `Bounded demo run: ${bits.join(", ")}; other sources used in full. `
      : "Full cleaned sources used. ") +
      `Tokenizer: ${t.tokenizer ? t.tokenizer.split(/[\\/]/).pop() : "SPM"} · vocab ${t.vocab_size} · ` +
      `split seed ${t.split_seed}.`;
  }

  renderSamples(stats.training);
  drawLossChart(stats.training);
}

function renderSamples(training) {
  clear(trainSamples);
  const samples = (training.samples || []).slice().reverse();
  if (!samples.length) {
    trainSamples.appendChild(el("p", "hint", "Samples appear once training starts."));
    return;
  }
  samples.slice(0, 3).forEach((s) => {
    const card = el("div", "sample");
    card.appendChild(el("span", "step", s.step ? `step ${s.step}` : "sample"));
    card.appendChild(el("div", "text", s.text));
    trainSamples.appendChild(card);
  });
}

function drawLossChart(training) {
  const points = (training && training.points) || [];
  const vals = (training && training.vals) || [];
  const dpr = window.devicePixelRatio || 1;
  const rect = lossChart.getBoundingClientRect();
  const w = Math.max(rect.width, 320);
  const h = 240;
  lossChart.width = Math.round(w * dpr);
  lossChart.height = Math.round(h * dpr);
  const ctx = lossChart.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  ctx.font = "12px Segoe UI, sans-serif";
  if (!points.length) {
    ctx.fillStyle = "#6c7f9f";
    ctx.textAlign = "center";
    ctx.fillText("Loss curve appears when training starts.", w / 2, h / 2);
    return;
  }

  const pad = { l: 52, r: 16, t: 14, b: 28 };
  const xMax = Math.max(
    training.total_steps || 0,
    points[points.length - 1][0],
    vals.length ? vals[vals.length - 1][0] : 0
  ) || 1;
  const losses = points.map((p) => p[1]).concat(vals.map((v) => v[1]));
  let yMin = Math.min(...losses);
  let yMax = Math.max(...losses);
  const margin = Math.max((yMax - yMin) * 0.08, 0.05);
  yMin = Math.max(0, yMin - margin);
  yMax = yMax + margin;
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const xOf = (x) => pad.l + (x / xMax) * plotW;
  const yOf = (y) => pad.t + (1 - (y - yMin) / (yMax - yMin || 1)) * plotH;

  ctx.strokeStyle = "#1a2740";
  ctx.fillStyle = "#6c7f9f";
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const value = yMin + ((yMax - yMin) * i) / 4;
    const y = yOf(value);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(value.toFixed(2), pad.l - 8, y + 4);
  }
  ctx.textAlign = "center";
  [0, 0.5, 1].forEach((frac) => {
    const x = pad.l + frac * plotW;
    ctx.fillText(String(Math.round(frac * xMax)), x, h - 8);
  });

  ctx.strokeStyle = "#46d7c0";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = xOf(p[0]);
    const y = yOf(p[1]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#ffb45e";
  vals.forEach((v) => {
    ctx.beginPath();
    ctx.arc(xOf(v[0]), yOf(v[1]), 3.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

async function loadData() {
  try {
    const stats = await (await fetch("/api/stats")).json();
    renderData(stats);
  } catch (e) {
    dataStatus.textContent = "Could not load stats: " + e.message;
  }
}

window.addEventListener("resize", () => {
  if (lastStats) drawLossChart(lastStats.training);
});

/* ---------------- init ---------------- */

tokInput.value = DEFAULT_SAMPLES[0];
updateCharCount();
refreshStatus();
setInterval(refreshStatus, 15000);

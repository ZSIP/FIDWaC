/* FIDWAC v2 web GUI: form state, presets, folder browser and job polling. No dependencies. */
(function () {
  "use strict";

  const initial = JSON.parse(document.getElementById("initial").textContent);
  const PRESETS = initial.presets;
  const METHOD_HELP = initial.methodHelp;
  const POLL_MS = 800;
  const STATE_LABELS = { idle: "Idle", running: "Running", done: "Done", stopped: "Stopped", error: "Error" };
  const BUTTON_HINTS = {
    run: "Start the compression job",
    stop: "Stop the current compression job",
    "clear-log": "Clear the current log output",
    "browser-use": "Use the selected path",
    "browser-up": "Go to the parent folder",
    "browser-close": "Close the folder browser",
  };

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const fieldEls = $$("[data-field]");
  const browser = { target: null, mode: "dir", path: "", parent: null };

  let pollTimer = null;
  let polling = false;
  let jobId = null;
  let logOffset = 0;
  let renderedResults = 0;
  let detectedInputKind = "unknown";
  let inputScanTimer = null;

  // ---------- form values ----------

  function readForm() {
    const form = { allowed_block_sizes: [], rgb_channels: [null, null, null], t_variants: [] };
    for (const el of fieldEls) {
      const key = el.dataset.field;
      if (el.dataset.list !== undefined) {
        if (el.checked) form[key].push(el.value);
      } else if (el.dataset.index !== undefined) {
        form[key][Number(el.dataset.index)] = el.value.trim() === "" ? null : el.value;
      } else if (el.type === "radio") {
        if (el.checked) form[key] = el.value;
      } else if (el.type === "checkbox") {
        form[key] = el.checked;
      } else {
        form[key] = el.value;
      }
    }
    return form;
  }

  function writeField(key, value) {
    for (const el of fieldEls) {
      if (el.dataset.field !== key) continue;
      if (el.dataset.list !== undefined) {
        el.checked = (value || []).map(String).includes(el.value);
      } else if (el.dataset.index !== undefined) {
        const v = (value || [])[Number(el.dataset.index)];
        el.value = v == null ? "" : String(v);
      } else if (el.type === "radio") {
        el.checked = el.value === String(value);
      } else if (el.type === "checkbox") {
        el.checked = Boolean(value);
      } else {
        el.value = value == null ? "" : String(value);
      }
    }
  }

  // ---------- enabling controls the way the selected method uses them ----------

  function setDisabled(el, disabled) {
    if (!el) return;
    el.disabled = disabled;
    const wrap = el.closest(".check, .field");
    if (wrap) wrap.classList.toggle("dim", disabled);
  }

  function setGroup(keys, enabled, extraIds = []) {
    for (const el of fieldEls) {
      if (keys.includes(el.dataset.field)) setDisabled(el, !enabled);
    }
    for (const id of extraIds) setDisabled(document.getElementById(id), !enabled);
  }

  // dct: quality slider in Quality mode; accuracy field and RGB bands in Accuracy mode.
  // pred / sz3: the 8-bit accuracy field only, because it sets the error bound of every band.
  function applyMethodState() {
    const f = readForm();
    const dct = f.method === "dct";
    const floatInput = detectedInputKind === "float";
    setGroup(["rgb_quality"], dct && f.uint8_mode === "quality" && !floatInput, ["rgb_quality_range"]);
    setGroup(["uint8_accuracy"], (!dct || f.uint8_mode === "accuracy") && !floatInput);
    setGroup(["rgb_channels", "ycbcr_per_block", "ycbcr_y", "ycbcr_cb", "ycbcr_cr"], dct && f.uint8_mode === "accuracy" && !floatInput);
    $("#method-help").textContent = METHOD_HELP[f.method] || "";

    const uint8Panel = document.getElementById("uint8-mode-panel");
    if (uint8Panel) {
      uint8Panel.classList.toggle("is-active", detectedInputKind === "uint8");
      uint8Panel.classList.toggle("is-dimmed", floatInput);
    }
  }

  function setDetectedInputKind(kind) {
    detectedInputKind = kind || "unknown";
    const badge = document.getElementById("input-kind-badge");
    if (!badge) return;
    if (detectedInputKind === "unknown") {
      badge.hidden = true;
      badge.textContent = "";
      badge.className = "input-kind";
      applyMethodState();
      return;
    }
    const labels = {
      uint8: "Detected: uint8 input — 8-bit path active",
      float: "Detected: floating-point input — float path active",
      other: "Detected: non-uint8 input — review the path settings",
    };
    badge.hidden = false;
    badge.className = `input-kind ${detectedInputKind}`;
    badge.textContent = labels[detectedInputKind] || "Detected input type";
    applyMethodState();
  }

  async function scanInputKind() {
    const path = readForm().input_path.trim();
    if (!path) {
      setDetectedInputKind("unknown");
      return;
    }
    try {
      const params = new URLSearchParams({ path });
      const res = await fetch(`/api/inspect-input?${params}`);
      if (!res.ok) {
        setDetectedInputKind("unknown");
        return;
      }
      const data = await res.json();
      setDetectedInputKind(data.kind || "unknown");
    } catch (err) {
      setDetectedInputKind("unknown");
    }
  }

  function scheduleInputScan() {
    clearTimeout(inputScanTimer);
    inputScanTimer = setTimeout(() => {
      scanInputKind();
    }, 220);
  }

  function applyAutoState() {
    const auto = readForm().auto_select;
    setGroup(["block_size"], !auto);
    setGroup(["allowed_block_sizes"], auto);
  }

  function applyBackendState() {
    const heuristic = readForm().backend === "heuristic";
    const adv = $("#advanced_heuristic");
    if (!heuristic) adv.checked = false;
    setDisabled(adv, !heuristic);
  }

  function applyInputHint() {
    const f = readForm();
    $("#op-hint").textContent = f.input_type === "file"
      ? "A raster (GeoTIFF, PNG, JPEG, BMP, ASC, IMG) or an archive (.7z, .fpred, .sz3)."
      : "Every raster and archive in this folder and its subfolders. The folder structure is kept in the output.";
  }

  // ---------- accuracy and quality controls ----------

  function syncRanges() {
    const acc = Number($("#accuracy").value);
    if (Number.isFinite(acc)) $("#accuracy-range").value = String(acc);
    const quality = Number($("#rgb_quality").value);
    if (Number.isFinite(quality)) $("#rgb_quality_range").value = String(quality);
  }

  function matchesPreset(f, p) {
    return Math.abs(Number(f.accuracy) - p.accuracy) < 1e-9
      && f.auto_select === p.auto_select
      && (p.auto_select || Number(f.block_size) === p.block_size)
      && Number(f.rgb_quality) === p.rgb_quality
      && f.uint8_mode === p.uint8_mode;
  }

  function markChoices() {
    const f = readForm();
    for (const b of $$("#accuracy-chips .chip")) {
      b.classList.toggle("active", Math.abs(Number(b.dataset.acc) - Number(f.accuracy)) < 1e-9);
    }
    let active = null;
    for (const [name, p] of Object.entries(PRESETS)) {
      if (matchesPreset(f, p)) active = name;
    }
    for (const b of $$("#preset-chips .chip")) {
      b.classList.toggle("active", b.dataset.preset === active);
    }
    $("#preset-desc").textContent = active ? PRESETS[active].description : "Custom settings.";
  }

  function refreshControls() {
    syncRanges();
    applyMethodState();
    applyAutoState();
    applyBackendState();
    applyInputHint();
    markChoices();
    // interp.js refreshes the Interpolation panel on this event.
    document.dispatchEvent(new Event("fidwac-state"));
  }

  function applyPreset(name) {
    const p = PRESETS[name];
    writeField("accuracy", p.accuracy);
    writeField("block_size", p.block_size);
    writeField("auto_select", p.auto_select);
    writeField("rgb_quality", p.rgb_quality);
    writeField("uint8_mode", p.uint8_mode);
    writeField("files_parallel", p.files_parallel);
    refreshControls();
  }

  function chip(label, data, onClick) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = label;
    Object.assign(b.dataset, data);
    b.addEventListener("click", onClick);
    return b;
  }

  function buildChips() {
    const accBox = $("#accuracy-chips");
    for (const v of initial.accuracyButtons) {
      accBox.appendChild(chip(String(v), { acc: String(v) }, () => {
        $("#accuracy").value = String(v);
        refreshControls();
      }));
    }
    const presetBox = $("#preset-chips");
    for (const name of Object.keys(PRESETS)) {
      presetBox.appendChild(chip(name, { preset: name }, () => applyPreset(name)));
    }
  }

  // ---------- folder browser ----------

  function baseName(path) {
    return String(path).split(/[\\/]/).pop();
  }

  function setBrowserError(message) {
    const el = $("#browser-error");
    el.textContent = message;
    el.hidden = !message;
  }

  async function loadBrowser(path) {
    const params = new URLSearchParams({ mode: browser.mode, path: path || "" });
    let res;
    let data;
    try {
      res = await fetch(`/api/browse?${params}`);
      data = await res.json();
    } catch (err) {
      setBrowserError(`Cannot read the folder: ${err}`);
      return false;
    }
    if (!res.ok) {
      setBrowserError(data.error || `HTTP ${res.status}`);
      return false;
    }
    browser.path = data.path;
    browser.parent = data.parent;
    $("#browser-path").textContent = data.path;
    $("#browser-up").disabled = !data.parent;
    setBrowserError("");
    renderEntries(data.entries);
    return true;
  }

  function renderEntries(entries) {
    const list = $("#browser-entries");
    list.replaceChildren();
    if (!entries.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = browser.mode === "dir" ? "No subfolders in this folder." : "No matching files in this folder.";
      list.appendChild(li);
      return;
    }
    for (const entry of entries) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = entry.kind === "dir" ? `${entry.name}/` : entry.name;
      btn.addEventListener("click", () => (entry.kind === "dir" ? loadBrowser(entry.path) : chooseValue(entry.path)));
      li.appendChild(btn);
      list.appendChild(li);
    }
  }

  // data-browse-type names the field that selects folder or file mode; data-browse-mode sets the file kinds to list.
  async function openBrowser(button) {
    const f = readForm();
    const target = button.dataset.browse;
    const typeKey = button.dataset.browseType;
    browser.target = target;
    browser.mode = typeKey && f[typeKey] !== "file" ? "dir" : button.dataset.browseMode || "dir";
    $("#browser-title").textContent = browseTitle(target, browser.mode);
    $("#browser-use").hidden = browser.mode !== "dir";
    $("#browser").hidden = false;
    // data-browse="#element-id" points the browser at a dynamic input outside data-field.
    const startPath = target.startsWith("#") ? ((document.querySelector(target) || {}).value || "") : f[target];
    if (!(await loadBrowser(startPath))) {
      await loadBrowser("");
    }
  }

  function browseTitle(target, mode) {
    if (target && target.startsWith("#")) return "Choose a dataset folder";
    if (mode === "geoid") return "Choose a geoid model file";
    if (mode === "points") return "Choose an input point file";
    if (mode === "file") return "Choose an input file";
    return target.endsWith("output_dir") ? "Choose an output folder" : "Choose an input folder";
  }

  function closeBrowser() {
    $("#browser").hidden = true;
  }

  function chooseValue(value) {
    if (browser.target && browser.target.startsWith("#")) {
      const el = document.querySelector(browser.target);
      if (el) el.value = value;
    } else {
      writeField(browser.target, value);
    }
    closeBrowser();
    refreshControls();
    if (browser.target === "input_path") scheduleInputScan();
  }

  // ---------- run, stop and job status ----------

  function setError(message) {
    const el = $("#error-line");
    el.textContent = message;
    el.hidden = !message;
  }

  function setBadge(state) {
    const el = $("#state");
    el.className = `state ${state}`;
    el.textContent = STATE_LABELS[state] || state;
  }

  function fmtBytes(n) {
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
    return `${n} B`;
  }

  function cell(text, title, className) {
    const td = document.createElement("td");
    td.textContent = text;
    if (title) td.title = title;
    if (className) td.className = className;
    return td;
  }

  // Insert zero-width spaces after "_", "." and "-" so that long file names wrap at separators.
  function softBreaks(name) {
    return name.replace(/([_.-])/g, "$1\u200B");
  }

  function validityCell(tag) {
    const td = document.createElement("td");
    if (tag) {
      const badge = document.createElement("span");
      badge.className = `badge ${tag.toLowerCase()}`;
      badge.textContent = tag;
      td.appendChild(badge);
    } else {
      td.textContent = "—";
    }
    return td;
  }

  function resultRow(r) {
    const tr = document.createElement("tr");
    tr.append(
      cell(softBreaks(baseName(r.input)), r.input, "name"),
      cell(softBreaks(baseName(r.output)), r.output, "name"),
      validityCell(r.validity),
      cell(`${fmtBytes(r.size_in)} → ${fmtBytes(r.size_out)}`, null, "num"),
      cell(r.ratio ? `${r.ratio}×` : "—", null, "num"),
    );
    return tr;
  }

  function renderResults(results) {
    $("#results-empty").hidden = results.length > 0;
    if (results.length === renderedResults) return;
    $("#results").replaceChildren(...results.map(resultRow));
    renderedResults = results.length;
  }

  function appendLog(lines) {
    if (!lines.length) return;
    const pre = $("#log");
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
    pre.textContent += `${lines.join("\n")}\n`;
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }

  function resetJobView() {
    jobId = null;
    logOffset = 0;
    renderedResults = 0;
    $("#log").textContent = "";
    $("#results").replaceChildren();
    setError("");
  }

  function statusText(s) {
    const total = s.total || 0;
    const done = s.done || 0;
    if (s.state === "running") {
      const names = (s.current || []).join(", ");
      return `Running: ${done} of ${total} done${names ? ` · ${names}` : ""}`;
    }
    if (s.state === "stopped") return `Stopped after ${done} of ${total} files.`;
    return `Finished: ${done} of ${total} files.`;
  }

  function renderJob(s) {
    const running = s.state === "running";
    setBadge(s.state);
    $("#status-line").textContent = s.id ? statusText(s) : "Idle";
    $("#run").disabled = running;
    $("#stop").disabled = !running;
    if (!s.id) return;
    const total = s.total || 0;
    const done = s.done || 0;
    const pct = total ? Math.round((100 * done) / total) : 0;
    const bar = $("#bar");
    const waiting = running && done === 0;
    bar.classList.toggle("indeterminate", waiting);
    bar.style.width = waiting ? "" : `${pct}%`;
    $("#progress").setAttribute("aria-valuenow", String(pct));
    setError(s.errors.join("\n"));
    renderResults(s.results);
  }

  function startPolling() {
    if (pollTimer === null) pollTimer = setInterval(pollStatus, POLL_MS);
  }

  function stopPolling() {
    if (pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  async function pollStatus() {
    if (polling) return;
    polling = true;
    try {
      const params = new URLSearchParams({ offset: String(logOffset), job: jobId || "" });
      const res = await fetch(`/api/status?${params}`);
      const s = await res.json();
      if (s.id && s.id !== jobId) {
        resetJobView();
        jobId = s.id;
      }
      if (s.id) {
        appendLog(s.log);
        logOffset = s.log_offset;
      }
      renderJob(s);
      if (s.state === "running") startPolling();
      else stopPolling();
    } catch (err) {
      setError(`Lost connection to the server: ${err}`);
    } finally {
      polling = false;
    }
  }

  async function startRun() {
    setError("");
    $("#run").disabled = true;
    let res;
    let data;
    try {
      const TAB_KIND = { interpolate: "interpolate", tests: "test" };
      const kind = TAB_KIND[document.body.dataset.activeTab] || "compress";
      // tests.js collects the dynamic dataset rows into the payload.
      const extra = kind === "test" && window.fidwacTestsPayload ? window.fidwacTestsPayload() : {};
      res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...readForm(), kind, ...extra }),
      });
      data = await res.json();
    } catch (err) {
      $("#run").disabled = false;
      setError(`Cannot reach the server: ${err}`);
      return;
    }
    if (!res.ok) {
      $("#run").disabled = false;
      setError(data.error || `Run failed (HTTP ${res.status}).`);
      return;
    }
    resetJobView();
    startPolling();
    pollStatus();
  }

  async function stopRun() {
    $("#stop").disabled = true;
    try {
      await fetch("/api/stop", { method: "POST" });
    } catch (err) {
      setError(`Cannot reach the server: ${err}`);
    }
    pollStatus();
  }

  // ---------- wiring ----------

  function attachFieldHelp() {
    document.querySelectorAll("[data-help]").forEach((el) => {
      const help = el.getAttribute("data-help");
      if (!help) return;
      el.dataset.tooltip = help;
      el.setAttribute("tabindex", "0");
      el.title = help;
      el.setAttribute("aria-label", help);
    });

    for (const [id, help] of Object.entries(BUTTON_HINTS)) {
      const el = document.getElementById(id);
      if (el) el.title = help;
    }

    for (const btn of $$("[data-browse]")) {
      btn.title = "Choose a path";
    }
  }

  // ---------- tests tab: environment summary ----------

  async function loadTestsInfo() {
    const box = document.getElementById("tests-env");
    if (!box) return;
    let info;
    try {
      const res = await fetch("/api/tests-info");
      info = await res.json();
    } catch (err) {
      box.className = "hint status-warn";
      box.textContent = `Cannot check the test environment: ${err}`;
      return;
    }
    const p = info.paths || {};
    const lines = [
      `data_root: ${p.data_root || "?"}`,
      `sz3: ${p.sz3_bin || "not found"}`,
      `results: ${info.results_dir || "?"}`,
    ];
    for (const prob of info.problems || []) lines.push(`! ${prob}`);
    box.textContent = lines.join("\n");
    box.className = (info.problems || []).length ? "hint status-warn" : "hint status-ok";
  }

  function init() {
    writeAll(initial.defaults);
    writeAll(initial.interpDefaults);
    writeAll(initial.testsDefaults || {});
    $("#processes").max = String(initial.cpuCount);
    $("#files_parallel").max = String(initial.cpuCount);
    $("#t_workers").max = String(initial.cpuCount);
    loadTestsInfo();
    document.addEventListener("click", (e) => {
      if (e.target.closest('[data-tab="tests"]')) loadTestsInfo();
    });
    buildChips();
    attachFieldHelp();
    scheduleInputScan();
    refreshControls();
    pollStatus();

    const layout = document.querySelector("main.layout");
    layout.addEventListener("input", (e) => {
      if (e.target && e.target.id === "input_path") scheduleInputScan();
      refreshControls();
    });
    layout.addEventListener("change", (e) => {
      if (e.target && e.target.id === "input_path") scheduleInputScan();
      refreshControls();
    });
    $("#accuracy-range").addEventListener("input", (e) => {
      $("#accuracy").value = e.target.value;
      refreshControls();
    });
    $("#rgb_quality_range").addEventListener("input", (e) => {
      $("#rgb_quality").value = e.target.value;
      refreshControls();
    });

    document.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-browse]");
      if (btn) openBrowser(btn);
    });
    $("#browser-use").addEventListener("click", () => chooseValue(browser.path));
    $("#browser-up").addEventListener("click", () => {
      if (browser.parent) loadBrowser(browser.parent);
    });
    $("#browser-close").addEventListener("click", closeBrowser);
    $("#browser").addEventListener("click", (e) => {
      if (e.target.id === "browser") closeBrowser();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !$("#browser").hidden) closeBrowser();
    });

    $("#run").addEventListener("click", startRun);
    $("#stop").addEventListener("click", stopRun);
    $("#clear-log").addEventListener("click", () => {
      $("#log").textContent = "";
    });
  }

  function writeAll(values) {
    for (const key of Object.keys(values)) writeField(key, values[key]);
  }

  init();
})();

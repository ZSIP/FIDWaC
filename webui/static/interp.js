/* Interpolation tab: switches between the Compression and Interpolation panels, keeps dependent fields in step and checks the geoid model path. */
(function () {
  "use strict";

  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const initial = JSON.parse(document.getElementById("initial").textContent);
  const geoidInput = document.getElementById("ip_geoid_file");
  const geoidToggle = document.getElementById("ip_geoid");
  const geoidHint = document.getElementById("ip-geoid-hint");
  let geoidChecked = null;
  let geoidInfo = null;
  let geoidTimer = null;
  let geoidSeq = 0;
  const INPUT_HINTS = {
    file: "Supported formats: TXT, CSV, LAS, LAZ and SHP (points).",
    directory: "Every point file below this folder is interpolated. Earlier results (names containing -dist-) are skipped.",
  };

  function showTab(name) {
    document.body.dataset.activeTab = name;
    for (const btn of $$("[data-tab]")) {
      const active = btn.dataset.tab === name;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", String(active));
    }
    for (const panel of $$("[data-panel]")) {
      panel.hidden = panel.dataset.panel !== name;
    }
  }

  // A control with data-enabled-by is usable only while the named checkbox is ticked.
  function applyDependencies() {
    for (const el of $$("[data-enabled-by]")) {
      const controller = document.getElementById(el.dataset.enabledBy);
      el.disabled = !(controller && controller.checked);
    }
  }

  function formatSize(bytes) {
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
    if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
    return `${bytes} B`;
  }

  // The path stays editable whether or not the correction is on; the hint shows whether the file exists.
  function renderGeoid() {
    if (!geoidInfo) return;
    const info = geoidInfo;
    let ok = false;
    let text;
    if (!info.path) {
      text = "Enter the path to a geoid model (.7z, .csv or .txt).";
    } else if (!info.supported) {
      text = `Unsupported file type. Use a .7z, .csv or .txt geoid model: ${info.path}`;
    } else if (!info.exists) {
      text = `File not found: ${info.path}`;
    } else {
      ok = true;
      text = `Found (${formatSize(info.size)}): ${info.path}`;
      if (!geoidToggle.checked) text += ". The correction is off, so this file is not used by the run.";
    }
    geoidHint.className = `hint ${ok ? "status-ok" : "status-warn"}`;
    geoidHint.textContent = text;
  }

  function checkGeoid() {
    const path = geoidInput.value.trim();
    if (path === geoidChecked) return;
    geoidChecked = path;
    geoidInfo = null;
    geoidHint.className = "hint";
    geoidHint.textContent = "Checking the path…";
    clearTimeout(geoidTimer);
    geoidTimer = setTimeout(async () => {
      const seq = ++geoidSeq;
      try {
        const res = await fetch(`/api/geoid-info?${new URLSearchParams({ path })}`);
        const info = await res.json();
        if (seq !== geoidSeq) return;
        geoidInfo = info;
      } catch (err) {
        if (seq !== geoidSeq) return;
        geoidHint.className = "hint status-warn";
        geoidHint.textContent = `Cannot check the file: ${err}`;
        return;
      }
      renderGeoid();
    }, 250);
  }

  function resetGeoid() {
    geoidInput.value = initial.bundledGeoid;
    geoidInput.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function refresh() {
    applyDependencies();
    const type = document.querySelector('input[name="ip_input_type"]:checked');
    document.getElementById("ip-input-hint").textContent = INPUT_HINTS[type ? type.value : "file"];
    const method = document.getElementById("method").value;
    const accuracy = document.getElementById("accuracy").value;
    document.getElementById("ip-compress-summary").textContent =
      `Compression step: method ${method}, accuracy ${accuracy}, set in the Compression tab.`;
    checkGeoid();
    renderGeoid();
  }

  for (const btn of $$("[data-tab]")) {
    btn.addEventListener("click", () => showTab(btn.dataset.tab));
  }
  document.addEventListener("fidwac-state", refresh);
  document.getElementById("ip-geoid-reset").addEventListener("click", resetGeoid);
  showTab("compress");
  refresh();
})();

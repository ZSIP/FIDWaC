/* FIDWAC v2 Tests tab: dynamic dataset rows with a per-dataset Scan button.
   Scan calls /api/tests-scan which inspects the folder's rasters and proposes
   kind, thresholds, rgb/dem tests and fidwac flags. Collected into the run
   payload via window.fidwacTestsPayload. */
(function () {
  "use strict";

  const initial = JSON.parse(document.getElementById("initial").textContent);
  const METRICS = [
    ["grad", "grad"],
    ["ssim", "SSIM"],
    ["dist", "dist"],
    ["de", "\u0394E (RGB)"],
    ["dem", "DEM"],
  ];
  const container = document.getElementById("t-datasets");
  const addBtn = document.getElementById("t-add-ds");
  if (!container || !addBtn) return;
  let seq = 0;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function field(labelText, control) {
    const box = el("div", "field");
    const label = el("label", "label", labelText);
    box.append(label, control);
    return box;
  }

  function bandsInput(id, value, placeholder) {
    const input = el("input");
    input.type = "text";
    input.id = id;
    input.dataset.ds = id.split("-").slice(2).join("-");
    input.autocomplete = "off";
    input.spellcheck = false;
    input.placeholder = placeholder;
    if (Array.isArray(value)) value = value.join(",");
    input.value = value || "";
    return input;
  }

  function addRow(ds) {
    const id = `ds-${++seq}`;
    ds = ds || {};
    const row = el("div", "ds-row");

    const head = el("div", "ds-head");
    const name = el("input", "ds-name");
    name.type = "text";
    name.id = `${id}-name`;
    name.dataset.ds = "name";
    name.placeholder = "dataset name";
    name.autocomplete = "off";
    name.spellcheck = false;
    name.value = ds.name || "";
    const remove = el("button", "button secondary ds-remove", "\u00d7");
    remove.type = "button";
    remove.title = "Remove dataset";
    remove.addEventListener("click", () => row.remove());
    head.append(name, remove);

    const dirWrap = el("div", "with-button");
    const dir = el("input");
    dir.type = "text";
    dir.id = `${id}-dir`;
    dir.dataset.ds = "dir";
    dir.placeholder = "/path/to/raster/folder";
    dir.autocomplete = "off";
    dir.spellcheck = false;
    dir.value = ds.dir || "";
    const browse = el("button", "button secondary", "Browse\u2026");
    browse.type = "button";
    browse.dataset.browse = `#${dir.id}`;
    browse.dataset.browseMode = "dir";
    browse.title = "Choose a path";
    const scan = el("button", "button secondary", "Scan");
    scan.type = "button";
    scan.title = "Inspect the files and fill kind, thresholds and tests";
    scan.addEventListener("click", () => scanRow(row, scan));
    dirWrap.append(dir, browse, scan);
    const dirField = field("Folder with .tif/.png files", dirWrap);
    const info = el("div", "hint ds-info");

    const kind = el("select");
    kind.id = `${id}-kind`;
    kind.dataset.ds = "kind";
    for (const k of ["float", "uint8"]) {
      const opt = el("option", "", k);
      opt.value = k;
      kind.appendChild(opt);
    }
    kind.value = ds.kind || "float";
    const thr = bandsInput(`${id}-thresholds`, ds.thresholds || [0.01, 0.05, 0.1, 0.5, 1.0], "0.01, 0.05, 0.1");
    thr.dataset.ds = "thresholds";
    const params = el("div", "grid-2");
    params.append(field("Data kind", kind), field("Error thresholds", thr));

    const testField = el("div", "field");
    testField.appendChild(el("label", "label", "Tests"));
    const chips = el("div", "chips");
    const chosen = ds.tests || METRICS.map(([k]) => k);
    for (const [key, label] of METRICS) {
      const lab = el("label", "check");
      const cb = el("input");
      cb.type = "checkbox";
      cb.dataset.dsTests = key;
      cb.checked = chosen.includes(key);
      lab.append(cb, ` ${label}`);
      chips.appendChild(lab);
    }
    testField.appendChild(chips);

    const rgb = bandsInput(`${id}-rgb_bands`, ds.rgb_bands || [], "1,2,3");
    const dem = bandsInput(`${id}-dem_bands`, ds.dem_bands || [], "1 or 4,5");
    const bandGrid = el("div", "grid-2");
    bandGrid.append(field("RGB bands (for \u0394E)", rgb), field("DEM bands", dem));

    const extra = bandsInput(`${id}-fidwac_extra`, null, "--rgb-channel-indices 1,2,3 --ycbcr-per-block");
    extra.value = (ds.fidwac_extra || []).join(" ");

    row.append(head, dirField, info, params, testField, bandGrid, field("Extra fidwac flags", extra));
    container.appendChild(row);
    if (ds.dir) infoFromScan(row, { info: ds.info });
    return row;
  }

  function infoFromScan(row, d) {
    const info = row.querySelector(".ds-info");
    const i = d && d.info;
    if (!i || !i.files) return;
    const parts = [
      `${i.files} file(s)`,
      `bands: ${(i.bands || []).join("/")}`,
      `dtype: ${(i.dtypes || []).join("/")}`,
    ];
    if (i.nodata && i.nodata.length) parts.push(`nodata: ${i.nodata.join("/")}`);
    if (i.crs && i.crs.length && i.crs[0]) parts.push(`crs: ${i.crs[0]}`);
    for (const n of i.notes || []) parts.push(`! ${n}`);
    info.textContent = parts.join("  \u00b7  ");
    info.classList.toggle("status-warn", (i.notes || []).length > 0);
  }

  async function scanRow(row, btn) {
    const dir = row.querySelector('[data-ds="dir"]').value.trim();
    const info = row.querySelector(".ds-info");
    if (!dir) {
      info.textContent = "Enter the dataset folder path first.";
      info.classList.add("status-warn");
      return;
    }
    btn.disabled = true;
    btn.textContent = "Scanning\u2026";
    info.classList.remove("status-warn");
    let d;
    try {
      const res = await fetch(`/api/tests-scan?path=${encodeURIComponent(dir)}`);
      d = await res.json();
      if (!res.ok) throw new Error(d.error || `HTTP ${res.status}`);
    } catch (err) {
      info.textContent = `Scan failed: ${err.message}`;
      info.classList.add("status-warn");
      btn.disabled = false;
      btn.textContent = "Scan";
      return;
    }
    btn.disabled = false;
    btn.textContent = "Scan";
    const name = row.querySelector('[data-ds="name"]');
    if (!name.value.trim()) name.value = d.name;
    row.querySelector('[data-ds="kind"]').value = d.kind;
    row.querySelector('[data-ds="thresholds"]').value = (d.thresholds || []).join(", ");
    for (const cb of row.querySelectorAll("[data-ds-tests]")) {
      cb.checked = (d.tests || []).includes(cb.dataset.dsTests);
    }
    row.querySelector('[data-ds="rgb_bands"]').value = (d.rgb_bands || []).join(",");
    row.querySelector('[data-ds="dem_bands"]').value = (d.dem_bands || []).join(",");
    row.querySelector('[data-ds="fidwac_extra"]').value = (d.fidwac_extra || []).join(" ");
    infoFromScan(row, d);
  }

  function collect() {
    const datasets = [];
    for (const row of container.querySelectorAll(".ds-row")) {
      const dir = row.querySelector('[data-ds="dir"]').value.trim();
      if (!dir) continue;
      datasets.push({
        name: row.querySelector('[data-ds="name"]').value.trim(),
        dir,
        kind: row.querySelector('[data-ds="kind"]').value,
        thresholds: row.querySelector('[data-ds="thresholds"]').value,
        tests: Array.from(row.querySelectorAll("[data-ds-tests]:checked")).map((cb) => cb.dataset.dsTests),
        rgb_bands: row.querySelector('[data-ds="rgb_bands"]').value,
        dem_bands: row.querySelector('[data-ds="dem_bands"]').value,
        fidwac_extra: row.querySelector('[data-ds="fidwac_extra"]').value.trim().split(/\s+/).filter(Boolean),
      });
    }
    return { datasets };
  }

  window.fidwacTestsPayload = collect;

  addBtn.addEventListener("click", () => addRow({}).querySelector('[data-ds="dir"]').focus());
  const seed = initial.testsDatasets || [];
  if (seed.length) {
    for (const ds of seed) addRow(ds);
  } else {
    addRow({});
  }
})();

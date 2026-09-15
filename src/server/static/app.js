/* traffic-detection dashboard.
 *
 * Two live channels, stacked. The <img> is an MJPEG stream of the CLEAN frame
 * and the <canvas> over it draws boxes from the metadata WebSocket. They line
 * up because the canvas is sized to the camera's SOURCE resolution while both
 * elements are stretched to the same box by CSS, and because the encoder
 * preserves aspect ratio (src/server/video.py says so at more length).
 *
 * The reason for keeping boxes as data rather than letting OpenCV burn them
 * into the JPEG is everything in the toolbar: filtering to flagged objects,
 * hiding labels, showing attributes, trails. None of that is possible once
 * the boxes are pixels.
 *
 * HONESTY IS A FEATURE HERE. This backend is careful never to overclaim, and
 * several renderers below exist only to carry that through to the screen:
 * search results say "nearest N of M", not "found"; a journey shows observed
 * hops and names its gaps; an unread plate is absent rather than blank. If
 * you simplify one of those away, you have changed what the tool asserts.
 */
"use strict";

/* ---- tiny helpers ------------------------------------------------------ */
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));

const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const num = (v, d = 0) => (v == null || v === "" || Number.isNaN(+v))
  ? "–" : (+v).toFixed(d);

function toast(msg, kind = "bad") {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 5200);
}

/* Errors are shown, never swallowed. The previous dashboard caught every
 * failure into one "server unreachable" string, which made a broken endpoint
 * indistinguishable from a stopped server. */
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    let detail = r.status + "";
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    throw new Error(path.split("?")[0] + ": " + detail);
  }
  return r.json();
}

/* Times are ambiguous in this system and pretending otherwise is a bug:
 * `time_base` of 'clip' means seconds from the start of a file, 'wall' means
 * a unix epoch. Rendering a clip offset as a wall clock would invent a date. */
function when(value, base) {
  if (value == null) return "–";
  if (base === "wall") return new Date(value * 1000).toLocaleTimeString();
  const s = +value;
  return (s < 60 ? s.toFixed(1) + "s"
    : Math.floor(s / 60) + "m" + String(Math.round(s % 60)).padStart(2, "0") + "s");
}
const clock = t => t ? new Date(t * 1000).toLocaleTimeString() : "–";
const dur = s => s == null ? "–"
  : s < 90 ? (+s).toFixed(1) + "s"
  : s < 5400 ? (s / 60).toFixed(1) + "m" : (s / 3600).toFixed(1) + "h";

function tile(k, v, cls) {
  return `<div class="tile"><div class="k">${esc(k)}</div>
    <div class="v ${cls || ""}">${esc(v)}</div></div>`;
}
function tileSmall(k, v, cls) {
  return `<div class="tile"><div class="k">${esc(k)}</div>
    <div class="v sm ${cls || ""}">${esc(v)}</div></div>`;
}
function table(rows, cols, emptyMsg = "nothing yet") {
  if (!rows || !rows.length) return `<div class="empty">${esc(emptyMsg)}</div>`;
  const head = cols.map(c => `<th>${esc(c[0])}</th>`).join("");
  const body = rows.map((r, i) =>
    `<tr${c_attr(cols, r)}>` + cols.map(c => `<td>${c[1](r, i)}</td>`).join("") + "</tr>"
  ).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}
function c_attr(cols, r) { return cols.__click ? ' class="clickable"' : ""; }

/* Attributes arrive in two shapes: live boxes carry {key: value} and stored
 * rows carry {key: {value, conf}}. One renderer, so a card and a box agree. */
function attrPairs(attrs) {
  if (!attrs) return [];
  return Object.entries(attrs).map(([k, v]) =>
    [k, (v && typeof v === "object") ? v.value : v]).filter(([, v]) => v !== "" && v != null);
}
/* An ABSENCE is not a description. The person enricher reports every attribute
 * it looked for, so a pedestrian carrying nothing comes back as bag=none,
 * hat=no, glasses=no, holding=no - all true, and all useless as a label.
 * Rendering those fills a card with the word "no" and buries the two facts
 * that actually identify somebody: what they are wearing.
 *
 * Filtered HERE rather than on the server, because the reading itself is real
 * and the detail drawer shows all of it. "No hat" is worth knowing when you
 * are looking at one object, just not when you are scanning ninety. */
const ABSENT = new Set(["no", "none", "unknown", "n/a", "false"]);

/* Most identifying first, so a 3-badge cap keeps the useful ones. */
const ATTR_RANK = ["upper_color", "lower_color", "color", "bag", "lower",
                   "sleeves", "headwear", "hat", "glasses", "sunglasses",
                   "holding", "facing"];
const rankOf = k => {
  const i = ATTR_RANK.indexOf(String(k).replace(/^garment_/, ""));
  return i < 0 ? ATTR_RANK.length : i;
};
const describing = attrs => attrPairs(attrs)
  .filter(([, v]) => !ABSENT.has(String(v).toLowerCase()))
  .sort((a, b) => rankOf(a[0]) - rankOf(b[0]));

function attrBadges(attrs, limit = 99) {
  const p = describing(attrs);
  if (!p.length) return "";
  const shown = p.slice(0, limit).map(([k, v]) =>
    `<span class="badge" title="${esc(k)}">${esc(v)}</span>`).join("");
  const restTitle = p.slice(limit).map(x => x[0] + "=" + x[1]).join(", ");
  return `<div class="badges">${shown}${p.length > limit
    ? `<span class="badge" title="${esc(restTitle)}">+${p.length - limit}</span>`
    : ""}</div>`;
}

/* The overlay label wants the same filtering: "person no no no" stacked over a
 * small box is noise. */
const attrLabel = (attrs, limit = 4) =>
  describing(attrs).slice(0, limit).map(([, v]) => v);

/* ---- state ------------------------------------------------------------- */
const S = {
  view: "live",
  camera: null,
  cams: [],
  frame: null,
  ws: null,
  vocab: { cameras: [], classes: [], groups: [], attributes: {} },
  health: null,
  trails: new Map(),        // track_id -> [[x,y], ...]
  objPage: 0,
  vehicle: null,
};

const T = {          // toolbar toggles, remembered per browser
  boxes: true, labels: true, plates: true, attrs: false,
  trails: false, flagged: false, video: true,
};
try { Object.assign(T, JSON.parse(localStorage.getItem("td.toggles") || "{}")); }
catch (_) {}
const saveToggles = () => {
  try { localStorage.setItem("td.toggles", JSON.stringify(T)); } catch (_) {}
};

/* ---- routing ----------------------------------------------------------- */
const VIEWS = ["live", "search", "objects", "vehicles", "incidents", "system"];

function showView(name) {
  if (!VIEWS.includes(name)) name = "live";
  S.view = name;
  $$(".view").forEach(v => v.classList.toggle("on", v.id === "v-" + name));
  $$("#nav a").forEach(a => a.classList.toggle("on", a.dataset.view === name));
  if (name === "objects" && !$("#obj-results").children.length) loadObjects();
  if (name === "vehicles" && !$("#veh-list").children.length) loadVehicles();
  if (name === "incidents") loadIncidents();
  if (name === "system") renderSystem();
  if (name === "search") { renderIndex(); $("#q").focus(); }
}
window.addEventListener("hashchange",
  () => showView((location.hash || "#live").slice(1)));

/* ====================================================================== *
 *  LIVE
 * ====================================================================== */
const COLOR = { wrong_way: "#f85149", wrong_lane: "#d29922" };

function sourceSize() {
  const cam = S.cams.find(c => c.camera === S.camera);
  if (cam && cam.resolution) {
    const [w, h] = String(cam.resolution).split("x").map(Number);
    if (w && h) return [w, h];
  }
  return null;
}

function drawOverlay() {
  const c = $("#overlay"), g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);

  const f = S.frame;
  if (!f) return;
  const boxes = (f.boxes || []).filter(b =>
    !T.flagged || (b.lane_flag && b.lane_flag !== "ok"));

  if (T.trails) {
    g.lineWidth = 2;
    for (const [tid, pts] of S.trails) {
      if (pts.length < 2) continue;
      g.strokeStyle = "rgba(88,166,255,.45)";
      g.beginPath();
      g.moveTo(pts[0][0], pts[0][1]);
      for (const [x, y] of pts.slice(1)) g.lineTo(x, y);
      g.stroke();
    }
  }
  if (!T.boxes) return;

  for (const b of boxes) {
    const [x1, y1, x2, y2] = b.xyxy;
    // Flag colour wins over class colour: a wrong-way car must not read as a
    // healthy green box because it happens to be a vehicle.
    const stroke = COLOR[b.lane_flag] || (b.group === "vehicle" ? "#3fb950" : "#58a6ff");
    g.strokeStyle = stroke;
    g.lineWidth = (b.lane_flag && b.lane_flag !== "ok") ? 3 : 2;
    g.strokeRect(x1, y1, x2 - x1, y2 - y1);

    if (!T.labels) continue;
    const lines = [];
    const id = b.identity_id != null ? "V" + b.identity_id : "#" + b.track_id;
    let head = `${id} ${b.cls}`;
    if (T.plates && b.plate) head += `  ${b.plate}`;
    if (b.lane_flag && b.lane_flag !== "ok") head += `  !${b.lane_flag}`;
    lines.push(head);
    if (T.attrs) {
      const p = attrLabel(b.attrs);
      if (p.length) lines.push(p.join(" · "));
    }

    g.font = "13px ui-monospace, Menlo, monospace";
    const w = Math.max(...lines.map(l => g.measureText(l).width)) + 9;
    const h = lines.length * 15 + 3;
    const ty = Math.max(0, y1 - h);
    g.fillStyle = "rgba(8,10,14,.86)";
    g.fillRect(x1, ty, w, h);
    g.fillStyle = stroke;
    lines.forEach((l, i) => g.fillText(l, x1 + 4, ty + 13 + i * 15));
  }

  g.fillStyle = "rgba(139,147,167,.9)";
  g.font = "12px ui-monospace, Menlo, monospace";
  g.fillText(`frame ${f.frame_no}  ·  ${boxes.length}/${(f.boxes || []).length} shown`,
    10, c.height - 10);
}

function pushTrails(f) {
  if (!T.trails) { S.trails.clear(); return; }
  const live = new Set();
  for (const b of f.boxes || []) {
    live.add(b.track_id);
    const [x1, y1, x2, y2] = b.xyxy;
    const pt = [(x1 + x2) / 2, y2];             // ground point, as the analyses use
    const pts = S.trails.get(b.track_id) || [];
    pts.push(pt);
    if (pts.length > 40) pts.shift();
    S.trails.set(b.track_id, pts);
  }
  for (const tid of Array.from(S.trails.keys())) {
    if (!live.has(tid)) S.trails.delete(tid);
  }
}

function setStage() {
  const cam = S.cams.find(c => c.camera === S.camera) || {};
  const running = cam.state === "running";
  const img = $("#video"), stage = $("#stage");
  const wantVideo = T.video && running;

  stage.classList.toggle("novideo", !wantVideo);
  const url = wantVideo ? `/stream/${encodeURIComponent(S.camera)}.mjpg` : "";
  if (wantVideo) {
    // Only reassign when the target actually changes: writing src restarts the
    // MJPEG connection, so doing it on every poll would rebuild the stream
    // every two seconds.
    if (!img.dataset.cam || img.dataset.cam !== S.camera || !img.getAttribute("src")) {
      img.dataset.cam = S.camera;
      img.src = url;
    }
  } else if (img.getAttribute("src")) {
    img.removeAttribute("src");                 // closes the connection
    delete img.dataset.cam;
  }

  const size = sourceSize();
  const c = $("#overlay");
  if (size && (c.width !== size[0] || c.height !== size[1])) {
    c.width = size[0]; c.height = size[1];
  }
  $("#live-res").textContent = cam.resolution
    ? `${cam.resolution}${cam.is_live ? " live" : " file"}` : "";
  $("#live-title").textContent = S.camera ? `live — ${S.camera}` : "live";

  const ph = $("#ph");
  if (!running) {
    ph.hidden = false;
    ph.innerHTML = `camera <code>${esc(S.camera || "?")}</code> is
      <b>${esc(cam.state || "idle")}</b>${cam.error
        ? `<br><span class="bad">${esc(cam.error)}</span>` : ""}
      <br><span class="tiny">press the start button in the header</span>`;
  } else if (!S.frame) {
    ph.hidden = false;
    ph.innerHTML = `<span class="spin"></span>&nbsp; waiting for frames…`;
  } else {
    ph.hidden = true;
  }
}

$("#video").addEventListener("error", () => {
  // A stream that dies because the camera stopped is normal; say so rather
  // than leaving a broken-image icon.
  const img = $("#video");
  if (img.getAttribute("src")) {
    img.removeAttribute("src");
    delete img.dataset.cam;
    setTimeout(setStage, 1500);
  }
});

function connectWs(id) {
  if (S.ws) { S.ws.onclose = null; S.ws.close(); }
  S.frame = null; S.trails.clear(); drawOverlay();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = S.ws = new WebSocket(
    `${proto}://${location.host}/live/${encodeURIComponent(id)}`);
  ws.onopen = () => { $("#conn").innerHTML = `<span class="ok">●</span> live`; };
  ws.onclose = () => {
    $("#conn").innerHTML = `<span class="warn">●</span> reconnecting`;
    // The camera may simply have been stopped; retry so starting it from this
    // page brings the view back without a reload.
    setTimeout(() => { if (S.camera === id) connectWs(id); }, 2000);
  };
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.error) { $("#conn").innerHTML = `<span class="bad">●</span> ${esc(m.error)}`; return; }
    S.frame = m;
    pushTrails(m);
    setStage();
    drawOverlay();
    renderTiles(m);
    renderInFrame(m);
  };
}

/* Plate OCR either loaded or it did not, and the dashboard has to say
 * which. An empty plate column is ambiguous between "no plate was
 * readable in this frame" (normal, constantly) and "the reader never
 * loaded" (a setup problem that will not fix itself). Telling those two
 * apart is the entire point of surfacing this. */
function plateOcr() {
  const cam = S.cams.find(c => c.camera === S.camera) || {};
  return cam.plate_ocr || {};
}
/* Only TRUE when a backend was named and failed. Unknown stays quiet: the
 * engine initialises on the first frame carrying a plate box, so early in
 * a run there is legitimately nothing to report yet. */
const plateOcrBroken = () => {
  const p = plateOcr();
  return Boolean(p.backend) && p.ok === false;
};

function renderPlateBanner() {
  const el = $("#plate-banner");
  if (!el) return;
  if (!plateOcrBroken()) { el.hidden = true; return; }
  const p = plateOcr();
  el.hidden = false;
  el.innerHTML = `<strong>Plate reading is off, so every plate column will
    stay empty.</strong> The <code>${esc(p.backend)}</code> OCR backend did
    not load:<br><span class="tiny">${esc(p.error || "reason not reported")}</span>
    <br><br>Either finish installing it, or set
    <code>perception.attributes.plate.ocr_backend: fast_plate</code> in
    config.yaml - that backend ships with the pip install and needs no
    download.`;
}

function renderTiles(m) {
  const h = m.health || {}, cam = S.cams.find(x => x.camera === m.camera) || {};
  const fl = m.flagged || {};
  $("#tiles").innerHTML = [
    tile("fps", num(cam.fps, 1)),
    tile("tracked", h.tracked ?? "–"),
    tile("queue depth", h.queue_depth ?? "–", h.queue_depth > 5000 ? "warn" : ""),
    tile("rows shed", h.rows_dropped ?? 0, h.rows_dropped ? "warn" : ""),
    tile("rows lost", h.rows_failed ?? 0, h.rows_failed ? "bad" : ""),
    tile("frames dropped", h.frames_dropped ?? 0, h.frames_dropped ? "warn" : ""),
    tile("wrong way", fl.wrong_way ?? 0, fl.wrong_way ? "bad" : ""),
    tile("wrong lane", fl.wrong_lane ?? 0, fl.wrong_lane ? "warn" : ""),
    tileSmall("congestion", m.congestion ?? "–",
      m.congestion === "congested" ? "bad" : m.congestion ? "ok" : ""),
    tile("meta viewers", m.viewers ?? 0),
    tile("video viewers", cam.video_viewers ?? 0),
    tile("state size", h.state_size ?? "–"),
    tileSmall("writer", h.write_alarm ? "ALARM" : "ok", h.write_alarm ? "bad" : "ok"),
  ].join("");

  const counts = m.counts || {}, cross = m.crossings || {};
  const cells = Object.entries(counts).map(([k, v]) => tile(k, v));
  cells.push(tile("a → b", cross.a_to_b ?? 0), tile("b → a", cross.b_to_a ?? 0));
  $("#counts").innerHTML = cells.length
    ? cells.join("")
    : `<div class="empty">no counting lines configured for this camera</div>`;
}

function renderInFrame(m) {
  const boxes = (m.boxes || []).slice().sort((a, b) =>
    (a.identity_id ?? 1e9) - (b.identity_id ?? 1e9) || a.track_id - b.track_id);
  $("#inframe-n").textContent = boxes.length ? `${boxes.length} object(s)` : "";
  const cols = [
    ["id", b => b.identity_id != null
      ? `<span class="badge b-accent">V${b.identity_id}</span>`
      : `<span class="muted">#${esc(b.track_id)}</span>`],
    ["class", b => esc(b.cls)],
    ["plate", b => b.plate ? `<code>${esc(b.plate)}</code>`
      : (b.group === "vehicle" && plateOcrBroken()
          ? `<span class="tiny warn">reader off</span>`
          : `<span class="muted">&ndash;</span>`)],
    ["flag", b => b.lane_flag && b.lane_flag !== "ok"
      ? `<span class="badge b-bad">${esc(b.lane_flag)}</span>`
      : (b.lane_id ? `<span class="muted">${esc(b.lane_id)}</span>` : "–")],
    ["attrs", b => attrBadges(b.attrs, 4) || "–"],
  ];
  cols.__click = true;
  $("#inframe").innerHTML = table(boxes, cols, "nothing in frame");
  $$("#inframe tbody tr").forEach((tr, i) => {
    const oid = boxes[i] && boxes[i].object_id;
    if (oid != null) tr.onclick = () => openObject(oid);
  });
}

function renderLiveCameras() {
  const cols = [
    ["camera", r => `<span class="dot s-${esc(r.state)}"></span>${esc(r.camera)}`],
    ["state", r => esc(r.state) + (r.error ? ` <span class="bad">${esc(r.error)}</span>` : "")],
    ["fps", r => num(r.fps, 1)],
    ["last", r => r.seconds_since_frame == null ? "\u2013"
      : `<span class="${r.seconds_since_frame > 10 ? "warn" : ""}">${
          r.seconds_since_frame}s</span>`],
  ];
  cols.__click = true;
  $("#cams").innerHTML = table(S.cams, cols, "no cameras configured");
  $$("#cams tbody tr").forEach((tr, i) => tr.onclick = () => {
    S.camera = S.cams[i].camera;
    $("#pick").value = S.camera;
    connectWs(S.camera);
    setStage();
  });
}

/* ====================================================================== *
 *  SEARCH  (open vocabulary, over embedded crops)
 * ====================================================================== */
function cropCard(o, opts = {}) {
  const id = o.object_id ?? o.id;
  const size = (o.crop_w && o.crop_h) ? `${o.crop_w}×${o.crop_h}` : "";
  const thumb = o.has_crop
    ? `<img loading="lazy" src="/crops/${id}.jpg" alt="object ${id}">`
    : `<div class="no">no image<br><span class="tiny">crop reaped or never taken</span></div>`;
  const flag = o.lane_flag && o.lane_flag !== "ok"
    ? `<span class="badge b-bad">${esc(o.lane_flag)}</span>` : "";
  const plate = o.plate
    ? `<span class="badge ${o.plate_unbound ? "b-warn" : "b-accent"}"
        title="${o.plate_unbound ? "read, but no durable vehicle row" : "vehicle id bound"}"
        >${esc(o.plate)}</span>` : "";
  return `<div class="card" data-id="${id}">
    <div class="thumb">${thumb}
      ${opts.score != null ? `<span class="score">${opts.score.toFixed(3)}</span>` : ""}
      ${opts.rank != null ? `<span class="rank">#${opts.rank}</span>` : ""}
    </div>
    <div class="meta">
      <div class="cls">${esc(o.cls_name || o.cls || "?")} <span class="muted">${esc(id)}</span></div>
      <div class="sub">${esc(o.camera || "")} · ${esc(when(o.first_seen_s, o.time_base))}${
        size ? " · " + size : ""}</div>
      ${plate || flag ? `<div class="badges">${plate}${flag}</div>` : ""}
      ${attrBadges(o.attributes, 3)}
    </div></div>`;
}

function wireCards(root) {
  $$(root + " .card").forEach(el =>
    el.onclick = () => openObject(+el.dataset.id));
}

/* INDEX COVERAGE, always visible - not only when a search comes back
 * empty. A partial index is the dangerous state: search returns results,
 * they look fine, and nothing says 60% of the corpus was never indexed.
 * Embeddings are built by a separate step, so this drifts by design as
 * the cameras run. */
function renderIndex() {
  const sr = (S.health || {}).search || {};
  const idx = sr.index || {}, emb = sr.embedder || {};
  const tiles = $("#index-tiles"), note = $("#index-note");
  if (!tiles) return;
  if (!idx.crops && !idx.embedded) {
    tiles.innerHTML = ""; note.innerHTML = "";
    return;
  }
  const pct = idx.crops ? Math.round(100 * idx.embedded / idx.crops) : 0;
  tiles.innerHTML = [
    tile("indexed", `${idx.embedded} / ${idx.crops}`, pct >= 99 ? "ok" : ""),
    tile("coverage", pct + "%", pct >= 99 ? "ok" : pct >= 50 ? "warn" : "bad"),
    tile("no vectors", idx.unembedded ?? 0, idx.unembedded ? "warn" : "ok"),
    tileSmall("space", idx.model || "-"),
    tileSmall("embedder", emb.enabled === false ? "off" : (emb.state || "-"),
      emb.state === "caught up" ? "ok" : emb.state === "embedding" ? "" : 
      (emb.enabled === false ? "warn" : "")),
    tile("embedded now", emb.embedded ?? 0),
  ].join("");

  // The tiles are the number; this line is what to DO about it.
  if (!idx.unembedded) { note.innerHTML = ""; return; }
  if (emb.enabled === false) {
    note.innerHTML = `<div class="notice warn"><strong>${idx.unembedded}
      crop(s) have no vectors and nothing is indexing them.</strong>
      Background embedding is off, so search covers only what was indexed
      by hand. Run it, or set <code>server.search.embed_in_background:
      true</code>:<br><code>python tools/embed_crops.py --model
      ${esc(idx.model)}</code></div>`;
  } else if (emb.state === "unavailable" || emb.state === "failed") {
    note.innerHTML = `<div class="notice bad"><strong>The background
      embedder is not running, so the index will not grow.</strong>
      <br><span class="tiny">${esc(emb.last_error || "reason not reported")}</span></div>`;
  } else {
    note.innerHTML = `<div class="notice"><strong>Indexing in progress:
      ${idx.unembedded} crop(s) still have no vectors.</strong> Results
      below cover the ${idx.embedded} already embedded, so a thing that is
      in the footage may not be findable yet. The embedder runs on a duty
      cycle (batch ${emb.batch}, ${emb.pause_s}s pause) so it does not take
      frame rate from the cameras.</div>`;
  }
}

async function doSearch(ev) {
  if (ev) ev.preventDefault();
  const q = $("#q").value.trim();
  if (!q) return;
  const p = new URLSearchParams({
    q, top: $("#s-top").value, model: $("#s-model").value,
    min_px: $("#s-minpx").value, min_conf: $("#s-minconf").value,
  });
  if ($("#s-camera").value) p.set("camera", $("#s-camera").value);

  $("#search-status").innerHTML =
    `<div class="notice"><span class="spin"></span>&nbsp; encoding and ranking…</div>`;
  $("#search-results").innerHTML = "";
  let r;
  try { r = await api("/search?" + p); }
  catch (e) {
    $("#search-status").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`;
    return;
  }

  if (r.error) {
    $("#search-status").innerHTML = `<div class="notice bad"><strong>cannot search.</strong>
      ${esc(r.error)}</div>`;
    return;
  }

  const bits = [];
  /* Nothing indexed is the most likely reason a search comes back empty,
   * and it is fixable in one command. Lead with that, and skip the ranker
   * disclaimer: "the nearest 0 of 0 candidates" is true and tells the user
   * nothing about what to do next. */
  if (!r.candidates) {
    // Distinguish "nothing embedded at all" from "this model is empty
    // while another one is full" - the second is a dropdown away from
    // working, and telling the user it is unindexed would be false.
    const idx = r.indexed_models || {};
    const others = Object.entries(idx).filter(([m, n]) => m !== r.model && n > 0);
    const body = others.length
      ? `<strong>No vectors in <code>${esc(r.model)}</code>, but ${others.map(([m, n]) => `<code>${esc(m)}</code> has ${n}`).join(" and ")}.</strong><br><br>Cosine similarity only means anything within one embedding space, so the two indexes cannot be searched together. Switch the model dropdown, or embed this one:<br><code>python tools/embed_crops.py --model ${esc(r.model)}</code>`
      : `<strong>Nothing is indexed yet, so there is nothing to search.
         </strong><br><br>Search ranks pre-computed CLIP vectors over saved
         crops; the pipeline does not build them while it runs,
         deliberately - no real-time decision needs an embedding.
         <br><br>${esc(r.note || "")}`;
    $("#search-status").innerHTML = `<div class="notice bad">${body}</div>`;
    $("#search-results").innerHTML = "";
    return;
  }
  /* The ranker framing, verbatim from the API. This is not decoration: the
   * measurement found no similarity floor separating present concepts from
   * absent ones, so a top hit is the nearest crop whether or not the thing
   * asked for is in the footage. Presenting these as detections would be a
   * false claim the backend deliberately refuses to make. */
  bits.push(`<div class="notice"><strong>Ranked, not detected.</strong>
    ${esc(r.ranker_note || "")}${r.score_spread
      ? ` Scores ${r.score_spread[0].toFixed(3)}–${r.score_spread[1].toFixed(3)}.` : ""}</div>`);
  if (r.note) bits.push(`<div class="notice warn">${esc(r.note)}</div>`);
  for (const w of r.warnings || [])
    bits.push(`<div class="notice warn"><strong>Cheaper exact answer exists.</strong> ${esc(w)}</div>`);
  if (r.scope && (r.scope.camera || r.scope.t_from || r.scope.t_to)) {
    const s = [];
    if (r.scope.camera) s.push("camera " + r.scope.camera);
    if (r.scope.t_from) s.push("after " + clock(r.scope.t_from));
    if (r.scope.t_to) s.push("before " + clock(r.scope.t_to));
    bits.push(`<div class="notice">Understood scope: ${esc(s.join(", "))}.
      Subject searched: <code>${esc(r.subject)}</code></div>`);
  }
  if ((r.excluded_runs || []).length) {
    /* Reported, never silently filtered - matching how journeys.py treats
     * sightings it cannot order. */
    bits.push(`<div class="notice warn"><strong>${r.excluded_runs.length} run(s)
      use a clip clock</strong>, so a wall-clock window cannot honestly be
      applied to them. They were not excluded from ranking; their timestamps
      are just not comparable.</div>`);
  }
  bits.push(`<div class="muted" style="margin:8px 0">${r.hits.length} shown of
    ${r.candidates} candidates in <code>${esc(r.model)}</code></div>`);
  $("#search-status").innerHTML = bits.join("");

  $("#search-results").innerHTML = r.hits.length
    ? r.hits.map((h, i) => cropCard(h, { score: h.score, rank: i + 1 })).join("")
    : `<div class="empty">no candidates in scope. Embed crops first:
       <code>python tools/embed_crops.py</code></div>`;
  wireCards("#search-results");
}

/* ====================================================================== *
 *  OBJECTS
 * ====================================================================== */
const OBJ_PAGE = 60;

async function loadObjects() {
  const p = new URLSearchParams({ limit: OBJ_PAGE, offset: S.objPage * OBJ_PAGE });
  const map = { camera: "#o-camera", cls: "#o-class", group: "#o-group",
                attr_key: "#o-attrkey", attr_value: "#o-attrval" };
  for (const [k, sel] of Object.entries(map)) if ($(sel).value) p.set(k, $(sel).value);
  if ($("#o-flagged").checked) p.set("flagged", "true");
  if ($("#o-crop").checked) p.set("has_crop", "true");

  $("#obj-status").innerHTML = `<div class="notice"><span class="spin"></span>&nbsp; loading…</div>`;
  let r;
  try { r = await api("/objects?" + p); }
  catch (e) { $("#obj-status").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; return; }

  $("#obj-total").textContent = `${r.total} match(es)`;
  $("#obj-status").innerHTML = ($("#o-flagged").checked && !r.total)
    ? `<div class="notice warn"><strong>No flagged objects.</strong> That is not the
       same as no violations: a lane whose traffic was never observed during
       warmup reports <code>no-data</code> and its direction stays unverified.</div>`
    : "";
  $("#obj-results").innerHTML = r.objects.length
    ? r.objects.map(o => cropCard(o)).join("")
    : `<div class="empty">nothing matches these filters</div>`;
  wireCards("#obj-results");

  const pages = Math.max(1, Math.ceil(r.total / OBJ_PAGE));
  $("#obj-pager").hidden = pages < 2;
  $("#o-page").textContent = `page ${S.objPage + 1} / ${pages}`;
  $("#o-prev").disabled = S.objPage === 0;
  $("#o-next").disabled = S.objPage + 1 >= pages;
}

/* ====================================================================== *
 *  OBJECT DRAWER
 * ====================================================================== */
function closeDrawer() { $("#drawer-host").innerHTML = ""; }

async function openObject(id) {
  const host = $("#drawer-host");
  host.innerHTML = `<div class="drawer-bg"></div><aside class="drawer">
    <header><b>object ${esc(id)}</b><span class="spacer"></span>
      <button data-x>close</button></header>
    <div class="body"><span class="spin"></span> loading…</div></aside>`;
  host.querySelector(".drawer-bg").onclick = closeDrawer;
  host.querySelector("[data-x]").onclick = closeDrawer;

  let o;
  try { o = await api("/objects/" + id); }
  catch (e) {
    host.querySelector(".drawer .body").innerHTML =
      `<div class="notice bad">${esc(e.message)}</div>`;
    return;
  }

  const kv = [
    ["class", `${esc(o.cls_name)} <span class="muted">(${esc(o.cls_group)})</span>`],
    ["camera", esc(o.camera)],
    ["track id", esc(o.track_id)],
    ["identity id", o.identity_id != null ? "V" + o.identity_id
      : `<span class="muted">none — no confident plate</span>`],
    ["plate", o.plate ? `<code>${esc(o.plate)}</code>${o.plate_unbound
      ? ` <span class="badge b-warn">unbound</span>` : ""}`
      : `<span class="muted">never read</span>`],
    ["first seen", esc(when(o.first_seen_s, o.time_base))],
    ["last seen", esc(when(o.last_seen_s, o.time_base))],
    ["frames seen", esc(o.frames_seen)],
    ["best conf", num(o.best_conf, 3)],
    ["lane", o.lane_id ? esc(o.lane_id) : `<span class="muted">off-lane</span>`],
    ["flag", o.lane_flag && o.lane_flag !== "ok"
      ? `<span class="badge b-bad">${esc(o.lane_flag)}</span>`
      : esc(o.lane_flag || "–")],
  ];
  const attrs = attrPairs(o.attributes);
  const attrRows = attrs.length
    ? attrs.map(([k, v]) => {
        const conf = o.attributes[k] && o.attributes[k].conf;
        return `<dt>${esc(k)}</dt><dd>${esc(v)}${conf != null
          ? ` <span class="tiny">${(+conf).toFixed(2)}</span>` : ""}</dd>`;
      }).join("")
    : `<dt class="muted">attributes</dt><dd class="muted">none read</dd>`;

  host.querySelector(".drawer .body").innerHTML = `
    ${o.has_crop ? `<img class="full" src="/crops/${o.id}.jpg" alt="">`
      : `<div class="notice warn">No image for this object — either the crop was
         reaped by retention, or the detection never met
         <code>objects.crop_min_conf</code>.</div>`}
    <dl class="kv">${kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>
    <h2 style="font-size:11px;color:var(--dim);margin:14px 0 6px;text-transform:uppercase">attributes</h2>
    <dl class="kv">${attrRows}</dl>
    `;
}

/* ====================================================================== *
 *  VEHICLES + JOURNEYS
 * ====================================================================== */
async function loadVehicles() {
  const p = new URLSearchParams({ limit: 200 });
  if ($("#v-plate").value.trim()) p.set("key", $("#v-plate").value.trim());
  let r, y;
  try { [r, y] = await Promise.all([api("/identities?" + p), api("/yield")]); }
  catch (e) { $("#veh-list").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; return; }

  $("#veh-total").textContent = `${r.total} vehicle(s)`;
  const cols = [
    ["plate", v => `<code>${esc(v.plate)}</code>`],
    ["sightings", v => esc(v.sightings)],
    ["cameras", v => v.cameras.length
      ? v.cameras.map(c => `<span class="badge">${esc(c)}</span>`).join(" ")
      : "–"],
    ["last seen", v => esc(clock(v.last_seen_at))],
  ];
  cols.__click = true;
  $("#veh-list").innerHTML = table(r.identities, cols,
    "no vehicles yet — no plate has been read confidently enough to bind an identity");
  $$("#veh-list tbody tr").forEach((tr, i) =>
    tr.onclick = () => openJourney(r.identities[i]));

  $("#yield-tiles").innerHTML = [
    tile("cameras", y.cameras),
    tile("vehicles", y.vehicles),
    tile("multi-camera", y.multi_camera_vehicles,
      y.multi_camera_vehicles ? "ok" : "warn"),
    tile("unanchored runs", y.unanchored_runs, y.unanchored_runs ? "warn" : ""),
    tile("runs w/o location", y.runs_without_location,
      y.runs_without_location ? "warn" : ""),
  ].join("") + (y.unanchored_runs
    ? `<div class="notice warn" style="margin-top:10px"><strong>${y.unanchored_runs}
       run(s) have no anchored clock.</strong> A file run without
       <code>--recorded-at</code> contributes sightings no journey can order, so a
       low hop count here is a configuration problem rather than a plate-accuracy
       one. The two need different work.</div>`
    : "");
}

async function openJourney(v) {
  S.vehicle = v;
  $("#journey-for").textContent = v.plate;
  $("#journey").innerHTML = `<span class="spin"></span> loading…`;
  let p;
  try { p = await api(`/identities/${v.id}/path`); }
  catch (e) { $("#journey").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; return; }

  const out = [];
  out.push(`<dl class="kv">
    <dt>plate</dt><dd><code>${esc(p.plate)}</code></dd>
    <dt>sightings</dt><dd>${esc(p.total_sightings)}</dd>
    <dt>cameras</dt><dd>${p.cameras.length ? esc(p.cameras.join(" → "))
      : `<span class="muted">none orderable</span>`}</dd></dl>`);

  /* Observed hops only. The backend cannot claim an A->D journey skipped B and
   * C because it never claims intermediate positions at all - so gaps are
   * drawn as gaps and nothing is interpolated. */
  if (p.journeys.length) {
    for (const j of p.journeys) {
      const items = [];
      j.visits.forEach((vis, i) => {
        items.push(`<li><div class="where"><b>${esc(vis.camera)}</b>
          <span class="muted">${esc(vis.cls_name)} · ${esc(vis.sightings)} sighting(s)</span></div>
          <div class="when">${esc(clock(vis.start))} → ${esc(clock(vis.end))}
            · dwell ${esc(dur(vis.dwell_s))}</div></li>`);
        const hop = j.hops[i];
        if (hop) {
          items.push(`<li class="gap"><div class="where muted">unobserved stretch</div>
            <div class="when">${esc(dur(hop.elapsed_s))}
              · ${hop.distance_km == null
                ? `<span class="warn">distance unknown (no camera location)</span>`
                : esc(hop.distance_km + " km")}
              ${hop.speed_kmh != null ? `· ~${esc(hop.speed_kmh)} km/h` : ""}
              ${hop.suspect ? `<span class="badge b-bad">suspect: ${
                esc((hop.suspect_reasons || []).join(", "))}</span>` : ""}</div></li>`);
        }
      });
      out.push(`<div class="muted" style="margin:10px 0 6px">journey ·
        ${esc(dur(j.duration_s))}${j.distance_km != null
          ? " · " + esc(j.distance_km) + " km"
          : ` · <span class="warn">length unmeasurable</span>`}
        ${j.suspect ? ` · <span class="bad">suspect</span>` : ""}</div>
        <ul class="hops">${items.join("")}</ul>`);
    }
  } else {
    out.push(`<div class="empty">no orderable journey for this vehicle</div>`);
  }

  if ((p.unorderable || []).length) {
    out.push(`<div class="notice warn"><strong>${p.unorderable.length}
      sighting(s) could not be ordered</strong> and are listed rather than
      silently placed in sequence:</div>` +
      table(p.unorderable, [
        ["object", u => `<a href="#" data-obj="${u.object_id}">${esc(u.object_id)}</a>`],
        ["camera", u => esc(u.camera)],
        ["reason", u => `<span class="tiny">${esc(u.reason)}</span>`],
      ]));
  }
  if ((p.conflicts || []).length) {
    out.push(`<div class="notice bad"><strong>${p.conflicts.length} conflict(s):</strong>
      this plate was seen at two cameras at overlapping times, which is
      physically impossible. Either the plate is cloned or OCR read two
      different vehicles as the same string — treat this identity as
      untrustworthy.</div>`);
  }
  $("#journey").innerHTML = out.join("");
  $$("#journey [data-obj]").forEach(a =>
    a.onclick = e => { e.preventDefault(); openObject(+a.dataset.obj); });
}

/* ====================================================================== *
 *  INCIDENTS + EVENTS
 * ====================================================================== */
async function loadIncidents() {
  const ip = new URLSearchParams({ limit: 60 });
  if ($("#i-kind").value) ip.set("kind", $("#i-kind").value);
  if ($("#i-camera").value) ip.set("camera", $("#i-camera").value);

  let inc;
  try { inc = await api("/incidents?" + ip); }
  catch (e) { $("#inc-list").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; return; }

  const cols = [
    ["at", r => esc(clock(r.created_at))],
    ["kind", r => `<span class="badge ${r.kind === "wrong_way" ? "b-bad" : "b-warn"}">${esc(r.kind)}</span>`],
    ["camera", r => esc(r.camera)],
    ["plate", r => { const p = r.payload && r.payload.vehicle && r.payload.vehicle.plate;
      return p ? `<code>${esc(p)}</code>` : `<span class="muted">none</span>`; }],
    ["image", r => r.object_id != null
      ? `<img src="/crops/${r.object_id}.jpg" alt="" style="height:34px;border-radius:3px"
           onerror="this.replaceWith(document.createTextNode('reaped'))">`
      : "–"],
    ["delivered", r => {
      const tot = r.sent + r.pending + r.dead;
      return `${r.sent}/${tot}` + (r.dead ? ` <span class="bad">${r.dead} dead</span>` : "");
    }],
  ];
  cols.__click = true;
  $("#inc-list").innerHTML = table(inc.incidents, cols,
    "no incidents raised yet");
  $$("#inc-list tbody tr").forEach((tr, i) => {
    const r = inc.incidents[i];
    tr.onclick = () => showIncident(r);
  });

}

function showIncident(r) {
  const host = $("#drawer-host");
  host.innerHTML = `<div class="drawer-bg"></div><aside class="drawer">
    <header><b>${esc(r.kind)}</b><span class="muted">${esc(r.camera)}</span>
      <span class="spacer"></span><button data-x>close</button></header>
    <div class="body">
      ${r.object_id != null ? `<img class="full" src="/crops/${r.object_id}.jpg" alt=""
        onerror="this.replaceWith(document.createTextNode(''))">` : ""}
      <dl class="kv">
        <dt>incident</dt><dd><code>${esc(r.id)}</code></dd>
        <dt>raised</dt><dd>${esc(clock(r.created_at))}</dd>
        <dt>delivered</dt><dd>${esc(r.sent)} sent, ${esc(r.pending)} pending,
          ${r.dead ? `<span class="bad">${esc(r.dead)} dead</span>` : "0 dead"}</dd>
      </dl>
      <pre class="json">${esc(JSON.stringify(r.payload, null, 2))}</pre>
    </div></aside>`;
  host.querySelector(".drawer-bg").onclick = closeDrawer;
  host.querySelector("[data-x]").onclick = closeDrawer;
}

/* ====================================================================== *
 *  SYSTEM
 * ====================================================================== */
async function renderSystem() {
  let del;
  try { del = await api("/deliveries?limit=40"); }
  catch (e) { del = { summary: {}, deliveries: [] }; toast(e.message); }
  const h = S.health || {};

  $("#sys-cams").innerHTML = table(S.cams, [
    ["camera", r => `<span class="dot s-${esc(r.state)}"></span>${esc(r.camera)}`],
    ["state", r => esc(r.state) + (r.error ? ` <span class="bad">${esc(r.error)}</span>` : "")],
    ["fps", r => num(r.fps, 1)],
    ["frames", r => esc(r.frames)],
    ["last frame", r => r.seconds_since_frame == null ? "–" : r.seconds_since_frame + "s ago"],
    ["restarts", r => esc(r.restarts)],
    ["meta", r => esc(r.viewers)],
    ["video", r => esc(r.video_viewers ?? 0)],
  ], "no cameras configured");

  const st = h.storage || {};
  $("#sys-store").innerHTML = Object.entries(st).map(([k, v]) =>
    tileSmall(k.replace(/_/g, " "), typeof v === "object" ? JSON.stringify(v) : v,
      (/fail|alarm|drop/.test(k) && v) ? "bad" : "")).join("")
    || `<div class="empty">no storage stats</div>`;

  const ds = del.summary || {};
  $("#sys-del").innerHTML =
    `<div class="muted" style="margin-bottom:6px">sent ${ds.sent || 0} ·
      pending ${ds.pending || 0} ·
      <span class="${ds.dead ? "bad" : ""}">dead ${ds.dead || 0}</span></div>` +
    table(del.deliveries, [
      ["endpoint", r => esc(String(r.endpoint).replace(/^https?:\/\//, ""))],
      ["status", r => `<span class="badge ${r.status === "dead" ? "b-bad"
        : r.status === "sent" ? "b-ok" : "b-warn"}">${esc(r.status)}</span>`],
      ["tries", r => esc(r.attempts)],
      ["error", r => `<span class="tiny">${esc((r.last_error || "").slice(0, 60))}</span>`],
    ], "no deliveries — no subscriptions configured, or nothing has fired");

  const crops = (h.retention || {}).crops || {};
  $("#sys-disk").innerHTML = [
    tile("pinned crops", crops.pinned_crops ?? 0),
    tile("pinned MB", num(crops.pinned_mb, 1), "warn"),
    tile("reclaimable", crops.reclaimable_crops ?? 0),
    tile("reclaimable MB", num(crops.reclaimable_mb, 1)),
    tile("reaped MB", num(crops.deleted_mb, 1)),
    tileSmall("retention", crops.enabled ? "on" : "off", crops.enabled ? "ok" : ""),
  ].join("");

  const vid = h.video || {}, hub = h.hub || {};
  $("#sys-stream").innerHTML = [
    tileSmall("video", vid.enabled ? "on" : "off", vid.enabled ? "ok" : "warn"),
    tile("video fps", vid.fps ?? "–"),
    tile("jpeg quality", vid.quality ?? "–"),
    tile("max width", vid.max_width ?? "–"),
    tile("frames encoded", vid.encoded ?? 0),
    tile("avg encode ms", num(vid.avg_encode_ms, 2)),
    tile("push hz", hub.push_hz ?? "–"),
    tile("meta published", hub.published ?? 0),
    tile("viewer drops", (hub.viewer_drops ?? 0) + (vid.viewer_drops ?? 0)),
    tileSmall("search model", (h.search || {}).default_model || "–"),
  ].join("");

  $("#sys-raw").textContent = JSON.stringify(h, null, 2);
}

/* ====================================================================== *
 *  POLLING, WIRING, KEYBOARD
 * ====================================================================== */
function fillSelect(sel, values, allLabel, keep) {
  const el = $(sel);
  const was = el.value;
  const opts = [`<option value="">${esc(allLabel)}</option>`].concat(
    values.map(v => `<option value="${esc(v)}">${esc(v)}</option>`));
  const next = opts.join("");
  if (el.innerHTML !== next) {
    el.innerHTML = next;
    if (keep && values.includes(was)) el.value = was;
  }
}

async function poll() {
  let cams, health;
  try {
    [cams, health] = await Promise.all([api("/cameras"), api("/health")]);
  } catch (e) {
    $("#conn").innerHTML = `<span class="bad">●</span> server unreachable`;
    return;
  }
  S.cams = cams.cameras || [];
  S.health = health;

  if (!S.camera && S.cams.length) {
    S.camera = S.cams[0].camera;
    connectWs(S.camera);
  }
  fillSelect("#pick", S.cams.map(c => c.camera), "— no cameras —", true);
  if (S.camera) $("#pick").value = S.camera;

  const cam = S.cams.find(c => c.camera === S.camera) || {};
  const running = cam.state === "running";
  $("#toggle").textContent = running ? "stop" : "start";
  $("#toggle").className = running ? "danger" : "primary";
  $("#toggle").disabled = !S.camera;

  $("#uptime").textContent = health.uptime_s != null
    ? `· up ${dur(health.uptime_s)} · ${health.cameras} camera(s)` : "";
  const bad = (health.unhealthy || []).length;
  const nsys = $("#n-sys");
  nsys.hidden = !bad;
  nsys.textContent = bad;
  nsys.className = "n hot";

  setStage();
  renderLiveCameras();
  renderPlateBanner();
  renderIndex();
  if (S.view === "live" && S.frame) renderTiles(S.frame);
  if (S.view === "system") renderSystem();
}

async function pollIncidentCount() {
  try {
    const r = await api("/incidents?limit=1");
    const dead = (S.health && S.health.webhooks && S.health.webhooks.dead) || 0;
    const n = $("#n-inc");
    const any = r.incidents.length;
    n.hidden = !any;
    n.textContent = any ? (dead ? dead + " dead" : "●") : "";
    n.className = "n" + (dead ? " hot" : "");
  } catch (_) { /* the badge is cosmetic; poll() already reports outages */ }
}

async function loadVocabulary() {
  try { S.vocab = await api("/vocabulary"); }
  catch (e) { toast("could not load filter vocabulary: " + e.message); return; }
  const v = S.vocab;
  fillSelect("#s-camera", v.cameras, "all cameras", true);
  fillSelect("#o-camera", v.cameras, "all cameras", true);
  fillSelect("#i-camera", v.cameras, "all cameras", true);
  fillSelect("#o-class", v.classes, "all classes", true);
  fillSelect("#o-group", v.groups, "all groups", true);
  fillSelect("#o-attrkey", Object.keys(v.attributes), "any attribute", true);
  fillSelect("#i-kind", ["wrong_way", "wrong_lane", "congestion", "face_match",
    "intrusion", "loitering", "crowd", "running"], "all kinds", true);
  const models = (S.health && S.health.search && S.health.search.models) || [];
  const def = (S.health && S.health.search && S.health.search.default_model);
  if (models.length) {
    $("#s-model").innerHTML = models.map(m =>
      `<option value="${esc(m)}"${m === def ? " selected" : ""}>${esc(m)}</option>`).join("");
  }
}

/* ---- wiring ------------------------------------------------------------ */
$("#pick").onchange = e => {
  S.camera = e.target.value;
  S.frame = null;
  connectWs(S.camera);
  setStage();
};
$("#toggle").onclick = async () => {
  const cam = S.cams.find(c => c.camera === S.camera);
  if (!cam) return;
  const action = cam.state === "running" ? "stop" : "start";
  $("#toggle").disabled = true;
  try {
    const r = await fetch(`/cameras/${encodeURIComponent(S.camera)}/${action}`,
      { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    // Stopping finalises every live track, so it is not instant by design.
    toast(action === "stop"
      ? "stopping — finalising tracks in flight" : "starting…", "info");
  } catch (e) { toast("could not " + action + ": " + e.message); }
  setTimeout(poll, 600);
};

for (const [key, id] of Object.entries({
  boxes: "#t-boxes", labels: "#t-labels", plates: "#t-plates",
  attrs: "#t-attrs", trails: "#t-trails", flagged: "#t-flagged",
  video: "#t-video",
})) {
  $(id).checked = T[key];
  $(id).onchange = e => {
    T[key] = e.target.checked;
    saveToggles();
    if (key === "video") setStage();
    if (key === "trails") S.trails.clear();
    drawOverlay();
  };
}

$("#fs").onclick = () => {
  const st = $("#stage");
  st.classList.toggle("fs");
  drawOverlay();
};
$("#search-form").onsubmit = doSearch;
$("#o-reload").onclick = () => { S.objPage = 0; loadObjects(); };
$("#o-prev").onclick = () => { S.objPage = Math.max(0, S.objPage - 1); loadObjects(); };
$("#o-next").onclick = () => { S.objPage += 1; loadObjects(); };
$$("#o-camera,#o-class,#o-group,#o-flagged,#o-crop,#o-attrkey").forEach(
  el => el.onchange = () => { S.objPage = 0; loadObjects(); });
$("#o-attrkey").addEventListener("change", () => {
  const vals = S.vocab.attributes[$("#o-attrkey").value] || [];
  fillSelect("#o-attrval", vals, "any value", false);
});
$("#o-attrval").onchange = () => { S.objPage = 0; loadObjects(); };
$("#v-reload").onclick = loadVehicles;
$("#v-plate").addEventListener("keydown", e => { if (e.key === "Enter") loadVehicles(); });
$("#i-kind").onchange = loadIncidents;
$("#i-camera").onchange = loadIncidents;

document.addEventListener("keydown", e => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) {
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (e.key === "Escape") { closeDrawer(); $("#stage").classList.remove("fs"); return; }
  const n = "123456".indexOf(e.key);
  if (n >= 0) { location.hash = "#" + VIEWS[n]; return; }
  if (e.key === "c" && S.cams.length > 1) {
    const i = S.cams.findIndex(c => c.camera === S.camera);
    S.camera = S.cams[(i + 1) % S.cams.length].camera;
    $("#pick").value = S.camera;
    connectWs(S.camera);
    setStage();
  }
  if (e.key === "f") $("#fs").click();
  if (e.key === "/") { location.hash = "#search"; e.preventDefault(); $("#q").focus(); }
});

/* ---- go ---------------------------------------------------------------- */
(async function start() {
  showView((location.hash || "#live").slice(1));
  drawOverlay();
  await poll();
  await loadVocabulary();
  pollIncidentCount();
  setInterval(poll, 2000);
  setInterval(pollIncidentCount, 8000);
})();

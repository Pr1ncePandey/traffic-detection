"""The live dashboard, served as one self-contained page.

Inlined as a string rather than shipped as a static directory because it is one
file with no build step and no dependencies, and a StaticFiles mount would be
more moving parts than the thing it serves.

WHAT IT DRAWS, AND WHAT IT DOES NOT

Metadata only, per the decision in docs/webserver-and-incidents.md: boxes are
drawn client-side on a schematic canvas, not over video. Video is deliberately
out of scope - `ctx.annotated` already holds the drawn frame, so an MJPEG
endpoint is a small addition later, it just is not on the critical path.

The health tiles are the point as much as the boxes are. fps, write-queue
depth, dropped frames and writer failures are how a silently degrading feed
becomes visible, and a camera that is quietly dead is the failure mode this
whole layer is built against.
"""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>traffic-detection</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --line:#252a35; --ink:#e6e9ef;
    --dim:#8b93a7; --ok:#3fb950; --warn:#d29922; --bad:#f85149;
    --accent:#58a6ff;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { padding:14px 18px; border-bottom:1px solid var(--line);
           display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.02em; }
  .muted { color:var(--dim); font-size:12px; }
  main { padding:18px; display:grid; gap:18px;
         grid-template-columns:minmax(0,2fr) minmax(280px,1fr); }
  @media (max-width:900px){ main { grid-template-columns:1fr; } }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:8px; overflow:hidden; }
  .panel > h2 { font-size:12px; margin:0; padding:9px 12px; color:var(--dim);
                border-bottom:1px solid var(--line); font-weight:600;
                text-transform:uppercase; letter-spacing:.08em; }
  .body { padding:12px; }
  canvas { width:100%; height:auto; display:block; background:#0a0c10; }
  select, button { background:#1e222b; color:var(--ink);
                   border:1px solid var(--line); border-radius:6px;
                   padding:5px 10px; font:inherit; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th, td { text-align:left; padding:4px 6px; border-bottom:1px solid var(--line);
           white-space:nowrap; }
  th { color:var(--dim); font-weight:600; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(96px,1fr));
           gap:8px; }
  .tile { background:#12151c; border:1px solid var(--line); border-radius:6px;
          padding:8px 10px; }
  .tile .k { color:var(--dim); font-size:10px; text-transform:uppercase;
             letter-spacing:.07em; }
  .tile .v { font-size:17px; margin-top:2px; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
         margin-right:6px; vertical-align:middle; }
  .s-running { background:var(--ok); } .s-idle,.s-stopped { background:var(--dim); }
  .s-retrying,.s-starting,.s-stopping { background:var(--warn); }
  .s-failed { background:var(--bad); }
  .bad { color:var(--bad); } .warn { color:var(--warn); } .ok { color:var(--ok); }
  .empty { color:var(--dim); padding:8px 2px; font-size:12px; }
  code { color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>traffic-detection</h1>
  <span class="muted" id="conn">connecting…</span>
  <span class="muted" id="uptime"></span>
  <span style="flex:1"></span>
  <select id="pick" title="camera"></select>
  <button id="toggle">start / stop</button>
</header>

<main>
  <div>
    <div class="panel">
      <h2>live &mdash; metadata drawn client-side, not video</h2>
      <canvas id="view" width="1280" height="720"></canvas>
    </div>
    <div class="panel" style="margin-top:18px">
      <h2>health</h2>
      <div class="body"><div class="tiles" id="tiles"></div></div>
    </div>
    <div class="panel" style="margin-top:18px">
      <h2>cameras</h2>
      <div class="body" style="overflow-x:auto"><div id="cams"></div></div>
    </div>
  </div>

  <div>
    <div class="panel">
      <h2>incidents</h2>
      <div class="body" style="overflow-x:auto"><div id="incidents"></div></div>
    </div>
    <div class="panel" style="margin-top:18px">
      <h2>webhook deliveries</h2>
      <div class="body" style="overflow-x:auto"><div id="deliveries"></div></div>
    </div>
    <div class="panel" style="margin-top:18px">
      <h2>disk &mdash; pinned crops are a floor the budget cannot reclaim</h2>
      <div class="body"><div class="tiles" id="disk"></div></div>
    </div>
  </div>
</main>

<script>
const $ = s => document.querySelector(s);
let camera = null, ws = null, frame = null, cams = [];

const esc = s => String(s ?? "").replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));

function tile(k, v, cls) {
  return `<div class="tile"><div class="k">${esc(k)}</div>
          <div class="v ${cls||''}">${esc(v)}</div></div>`;
}

// ---- canvas -------------------------------------------------------------
// Boxes arrive in SOURCE pixel coordinates, so the canvas is sized to the
// camera's real resolution and CSS scales it. Drawing in source space means
// no coordinate maths here and none in the pipeline either.
const COLOR = { wrong_way:"#f85149", wrong_lane:"#d29922" };
function draw() {
  const c = $("#view"), g = c.getContext("2d");
  g.fillStyle = "#0a0c10"; g.fillRect(0, 0, c.width, c.height);
  if (!frame) {
    g.fillStyle = "#8b93a7"; g.font = "16px ui-monospace";
    g.fillText("waiting for frames…", 18, 30);
    return;
  }
  g.strokeStyle = "#252a35"; g.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = c.height * i / 4;
    g.beginPath(); g.moveTo(0, y); g.lineTo(c.width, y); g.stroke();
  }
  for (const b of frame.boxes || []) {
    const [x1, y1, x2, y2] = b.xyxy;
    const col = COLOR[b.lane_flag] || (b.group === "vehicle" ? "#3fb950" : "#58a6ff");
    g.strokeStyle = col; g.lineWidth = 2;
    g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const id = b.vehicle_id != null ? "V" + b.vehicle_id : "#" + b.track_id;
    let label = `${id} ${b.cls}`;
    if (b.plate) label += ` ${b.plate}`;
    if (b.lane_flag && b.lane_flag !== "ok") label += ` !${b.lane_flag}`;
    g.font = "13px ui-monospace";
    const w = g.measureText(label).width + 8;
    g.fillStyle = "rgba(10,12,16,.85)";
    g.fillRect(x1, Math.max(0, y1 - 17), w, 16);
    g.fillStyle = col; g.fillText(label, x1 + 4, Math.max(11, y1 - 5));
  }
  g.fillStyle = "#8b93a7"; g.font = "12px ui-monospace";
  g.fillText(`frame ${frame.frame_no}  ·  ${(frame.boxes||[]).length} objects`,
             10, c.height - 10);
}

// ---- websocket ----------------------------------------------------------
function connect(id) {
  if (ws) { ws.onclose = null; ws.close(); }
  frame = null; draw();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/live/${encodeURIComponent(id)}`);
  ws.onopen  = () => $("#conn").textContent = `live: ${id}`;
  ws.onclose = () => {
    $("#conn").textContent = "disconnected — retrying";
    // The camera may simply have been stopped; reconnect rather than going
    // dark, so restarting it from this page brings the view back by itself.
    setTimeout(() => { if (camera === id) connect(id); }, 2000);
  };
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.error) { $("#conn").textContent = m.error; return; }
    frame = m;
    const c = $("#view");
    const cam = cams.find(x => x.camera === id);
    if (cam && cam.resolution) {
      const [w, h] = cam.resolution.split("x").map(Number);
      if (w && h && (c.width !== w || c.height !== h)) { c.width = w; c.height = h; }
    }
    draw(); renderTiles(m);
  };
}

function renderTiles(m) {
  const h = m.health || {}, cam = cams.find(x => x.camera === m.camera) || {};
  $("#tiles").innerHTML = [
    tile("fps", cam.fps ?? "–"),
    tile("tracked", h.tracked ?? "–"),
    tile("queue depth", h.queue_depth ?? "–",
         (h.queue_depth > 5000) ? "warn" : ""),
    tile("rows shed", h.rows_dropped ?? 0, h.rows_dropped ? "warn" : ""),
    tile("rows lost", h.rows_failed ?? 0, h.rows_failed ? "bad" : ""),
    tile("frames dropped", h.frames_dropped ?? 0),
    tile("wrong way", (m.flagged||{}).wrong_way ?? 0,
         (m.flagged||{}).wrong_way ? "bad" : ""),
    tile("wrong lane", (m.flagged||{}).wrong_lane ?? 0,
         (m.flagged||{}).wrong_lane ? "warn" : ""),
    tile("congestion", m.congestion ?? "–",
         m.congestion === "congested" ? "bad" : "ok"),
    tile("viewers", m.viewers ?? 0),
    tile("state size", h.state_size ?? "–"),
    tile("writer", h.write_alarm ? "ALARM" : "ok", h.write_alarm ? "bad" : "ok"),
  ].join("");
}

// ---- polling ------------------------------------------------------------
async function get(p) {
  const r = await fetch(p);
  if (!r.ok) throw new Error(p + " -> " + r.status);
  return r.json();
}

function table(rows, cols) {
  if (!rows.length) return '<div class="empty">nothing yet</div>';
  const head = cols.map(c => `<th>${esc(c[0])}</th>`).join("");
  const body = rows.map(r =>
    "<tr>" + cols.map(c => `<td>${c[1](r)}</td>`).join("") + "</tr>").join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

const clock = t => t ? new Date(t * 1000).toLocaleTimeString() : "–";

async function poll() {
  try {
    const [c, h, inc, del] = await Promise.all([
      get("/cameras"), get("/health"),
      get("/incidents?limit=12"), get("/deliveries?limit=12")]);
    cams = c.cameras;
    if (!camera && cams.length) { camera = cams[0].camera; connect(camera); }

    const pick = $("#pick");
    if (pick.options.length !== cams.length) {
      pick.innerHTML = cams.map(x =>
        `<option value="${esc(x.camera)}">${esc(x.camera)}</option>`).join("");
      pick.value = camera;
    }
    $("#uptime").textContent = `up ${Math.round(h.uptime_s)}s · `
      + `${h.cameras} camera(s)`
      + (h.unhealthy.length ? ` · unhealthy: ${h.unhealthy.join(", ")}` : "");

    $("#cams").innerHTML = table(cams, [
      ["camera", r => `<span class="dot s-${esc(r.state)}"></span>${esc(r.camera)}`],
      ["state",  r => esc(r.state) + (r.error ? ` <span class="bad">${esc(r.error)}</span>` : "")],
      ["fps",    r => esc(r.fps)],
      ["frames", r => esc(r.frames)],
      ["last",   r => r.seconds_since_frame == null ? "–" :
                      `${r.seconds_since_frame}s ago`],
      ["restarts", r => esc(r.restarts)],
      ["viewers", r => esc(r.viewers)],
    ]);

    $("#incidents").innerHTML = table(inc.incidents, [
      ["at",     r => clock(r.created_at)],
      ["kind",   r => `<span class="${r.kind==='wrong_way'?'bad':'warn'}">${esc(r.kind)}</span>`],
      ["camera", r => esc(r.camera)],
      ["plate",  r => esc((r.payload?.vehicle?.plate) ?? "–")],
      ["img",    r => r.object_id != null
                      ? `<a href="/crops/${r.object_id}.jpg" target="_blank">view</a>` : "–"],
      ["sent",   r => `${r.sent}/${r.sent + r.pending + r.dead}`
                      + (r.dead ? ` <span class="bad">${r.dead} dead</span>` : "")],
    ]);

    const dsum = del.summary || {};
    $("#deliveries").innerHTML =
      `<div class="muted" style="margin-bottom:6px">`
      + `sent ${dsum.sent||0} · pending ${dsum.pending||0} · `
      + `<span class="${dsum.dead?'bad':''}">dead ${dsum.dead||0}</span></div>`
      + table(del.deliveries, [
        ["endpoint", r => esc(String(r.endpoint).replace(/^https?:\/\//, ""))],
        ["status",   r => `<span class="${r.status==='dead'?'bad':r.status==='sent'?'ok':'warn'}">${esc(r.status)}</span>`],
        ["tries",    r => esc(r.attempts)],
        ["error",    r => esc((r.last_error||"").slice(0, 40))],
      ]);

    const crops = h.retention?.crops || {};
    $("#disk").innerHTML = [
      tile("pinned crops", crops.pinned_crops ?? 0),
      tile("pinned MB", crops.pinned_mb ?? 0, "warn"),
      tile("reclaimable", crops.reclaimable_crops ?? 0),
      tile("reclaimable MB", crops.reclaimable_mb ?? 0),
      tile("reaped MB", crops.deleted_mb ?? 0),
      tile("retention", crops.enabled ? "on" : "off",
           crops.enabled ? "ok" : ""),
    ].join("");
  } catch (e) {
    $("#conn").textContent = "server unreachable";
  }
}

$("#pick").onchange = e => { camera = e.target.value; connect(camera); };
$("#toggle").onclick = async () => {
  const cam = cams.find(x => x.camera === camera);
  const action = (cam && cam.state === "running") ? "stop" : "start";
  await fetch(`/cameras/${encodeURIComponent(camera)}/${action}`, {method:"POST"});
  poll();
};

draw(); poll(); setInterval(poll, 2000);
</script>
</body>
</html>
"""

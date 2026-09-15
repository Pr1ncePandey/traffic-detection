"""Dummy webhook consumer: see exactly what the incident API sends out.

    python tools/dummy_receiver.py                 listen on 127.0.0.1:9000
    python tools/dummy_receiver.py --reply 500     answer every POST with 500,
                                                   to watch the retry ladder
    python tools/dummy_receiver.py --write-config  write a serve.py config that
                                                   points its webhooks here

Stands in for the "ops system" on the other end of a subscription. Open
http://127.0.0.1:9000/ and every POST it receives is listed live: the headers,
the parsed payload, whether the signature verifies, how old the timestamp is,
and whether the incident_id is a repeat (delivery is at-least-once).

TWO WAYS TO FEED IT

1. "Send sample incidents" on the page. Payloads are built by the REAL
   src/incidents.py policy from synthetic pipeline events, then serialised,
   signed and headed exactly as src/server/webhooks.py does. No video, no
   YOLO, no fastapi - seconds, and every incident kind shows up.

2. The real service, end to end:
       python tools/dummy_receiver.py --write-config
       set TRAFFIC_HOOK_SECRET=dummy-local-secret      (PowerShell: $env:...)
       python serve.py --config outputs/_receiver_config.yaml --camera indian_road
   config.yaml is not touched; the generated file lives under outputs/.

Stdlib only for the server itself, so it runs on any interpreter.
"""

import argparse
import hashlib
import hmac
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from src.incidents import IncidentPolicy, serialise, sign  # noqa: E402

HOOK_PATH = "/hooks/traffic"
SECRET_ENV = "TRAFFIC_HOOK_SECRET"
DEFAULT_SECRET = "dummy-local-secret"
TIMESTAMP_TOLERANCE_S = 300

# Keys the payload contract promises. Present even when null.
ALWAYS = ("incident_id", "kind", "state", "camera", "detected_at", "detail")
PER_VEHICLE = ("vehicle", "sighting", "image_url")

RECEIVED = []            # newest last
SEEN_IDS = {}            # incident_id -> times received
LOCK = threading.Lock()
ARGS = None


# --- what a careful consumer checks --------------------------------------
def inspect(headers: dict, raw: bytes) -> dict:
    """Everything a real consumer should verify, reported instead of enforced."""
    notes = []
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        return {"payload": None, "raw": raw.decode("utf-8", "replace"),
                "signature": "n/a", "notes": [f"body is not JSON: {e}"],
                "missing": [], "age_s": None, "duplicate": False}

    stamp = headers.get("x-timestamp", "")
    given = headers.get("x-signature", "")
    if not given:
        signature = "unsigned"
        notes.append("no X-Signature: the subscription has no secret_env set, "
                     "or that env var was empty when serve.py started")
    else:
        expected = sign(raw, ARGS.secret, stamp)
        signature = "valid" if hmac.compare_digest(given, expected) else "INVALID"
        if signature == "INVALID":
            notes.append("signature does not match: different secret, or the "
                         "body/timestamp was altered in transit")

    age = None
    try:
        age = round(time.time() - int(stamp), 1)
        if abs(age) > TIMESTAMP_TOLERANCE_S:
            notes.append(f"timestamp is {age:.0f}s old; a real consumer rejects "
                         f"this as a possible replay")
    except ValueError:
        notes.append("missing or non-numeric X-Timestamp")

    missing = [k for k in ALWAYS if k not in payload]
    if payload.get("kind") != "congestion":
        missing += [k for k in PER_VEHICLE if k not in payload]
    if headers.get("x-incident-id") not in (None, payload.get("incident_id")):
        notes.append("X-Incident-Id header disagrees with the body")

    iid = payload.get("incident_id")
    with LOCK:
        SEEN_IDS[iid] = SEEN_IDS.get(iid, 0) + 1
        duplicate = SEEN_IDS[iid] > 1
    if duplicate:
        notes.append("repeat incident_id: a retry. Process it once, ack it again")

    return {"payload": payload, "raw": raw.decode("utf-8"),
            "signature": signature, "notes": notes, "missing": missing,
            "age_s": age, "duplicate": duplicate}


# --- synthetic incidents, built by the real policy -----------------------
def build_samples() -> list:
    """Drive IncidentPolicy with the events the pipeline emits, collect payloads.

    Event shapes mirror src/analysis/lanes.py (verdict.detail() + cls),
    src/pipeline.py (track_retired) and src/analysis/congestion.py.
    """
    policy = IncidentPolicy(
        {"incidents": {"base_url": "http://127.0.0.1:8000",
                       "kinds": {"wrong_way": True, "wrong_lane": True,
                                 "congestion": True}}},
        store=None,
        cameras={"indian_road": {"name": "Indian road (sample)",
                                 "lat": 28.6139, "lon": 77.2090}})
    ctx = {"camera": "indian_road"}
    out = []

    def verdict(flag):
        return {"lane": "carriageway", "flag": flag, "alignment": -0.912,
                "travel_px": 184.3, "heading": [-0.251, -0.968], "cls": "car"}

    def retire(tid, oid, flag, ts, plate=None, conf=0.0, vid=None, colour=None):
        return {"kind": "track_retired", "track_id": tid, "ts": ts, "detail": {
            "object_id": oid, "identity_id": vid, "plate": plate,
            "plate_conf": conf, "cls_name": "car", "colour": colour,
            "first_seen_s": round(ts - 6.4, 2), "last_seen_s": ts, "frames_seen": 160,
            "crop_path": f"outputs/crops/{oid}.jpg", "lane_id": "carriageway",
            "lane_flag": flag}}

    # 1. Wrong way, plate read confidently -> closed, fully populated.
    policy.handle({"kind": "wrong_way", "track_id": 11, "ts": 12.0,
                   "detail": verdict("wrong_way")}, ctx)
    out += policy.handle(retire(11, 4101, "wrong_way", 18.4, "DL3CAF4521",
                                0.93, 57, "white"), ctx)

    # 2. Wrong way + wrong lane, plate never read -> two incidents, null plate.
    both = "wrong_way+wrong_lane"
    policy.handle({"kind": "wrong_way", "track_id": 12, "ts": 30.0,
                   "detail": verdict(both)}, ctx)
    policy.handle({"kind": "wrong_lane", "track_id": 12, "ts": 30.0,
                   "detail": verdict(both)}, ctx)
    out += policy.handle(retire(12, 4102, both, 36.9), ctx)

    # 3. Stuck track: flagged for longer than max_dwell_s -> fires "ongoing".
    policy.handle({"kind": "wrong_way", "track_id": 13, "ts": 40.0,
                   "detail": verdict("wrong_way")}, {**ctx, "object_id": 4103})
    out += policy.handle({"kind": "wrong_way", "track_id": 13, "ts": 165.0,
                          "detail": verdict("wrong_way")},
                         {**ctx, "object_id": 4103})

    # 4. Congestion onset, then clearing. The policy needs the level observed
    #    twice, congestion_dwell_s apart, before it publishes an edge.
    jam = {"level": "jammed", "count": 23, "occupancy": 0.412, "motion_px": 0.8}
    free = {"level": "free", "count": 3, "occupancy": 0.041, "motion_px": 9.6}
    for ts, detail in ((200.0, jam), (231.0, jam), (300.0, free), (331.0, free)):
        out += policy.handle({"kind": "congestion", "ts": ts, "detail": detail},
                             ctx)
    return [i["payload"] for i in out]


def post(payload: dict, secret: str | None, tamper=False) -> int:
    """POST one payload to ourselves exactly as WebhookDispatcher._attempt does."""
    body = serialise(payload)
    stamp = str(int(time.time()))
    headers = {"content-type": "application/json",
               "x-incident-id": str(payload["incident_id"]),
               "x-incident-kind": str(payload["kind"]),
               "x-timestamp": stamp,
               "user-agent": "traffic-detection/incidents"}
    if secret:
        headers["x-signature"] = sign(body, secret, stamp)
    if tamper:
        body = body.replace(b'"state":"closed"', b'"state":"cancelled"')
    req = urllib.request.Request(
        f"http://{ARGS.host}:{ARGS.port}{HOOK_PATH}", data=body,
        headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def send_samples() -> dict:
    payloads = build_samples()
    for p in payloads:
        post(p, ARGS.secret)
    # The three things a consumer must cope with, demonstrated on purpose.
    post(payloads[0], ARGS.secret)                 # a retry: same incident_id
    post(payloads[0], ARGS.secret, tamper=True)    # altered after signing
    post(payloads[-1], None)                       # unsigned
    return {"sent": len(payloads) + 3}


# --- HTTP -----------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, status, body: bytes, ctype="application/json"):
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj, default=str).encode("utf-8"))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path.startswith("/api/received"):
            with LOCK:
                self._json({"reply": ARGS.reply, "hook": HOOK_PATH,
                            "secret_source": ARGS.secret_source,
                            "items": list(reversed(RECEIVED))})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.path == "/api/samples":
            return self._json(send_samples())
        if self.path == "/api/clear":
            with LOCK:
                RECEIVED.clear()
                SEEN_IDS.clear()
            return self._json({"cleared": True})

        headers = {k.lower(): v for k, v in self.headers.items()}
        entry = {"n": 0, "received_at": time.time(), "path": self.path,
                 "status": ARGS.reply,
                 "headers": {k: v for k, v in headers.items()
                             if k.startswith("x-") or k in
                             ("content-type", "user-agent")},
                 **inspect(headers, raw)}
        with LOCK:
            entry["n"] = len(RECEIVED) + 1
            RECEIVED.append(entry)
            del RECEIVED[:-500]                      # bounded, like everything else
        p = entry["payload"] or {}
        print(f"[receiver] #{entry['n']} {p.get('kind', '?'):<11} "
              f"{p.get('state', ''):<9} sig={entry['signature']:<8} "
              f"{'DUP ' if entry['duplicate'] else ''}-> {ARGS.reply}")
        self._json({"ok": ARGS.reply < 300}, ARGS.reply)


def write_config(path: str):
    import yaml
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["incidents"] = {
        "enabled": True,
        "base_url": "http://127.0.0.1:8000",
        "kinds": {"wrong_way": True, "congestion": True, "wrong_lane": True},
        "subscriptions": [{"endpoint": f"http://127.0.0.1:{ARGS.port}{HOOK_PATH}",
                           "secret_env": SECRET_ENV, "kinds": [], "cameras": []}],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# GENERATED by tools/dummy_receiver.py --write-config.\n"
                "# config.yaml plus an incidents block aimed at the dummy "
                "receiver. Safe to delete.\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {path}\n\nthen, in a second terminal:\n"
          f"  PowerShell:  $env:{SECRET_ENV}=\"{ARGS.secret}\"\n"
          f"  cmd:         set {SECRET_ENV}={ARGS.secret}\n"
          f"  python serve.py --config {path} --camera indian_road")


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--reply", type=int, default=200,
                    help="HTTP status to answer webhooks with (try 500 or 400)")
    ap.add_argument("--secret", default=None,
                    help=f"HMAC secret to verify with (default: ${SECRET_ENV}, "
                         f"else {DEFAULT_SECRET!r})")
    ap.add_argument("--write-config", nargs="?", metavar="PATH",
                    const=os.path.join("outputs", "_receiver_config.yaml"),
                    help="write a serve.py config pointing here, then exit")
    ARGS = ap.parse_args()
    if ARGS.secret:
        ARGS.secret_source = "--secret"
    elif os.environ.get(SECRET_ENV):
        ARGS.secret, ARGS.secret_source = os.environ[SECRET_ENV], f"${SECRET_ENV}"
    else:
        ARGS.secret, ARGS.secret_source = DEFAULT_SECRET, "built-in default"

    if ARGS.write_config:
        return write_config(ARGS.write_config)

    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"dummy receiver on http://{ARGS.host}:{ARGS.port}/\n"
          f"  webhooks: POST http://{ARGS.host}:{ARGS.port}{HOOK_PATH}\n"
          f"  replying: {ARGS.reply}   secret from: {ARGS.secret_source}\n"
          f"  Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Webhook receiver</title>
<style>
:root{--bg:#f3f5f6;--panel:#fff;--ink:#14191b;--muted:#56626a;--rule:#d5dbde;
--ok:#1d6b3a;--bad:#b0330f;--warn:#8a5a00;--code:#eef1f2}
@media (prefers-color-scheme:dark){:root{--bg:#111517;--panel:#1a2023;--ink:#e6eaec;
--muted:#9aa6ad;--rule:#2c3438;--ok:#5fc787;--bad:#ff8a66;--warn:#e7b44d;--code:#12171a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--rule);
padding:14px 20px;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;z-index:1}
h1{font-size:17px;margin:0;font-weight:600}
.meta{color:var(--muted);font-size:13px}
button{font:inherit;padding:7px 14px;border-radius:6px;border:1px solid var(--rule);
background:var(--panel);color:var(--ink);cursor:pointer}
button.primary{background:var(--ink);color:var(--bg);border-color:var(--ink)}
main{max-width:1000px;margin:0 auto;padding:18px 20px 60px}
.empty{color:var(--muted);padding:40px 0;text-align:center}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:8px;
margin:0 0 14px;padding:14px 16px}
.top{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}
.kind{font-weight:600;font-size:16px}
.tag{font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid currentColor}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.mut{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:4px 20px;
margin:10px 0 0}
.kv{font-size:14px;overflow-wrap:anywhere}.kv b{color:var(--muted);font-weight:500}
.null{color:var(--muted);font-style:italic}
ul.notes{margin:10px 0 0;padding-left:18px;font-size:14px}
details{margin-top:10px}summary{cursor:pointer;color:var(--muted);font-size:13px}
pre{background:var(--code);padding:10px;border-radius:6px;overflow-x:auto;
font:12.5px/1.45 ui-monospace,Consolas,monospace;margin:6px 0 0}
</style></head><body>
<header>
  <h1>Webhook receiver</h1>
  <span class="meta" id="meta">connecting...</span>
  <span style="flex:1"></span>
  <button class="primary" id="sample">Send sample incidents</button>
  <button id="clear">Clear</button>
</header>
<main id="list"><p class="empty">Nothing received yet.</p></main>
<script>
const $ = s => document.querySelector(s);
const esc = v => String(v).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const val = v => v === null || v === undefined ? '<span class="null">null</span>'
  : typeof v === "object" ? esc(JSON.stringify(v)) : esc(v);
const kv = (k, v) => `<div class="kv"><b>${esc(k)}</b> ${val(v)}</div>`;
let lastKey = "";

function card(it) {
  const p = it.payload || {};
  const sig = {valid: "ok", INVALID: "bad", unsigned: "warn"}[it.signature] || "mut";
  const when = new Date(it.received_at * 1000).toLocaleTimeString();
  const rows = [kv("incident_id", p.incident_id), kv("state", p.state),
    kv("camera", p.camera && `${p.camera.name} (${p.camera.id})`),
    kv("location", p.camera && (p.camera.lat == null ? null : `${p.camera.lat}, ${p.camera.lon}`)),
    kv("detected_at (s into run)", p.detected_at)];
  if ("dwell_s" in p) rows.push(kv("dwell_s", p.dwell_s));
  if (p.vehicle) for (const [k, v] of Object.entries(p.vehicle)) rows.push(kv("vehicle." + k, v));
  if (p.sighting) for (const [k, v] of Object.entries(p.sighting)) rows.push(kv("sighting." + k, v));
  if ("image_url" in p) rows.push(kv("image_url", p.image_url));
  for (const [k, v] of Object.entries(p.detail || {})) rows.push(kv("detail." + k, v));
  const notes = [...(it.missing.length ? [`missing keys: ${it.missing.join(", ")}`] : []), ...it.notes];
  return `<article class="card">
    <div class="top"><span class="mut">#${it.n}</span>
      <span class="kind">${esc(p.kind || "unparseable body")}</span>
      <span class="tag ${sig}">signature ${esc(it.signature)}</span>
      ${it.duplicate ? '<span class="tag warn">duplicate</span>' : ""}
      <span class="tag ${it.status < 300 ? "ok" : "bad"}">replied ${it.status}</span>
      <span class="mut" style="margin-left:auto;font-size:13px">${when}${it.age_s == null ? "" : ` · timestamp age ${it.age_s}s`}</span></div>
    ${it.payload ? `<div class="grid">${rows.join("")}</div>` : ""}
    ${notes.length ? `<ul class="notes">${notes.map(n => `<li class="warn">${esc(n)}</li>`).join("")}</ul>` : ""}
    <details><summary>headers and raw body</summary>
      <pre>${esc(JSON.stringify(it.headers, null, 2))}</pre>
      <pre>${esc(it.payload ? JSON.stringify(it.payload, null, 2) : it.raw)}</pre></details>
  </article>`;
}

async function refresh() {
  try {
    const d = await (await fetch("/api/received")).json();
    $("#meta").textContent = `POST ${location.host}${d.hook} · replying ${d.reply} · secret from ${d.secret_source} · ${d.items.length} received`;
    const key = d.items.length + ":" + (d.items[0] && d.items[0].n);
    if (key === lastKey) return;
    lastKey = key;
    $("#list").innerHTML = d.items.length ? d.items.map(card).join("")
      : '<p class="empty">Nothing received yet. Press "Send sample incidents", or point serve.py here.</p>';
  } catch (e) { $("#meta").textContent = "receiver not reachable"; }
}
$("#sample").onclick = async () => { await fetch("/api/samples", {method: "POST"}); refresh(); };
$("#clear").onclick = async () => { await fetch("/api/clear", {method: "POST"}); lastKey = ""; refresh(); };
refresh(); setInterval(refresh, 1500);
</script></body></html>
"""

if __name__ == "__main__":
    main()

// Shape panel: every frame's points of the selected track overlaid in the BOX frame (the aggregate
// view). A crisp blob means the per-frame boxes are right; a smear means a frame is off. Drag a face
// to set the shared shape (the opposite face stays put), drag the body to shift the box on every
// frame. The current frame's points are drawn darker.

import { api } from "../api.js";
import * as ops from "../ops.js";
import { S, on, preview, commitDrag, reportError } from "../store.js";
import { el, icon } from "../ui.js";

const VIEWS = {               // which box-frame axes each tab shows (horizontal, vertical)
  top: { h: 0, v: 1, dims: ["L", "W"], label: "Top" },
  side: { h: 0, v: 2, dims: ["L", "H"], label: "Side" },
  front: { h: 1, v: 2, dims: ["W", "H"], label: "Front" },
};
const DIM_I = { L: 0, W: 1, H: 2 };

export function createShapePanel(host) {
  const seg = el("div.seg");
  const head = el("div.panel-h", {}, el("span.title", {}, "Shape"), el("span.grow"), seg);
  const canvas = el("canvas");
  const body = el("div.panel-b", {}, el("div.ortho", {}, canvas));
  const empty = el("div.empty", {}, icon("cube"), "Select a track to shape it.", el("br"),
    el("span", {}, "All frames are overlaid in the box frame."));
  host.append(head, body);

  let agg = null, aggTid = null, aggSig = "", loading = false, drag = null, scale = 1, cx = 0, cy = 0;
  const sig = (t) => (t ? JSON.stringify([t.shape, t.poses]) : "");

  function renderSeg() {
    seg.replaceChildren(...Object.entries(VIEWS).map(([k, v]) =>
      el("button" + (S.view.shape === k ? ".on" : ""), { onclick: () => { S.view.shape = k; renderSeg(); draw(); } }, v.label)));
  }

  async function load(force = false) {
    if (!S.track) { agg = null; aggTid = null; draw(); return; }
    if (!force && aggTid === S.tid && agg) { draw(); return; }
    loading = true; draw();
    const tid = S.tid;
    try {
      const a = await api.aggregate(tid);
      if (tid !== S.tid) return;
      agg = a; aggTid = tid; aggSig = sig(S.track);
    } catch (e) { reportError(e); }
    loading = false;
    draw();
  }

  function draw() {
    const box = body.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio, 2);
    canvas.width = Math.max(1, box.width * dpr); canvas.height = Math.max(1, box.height * dpr);
    canvas.style.width = box.width + "px"; canvas.style.height = box.height + "px";
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, box.width, box.height);
    if (!S.track) { if (!empty.isConnected) body.firstChild.append(empty); canvas.style.display = "none"; return; }
    empty.remove(); canvas.style.display = "";
    const v = VIEWS[S.view.shape];
    const [L, W, H] = S.track.shape;
    const ext = [L, W, H];
    const hw = ext[v.h] / 2, hv = ext[v.v] / 2;
    const pad = 1.4;
    scale = Math.min((box.width - 40) / (2 * (hw + pad)), (box.height - 40) / (2 * (hv + pad)));
    cx = box.width / 2; cy = box.height / 2;
    const X = (a) => cx + a * scale, Y = (b) => cy - b * scale;

    // grid: 1 m ticks, very faint
    g.strokeStyle = "#efeeea"; g.lineWidth = 1;
    for (let m = -Math.ceil(hw + pad); m <= hw + pad; m++) { g.beginPath(); g.moveTo(X(m), 0); g.lineTo(X(m), box.height); g.stroke(); }
    for (let m = -Math.ceil(hv + pad); m <= hv + pad; m++) { g.beginPath(); g.moveTo(0, Y(m)); g.lineTo(box.width, Y(m)); g.stroke(); }

    // points: other frames light, the current frame dark
    if (agg) {
      const fr = ops.framesOf(S.track), cur = fr.indexOf(S.frame);
      const { n, xyz, fidx } = agg;
      g.fillStyle = "rgba(110, 118, 132, 0.28)";
      for (let i = 0; i < n; i++) if (fidx[i] !== cur) g.fillRect(X(xyz[3 * i + v.h]) - 0.7, Y(xyz[3 * i + v.v]) - 0.7, 1.4, 1.4);
      g.fillStyle = "#1a3fa8";
      for (let i = 0; i < n; i++) if (fidx[i] === cur) g.fillRect(X(xyz[3 * i + v.h]) - 1.2, Y(xyz[3 * i + v.v]) - 1.2, 2.4, 2.4);
    }

    // the box
    g.strokeStyle = "#2563eb"; g.lineWidth = 2;
    g.fillStyle = "rgba(37, 99, 235, 0.05)";
    g.fillRect(X(-hw), Y(hv), 2 * hw * scale, 2 * hv * scale);
    g.strokeRect(X(-hw), Y(hv), 2 * hw * scale, 2 * hv * scale);
    if (v.h === 0) {                        // heading arrow on views that show the length axis
      g.beginPath(); g.moveTo(X(hw - 0.5), Y(0)); g.lineTo(X(hw), Y(0)); g.stroke();
    }
    // face handles
    for (const [hx, hy] of [[hw, 0], [-hw, 0], [0, hv], [0, -hv]]) {
      g.fillStyle = "#fff"; g.strokeStyle = "#2563eb"; g.lineWidth = 1.5;
      g.fillRect(X(hx) - 4.5, Y(hy) - 4.5, 9, 9); g.strokeRect(X(hx) - 4.5, Y(hy) - 4.5, 9, 9);
    }
    // dimension labels
    g.font = "12px ui-sans-serif, system-ui, sans-serif"; g.fillStyle = "#5d5c58"; g.textAlign = "center";
    g.fillText(`${v.dims[0]} ${ext[v.h].toFixed(2)} m`, X(0), Y(-hv) + 18);
    g.save(); g.translate(X(hw) + 16, Y(0)); g.rotate(-Math.PI / 2);
    g.fillText(`${v.dims[1]} ${ext[v.v].toFixed(2)} m`, 0, 0); g.restore();
    canvas.dataset.tid = aggTid ?? ""; canvas.dataset.loading = loading ? "1" : "0";
    if (loading) { g.fillStyle = "#8f8d87"; g.textAlign = "left"; g.fillText("Loading points…", 12, 20); }
    else if (agg && agg.n === 0) { g.fillStyle = "#8f8d87"; g.textAlign = "left"; g.fillText("No LiDAR points near this vehicle", 12, 20); }
  }

  // ------------------------------------------------------------ dragging
  function hit(e) {
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const v = VIEWS[S.view.shape];
    const ext = S.track.shape;
    const hw = ext[DIM_I[v.dims[0]]] / 2, hv = ext[DIM_I[v.dims[1]]] / 2;
    const a = (px - cx) / scale, b = -(py - cy) / scale;
    const tol = 8 / scale;
    if (Math.abs(a - hw) < tol && Math.abs(b) < hv) return { kind: "h+", a, b };
    if (Math.abs(a + hw) < tol && Math.abs(b) < hv) return { kind: "h-", a, b };
    if (Math.abs(b - hv) < tol && Math.abs(a) < hw) return { kind: "v+", a, b };
    if (Math.abs(b + hv) < tol && Math.abs(a) < hw) return { kind: "v-", a, b };
    if (Math.abs(a) < hw && Math.abs(b) < hv) return { kind: "body", a, b };
    return null;
  }

  canvas.addEventListener("pointermove", (e) => {
    if (!S.track) return;
    if (!drag) {
      const h = hit(e);
      canvas.style.cursor = !h ? "" : h.kind === "body" ? "move" : h.kind[0] === "h" ? "ew-resize" : "ns-resize";
      return;
    }
    const r = canvas.getBoundingClientRect();
    const a = (e.clientX - r.left - cx) / scale, b = -(e.clientY - r.top - cy) / scale;
    const v = VIEWS[S.view.shape];
    const t0 = drag.before;
    const [hd, vd] = v.dims;
    const e0 = t0.shape;
    let t;
    if (drag.kind === "body") {
      const da = a - drag.a, db = b - drag.b;
      const d = [0, 0, 0]; d[v.h] = da; d[v.v] = db;
      t = ops.shiftLocal(t0, d[0], d[1], d[2]);
    } else if (drag.kind === "h+") t = ops.resize(t0, hd, a + e0[DIM_I[hd]] / 2, +1);
    else if (drag.kind === "h-") t = ops.resize(t0, hd, e0[DIM_I[hd]] / 2 - a, -1);
    else if (drag.kind === "v+") t = ops.resize(t0, vd, b + e0[DIM_I[vd]] / 2, +1);
    else t = ops.resize(t0, vd, e0[DIM_I[vd]] / 2 - b, -1);
    preview(t);
  });

  canvas.addEventListener("pointerdown", (e) => {
    if (!S.track || e.button !== 0) return;
    const h = hit(e);
    if (!h) return;
    drag = { ...h, before: S.track };
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointerup", () => {
    if (!drag) return;
    const d = drag; drag = null;
    if (S.track !== d.before) commitDrag(d.kind === "body" ? "shift track" : "resize", d.before);
  });

  new ResizeObserver(draw).observe(body);
  renderSeg();
  let refetch = null;
  on("select", () => load());
  on("frame", draw);
  on("track", (info) => {
    draw();
    if ((!info || !info.preview) && sig(S.track) !== aggSig) {   // geometry changed: points move
      clearTimeout(refetch);
      refetch = setTimeout(() => load(true), 450);
    }
  });
  return { draw };
}

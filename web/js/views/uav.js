// UAV view: the drone's own image of the selected vehicle at this frame, with every box projected
// through the release's camera model (distortion included). Where the LiDAR sees one side of a car,
// the drone still sees its whole footprint.

import { api } from "../api.js";
import { S, on, reportError, selectTrack, finishPlacing } from "../store.js";
import { el, icon } from "../ui.js";

const ZOOM = { close: 1.5, medium: 2.4, wide: 4.2 };

export function createUavPanel(host) {
  const camSeg = el("div.seg");
  const zoomSeg = el("div.seg");
  const head = el("div.panel-h", {}, el("span.title", {}, "UAV view"), el("span.grow"), zoomSeg, camSeg);
  const canvas = el("canvas");
  const wrap = el("div.uav", {}, canvas);
  const body = el("div.panel-b", {}, wrap);
  const empty = el("div.empty");
  host.append(head, body);

  let zoom = "medium", seq = 0, view = null, img = null, fit = null;   // fit: window -> canvas mapping

  function renderSegs() {
    const uavs = S.project?.session.uav_agents || [];
    camSeg.replaceChildren(...uavs.map((u) => el("button" + (S.view.uav === u.id ? ".on" : ""),
      { onclick: () => { S.view.uav = u.id; renderSegs(); load(); } }, u.id === 3 ? "Hovering" : "Escort")));
    zoomSeg.replaceChildren(...Object.keys(ZOOM).map((z) => el("button" + (zoom === z ? ".on" : ""),
      { onclick: () => { zoom = z; renderSegs(); load(); } }, { close: "Close", medium: "Mid", wide: "Wide" }[z])));
  }

  function showEmpty(msg) {
    view = null; img = null;
    canvas.style.display = "none";
    empty.replaceChildren(icon("drone"), msg);
    if (!empty.isConnected) wrap.append(empty);
  }

  async function load() {
    if (!S.project) return;
    if (S.view.uav === null || S.view.uav === undefined) return showEmpty("This session has no UAV imagery.");
    const s = ++seq, f = S.frame, a = S.view.uav;
    const [cw, ch] = area();
    try {
      const v = await api.uavView(a, f, S.tid ?? undefined, ZOOM[zoom], cw / ch);
      if (s !== seq) return;
      if (!v.available) return showEmpty("No drone image for this frame.");
      const out = Math.round(Math.min(1800, Math.max(320, cw * Math.min(devicePixelRatio, 2))));
      const im = new Image();
      im.src = api.uavImageURL(a, f, v.window, out);
      await im.decode();
      if (s !== seq) return;
      view = { ...v, frame: f, tid: S.tid }; img = im;
      empty.remove();
      canvas.style.display = "";
      draw();
    } catch (e) { reportError(e); }
  }

  const area = () => [Math.max(64, wrap.clientWidth - 16), Math.max(64, wrap.clientHeight - 16)];

  function draw() {
    if (!view || !img) return;
    const [x0, y0, w, h] = view.window;
    const [aw, ah] = area();
    const k = Math.min(aw / w, ah / h);              // fit the window, keep its aspect
    const cw = Math.round(w * k), ch = Math.round(h * k);
    const dpr = Math.min(devicePixelRatio, 2);
    canvas.width = cw * dpr; canvas.height = ch * dpr;
    canvas.style.width = cw + "px"; canvas.style.height = ch + "px";
    canvas.style.borderRadius = "6px";
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.drawImage(img, 0, 0, cw, ch);
    const P = ([u, v]) => [(u - x0) * k, (v - y0) * k];
    fit = { x0, y0, k, polys: view.boxes.map((b) => ({ tid: b.tid, hull: hull(b.corners.map(P)) })) };
    canvas.dataset.frame = view.frame; canvas.dataset.tid = view.tid ?? "";     // what is on screen
    const others = view.boxes.filter((b) => b.tid !== S.tid);
    const sel = view.boxes.find((b) => b.tid === S.tid);
    for (const b of others) face(g, b.corners.map(P), "rgba(255,255,255,0.78)", 1);
    if (sel) {
      const c = sel.corners.map(P);
      face(g, c.slice(0, 4), "rgba(37,99,235,0.55)", 1.2, true);      // footprint (bottom)
      face(g, c.slice(4), "#2563eb", 2.2);                              // roof outline
      g.strokeStyle = "rgba(37,99,235,0.55)"; g.lineWidth = 1;
      for (let i = 0; i < 4; i++) { g.beginPath(); g.moveTo(...c[i]); g.lineTo(...c[i + 4]); g.stroke(); }
      const top = c.slice(4).reduce((a, p) => (p[1] < a[1] ? p : a));
      g.font = "600 11.5px ui-sans-serif, system-ui, sans-serif";
      const label = `#${S.tid}`;
      const w = g.measureText(label).width + 10;
      const lx = Math.max(w / 2 + 3, Math.min(cw - w / 2 - 3, top[0]));      // keep the tag on the canvas
      const ly = Math.max(20, Math.min(ch - 3, top[1]));
      g.fillStyle = "#2563eb"; roundRect(g, lx - w / 2, ly - 22, w, 17, 4); g.fill();
      g.fillStyle = "#fff"; g.textAlign = "center"; g.fillText(label, lx, ly - 9.5);
    }
  }

  // ---- interaction: click a box to select it; while placing, click the vehicle to add it there
  function local(e) { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  function boxAt(px, py) {
    if (!fit) return null;
    const hits = fit.polys.filter((p) => inside(p.hull, px, py));
    return hits.length ? hits[hits.length - 1].tid : null;
  }
  canvas.addEventListener("click", async (e) => {
    if (!fit || !view) return;
    const [px, py] = local(e);
    if (S.placing) {
      try {
        const r = await api.uavLocate(S.view.uav, S.frame, fit.x0 + px / fit.k, fit.y0 + py / fit.k);
        finishPlacing(r.xy);
      } catch (err) { reportError(err); }
      return;
    }
    const tid = boxAt(px, py);
    if (tid !== null && tid !== S.tid) selectTrack(tid).catch(reportError);
  });
  canvas.addEventListener("mousemove", (e) => {
    const [px, py] = local(e);
    canvas.style.cursor = S.placing ? "crosshair" : boxAt(px, py) !== null ? "pointer" : "";
  });
  on("placing", () => { canvas.style.cursor = S.placing ? "crosshair" : ""; wrap.classList.toggle("placing", S.placing); });

  function hull(pts) {                       // convex hull (monotone chain) of the projected corners
    const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], up = [];
    for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
    for (const q of p.reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
    return lo.slice(0, -1).concat(up.slice(0, -1));
  }
  function inside(poly, x, y) {
    let ins = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) ins = !ins;
    }
    return ins;
  }

  function face(g, pts, color, width, dashed = false) {
    const poly = pts.length === 8 ? pts.slice(4) : pts;
    g.strokeStyle = color; g.lineWidth = width;
    g.setLineDash(dashed ? [4, 3] : []);
    g.beginPath(); poly.forEach((p, i) => (i ? g.lineTo(...p) : g.moveTo(...p))); g.closePath(); g.stroke();
    g.setLineDash([]);
  }

  function roundRect(g, x, y, w, h, r) {
    g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
  }

  let lastAspect = null, rz = null;
  new ResizeObserver(() => {
    draw();
    const [aw, ah] = area(), asp = aw / ah;
    if (view && lastAspect && Math.abs(asp - lastAspect) / lastAspect > 0.12) { clearTimeout(rz); rz = setTimeout(load, 200); }
    lastAspect = asp;
  }).observe(wrap);
  on("project", () => { renderSegs(); load(); });
  on("frame", load);
  on("select", load);
  let tt = null;
  on("tracks", () => { clearTimeout(tt); tt = setTimeout(load, 300); });
  let t = null;
  on("track", (info) => { if (!info || !info.preview) { clearTimeout(t); t = setTimeout(load, 250); } });
  return { reload: load, geometry: () => fit };
}

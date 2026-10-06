// The 3D viewport: the current frame's LiDAR, every vehicle box, and direct manipulation in Top view.
//
//   3D   orbit (left drag), pan (right drag), zoom (wheel); click a box to select it.
//   Top  orthographic bird's-eye view with handles on the selected box:
//        drag the body to move this frame (Shift: the whole track), an edge to resize one-sided
//        (shared shape), the ring to rotate this frame (Shift: the whole track). Pan with left
//        drag on empty space or right drag.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

import { api } from "../api.js";
import * as ops from "../ops.js";
import { S, on, emit, edit, preview, commitDrag, selectTrack, reportError, finishPlacing, cancelPlacing } from "../store.js";
import { el, icon } from "../ui.js";

const SENSOR = {               // humble, low-saturation: the boxes carry the colour
  0: { name: "Ford", rgb: [0.40, 0.44, 0.50] },
  1: { name: "Lexus", rgb: [0.42, 0.50, 0.44] },
  2: { name: "Roadside", rgb: [0.62, 0.55, 0.46] },
};
const HILITE = [0.10, 0.26, 0.70];           // points inside the selected box
const BOX = { unreviewed: 0xa3a19a, accepted: 0x7aa384, edited: 0x8f7cc8, new: 0x6aa3b3, flagged: 0xd39a4e };
const ACCENT = 0x2563eb;
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];

export function createViewport(host) {
  const root = el("div.viewport", { style: { position: "absolute", inset: "0" } });
  host.append(root);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0xffffff, 1);
  root.append(renderer.domElement);

  const scene = new THREE.Scene();
  const persp = new THREE.PerspectiveCamera(50, 1, 0.1, 2000);
  persp.up.set(0, 0, 1);
  const ortho = new THREE.OrthographicCamera(-20, 20, 20, -20, -500, 500);
  ortho.up.set(0, 1, 0);
  let camera = persp;
  let controls = makeControls(persp);

  // ---------------------------------------------------------------- scene objects
  const pts = new THREE.Points(new THREE.BufferGeometry(),       // 2 CSS px at any screen density
    new THREE.PointsMaterial({ size: 2 * renderer.getPixelRatio(), sizeAttenuation: false, vertexColors: true }));
  pts.frustumCulled = false;
  scene.add(pts);

  const lineMat = (color, width, opacity = 1) => new LineMaterial({
    color, linewidth: width, transparent: opacity < 1, opacity, worldUnits: false, depthTest: true });
  const others = new LineSegments2(new LineSegmentsGeometry(), lineMat(0xffffff, 1.3));
  others.material.vertexColors = true;
  const selLines = new LineSegments2(new LineSegmentsGeometry(), lineMat(ACCENT, 2.6));
  scene.add(others, selLines);
  const fill = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.07, depthWrite: false }));
  fill.visible = false;
  scene.add(fill);
  const pick = new THREE.Group();
  pick.visible = false;
  scene.add(pick);
  const handles = new THREE.Group();
  scene.add(handles);

  // ---------------------------------------------------------------- overlays
  const modeSeg = el("div.seg", {},
    el("button", { onclick: () => setMode("3d") }, "3D"),
    el("button", { onclick: () => setMode("top") }, "Top"));
  const sensorChips = el("div.legend.glass");
  const toggles = el("div.row");
  const frameBadge = el("div.frame-badge.glass");
  const hint = el("div.hint.glass");
  const notice = el("div.notice.glass.hidden");
  const placebar = el("div.placebar.hidden", {},
    icon("crosshair"), el("span", {}, "Click the new vehicle — here or in the UAV image"),
    el("button.btn.sm", { onclick: () => cancelPlacing() }, "Cancel", el("span.kbd", {}, "Esc")));
  root.append(placebar,
    el("div.overlay.ov-tl", {}, el("div.glass", { style: { padding: "3px" } }, modeSeg), sensorChips),
    el("div.overlay.ov-tr", {}, toggles),
    el("div.overlay.ov-bl", {}, frameBadge, notice),
    el("div.overlay.ov-br", {}, hint));

  function renderOverlays() {
    [...modeSeg.children].forEach((b, i) => b.classList.toggle("on", (i === 0) === (S.view.mode === "3d")));
    sensorChips.replaceChildren(...(S.project?.session.lidar_agents || []).map(({ id }) => {
      const onState = S.view.agents.includes(id);
      const c = SENSOR[id]?.rgb || [0.5, 0.5, 0.5];
      return el("span", { style: { cursor: "pointer", opacity: onState ? 1 : 0.4 }, title: "Show / hide (1-3)",
        onclick: () => toggleSensor(id) },
        el("i", { style: { background: `rgb(${c.map((v) => Math.round(v * 255)).join(",")})` } }), SENSOR[id]?.name || id);
    }));
    toggles.replaceChildren(
      el("button.btn.sm.glass" + (S.view.follow ? "" : ".ghost"), { title: "Camera follows the vehicle", onclick: () => { S.view.follow = !S.view.follow; renderOverlays(); } }, icon("crosshair"), "Follow"),
      S.project?.session.ground_model === false
        ? el("button.btn.sm.glass.ghost", { disabled: true, title: "No ground model: studio/assets/dtm.npz is missing, so ground points cannot be removed" }, icon("layers"), "No ground model")
        : el("button.btn.sm.glass" + (S.view.ground ? "" : ".ghost"), { title: "Show ground points (G)", onclick: () => toggleGround() }, icon("layers"), "Ground"),
      el("button.btn.sm.icon.glass", { title: "Fit to the vehicle (F)", onclick: () => fit() }, icon("eye")));
    hint.textContent = S.view.mode === "top"
      ? "Drag box to move · edges resize · ring rotates · Shift = whole track"
      : "Drag to orbit · right-drag to pan · click a box to select";
  }

  // ---------------------------------------------------------------- data
  const cache = new Map();                  // "frame|agents|ground" -> points
  let frameBoxes = [];
  let loadSeq = 0;
  const key = (f) => `${f}|${S.view.agents.join(",")}|${S.view.ground ? 1 : 0}`;

  async function points(f) {
    const k = key(f);
    if (!cache.has(k)) {
      cache.set(k, api.framePoints(f, S.view.agents, S.view.ground));
      if (cache.size > 48) cache.delete(cache.keys().next().value);
    }
    return cache.get(k);
  }

  function prefetch() {
    const fr = S.track ? ops.framesOf(S.track) : S.project.frames;
    const i = fr.indexOf(S.frame);
    for (const d of [1, 2, 3, -1]) if (fr[i + d] !== undefined) points(fr[i + d]).catch(() => {});
  }

  let current = null;                        // the loaded points of the shown frame
  let lastLoad = Promise.resolve();
  function loadFrame() { lastLoad = loadFrameNow(); return lastLoad; }
  async function loadFrameNow() {
    if (!S.project) return;
    const seq = ++loadSeq, f = S.frame;
    try {
      const [p, boxes] = await Promise.all([points(f), api.frameBoxes(f)]);
      if (seq !== loadSeq) return;
      current = p;
      frameBoxes = boxes;
      drawPoints();
      drawBoxes();
      renderBadge();
      prefetch();
    } catch (e) { reportError(e); }
  }

  function drawPoints() {
    if (!current) return;
    const { n, xyz, ego } = current;
    const col = new Float32Array(n * 3);
    const sel = selectedBox();
    let c = 1, s = 0, hx = 0, hy = 0, hz = 0;
    if (sel) { c = Math.cos(sel[6]); s = Math.sin(sel[6]); hx = sel[3] / 2 + 0.15; hy = sel[4] / 2 + 0.15; hz = sel[5] / 2 + 0.2; }
    for (let i = 0; i < n; i++) {
      let rgb = SENSOR[ego[i]]?.rgb || [0.5, 0.5, 0.5];
      if (sel) {
        const dx = xyz[3 * i] - sel[0], dy = xyz[3 * i + 1] - sel[1], dz = xyz[3 * i + 2] - sel[2];
        if (Math.abs(c * dx + s * dy) <= hx && Math.abs(-s * dx + c * dy) <= hy && Math.abs(dz) <= hz) rgb = HILITE;
      }
      col[3 * i] = rgb[0]; col[3 * i + 1] = rgb[1]; col[3 * i + 2] = rgb[2];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    pts.geometry.dispose();
    pts.geometry = g;
    requestRender();
  }

  function selectedBox() {
    return S.track && ops.has(S.track, S.frame) ? ops.box(S.track, S.frame) : null;
  }

  function boxSegments(b) {
    const c = ops.corners(b);
    const out = [];
    for (const [i, j] of EDGES) out.push(...c[i], ...c[j]);
    // heading tick: from the top centre to the middle of the front edge
    const [cx, cy, cz, L, , H, yaw] = b;
    out.push(cx, cy, cz + H / 2, cx + Math.cos(yaw) * L / 2, cy + Math.sin(yaw) * L / 2, cz + H / 2);
    return out;
  }

  function drawBoxes() {
    const pos = [], col = [];
    pick.clear();
    for (const fb of frameBoxes) {
      if (fb.tid === S.tid) continue;
      const seg = boxSegments(fb.box);
      pos.push(...seg);
      const color = new THREE.Color(BOX[S.tracks.get(fb.tid)?.status ?? fb.status] ?? BOX.unreviewed);
      for (let k = 0; k < seg.length / 3; k++) col.push(color.r, color.g, color.b);
      addPick(fb.tid, fb.box);
    }
    setSegs(others, pos, col);
    const sel = selectedBox();
    if (sel) {
      setSegs(selLines, boxSegments(sel));
      fill.visible = true;
      fill.position.set(sel[0], sel[1], sel[2]);
      fill.scale.set(sel[3], sel[4], sel[5]);
      fill.rotation.set(0, 0, sel[6]);
      addPick(S.tid, sel);
    } else {
      setSegs(selLines, []);
      fill.visible = false;
    }
    drawHandles();
    requestRender();
  }

  function setSegs(obj, pos, col) {
    const g = new LineSegmentsGeometry();
    if (pos.length) {
      g.setPositions(pos);
      if (col) g.setColors(col);
    }
    obj.geometry.dispose();
    obj.geometry = g;
    obj.visible = pos.length > 0;
  }

  function addPick(tid, b) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(b[3], b[4], b[5]), new THREE.MeshBasicMaterial());
    m.position.set(b[0], b[1], b[2]);
    m.rotation.set(0, 0, b[6]);
    m.userData.tid = tid;
    m.updateMatrixWorld();
    pick.add(m);
  }

  function renderBadge() {
    const n = frameBoxes.length;
    const present = S.track ? ops.has(S.track, S.frame) : true;
    frameBadge.replaceChildren(el("span", {}, "Frame ", el("b", {}, S.frame)), ` · ${n} vehicle${n === 1 ? "" : "s"}`);
    notice.classList.toggle("hidden", present);
    notice.textContent = `Track ${S.tid} has no box in this frame`;
  }

  // ---------------------------------------------------------------- handles (Top view)
  let pxPerM = 10;
  function drawHandles() {
    handles.clear();
    const b = selectedBox();
    if (!b || S.view.mode !== "top") return;
    const [cx, cy, cz, L, W, H, yaw] = b;
    const s = 9 / pxPerM;                    // ~9 px on screen at any zoom
    const z = cz + H / 2 + 0.05;
    const c = Math.cos(yaw), sn = Math.sin(yaw);
    const at = (lx, ly) => [cx + c * lx - sn * ly, cy + sn * lx + c * ly];
    const sq = (name, lx, ly) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(s, s), new THREE.MeshBasicMaterial({ color: 0xffffff }));
      const ring = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(s, s)),
        new THREE.LineBasicMaterial({ color: ACCENT }));
      const [x, y] = at(lx, ly);
      m.position.set(x, y, z); ring.position.set(x, y, z + 0.01);
      m.rotation.z = ring.rotation.z = yaw;
      m.userData.handle = name; handles.add(m, ring);
    };
    sq("front", L / 2, 0); sq("back", -L / 2, 0); sq("left", 0, W / 2); sq("right", 0, -W / 2);
    const rr = 1.1 + 6 / pxPerM;
    const [rx, ry] = at(L / 2 + rr, 0);
    const knob = new THREE.Mesh(new THREE.CircleGeometry(s * 0.62, 20), new THREE.MeshBasicMaterial({ color: ACCENT }));
    knob.position.set(rx, ry, z); knob.userData.handle = "rotate";
    const [fx, fy] = at(L / 2, 0);
    const stem = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(fx, fy, z), new THREE.Vector3(rx, ry, z)]),
      new THREE.LineBasicMaterial({ color: ACCENT }));
    handles.add(stem, knob);
  }

  // ---------------------------------------------------------------- interaction
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let drag = null;

  function setNdc(e) {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
  }

  function groundAt(e, z) {
    setNdc(e);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -z);
    const p = new THREE.Vector3();
    return ray.ray.intersectPlane(plane, p) ? p : null;
  }

  renderer.domElement.addEventListener("pointerdown", (e) => {
    down = { x: e.clientX, y: e.clientY };
    if (S.placing) return;
    if (e.button !== 0 || S.view.mode !== "top" || !S.track) return;
    const b = selectedBox();
    if (!b) return;
    setNdc(e);
    const h = ray.intersectObjects(handles.children.filter((o) => o.userData.handle), false)[0];
    let kind = h ? h.object.userData.handle : null;
    if (!kind) {
      const hit = ray.intersectObjects(pick.children, false).find((x) => x.object.userData.tid === S.tid);
      if (hit) kind = "body";
    }
    if (!kind) return;
    const p0 = groundAt(e, b[2]);
    if (!p0) return;
    drag = { kind, before: S.track, b0: b, p0, all: e.shiftKey };
    controls.enabled = false;
    renderer.domElement.setPointerCapture(e.pointerId);
    e.preventDefault();
  });

  renderer.domElement.addEventListener("pointermove", (e) => {
    if (!drag) return hover(e);
    const p = groundAt(e, drag.b0[2]);
    if (!p) return;
    preview(dragResult(drag, p));
  });

  renderer.domElement.addEventListener("pointerup", async (e) => {
    if (drag) {
      const d = drag; drag = null;
      controls.enabled = true;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y) > 2;
      if (moved) commitDrag(opName(d), d.before); else preview(d.before);
      return;
    }
    if (S.placing) {
      if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4 && e.button === 0) {
        const p = groundAt(e, controls.target.z);
        if (p) finishPlacing([p.x, p.y]);
      }
      return;
    }
    if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4 && e.button === 0) {
      setNdc(e);
      const hit = ray.intersectObjects(pick.children, false)[0];
      if (hit && hit.object.userData.tid !== S.tid) selectTrack(hit.object.userData.tid).catch(reportError);
    }
  });
  let down = null;

  function hover(e) {
    if (S.placing) { renderer.domElement.style.cursor = "crosshair"; return; }
    if (S.view.mode !== "top" || !S.track) { renderer.domElement.style.cursor = ""; return; }
    setNdc(e);
    const h = ray.intersectObjects(handles.children.filter((o) => o.userData.handle), false)[0];
    const kind = h && h.object.userData.handle;
    const onBody = !kind && ray.intersectObjects(pick.children, false).some((x) => x.object.userData.tid === S.tid);
    renderer.domElement.style.cursor = kind === "rotate" ? "grab" : kind ? "pointer" : onBody ? "move" : "";
  }

  const opName = (d) => ({ body: d.all ? "move track" : "move", rotate: d.all ? "rotate track" : "rotate",
    front: "resize length", back: "resize length", left: "resize width", right: "resize width" }[d.kind]);

  function dragResult(d, p) {
    const [cx, cy, , L, W, , yaw] = d.b0;
    const t = d.before;
    const dx = p.x - d.p0.x, dy = p.y - d.p0.y;
    if (d.kind === "body") return ops.move(t, d.all ? null : [S.frame], dx, dy, 0);
    if (d.kind === "rotate") {
      const a0 = Math.atan2(d.p0.y - cy, d.p0.x - cx), a1 = Math.atan2(p.y - cy, p.x - cx);
      return ops.rotate(t, d.all ? null : [S.frame], ops.wrap(a1 - a0));
    }
    const [lx, ly] = ops.toBox(d.b0, p.x, p.y);
    if (d.kind === "front") return ops.resize(t, "L", lx + L / 2, +1);
    if (d.kind === "back") return ops.resize(t, "L", L / 2 - lx, -1);
    if (d.kind === "left") return ops.resize(t, "W", ly + W / 2, +1);
    return ops.resize(t, "W", W / 2 - ly, -1);
  }

  // ---------------------------------------------------------------- camera
  function makeControls(cam) {
    const c = new OrbitControls(cam, renderer.domElement);
    c.enableDamping = true;
    c.dampingFactor = 0.18;
    c.screenSpacePanning = true;
    c.addEventListener("change", () => { updatePxPerM(); requestRender(); });
    if (cam === ortho) {
      c.enableRotate = false;
      c.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    } else {
      c.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
      c.maxPolarAngle = Math.PI * 0.495;
    }
    return c;
  }

  function updatePxPerM() {
    const h = renderer.domElement.clientHeight || 1;
    const prev = pxPerM;
    if (camera === ortho) pxPerM = h / ((ortho.top - ortho.bottom) / ortho.zoom);
    if (Math.abs(prev - pxPerM) / prev > 0.05) drawHandles();
  }

  function setMode(m) {
    if (S.view.mode === m) return;
    const target = controls.target.clone();
    controls.dispose();
    S.view.mode = m;
    camera = m === "top" ? ortho : persp;
    controls = makeControls(camera);
    controls.target.copy(target);
    if (m === "top") { ortho.position.set(target.x, target.y, target.z + 200); ortho.lookAt(target); }
    fit(false);
    renderOverlays();
    drawBoxes();
    emit("view");
  }

  function fit(animateDist = true) {
    const b = selectedBox() || (frameBoxes[0] && frameBoxes[0].box);
    const tgt = b ? new THREE.Vector3(b[0], b[1], b[2]) : controls.target.clone();
    if (camera === ortho) {
      const span = b ? Math.max(b[3], b[4]) * 4 + 8 : 80;
      resize();
      ortho.zoom = (ortho.top - ortho.bottom) / span;
      ortho.position.set(tgt.x, tgt.y, tgt.z + 200);
      ortho.updateProjectionMatrix();
    } else {
      const dir = persp.position.clone().sub(controls.target);
      if (dir.lengthSq() < 1e-6 || animateDist) dir.set(-14, -14, 11);
      persp.position.copy(tgt).add(dir.setLength(b ? Math.max(15, b[3] * 3.2) : 70));
    }
    controls.target.copy(tgt);
    controls.update();
    updatePxPerM();
    drawHandles();
    requestRender();
  }

  let lastCentre = null;
  function follow() {
    const b = selectedBox();
    if (!b) { lastCentre = null; return; }
    const c = new THREE.Vector3(b[0], b[1], b[2]);
    if (S.view.follow && lastCentre) {
      const d = c.clone().sub(lastCentre);
      camera.position.add(d);
      controls.target.add(d);
      controls.update();
    }
    lastCentre = c;
  }

  // ---------------------------------------------------------------- render loop
  let need = true;
  function requestRender() { need = true; }
  function resize() {
    const w = root.clientWidth, h = root.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = w + "px";
    renderer.domElement.style.height = h + "px";
    persp.aspect = w / h; persp.updateProjectionMatrix();
    const half = 20;
    ortho.left = -half * (w / h); ortho.right = half * (w / h); ortho.top = half; ortho.bottom = -half;
    ortho.updateProjectionMatrix();
    for (const m of [others.material, selLines.material]) m.resolution.set(w, h);
    updatePxPerM();
    requestRender();
  }
  new ResizeObserver(resize).observe(root);
  (function loop() {
    requestAnimationFrame(loop);
    if (controls.enableDamping && controls.update()) need = true;
    if (!need) return;
    need = false;
    renderer.render(scene, camera);
  })();

  // ---------------------------------------------------------------- store wiring
  function toggleSensor(id) {
    const a = S.view.agents.includes(id) ? S.view.agents.filter((x) => x !== id) : [...S.view.agents, id].sort();
    if (!a.length) return;
    S.view.agents = a;
    renderOverlays();
    loadFrame();
  }
  function toggleGround() { S.view.ground = !S.view.ground; renderOverlays(); loadFrame(); }

  on("project", () => { cache.clear(); renderOverlays(); loadFrame().then(() => fit()); });
  on("frame", () => { loadFrame().then(follow); });
  on("select", () => { lastCentre = null; loadFrame().then(() => { fit(false); follow(); }); });
  on("track", () => { drawBoxes(); drawPoints(); renderBadge(); });
  on("placing", () => {
    placebar.classList.toggle("hidden", !S.placing);
    if (S.placing && S.view.mode !== "top") setMode("top");
    renderer.domElement.style.cursor = S.placing ? "crosshair" : "";
  });
  let tracksTimer = null;                   // tracks arrived or changed elsewhere (seeding, reload)
  on("tracks", () => { clearTimeout(tracksTimer); tracksTimer = setTimeout(loadFrame, 150); });

  return {
    fit, setMode, toggleSensor, toggleGround,
    soloSensor(id) { S.view.agents = id === null ? S.project.session.lidar_agents.map((a) => a.id) : [id]; renderOverlays(); loadFrame(); },
    reload() { cache.clear(); loadFrame(); },
    target: () => [controls.target.x, controls.target.y, controls.target.z],
    // screen position (client px) of a map point, and of the selected box's handles (Top view)
    toScreen([x, y, z]) {
      const v = new THREE.Vector3(x, y, z).project(camera);
      const r = renderer.domElement.getBoundingClientRect();
      return [r.left + ((v.x + 1) / 2) * r.width, r.top + ((1 - v.y) / 2) * r.height];
    },
    handleScreen(name) {
      const h = handles.children.find((o) => o.userData.handle === name);
      return h ? this.toScreen([h.position.x, h.position.y, h.position.z]) : null;
    },
    ready: () => lastLoad,
  };
}

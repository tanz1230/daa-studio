// Timeline: scrub the session; the selected track's span, its edited frames, keyframes and
// low-confidence frames are marked so problems are visible before you get to them.

import * as ops from "../ops.js";
import { S, on, setFrame, step, jumpEnd } from "../store.js";
import { el, icon } from "../ui.js";

export function createTimeline(host) {
  const playBtn = el("button.btn.icon.ghost", { title: "Play / pause (Space)", onclick: () => togglePlay() }, icon("play"));
  const ctrl = el("div.tl-ctrl", {},
    el("button.btn.icon.ghost", { title: "First frame (Home)", onclick: () => jumpEnd(-1) }, icon("first")),
    el("button.btn.icon.ghost", { title: "Previous frame ( , )", onclick: () => step(-1) }, icon("left")),
    playBtn,
    el("button.btn.icon.ghost", { title: "Next frame ( . )", onclick: () => step(1) }, icon("right")),
    el("button.btn.icon.ghost", { title: "Last frame (End)", onclick: () => jumpEnd(1) }, icon("last")));
  const canvas = el("canvas.track-canvas");
  const label = el("div.tl-frame");
  host.append(ctrl, canvas, label);

  let playTimer = null;
  function togglePlay(force) {
    const go = force ?? !playTimer;
    clearInterval(playTimer); playTimer = null;
    if (go) {
      playTimer = setInterval(() => {
        const fr = S.track ? ops.framesOf(S.track) : S.project.frames;
        if (S.frame === fr[fr.length - 1]) return togglePlay(false);
        step(1);
      }, 100);                                  // 10 Hz, the LiDAR rate
    }
    S.playing = !!playTimer;
    playBtn.replaceChildren(icon(playTimer ? "pause" : "play"));
  }

  // The axis is the session's frame ORDER, not the frame number: recordings have gaps (frames
  // without every sensor are not in the session), and a number axis would leave dead stretches.
  // Gaps show as breaks in the rail.
  let axis = { frames: null, index: new Map(), breaks: [] };
  function geom() {
    const fr = S.project.frames;
    if (axis.frames !== fr) {
      const steps = fr.slice(1).map((f, i) => f - fr[i]).sort((a, b) => a - b);
      const typical = steps.length ? steps[steps.length >> 1] : 1;
      axis = { frames: fr, index: new Map(fr.map((f, i) => [f, i])),
               breaks: fr.map((f, i) => (i && f - fr[i - 1] > 3 * typical ? i : -1)).filter((i) => i > 0) };
    }
    return axis;
  }
  function ord(f) {                               // frame -> position on the axis (fractional if off-session)
    const { frames, index } = geom();
    if (index.has(f)) return index.get(f);
    let lo = 0, hi = frames.length - 1;
    if (f <= frames[0]) return 0;
    if (f >= frames[hi]) return hi;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (frames[m] <= f) lo = m; else hi = m; }
    return lo + (f - frames[lo]) / (frames[hi] - frames[lo]);
  }

  function draw() {
    if (!S.project) return;
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio, 2);
    canvas.width = Math.max(1, r.width * dpr); canvas.height = Math.max(1, r.height * dpr);
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, r.width, r.height);
    const { frames, breaks } = geom();
    const n = Math.max(1, frames.length - 1);
    const X = (f) => 6 + (ord(f) / n) * (r.width - 12);
    const mid = 20;
    // session rail, broken where the recording has gaps
    g.fillStyle = "#eceae6"; g.fillRect(6, mid - 2, r.width - 12, 4);
    g.fillStyle = "#fff";
    for (const i of breaks) { const x = 6 + ((i - 0.5) / n) * (r.width - 12); g.fillRect(x - 1.5, mid - 3, 3, 6); }
    if (S.track) {
      const fr = ops.framesOf(S.track);
      // track span(s): runs of consecutive session frames
      g.fillStyle = "#c7d6fb";
      let a = fr[0];
      for (let i = 1; i <= fr.length; i++) {
        if (i === fr.length || ord(fr[i]) - ord(fr[i - 1]) > 1.01) {
          g.fillRect(X(a) - 1, mid - 4, Math.max(3, X(fr[i - 1]) - X(a) + 2), 8);
          a = fr[i];
        }
      }
      const conf = (S.track.seed && S.track.seed.conf) || {};
      for (const f of fr) {
        const fl = S.track.flags[String(f)] || [];
        if (fl.includes("edited")) { g.fillStyle = "#6d28d9"; g.fillRect(X(f) - 0.75, mid - 9, 1.5, 4); }
        if (conf[String(f)] !== undefined && conf[String(f)] < 0.25) { g.fillStyle = "#d39a4e"; g.fillRect(X(f) - 0.75, mid + 6, 1.5, 4); }
        if (fl.includes("keyframe")) diamond(g, X(f), mid - 13, "#1b1b1a");
      }
      if (S.keyframe !== null) diamond(g, X(S.keyframe), mid - 13, "#2563eb");
    }
    // playhead
    const x = X(S.frame);
    g.fillStyle = "#2563eb"; g.fillRect(x - 1, 4, 2, r.height - 12);
    g.beginPath(); g.arc(x, mid, 5, 0, 2 * Math.PI); g.fill();
    g.fillStyle = "#fff"; g.beginPath(); g.arc(x, mid, 2, 0, 2 * Math.PI); g.fill();
    // frame numbers at the ends
    g.font = "11px ui-sans-serif, system-ui, sans-serif"; g.fillStyle = "#8f8d87";
    g.textAlign = "left"; g.fillText(frames[0], 6, r.height - 3);
    g.textAlign = "right"; g.fillText(frames[frames.length - 1], r.width - 6, r.height - 3);
    renderLabel();
  }

  function diamond(g, x, y, c) {
    g.fillStyle = c; g.beginPath(); g.moveTo(x, y - 4); g.lineTo(x + 4, y); g.lineTo(x, y + 4); g.lineTo(x - 4, y); g.closePath(); g.fill();
  }

  function renderLabel() {
    const parts = [el("span", {}, "Frame "), el("b", {}, S.frame)];
    if (S.track) {
      const fr = ops.framesOf(S.track);
      const i = fr.indexOf(S.frame);
      parts.push(el("span", {}, i >= 0 ? ` · ${i + 1} / ${fr.length}` : " · not in track"));
    }
    label.replaceChildren(...parts);
  }

  let scrubbing = false;
  function scrub(e) {
    const r = canvas.getBoundingClientRect();
    const { frames } = geom();
    const u = Math.max(0, Math.min(1, (e.clientX - r.left - 6) / (r.width - 12))) * (frames.length - 1);
    const pool = S.track && !e.altKey ? ops.framesOf(S.track) : frames;
    let best = pool[0], bd = Infinity;
    for (const g of pool) { const d = Math.abs(ord(g) - u); if (d < bd) { bd = d; best = g; } }
    setFrame(best);
  }
  canvas.addEventListener("pointerdown", (e) => { scrubbing = true; canvas.setPointerCapture(e.pointerId); scrub(e); });
  canvas.addEventListener("pointermove", (e) => scrubbing && scrub(e));
  canvas.addEventListener("pointerup", () => (scrubbing = false));

  new ResizeObserver(draw).observe(canvas);
  for (const evt of ["project", "frame", "select", "track", "keyframe"]) on(evt, draw);
  return { togglePlay };
}

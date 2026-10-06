// Inspector: everything about the selected track — numbers you can type, where the seed came from
// and how much to trust it, the review decision, and the recent edit history.

import { api } from "../api.js";
import * as ops from "../ops.js";
import { S, on, edit, accept, flag, remove, restore, reportError } from "../store.js";
import { el, icon, statusBadge, fmt, popover } from "../ui.js";

const SEED = { daa: "DAA (UAV + LiDAR)", uav: "UAV detection", "uav-fallback": "UAV (DAA fell back)",
               import: "Imported labels", manual: "Added by hand" };
const isDaa = (seed) => seed.kind === "daa" || seed.kind === "uav-fallback";
export const FLAG_REASONS = ["Not a vehicle", "Two vehicles merged", "Occluded — extent unsure",
                             "Wrong heading", "Needs a second look"];

export function createInspector(host) {
  const root = el("div");
  host.append(root);
  let histTid = null, hist = [];

  function numField(label, unit, value, onCommit, step = 0.05, digits = 2) {
    const inp = el("input.input.num", { value: value === null ? "" : Number(value).toFixed(digits), spellcheck: false });
    const commit = () => {
      const v = parseFloat(inp.value);
      if (Number.isFinite(v) && Math.abs(v - value) > 1e-6) onCommit(v); else inp.value = Number(value).toFixed(digits);
    };
    inp.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { commit(); inp.blur(); }
      else if (e.key === "Escape") { inp.value = Number(value).toFixed(digits); inp.blur(); }
      else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        const d = (e.key === "ArrowUp" ? 1 : -1) * step * (e.shiftKey ? 4 : 1);
        inp.value = (parseFloat(inp.value) + d).toFixed(digits); commit();
      }
    });
    inp.addEventListener("blur", commit);
    return el("div.field", {}, el("label", {}, label), el("div.unit", { "data-unit": unit }, inp));
  }

  function render() {
    const t = S.track;
    if (!t) {
      root.replaceChildren(el("div.sec", {}, el("div.empty", { style: { padding: "40px 10px" } },
        icon("cube"), "No track selected.", el("br"),
        el("span", {}, "Press ", el("span.kbd", {}, "N"), " to start the review queue."))));
      return;
    }
    const has = ops.has(t, S.frame);
    const [L, W, H] = t.shape;
    const p = has ? t.poses[String(S.frame)] : null;
    const seed = t.seed || {}, d = seed.diag || {};
    const conf = seed.conf || {};
    const fconf = conf[String(S.frame)];

    const clsSel = el("select.select", { style: { width: "auto" }, onchange: (e) => edit("class", (x) => ops.setClass(x, e.target.value)) },
      ...["Car", "Truck"].map((c) => el("option", { value: c, selected: t.cls === c }, c)));

    const head = el("div.sec", {},
      el("div.ihead", {}, el("span.tid", {}, `#${t.tid}`), clsSel, el("span.grow"), statusBadge(t.status)),
      t.note ? el("div.muted", { style: { marginTop: "8px", fontSize: "12.5px" } }, t.note) : null);

    const shape = el("div.sec", {},
      el("div.sec-h", {}, el("span.label", {}, "Shape · all frames")),
      el("div.grid3", {},
        numField("Length", "m", L, (v) => edit("resize", (x) => ops.resize(x, "L", v, 0))),
        numField("Width", "m", W, (v) => edit("resize", (x) => ops.resize(x, "W", v, 0))),
        numField("Height", "m", H, (v) => edit("resize", (x) => ops.resize(x, "H", v, 0)))));

    const isKey = has && (t.flags[String(S.frame)] || []).includes("keyframe");
    const pose = el("div.sec", {},
      el("div.sec-h", {}, el("span.label", {}, `Pose · frame ${S.frame}`),
        has ? el("button.btn.sm" + (isKey ? "" : ".ghost"), { title: "Keyframe (M marks one for range tools)",
          onclick: () => edit("keyframe", (x) => ops.mark(x, S.frame, "keyframe", !isKey)) }, icon("diamond"), isKey ? "Keyframe" : "Set keyframe") : null),
      has ? el("div.grid2", {},
        numField("X", "m", p[0], (v) => edit("pose", (x) => ops.setPose(x, S.frame, { x: v }))),
        numField("Y", "m", p[1], (v) => edit("pose", (x) => ops.setPose(x, S.frame, { y: v }))),
        numField("Z", "m", p[2], (v) => edit("pose", (x) => ops.setPose(x, S.frame, { z: v }))),
        numField("Heading", "°", (p[3] * 180) / Math.PI, (v) => edit("pose", (x) => ops.setPose(x, S.frame, { yaw: (v * Math.PI) / 180 })), 0.5, 1))
        : el("div", {}, el("p.muted", { style: { margin: "0 0 10px" } }, "This track has no box in this frame."),
            el("div.tools", {},
              el("button.btn.sm", { title: "Copy the nearest frame's box here", onclick: () => edit("add frame", (x) => ops.setPose(x, S.frame, {})) }, icon("plus"), "Add box here"),
              between() ? el("button.btn.sm", { title: "Interpolate a box here from the frames around it (X)", onclick: () => interpolateHere() }, icon("interp"), "Interpolate here") : null)),
      has ? el("div.tools", { style: { marginTop: "10px" } },
        el("button.btn.sm", { title: "Interpolate this frame from its neighbours (X)", onclick: () => interpolateHere() }, icon("interp"), "Interpolate"),
        el("button.btn.sm", { title: "Turn this frame's heading by 180° (Y)", onclick: () => edit("flip", (x) => ops.flip(x, [S.frame])) }, icon("flip"), "Flip"),
        el("button.btn.sm", { title: "Fill gaps, copy to every frame, remove this frame", onclick: (e) => framesMenu(e.currentTarget) }, icon("layers"), "Frames…")) : null);

    const seedSec = el("div.sec", {},
      el("div.sec-h", {}, el("span.label", {}, "Seed")),
      el("dl.kv", {},
        el("dt", {}, "Source"), el("dd", {}, SEED[seed.kind] || seed.kind || "—"),
        d.conf !== undefined ? [el("dt", {}, "DAA confidence"), el("dd", { class: d.conf < 0.3 ? "warn" : "" }, fmt.m(d.conf))] : null,
        fconf !== undefined && isDaa(seed) ? [el("dt", {}, "This frame"), el("dd", { class: fconf < 0.25 ? "warn" : "" }, fmt.m(fconf))] : null,
        fconf !== undefined && seed.kind === "uav" ? [el("dt", {}, "Detection score"), el("dd", {}, fmt.m(fconf))] : null,
        d.pts !== undefined ? [el("dt", {}, "LiDAR points (median)"), el("dd", { class: d.pts < 20 ? "warn" : "" }, Math.round(d.pts))] : null,
        d.two_sided !== undefined ? [el("dt", {}, "Seen from both sides"), el("dd", { class: d.two_sided < 0.1 ? "warn" : "" }, d.two_sided < 0.1 ? "No — one side only" : `Yes (${fmt.pct(d.two_sided)})`)] : null,
        d.shift_m !== undefined ? [el("dt", {}, "Moved from UAV box"), el("dd", {}, `${fmt.m(d.shift_m)} m`)] : null,
        d.dL !== undefined ? [el("dt", {}, "Size change vs UAV"), el("dd", {}, `L ${d.dL >= 0 ? "+" : ""}${fmt.m(d.dL)} · W ${d.dW >= 0 ? "+" : ""}${fmt.m(d.dW)}`)] : null,
        el("dt", {}, "Frames"), el("dd", {}, ops.framesOf(t).length)),
      (() => {
        const why = d.failed || seed.kind === "uav-fallback"
          ? "DAA could not refine this vehicle, so the box is the UAV detection projected to the LiDAR map with a default height."
          : d.pts !== undefined && d.pts < 1
            ? "No LiDAR points on this vehicle: the box is the UAV detection projected to the LiDAR map with a default height."
            : null;
        return why ? el("div.infobox.warn", { style: { marginTop: "12px" } }, why) : null;
      })());

    const deleted = t.status === "deleted";
    const review = el("div.sec", {},
      el("div.sec-h", {}, el("span.label", {}, "Review")),
      deleted
        ? el("div.actions", {}, el("button.btn.wide", { onclick: () => restore() }, icon("undo"), "Restore track"))
        : el("div.actions", {},
          el("button.btn.primary.wide", { onclick: () => accept().catch(reportError) }, icon("check"),
            ["accepted", "edited", "new"].includes(t.status) ? "Next track" : "Accept", el("span.kbd", {}, "↵")),
          el("button.btn", { onclick: (e) => flagMenu(e.currentTarget) }, icon("flag"), "Flag", el("span.kbd", {}, "⇧F")),
          el("button.btn.danger", { onclick: () => remove() }, icon("trash"), "Delete")),
      el("div.field", { style: { marginTop: "12px" } }, el("label", {}, "Note"),
        (() => {
          const ta = el("textarea.input", { placeholder: "Anything the next reviewer should know", value: t.note || "" });
          ta.addEventListener("blur", () => { if ((ta.value || "") !== (S.track.note || "")) edit("note", (x) => { x.note = ta.value; return x; }); });
          return ta;
        })()));

    const histSec = el("div.sec", {}, el("div.sec-h", {}, el("span.label", {}, "History")),
      hist.length ? el("dl.kv", {}, hist.slice(-5).reverse().map((h) => [
        el("dt", {}, `${h.op} · ${h.user || "—"}`), el("dd.muted", {}, (h.ts || "").replace("T", " ").slice(5, 16))]))
        : el("div.muted", {}, "No edits yet."));

    root.replaceChildren(head, review, shape, pose, seedSec, histSec);
  }

  function flagMenu(anchor) {
    popover(anchor, FLAG_REASONS.map((r) => ({ label: r, fn: () => flag(r) })));
  }

  function between() {
    const fr = ops.framesOf(S.track);
    const prev = fr.filter((g) => g < S.frame).pop(), next = fr.find((g) => g > S.frame);
    return prev === undefined || next === undefined ? null : [prev, next];
  }

  function interpolateHere() {
    const b = between();
    if (b) edit("interpolate", (x) => ops.interpolate(x, b[0], b[1], [S.frame]));
  }

  function framesMenu(anchor) {
    const t = S.track, fr = ops.framesOf(t), all = S.project.frames;
    const gaps = all.filter((f) => f > fr[0] && f < fr[fr.length - 1] && !ops.has(t, f)).length;
    const missing = all.filter((f) => !ops.has(t, f)).length;
    const n = (k) => `${k} frame${k === 1 ? "" : "s"}`;
    popover(anchor, [
      { label: gaps ? `Fill ${n(gaps)} in the gaps (interpolate)` : "No gaps to fill", icon: "interp",
        fn: () => gaps && edit("fill gaps", (x) => ops.fillGaps(x, all)) },
      { label: missing ? `Parked: copy this box to ${n(missing)} more` : "Already in every frame", icon: "layers",
        fn: () => missing && edit("copy to all frames", (x) => ops.copyTo(x, S.frame, all)) },
      "-",
      { label: "Remove this frame's box", icon: "trash", fn: () => edit("remove frame", (x) => ops.removeFrame(x, S.frame)) },
    ]);
  }

  async function loadHistory() {
    if (S.tid === null) { hist = []; return; }
    histTid = S.tid;
    try { hist = await api.history(S.tid); } catch (e) { hist = []; }
    if (histTid === S.tid) render();
  }

  let t = null;
  on("select", () => { render(); loadHistory(); });
  on("frame", render);
  on("track", (info) => { if (!info || !info.preview) { render(); clearTimeout(t); t = setTimeout(loadHistory, 800); } });
  on("project", render);
  render();
  return { render, flagMenu };
}

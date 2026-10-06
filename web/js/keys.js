// Keyboard: n/p tracks, ,/. frames, [ ] { } rotate, arrows and - = nudge, i/k j/l u/o size, m/x/y range
// tools, Enter = accept & next, Shift+F = flag, Ctrl+Z / Ctrl+Shift+Z = undo / redo.

import * as ops from "./ops.js";
import { S, edit, step, jumpEnd, selectNext, accept, remove, undo, redo, emit, reportError, cancelPlacing } from "./store.js";
import { shortcutsDialog } from "./views/dialogs.js";
import { toast, confirmBox } from "./ui.js";

const DEG = Math.PI / 180;

export function installKeys({ viewport, timeline, tracks, inspector, addTrack }) {
  window.addEventListener("keydown", async (e) => {
    const tag = (e.target && e.target.tagName) || "";
    if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || document.querySelector(".scrim")) return;
    const k = e.key, mod = e.ctrlKey || e.metaKey;
    if (S.placing) { if (k === "Escape") { e.preventDefault(); cancelPlacing(); } return; }
    const t = S.track, f = S.frame;
    const hasBox = t && ops.has(t, f);
    const nudge = e.shiftKey ? 0.2 : 0.05;
    const run = (fn) => { e.preventDefault(); Promise.resolve(fn()).catch(reportError); };

    if (mod && (k === "z" || k === "Z")) return run(() => (e.shiftKey ? redo() : undo()));
    if (mod && (k === "y" || k === "Y")) return run(redo);
    if (mod) return;

    switch (k) {
      case "?": return run(shortcutsDialog);
      case "a": return run(addTrack);
      case "/": return run(() => tracks.focusSearch());
      case "n": return run(() => selectNext(1));
      case "p": return run(() => selectNext(-1));
      case ".": return run(() => step(1));
      case ",": return run(() => step(-1));
      case ">": return run(() => step(10));
      case "<": return run(() => step(-10));
      case "Home": return run(() => jumpEnd(-1));
      case "End": return run(() => jumpEnd(1));
      case " ": return run(() => timeline.togglePlay());
      case "f": return run(() => viewport.fit());
      case "g": return run(() => viewport.toggleGround());
      case "v": return run(() => viewport.setMode(S.view.mode === "3d" ? "top" : "3d"));
      case "0": return run(() => viewport.soloSensor(null));
      case "1": case "2": case "3": return run(() => viewport.soloSensor(Number(k) - 1));
    }
    if (!t) return;
    switch (k) {
      case "Enter": return run(accept);
      case "F": return run(() => inspector.flagMenu(document.querySelector(".sec .btn:not(.primary):not(.danger)") || document.body));
      case "Delete": case "Backspace":
        return run(async () => { if (await confirmBox({ title: `Delete track ${t.tid}?`, text: "It is kept in the project for the record and can be restored.", ok: "Delete", danger: true })) remove(); });
      case "m": return run(() => { S.keyframe = S.keyframe === f ? null : f; emit("keyframe"); toast(S.keyframe === null ? "Mark cleared" : `Marked frame ${f}`); });
      case "{": return run(() => edit("rotate track", (x) => ops.rotate(x, null, 0.5 * DEG), { coalesce: true }));
      case "}": return run(() => edit("rotate track", (x) => ops.rotate(x, null, -0.5 * DEG), { coalesce: true }));
      case "i": return run(() => edit("length", (x) => ops.resize(x, "L", x.shape[0] + 0.05), { coalesce: true }));
      case "k": return run(() => edit("length", (x) => ops.resize(x, "L", x.shape[0] - 0.05), { coalesce: true }));
      case "l": return run(() => edit("width", (x) => ops.resize(x, "W", x.shape[1] + 0.05), { coalesce: true }));
      case "j": return run(() => edit("width", (x) => ops.resize(x, "W", x.shape[1] - 0.05), { coalesce: true }));
      case "u": return run(() => edit("height", (x) => ops.resize(x, "H", x.shape[2] + 0.05), { coalesce: true }));
      case "o": return run(() => edit("height", (x) => ops.resize(x, "H", x.shape[2] - 0.05), { coalesce: true }));
      case "X": return run(() => {
        if (S.keyframe === null || S.keyframe === f) return toast("Press M on a good frame first, then go to another good frame");
        if (!ops.has(t, S.keyframe) || !hasBox) return toast("Both the marked frame and this one need a box");
        const [a, b] = [Math.min(S.keyframe, f), Math.max(S.keyframe, f)];
        edit("interpolate range", (x) => ops.interpolate(x, a, b, S.project.frames.filter((g) => g > a && g < b)));
      });
      case "x": return run(() => {
        // a box here: re-derive it from its neighbours; no box here: create one between them
        const fr = ops.framesOf(t);
        const prev = fr.filter((g) => g < f).pop(), next = fr.find((g) => g > f);
        if (prev === undefined || next === undefined) return toast("Needs a box on both sides of this frame");
        edit("interpolate", (x) => ops.interpolate(x, prev, next, [f]));
      });
      case "Y": return run(() => {
        if (S.keyframe === null) return toast("Press M on the first frame to flip first");
        const [a, b] = [Math.min(S.keyframe, f), Math.max(S.keyframe, f)];
        edit("flip range", (x) => ops.flip(x, ops.framesOf(x).filter((g) => g >= a && g <= b)));
      });
    }
    if (!hasBox) return;
    switch (k) {
      case "ArrowLeft": return run(() => edit("nudge", (x) => ops.move(x, [f], -nudge, 0, 0), { coalesce: true }));
      case "ArrowRight": return run(() => edit("nudge", (x) => ops.move(x, [f], nudge, 0, 0), { coalesce: true }));
      case "ArrowUp": return run(() => edit("nudge", (x) => ops.move(x, [f], 0, nudge, 0), { coalesce: true }));
      case "ArrowDown": return run(() => edit("nudge", (x) => ops.move(x, [f], 0, -nudge, 0), { coalesce: true }));
      case "=": return run(() => edit("nudge z", (x) => ops.move(x, [f], 0, 0, 0.05), { coalesce: true }));
      case "-": return run(() => edit("nudge z", (x) => ops.move(x, [f], 0, 0, -0.05), { coalesce: true }));
      case "[": return run(() => edit("rotate", (x) => ops.rotate(x, [f], 0.5 * DEG), { coalesce: true }));
      case "]": return run(() => edit("rotate", (x) => ops.rotate(x, [f], -0.5 * DEG), { coalesce: true }));
      case "y": return run(() => edit("flip", (x) => ops.flip(x, [f])));
    }
  });
}

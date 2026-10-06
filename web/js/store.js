// Application state, navigation, and the edit pipeline.
//
// Edits are optimistic: the change shows immediately, then a serial save queue sends it with the
// version the server last confirmed. A stale version (someone else saved this track) reloads the
// track and says so; a lost connection keeps the queue and retries. Undo/redo work on whole-track
// snapshots, so every operation is reversible the same way.

import { api, ApiError } from "./api.js";
import * as ops from "./ops.js";
import { toast } from "./ui.js";

export const S = {
  app: null,
  project: null,              // project payload (meta, session info, stats, frames, ...)
  tracks: new Map(),          // tid -> summary
  tid: null,
  track: null,                // selected track (full JSON, local truth)
  frame: null,
  view: { mode: "3d", agents: null, ground: false, follow: true, ghost: false, uav: null, shape: "top" },
  list: { filter: "review", sort: "risk", query: "" },
  save: "saved",              // saved | saving | offline | error
  jobs: [],
  playing: false,
  keyframe: null,             // the frame marked with "m" for range tools
  placing: false,             // waiting for a click that places a new vehicle
};

// ------------------------------------------------------------------ events
const subs = new Map();
export function on(evt, fn) { (subs.get(evt) || subs.set(evt, new Set()).get(evt)).add(fn); return () => subs.get(evt).delete(fn); }
export function emit(evt, data) { for (const fn of subs.get(evt) || []) { try { fn(data); } catch (e) { reportError(e); } } }

export function reportError(e) {
  console.error(e);
  api.log("error", String(e && e.message || e), e && e.stack);
}

// ------------------------------------------------------------------ project + list
export async function loadProject(payload) {
  S.project = payload || await api.project();
  const sensors = S.project.meta.sensors || S.project.session.lidar_agents.map((a) => a.id);
  S.view.agents = sensors.slice();
  S.view.uav = S.project.meta.uav_agent ?? (S.project.session.uav_agents[0]?.id ?? null);
  S.frame = S.project.frames[0];
  S.tid = null; S.track = null;
  undoStack.length = 0; redoStack.length = 0;
  await refreshTracks();
  emit("project");
  emit("frame");
}

export async function refreshTracks() {
  const list = await api.tracks();
  S.tracks = new Map(list.map((t) => [t.tid, t]));
  emit("tracks");
}

export async function refreshProject() {
  const p = await api.project();
  S.project.stats = p.stats; S.project.seed_runs = p.seed_runs; S.project.meta = p.meta;
  emit("stats");
}

const RISK = (t) => (t.diag && t.diag.risk !== undefined ? t.diag.risk : t.conf !== null ? 1 - t.conf : 0.5);

export function visibleTracks() {
  const { filter, sort, query } = S.list;
  let rows = [...S.tracks.values()];
  const want = {
    review: (t) => t.status === "unreviewed", flagged: (t) => t.status === "flagged",
    done: (t) => ["accepted", "edited", "new"].includes(t.status), deleted: (t) => t.status === "deleted",
    all: (t) => t.status !== "deleted",
  }[filter] || (() => true);
  rows = rows.filter(want);
  if (query.trim()) {
    const qs = query.trim().toLowerCase();
    rows = rows.filter((t) => String(t.tid).includes(qs) || (t.note || "").toLowerCase().includes(qs)
                              || (t.cls || "").toLowerCase().includes(qs));
  }
  const by = {
    risk: (a, b) => RISK(b) - RISK(a) || a.tid - b.tid,
    id: (a, b) => a.tid - b.tid,
    conf: (a, b) => (a.conf ?? 2) - (b.conf ?? 2) || a.tid - b.tid,
    length: (a, b) => b.n_frames - a.n_frames || a.tid - b.tid,
  }[sort];
  return rows.sort(by);
}

export function counts() {
  const c = { review: 0, flagged: 0, done: 0, deleted: 0, all: 0 };
  for (const t of S.tracks.values()) {
    if (t.status === "deleted") { c.deleted++; continue; }
    c.all++;
    if (t.status === "unreviewed") c.review++;
    else if (t.status === "flagged") c.flagged++;
    else c.done++;
  }
  return c;
}

// ------------------------------------------------------------------ navigation
export async function selectTrack(tid, { frame = null, keepFrame = true } = {}) {
  if (tid === null || tid === undefined) { S.tid = null; S.track = null; emit("select"); return; }
  await flush();
  const t = await api.track(tid);
  if (!pending.has(tid)) serverVersion.set(tid, t.version);    // the server's view is current
  S.tid = tid; S.track = t;
  S.keyframe = null;
  const fr = ops.framesOf(t);
  if (frame !== null && ops.has(t, frame)) S.frame = frame;
  else if (!(keepFrame && ops.has(t, S.frame))) S.frame = ops.nearestFrame(t, S.frame ?? fr[0]);
  emit("select");
  emit("frame");
}

export function setFrame(f) {
  if (f === S.frame || f === null || f === undefined) return;
  S.frame = f;
  emit("frame");
}

export function step(n) {
  const fr = S.track ? ops.framesOf(S.track) : S.project.frames;
  let i = fr.indexOf(S.frame);
  if (i < 0) i = fr.findIndex((g) => g >= S.frame);
  i = Math.max(0, Math.min(fr.length - 1, (i < 0 ? 0 : i) + n));
  setFrame(fr[i]);
}

export function jumpEnd(dir) {
  const fr = S.track ? ops.framesOf(S.track) : S.project.frames;
  setFrame(dir < 0 ? fr[0] : fr[fr.length - 1]);
}

export async function selectNext(dir = 1) {
  const rows = visibleTracks();
  if (!rows.length) return;
  let i = rows.findIndex((t) => t.tid === S.tid);
  i = i < 0 ? 0 : (i + dir + rows.length) % rows.length;
  await selectTrack(rows[i].tid);
}

// ------------------------------------------------------------------ edits
const undoStack = [], redoStack = [];
let lastEdit = { op: null, at: 0 };

export function preview(track) {            // during a drag: show, do not record or save
  S.track = track;
  emit("track", { preview: true });
}

export function edit(op, fn, { coalesce = false } = {}) {
  if (!S.track) return;
  const before = S.track;
  let after;
  try { after = fn(ops.clone(before)); } catch (e) { reportError(e); toast(String(e.message || e), { kind: "err" }); return; }
  if (!after) return;
  const now = performance.now();
  const merge = coalesce && lastEdit.op === op && now - lastEdit.at < 900 && undoStack.length
    && undoStack[undoStack.length - 1].tid === after.tid;
  if (merge) undoStack[undoStack.length - 1].after = after;
  else undoStack.push({ tid: after.tid, op, before, after });
  if (undoStack.length > 200) undoStack.shift();
  redoStack.length = 0;
  lastEdit = { op, at: now };
  commitLocal(after, op, coalesce ? 350 : 0);
}

// a drag ended: record one undo step from the pre-drag state
export function commitDrag(op, before) {
  if (!S.track || !before) return;
  undoStack.push({ tid: S.track.tid, op, before, after: S.track });
  redoStack.length = 0;
  commitLocal(S.track, op, 0);
}

function commitLocal(track, op, delay) {
  S.track = track;
  const sum = S.tracks.get(track.tid);
  if (sum) {
    Object.assign(sum, { status: track.status, shape: track.shape.map((v) => +v.toFixed(2)), cls: track.cls,
                         note: track.note, n_frames: Object.keys(track.poses).length });
  }
  emit("track", {});
  emit("tracks-light");
  queueSave(track, op, delay);
}

export async function undo() {
  const e = undoStack.pop();
  if (!e) return toast("Nothing to undo");
  redoStack.push(e);
  if (S.tid !== e.tid) await selectTrack(e.tid);
  commitLocal(ops.clone(e.before), "undo", 0);
}

export async function redo() {
  const e = redoStack.pop();
  if (!e) return toast("Nothing to redo");
  undoStack.push(e);
  if (S.tid !== e.tid) await selectTrack(e.tid);
  commitLocal(ops.clone(e.after), "redo", 0);
}

// ------------------------------------------------------------------ save queue
const pending = new Map();                 // tid -> { track, op }
let timer = null;
const serverVersion = new Map();           // tid -> last version the server confirmed

function setSave(s) { if (S.save !== s) { S.save = s; emit("save"); } }

function queueSave(track, op, delay) {
  if (!serverVersion.has(track.tid)) serverVersion.set(track.tid, track.version);
  pending.set(track.tid, { track: ops.clone(track), op });
  setSave("saving");
  clearTimeout(timer);
  timer = setTimeout(drain, delay);
}

let draining = null;
function drain() {                         // one drain at a time; callers can await the current one
  if (!draining) draining = drainLoop().finally(() => { draining = null; });
  return draining;
}

async function drainLoop() {
  let failed = false;
  while (pending.size) {
    const [tid, { track, op }] = pending.entries().next().value;
    pending.delete(tid);
    try {
      const saved = await api.saveTrack(track, serverVersion.get(tid), op);
      serverVersion.set(tid, saved.version);
      if (S.track && S.track.tid === tid) { S.track.version = saved.version; S.track.updated = saved.updated; }
      const sum = S.tracks.get(tid);
      if (sum) sum.version = saved.version;
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.data && e.data.current) {
        const cur = e.data.current;
        serverVersion.set(tid, cur.version);
        const sum = S.tracks.get(tid);
        if (sum) { sum.status = cur.status; sum.version = cur.version; }
        if (S.tid === tid) { S.track = cur; emit("track", {}); emit("tracks-light"); }
        toast(`Track ${tid} was changed by someone else — reloaded their version`, { kind: "err", ms: 5000 });
      } else if (e instanceof ApiError && e.status === 0) {
        if (!pending.has(tid)) pending.set(tid, { track, op });
        setSave("offline");
        clearTimeout(timer);
        timer = setTimeout(drain, 2500);             // keep the queue; retry later
        return;
      } else {
        failed = true;
        reportError(e);
        toast(`Could not save track ${tid}: ${e.message}`, { kind: "err", ms: 6000 });
      }
    }
  }
  setSave(failed ? "error" : "saved");
  refreshProject().catch(() => {});
}

export async function flush() {
  clearTimeout(timer);
  await drain();
  if (pending.size && S.save !== "offline") await drain();   // edits made while the last drain ran
}

window.addEventListener("beforeunload", (e) => {
  if (pending.size || draining) { e.preventDefault(); e.returnValue = ""; }
});

// ------------------------------------------------------------------ review actions
export async function accept() {
  if (!S.track) return;
  if (["unreviewed", "flagged"].includes(S.track.status)) edit("accept", (t) => ops.setStatus(t, "accepted"));
  await selectNextInQueue();
}

// every verdict moves on to the next track in the queue
export async function flag(note) { edit("flag", (t) => ops.setStatus(t, "flagged", note)); await selectNextInQueue(); }
export async function remove() { edit("delete", (t) => ops.setStatus(t, "deleted")); await selectNextInQueue(); }
export function restore() { edit("restore", (t) => ops.setStatus(t, "unreviewed")); }

async function selectNextInQueue() {
  const rows = visibleTracks().filter((t) => t.tid !== S.tid);
  const cur = visibleTracks().findIndex((t) => t.tid === S.tid);
  if (!rows.length) { emit("select"); return toast("Queue empty — nice work"); }
  const next = cur >= 0 && cur < rows.length ? rows[cur] : rows[0];
  await selectTrack(next.tid);
}

export function lastSaved(tid) { return serverVersion.get(tid); }

// ------------------------------------------------------------------ placing a new vehicle
// "Add" waits for one click: on the map (3D / Top view) or on the drone image. Esc cancels.
let placeResolve = null;
export function startPlacing() {
  cancelPlacing();
  S.placing = true;
  emit("placing");
  return new Promise((resolve) => { placeResolve = resolve; });
}
export function finishPlacing(xy) {
  const r = placeResolve;
  placeResolve = null;
  S.placing = false;
  emit("placing");
  if (r) r(xy);
}
export function cancelPlacing() { if (S.placing) finishPlacing(null); }

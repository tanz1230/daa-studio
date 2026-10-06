// DAA Studio — application shell.

import { api, user } from "./api.js";
import { S, on, emit, loadProject, refreshTracks, refreshProject, selectTrack, flush, reportError, startPlacing } from "./store.js";
import { el, icon, toast, popover, fmt } from "./ui.js";
import { installKeys } from "./keys.js";
import { renderStart } from "./views/start.js";
import { createTrackList } from "./views/tracks.js";
import { createViewport } from "./views/viewport.js";
import { createShapePanel } from "./views/shape.js";
import { createUavPanel } from "./views/uav.js";
import { createTimeline } from "./views/timeline.js";
import { createInspector } from "./views/inspector.js";
import { seedDialog, exportDialog, shortcutsDialog, askName } from "./views/dialogs.js";

const root = document.getElementById("app");

window.addEventListener("error", (e) => reportError(e.error || e.message));
// a clicked button gives focus back, so Space / Enter always reach the shortcuts
document.addEventListener("click", (e) => {
  const b = e.target.closest && e.target.closest("button");
  if (b && !b.closest(".scrim")) b.blur();
});
window.addEventListener("unhandledrejection", (e) => {
  reportError(e.reason);
  toast(String((e.reason && e.reason.message) || e.reason), { kind: "err" });
});

async function boot() {
  let app;
  try { app = await api.app(); } catch (e) {
    root.replaceChildren(el("div.start", {}, el("div.empty-card", {}, "The DAA Studio server is not reachable. Start it with ", el("span.mono", {}, "python -m studio serve"), ".")));
    return;
  }
  S.app = app;
  if (!user.get()) await askName(app.user || "");
  if (app.project) openEditor(app.project);
  else renderStart(root, { app, onOpen: (payload, job) => openEditor(payload, job) });
}

// ============================================================================ editor
let built = false, views = null;

function openEditor(payload, job = null) {
  if (!built) buildEditor();
  loadProject(payload).then(() => {
    renderHeader();
    if (job) { S.jobs = [job]; emit("jobs"); }
    pollJobs(true);
  }).catch(reportError);
}

function buildEditor() {
  built = true;
  const header = el("div.header", { id: "hdr" });
  const left = el("div.panel.side-l");
  const vp = el("div.panel-b");
  const viewportPanel = el("div.panel", { style: { position: "relative" } }, vp);
  const uavPanel = el("div.panel");
  const shapePanel = el("div.panel");
  const right = el("div.panel.side-r");
  const timeline = el("div.timeline");
  root.replaceChildren(el("div.shell", {}, header,
    el("div.body", {}, left, el("div.center", {}, viewportPanel, el("div.lower", {}, uavPanel, shapePanel)), right),
    timeline));

  const viewport = createViewport(vp);
  const tracks = createTrackList(left, { onAdd: addTrack });
  const uav = createUavPanel(uavPanel);
  createShapePanel(shapePanel);
  const tl = createTimeline(timeline);
  const inspector = createInspector(right);
  views = { viewport, tracks, timeline: tl, inspector, uav };
  window.__daa = { S, views };                 // for the UI smoke test and debugging
  installKeys({ ...views, addTrack });
  on("save", renderSave);
  on("stats", renderHeader);
  on("jobs", renderJob);
}

function renderHeader() {
  const h = document.getElementById("hdr");
  if (!h || !S.project) return;
  const meta = S.project.meta;
  document.title = `${meta.name} — DAA Studio`;
  h.replaceChildren(
    el("div.brand", {}, icon("logo"), "DAA Studio"),
    el("div.crumb", {}, el("b", {}, meta.name),
      meta.name !== S.project.session.name ? el("span", {}, `  ·  ${S.project.session.name}`) : null,
      el("span", {}, `  ·  ${S.project.frames.length} frames`)),
    el("div", { id: "jobslot" }),
    el("div.spacer"),
    el("div.savestate", { id: "savestate" }),
    el("button.btn", { onclick: () => seedDialog({ project: S.project, onStarted: (job) => { S.jobs = [job]; emit("jobs"); pollJobs(true); } }) }, icon("sparkle"), "Seed"),
    el("button.btn", { onclick: () => exportDialog({ project: S.project }) }, icon("download"), "Export"),
    el("button.btn.ghost", { title: "Your name (recorded with every edit)", onclick: async () => { await askName(user.get()); renderHeader(); } }, user.get() || "anonymous"),
    el("button.btn.icon.ghost", { title: "Keyboard shortcuts (?)", onclick: shortcutsDialog }, icon("keyboard")),
    el("button.btn.icon.ghost", { title: "More", onclick: (e) => popover(e.currentTarget, [
      { label: "Close project", icon: "close", fn: closeProject },
      { label: "Reload tracks", icon: "rotate", fn: () => refreshTracks() },
    ]) }, icon("more")));
  renderSave();
  renderJob();
}

function renderSave() {
  const s = document.getElementById("savestate");
  if (!s) return;
  const txt = { saved: "All saved", saving: "Saving…", offline: "Offline — retrying", error: "Save failed" }[S.save];
  s.className = "savestate " + S.save;
  s.replaceChildren(el("span.dot"), txt);
}

const doneAt = new Map();                     // seed job id -> when this page saw it finish
const PHASE = { reading: "Reading detections", crops: "Cropping LiDAR", refine: "DAA refining", import: "Loading" };

function renderJob() {
  const slot = document.getElementById("jobslot");
  if (!slot) return;
  const seeds = S.jobs.filter((j) => j.kind === "seed");
  const job = seeds.find((j) => j.state === "running")
    || seeds.filter((j) => doneAt.has(j.id) && Date.now() - doneAt.get(j.id) < 10000).pop();
  if (!job) { slot.replaceChildren(); return; }
  const pct = job.total ? (100 * job.done) / job.total : 0;
  const n = job.imported ? job.imported.added + job.imported.replaced : 0;
  const label = job.state === "running"
    ? `${PHASE[job.phase] || "Seeding"} · ${Math.round(pct)}%${job.eta_s ? ` · ${fmt.dur(job.eta_s)} left` : ""}`
    : { done: `Seeds ready · ${n} tracks`, cancelled: "Seeding cancelled" }[job.state] || "Seeding failed";
  slot.replaceChildren(el("div.jobpill" + (job.state === "done" ? ".done" : job.state === "running" ? "" : ".failed"),
    { title: job.state === "running" ? "Click to cancel" : job.error || "",
      onclick: (e) => job.state === "running" && popover(e.currentTarget, [{ label: "Cancel seeding", icon: "x", fn: () => api.cancelJob(job.id).then(() => pollJobs(true)) }]) },
    job.state === "running" ? el("div.spinner") : icon(job.state === "done" ? "check" : "alert"),
    label, job.state === "running" ? el("span.bar", {}, el("i", { style: { width: `${pct}%` } })) : null));
}

let pollTimer = null, lastTracksRefresh = 0;
async function pollJobs(force = false) {
  clearTimeout(pollTimer);
  try {
    const jobs = await api.jobs();
    const was = new Set(S.jobs.filter((j) => j.kind === "seed" && j.state === "running").map((j) => j.id));
    S.jobs = jobs;
    let finished = false;
    for (const j of jobs) {
      if (j.kind !== "seed" || !was.has(j.id) || j.state === "running") continue;
      finished = true;
      doneAt.set(j.id, Date.now());
      setTimeout(renderJob, 10500);
      if (j.state === "done") toast(`Seeding finished — ${j.imported.added + j.imported.replaced} tracks ready`);
      else if (j.state === "failed") toast(`Seeding failed: ${(j.error || "").trim().split("\n").pop()}`, { kind: "err", ms: 9000 });
    }
    const running = jobs.some((j) => j.kind === "seed" && j.state === "running");
    if (finished || (running && Date.now() - lastTracksRefresh > 3000)) {
      lastTracksRefresh = Date.now();
      await refreshTracks(); await refreshProject();
    }
    emit("jobs");
    if (running || force) pollTimer = setTimeout(() => pollJobs(), running ? 1500 : 4000);
  } catch (e) { pollTimer = setTimeout(() => pollJobs(), 5000); }
}

async function addTrack() {
  // one click places it (map or drone image); heading copied from the nearest vehicle, which is
  // almost always parallel (parking rows, lanes); z is set on the ground by the server
  const xy = await startPlacing();
  if (!xy) return;
  let yaw = 0;
  try {
    let best = 15;
    for (const b of await api.frameBoxes(S.frame)) {
      const d = Math.hypot(b.box[0] - xy[0], b.box[1] - xy[1]);
      if (d < best) { best = d; yaw = b.box[6]; }
    }
  } catch (e) { /* default heading */ }
  const t = { tid: 0, cls: "Car", shape: [4.6, 1.85, 1.55], poses: { [S.frame]: [xy[0], xy[1], null, yaw] },
              status: "new", flags: {}, note: "", seed: { kind: "manual" }, version: 0 };
  try {
    const made = await api.addTrack(t);
    await refreshTracks();
    await selectTrack(made.tid, { frame: S.frame });
    toast(`Added track ${made.tid} — fit it here, then Extend it to more frames`);
  } catch (e) { toast(e.message, { kind: "err" }); }
}

async function closeProject() {
  await flush();
  await api.closeProject();
  location.reload();
}

boot();

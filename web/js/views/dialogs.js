// Dialogs: file browser, new-project wizard, seeding, export, shortcuts, annotator name.

import { api, user } from "../api.js";
import { el, icon, modal, toast, confirmBox, fmt } from "../ui.js";

// ============================================================================ file browser
export function pickPath({ title = "Choose", start = "", want = "dir", hint = "" } = {}) {
  return new Promise((resolve) => {
    let cur = start, sel = null, listing = null;
    const pathInp = el("input.input.mono", { value: start, spellcheck: false });
    const list = el("div.fsl");
    const status = el("div.muted", { style: { fontSize: "12px", minHeight: "18px" } });
    const choose = el("button.btn.primary", { onclick: () => done(sel || (want === "dir" ? cur : null)) }, "Choose");
    const m = modal({
      title, subtitle: hint, wide: true,
      body: [el("div.pathrow", {}, el("button.btn.icon", { title: "Up", onclick: () => go(listing && listing.parent) }, icon("up")), pathInp,
        el("button.btn", { onclick: () => go(pathInp.value) }, "Go")), list, status],
      footer: [el("button.btn", { onclick: () => done(null) }, "Cancel"), choose],
      onClose: () => resolve(null),
    });
    const done = (v) => { resolve(v); m.close(); };
    pathInp.addEventListener("keydown", (e) => { if (e.key === "Enter") go(pathInp.value); });

    async function go(path) {
      if (!path) return;
      try { listing = await api.fs(path); } catch (e) { status.textContent = e.message; return; }
      cur = listing.path; sel = null; pathInp.value = cur;
      status.textContent = listing.is_session ? "This folder is an OpenCOOD session." : listing.is_project ? "This folder is a DAA Studio project." : "";
      list.replaceChildren(...listing.entries.filter((e) => e.dir || want === "file").map((e) => {
        const full = `${cur.replace(/\/$/, "")}/${e.name}`;
        const r = el("div.fsr", {}, icon(e.dir ? "folder" : "file"), e.name,
          e.session ? el("span.tag", {}, "session") : e.project ? el("span.tag", {}, "project") : null);
        r.addEventListener("click", () => {
          for (const x of list.children) x.classList.remove("on");
          r.classList.add("on"); sel = full;
        });
        r.addEventListener("dblclick", () => { if (e.dir && !(want === "dir" && (e.session || e.project))) go(full); else done(full); });
        return r;
      }));
      if (!listing.entries.length) list.append(el("div.empty", {}, "Empty folder"));
    }
    go(start || "~");
  });
}

function pathField(label, value, opts) {
  const inp = el("input.input.mono", { value: value || "", spellcheck: false, placeholder: opts.placeholder || "" });
  const wrap = el("div.field", {}, el("label", {}, label),
    el("div.pathrow", {}, inp, el("button.btn", { onclick: async () => {
      const v = await pickPath({ ...opts, start: inp.value ? inp.value.replace(/\/[^/]*$/, "") || inp.value : opts.start });
      if (v) { inp.value = v; inp.dispatchEvent(new Event("change")); }
    } }, icon("folder"), "Browse")));
  return { wrap, inp };
}

// ============================================================================ seeding options (shared)
const SEED_MODES = [
  ["daa", "DAA — UAV detections refined with ground LiDAR", "Amodal 3D boxes from the drone's detections, fitted to every LiDAR you select. The paper's adopted configuration.", true],
  ["uav", "UAV detections only", "The drone's boxes as they are, with default heights. Fast; for manual labeling with UAV assistance."],
  ["session", "The session's existing labels", "Review or correct the annotations already in the dataset."],
  ["import", "Import a label CSV", "Any labels in the DAA CSV format, e.g. from another run or tool."],
  ["empty", "Start empty", "No seeds. Add vehicles by hand."],
];

function seedOptions(state, onChange) {
  return el("div", { style: { display: "flex", flexDirection: "column", gap: "8px" } }, SEED_MODES.map(([k, t, d, rec]) => {
    const o = el("label.opt" + (state.mode === k ? ".on" : ""), {},
      el("input", { type: "radio", name: "seedmode", checked: state.mode === k, onchange: () => { state.mode = k; onChange(); } }),
      el("div", {}, el("div.t", {}, t, rec ? el("span.rec", {}, "Recommended") : null), el("div.d", {}, d)));
    return o;
  }));
}

// ============================================================================ new project wizard
export function newProjectWizard({ onCreated }) {
  const st = { step: 0, session: "", info: null, err: null, checking: false, mode: "daa", detections: "", labels: "",
               sensors: [], uav: null, name: "", project: "", busy: false,
               auto: { detections: true, name: true, project: true } };     // false once the user edits it
  const m = modal({ title: "New project", wide: true });
  const titles = ["Session", "Seeds", "Sensors", "Project"];
  let nextBtn = null, sessionNote = null, vTimer = null, vSeq = 0;

  const needsDets = () => st.mode === "daa" || st.mode === "uav";
  const canNext = () => (st.step === 0 ? !!st.info
    : st.step === 1 ? (needsDets() ? !!st.detections : st.mode === "import" ? !!st.labels : true)
    : st.step === 2 ? st.sensors.length > 0 : !!st.project && !!st.name.trim());
  const updateNext = () => { if (nextBtn) nextBtn.disabled = !canNext() || st.busy; };

  // ---- step 1: the session is checked as you type, without re-rendering the field you type in
  function sessionNoteContent() {
    if (st.checking) return el("div.infobox", {}, el("div.row", {}, el("div.spinner"), "Checking the session…"));
    if (st.err) return el("div.infobox.bad", {}, st.err);
    if (st.info) {
      const i = st.info;
      return el("div.infobox.ok", {}, el("b", {}, i.name), ` — ${i.frames} frames (${i.first}–${i.last}) · LiDAR: `,
        i.lidar_agents.map((a) => a.name).join(", "), " · UAV: ", i.uav_agents.map((a) => a.name).join(", ") || "none");
    }
    return el("div.infobox", {}, "Pick the session you want to label. The dataset is never modified — labels live in the project.");
  }
  const showSessionNote = () => { if (sessionNote) sessionNote.replaceChildren(sessionNoteContent()); updateNext(); };

  async function validateSession(path) {
    const seq = ++vSeq;
    Object.assign(st, { session: path, info: null, err: null, checking: !!path });
    showSessionNote();
    if (!path) return;
    try {
      const [info, sug] = await Promise.all([api.sessionInfo(path), api.suggest(path)]);
      if (seq !== vSeq) return;
      st.info = info;
      st.sensors = info.lidar_agents.map((a) => a.id);
      st.uav = info.uav_agents[0]?.id ?? null;
      if (st.auto.detections) st.detections = sug.detections || "";
      if (st.auto.project && sug.project) st.project = sug.project;
      if (st.auto.name) st.name = info.name;
    } catch (e) {
      if (seq !== vSeq) return;
      st.err = e.message;
    }
    st.checking = false;
    showSessionNote();
  }

  function render() {
    const body = [], foot = [];
    if (st.step === 0) {
      const f = pathField("Session folder", st.session, { title: "Choose a session", want: "dir", hint: "An OpenCOOD session folder, e.g. …/agi_coop/take_3" });
      f.inp.addEventListener("input", () => { clearTimeout(vTimer); vTimer = setTimeout(() => validateSession(f.inp.value.trim()), 450); });
      f.inp.addEventListener("change", () => {
        clearTimeout(vTimer);
        if (f.inp.value.trim() !== st.session || (!st.info && !st.checking)) validateSession(f.inp.value.trim());
      });
      sessionNote = el("div", {}, sessionNoteContent());
      body.push(f.wrap, sessionNote);
      setTimeout(() => f.inp.focus(), 30);
    } else if (st.step === 1) {
      body.push(seedOptions(st, render));
      if (needsDets()) {
        const f = pathField("UAV detections (Stage-1 CSV, map frame)", st.detections, { title: "Choose UAV detections", want: "file" });
        const set = () => { st.detections = f.inp.value.trim(); st.auto.detections = false; updateNext(); };
        f.inp.addEventListener("input", set); f.inp.addEventListener("change", set);
        body.push(f.wrap);
        if (!st.detections) body.push(el("div.infobox", {}, "No detections were found next to this session. Choose the CSV exported from the UAV detector (one row per vehicle and frame, in the LiDAR map frame)."));
      } else if (st.mode === "import") {
        const f = pathField("Label CSV", st.labels, { title: "Choose a label CSV", want: "file" });
        const set = () => { st.labels = f.inp.value.trim(); updateNext(); };
        f.inp.addEventListener("input", set); f.inp.addEventListener("change", set);
        body.push(f.wrap);
      }
    } else if (st.step === 2) {
      body.push(el("div.field", {}, el("label", {}, "Ground LiDARs" + (st.mode === "daa" ? " — used by DAA and shown in the editor" : "")),
        el("div.row", { style: { gap: "18px" } }, st.info.lidar_agents.map((a) => el("label.check", {},
          el("input", { type: "checkbox", checked: st.sensors.includes(a.id), onchange: (e) => {
            st.sensors = e.target.checked ? [...st.sensors, a.id].sort() : st.sensors.filter((x) => x !== a.id);
            updateNext();
          } }), a.name)))));
      if (st.mode === "daa") body.push(el("div.infobox", {}, "DAA fits every box to all the LiDARs you select: more viewpoints see more of each vehicle."));
      if (st.info.uav_agents.length) body.push(el("div.field", {}, el("label", {}, "UAV view in the editor"),
        el("div.row", { style: { gap: "18px" } }, st.info.uav_agents.map((a) => el("label.check", {},
          el("input", { type: "radio", name: "uavcam", checked: st.uav === a.id, onchange: () => { st.uav = a.id; } }), a.name)))));
    } else {
      const n = el("input.input", { value: st.name, spellcheck: false });
      n.addEventListener("input", () => { st.name = n.value; st.auto.name = false; updateNext(); });
      const f = pathField("Project folder", st.project, { title: "Where to keep the project", want: "dir" });
      const set = () => { st.project = f.inp.value.trim(); st.auto.project = false; updateNext(); };
      f.inp.addEventListener("input", set); f.inp.addEventListener("change", set);
      body.push(el("div.field", {}, el("label", {}, "Name"), n), f.wrap,
        el("div.infobox", {}, "The project folder holds labels, edit history, seed runs and exports. Nothing is written into the dataset."));
      if (st.err) body.push(el("div.infobox.bad", {}, st.err));
    }
    if (st.step > 0) foot.push(el("button.btn.left", { onclick: () => { st.step--; st.err = null; render(); } }, "Back"));
    foot.push(el("button.btn", { onclick: () => m.close() }, "Cancel"));
    nextBtn = st.step < 3
      ? el("button.btn.primary", { onclick: () => { if (canNext()) { st.step++; render(); } } }, "Continue")
      : el("button.btn.primary", { onclick: () => canNext() && !st.busy && create() },
        st.busy ? [el("div.spinner"), "Creating…"] : st.mode === "empty" ? "Create project" : "Create & seed");
    foot.push(nextBtn);
    updateNext();
    m.render({ title: "New project", subtitle: `Step ${st.step + 1} of 4 — ${titles[st.step]}`, body, footer: foot, steps: titles.map((_, i) => i <= st.step) });
  }

  async function create() {
    st.busy = true; st.err = null; render();
    try {
      const payload = await api.createProject({ path: st.project, session: st.session, name: st.name.trim(), sensors: st.sensors, uav_agent: st.uav });
      let job = null;
      if (st.mode !== "empty") {
        try { job = await api.seed({ mode: st.mode, detections: st.detections, labels: st.labels, sensors: st.sensors }); }
        catch (e) { toast(`Project created, but seeding did not start: ${e.message}`, { kind: "err", ms: 8000 }); }
      }
      m.close();
      onCreated(payload, job);
    } catch (e) {
      st.busy = false; st.err = e.message; render();
    }
  }
  render();
}

// ============================================================================ seed dialog (open project)
export function seedDialog({ project, onStarted }) {
  const st = { mode: "daa", detections: "", labels: "", sensors: (project.meta.sensors || []).slice() };
  api.suggest(project.meta.session).then((s) => { if (s.detections && !st.detections) { st.detections = s.detections; render(); } }).catch(() => {});
  const m = modal({ title: "Generate seeds", wide: true });
  function render() {
    const body = [seedOptions(st, render)];
    if (st.mode === "daa" || st.mode === "uav") {
      const f = pathField("UAV detections (Stage-1 CSV, map frame)", st.detections, { title: "Choose UAV detections", want: "file" });
      f.inp.addEventListener("change", () => { st.detections = f.inp.value.trim(); render(); });
      body.push(f.wrap);
    } else if (st.mode === "import") {
      const f = pathField("Label CSV", st.labels, { title: "Choose a label CSV", want: "file" });
      f.inp.addEventListener("change", () => { st.labels = f.inp.value.trim(); render(); });
      body.push(f.wrap);
    }
    if (st.mode !== "empty") body.push(el("div.infobox", {}, "Tracks you have already reviewed, edited, flagged or added are kept exactly as they are; only untouched tracks take the new seeds."));
    const ready = st.mode === "empty" ? false : st.mode === "import" ? !!st.labels : st.mode === "session" ? true : !!st.detections;
    m.render({ title: "Generate seeds", subtitle: `Session ${project.session.name}`, body,
      footer: [el("button.btn", { onclick: () => m.close() }, "Cancel"),
        el("button.btn.primary", { disabled: !ready, onclick: start }, icon("sparkle"), "Start")] });
  }
  async function start() {
    try { const job = await api.seed({ mode: st.mode, detections: st.detections, labels: st.labels, sensors: st.sensors }); m.close(); onStarted(job); }
    catch (e) { toast(e.message, { kind: "err" }); }
  }
  render();
}

// ============================================================================ export
export function exportDialog({ project }) {
  const st = { format: "yaml", scope: "reviewed", into: false, out: "", running: null };
  const m = modal({ title: "Export labels", wide: true });
  const defOut = () => `${project.path}/exports/${st.format === "yaml" ? "opencood" : "labels.csv"}`;
  function render() {
    const s = project.stats.by_status || {};
    const reviewed = (s.accepted || 0) + (s.edited || 0) + (s.new || 0);
    const all = reviewed + (s.unreviewed || 0) + (s.flagged || 0);
    const body = [
      el("div.field", {}, el("label", {}, "Format"), el("div.seg", {},
        el("button" + (st.format === "yaml" ? ".on" : ""), { onclick: () => { st.format = "yaml"; render(); } }, "OpenCOOD YAML"),
        el("button" + (st.format === "csv" ? ".on" : ""), { onclick: () => { st.format = "csv"; st.into = false; render(); } }, "DAA CSV"))),
      el("div.field", {}, el("label", {}, "Tracks"),
        el("label.opt" + (st.scope === "reviewed" ? ".on" : ""), {}, el("input", { type: "radio", name: "scope", checked: st.scope === "reviewed", onchange: () => { st.scope = "reviewed"; render(); } }),
          el("div", {}, el("div.t", {}, `Reviewed only (${reviewed})`), el("div.d", {}, "Accepted, edited and hand-added tracks. Human-verified labels."))),
        el("label.opt" + (st.scope === "all" ? ".on" : ""), {}, el("input", { type: "radio", name: "scope", checked: st.scope === "all", onchange: () => { st.scope = "all"; render(); } }),
          el("div", {}, el("div.t", {}, `Everything except deleted (${all})`), el("div.d", {}, "Also the tracks no one has reviewed yet, exactly as seeded (e.g. DAA pseudo labels).")))),
    ];
    if (!st.into) {
      const f = pathField(st.format === "yaml" ? "Output folder" : "Output file", st.out || defOut(), { title: "Choose where to export", want: "dir" });
      f.inp.addEventListener("change", () => { st.out = f.inp.value.trim(); });
      f.inp.addEventListener("input", () => { st.out = f.inp.value.trim(); });
      body.push(f.wrap);
    }
    if (st.format === "yaml") body.push(el("label.check", {}, el("input", { type: "checkbox", checked: st.into, onchange: (e) => { st.into = e.target.checked; render(); } }),
      "Write into the dataset itself (each file is backed up once before its first overwrite)"));
    if (st.running) body.push(el("div.infobox", {}, el("div.row", {}, el("div.spinner"), "Exporting…")));
    if (st.result) body.push(el("div.infobox.ok", {}, `Done — ${st.result.tracks} tracks · ${st.result.yaml_files ?? st.result.rows} ${st.result.yaml_files !== undefined ? "files" : "rows"}`, el("br"), el("span.mono", {}, st.result.target)));
    m.render({ title: "Export labels", subtitle: project.meta.name, body,
      footer: [el("button.btn", { onclick: () => m.close() }, st.result ? "Close" : "Cancel"),
        st.result ? null : el("button.btn.primary", { disabled: !!st.running, onclick: run }, icon("download"), st.into ? "Write into dataset" : "Export")] });
  }
  async function run() {
    if (st.into && !(await confirmBox({ title: "Write into the dataset?", text: "The session's label YAMLs will be overwritten with this project's labels. Each file is backed up once to the project's backups/ folder.", ok: "Write into dataset", danger: true }))) return;
    const statuses = st.scope === "all" ? ["unreviewed", "accepted", "edited", "flagged", "new"] : ["accepted", "edited", "new"];
    try {
      let job = await api.exportLabels({ format: st.format, statuses, out: st.out || defOut(), apply: st.into });
      st.running = job.id; render();
      while (job.state === "running") { await new Promise((r) => setTimeout(r, 700)); job = await api.job(job.id); }
      st.running = null;
      if (job.state === "failed") throw new Error(job.error || "export failed");
      st.result = job.result; render();
    } catch (e) { st.running = null; render(); toast(e.message, { kind: "err" }); }
  }
  render();
}

// ============================================================================ shortcuts
export const KEYMAP = [
  ["Review", [["Accept & next", "↵"], ["Flag", "⇧ F"], ["Delete track", "Del"], ["Next / previous track", "N / P"], ["Add a missed vehicle", "A"], ["Search tracks", "/"]]],
  ["Frames", [["Next / previous frame", ". / ,"], ["±10 frames", "> / <"], ["First / last frame", "Home / End"], ["Play / pause", "Space"]]],
  ["This frame", [["Nudge 5 cm (⇧ 20 cm)", "← → ↑ ↓"], ["Rotate ∓0.5°", "[ / ]"], ["Raise / lower 5 cm", "= / -"], ["Interpolate / create from neighbours", "X"], ["Flip heading", "Y"]]],
  ["Whole track", [["Rotate ∓0.5°", "{ / }"], ["Length / width / height", "I K · J L · U O"], ["Mark keyframe", "M"], ["Interpolate mark → here", "⇧ X"], ["Flip mark → here", "⇧ Y"]]],
  ["View", [["3D / Top", "V"], ["Fit to vehicle", "F"], ["Ground points", "G"], ["Solo sensor / all", "1 2 3 / 0"]]],
  ["Edit", [["Undo", "Ctrl Z"], ["Redo", "Ctrl ⇧ Z"], ["This help", "?"]]],
];

export function shortcutsDialog() {
  modal({ title: "Keyboard shortcuts", wide: true,
    body: el("div.keys", {}, KEYMAP.map(([h, rows]) => [el("h4", {}, h), rows.map(([l, k]) => el("div.k", {}, el("span", {}, l), el("span.kbd", {}, k)))])),
    footer: [el("button.btn.primary", { onclick: (e) => e.target.closest(".scrim").remove() }, "Got it")] });
}

// ============================================================================ annotator name
export function askName(current = "") {
  return new Promise((resolve) => {
    const inp = el("input.input", { value: current, placeholder: "e.g. Jordan", spellcheck: false });
    const save = () => { user.set(inp.value.trim()); resolve(inp.value.trim()); m.close(); };
    inp.addEventListener("keydown", (e) => { if (e.key === "Enter") save(); });
    const m = modal({ title: "Your name", subtitle: "Recorded with every edit, so reviews can be traced.",
      body: el("div.field", {}, el("label", {}, "Name"), inp),
      footer: [el("button.btn.primary", { onclick: save }, "Save")], onClose: () => resolve(user.get()) });
    setTimeout(() => { inp.focus(); inp.select(); }, 50);
  });
}

export { fmt };

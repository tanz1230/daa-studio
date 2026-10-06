// Start screen: what the product does, recent projects, new / open.

import { api } from "../api.js";
import { el, icon, toast, fmt } from "../ui.js";
import { newProjectWizard, pickPath } from "./dialogs.js";

export function renderStart(root, { app, onOpen }) {
  const recent = app.recent || [];
  const open = async (path) => {
    try { onOpen(await api.openProject(path)); } catch (e) { toast(e.message, { kind: "err" }); }
  };
  const feature = (ic, t, d) => el("div.feature", {}, el("div.t", {}, icon(ic), t), el("p", {}, d));
  root.replaceChildren(el("div.start", {}, el("div.start-inner", {},
    el("div.hero", {}, icon("logo"), el("h1", {}, "DAA Studio")),
    el("p.tagline", {}, "UAV-assisted 3D labeling for ground LiDAR"),
    el("div.cards3", {},
      feature("drone", "Seeded from the sky", "The drone's detections give every vehicle a box — even where the LiDAR sees one side or nothing."),
      feature("sparkle", "Refined by DAA", "Training-free EM refinement fits the boxes to every ground LiDAR: amodal, one shape per vehicle."),
      feature("check", "Review what matters", "A queue ordered by risk, the drone's view beside the LiDAR, every edit versioned and traceable.")),
    el("div.section-h", {}, el("h2", {}, "Projects"),
      el("div.row", {},
        el("button.btn", { onclick: async () => { const p = await pickPath({ title: "Open a project", want: "dir", hint: "Pick a DAA Studio project folder (it holds project.json)." }); if (p) open(p); } }, icon("folder"), "Open…"),
        el("button.btn.primary", { onclick: () => newProjectWizard({ onCreated: (payload, job) => onOpen(payload, job) }) }, icon("plus"), "New project"))),
    recent.length
      ? el("div.recent", {}, recent.map((r) => el("div.rrow", { onclick: () => open(r.path) },
          el("div", { style: { minWidth: 0 } }, el("div.name", {}, r.name), el("div.path", {}, r.path)),
          el("div", {}, el("div.meter", { style: { height: "4px" } }, el("i", { style: { width: `${r.tracks ? (100 * r.reviewed) / r.tracks : 0}%`, background: "#4f9c66" } })),
            el("div.muted", { style: { fontSize: "12px", marginTop: "4px" } }, `${r.reviewed} of ${r.tracks} reviewed`)),
          el("div.when", {}, fmt.when(r.opened)))))
      : el("div.empty-card", {}, "No projects yet. Create one to start labeling a session."))));
}

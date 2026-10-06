// Track list: the review queue. Ordered by risk by default (low DAA confidence, few LiDAR points,
// one-sided evidence, large move from the UAV box), so human time goes where the seeds are weakest.

import * as ops from "../ops.js";
import { S, on, visibleTracks, counts, selectTrack, reportError } from "../store.js";
import { el, icon, statusGlyph, STATUS } from "../ui.js";

const FILTERS = [["review", "To review"], ["flagged", "Flagged"], ["done", "Done"], ["all", "All"], ["deleted", "Deleted"]];
const SORTS = [["risk", "Review priority"], ["id", "Track ID"], ["conf", "Lowest confidence"], ["length", "Longest track"]];

export function createTrackList(host, { onAdd } = {}) {
  const search = el("input.input", { placeholder: "Search id, class or note", spellcheck: false, id: "track-search" });
  const chips = el("div.chips");
  const sortSel = el("select.select", { style: { height: "26px", fontSize: "12px", width: "auto" } },
    ...SORTS.map(([k, l]) => el("option", { value: k }, l)));
  const head = el("div.panel-h", {}, el("span.title", {}, "Tracks"), el("span.grow"),
    el("button.btn.sm.ghost", { title: "Add a vehicle the seeds missed (A)", onclick: () => onAdd && onAdd() }, icon("plus"), "Add"));
  const progress = el("div.progressline");
  const banner = el("div");
  const list = el("div.list", { tabindex: 0 });
  const tools = el("div.tl-tools", {}, el("div.search", {}, icon("search"), search), chips,
    el("div.row", {}, el("span.muted", { style: { fontSize: "12px" } }, "Sort"), sortSel));
  host.append(head, progress, tools, banner, list);

  search.addEventListener("input", () => { S.list.query = search.value; renderList(); });
  sortSel.addEventListener("change", () => { S.list.sort = sortSel.value; renderList(); });

  function renderChips() {
    const c = counts();
    chips.replaceChildren(...FILTERS.map(([k, l]) => el("button.chip" + (S.list.filter === k ? ".on" : ""),
      { onclick: () => { S.list.filter = k; renderChips(); renderList(); } }, l, el("span.n", {}, c[k]))));
  }

  function renderProgress() {
    const st = S.project?.stats;
    if (!st) return;
    const by = st.by_status || {};
    const total = Math.max(1, st.total);
    const seg = (k, color) => el("i", { style: { width: `${(100 * (by[k] || 0)) / total}%`, background: color }, title: `${STATUS[k].label}: ${by[k] || 0}` });
    progress.replaceChildren(
      el("div.meta", {}, el("span", {}, el("b", {}, st.reviewed), ` of ${st.total} reviewed`),
        el("span", {}, `${Math.round((100 * st.reviewed) / total)}%`)),
      el("div.meter", {}, seg("accepted", "#4f9c66"), seg("edited", "#8b6fd4"), seg("new", "#4f9fb2"), seg("flagged", "#d39a4e")));
  }

  function row(t, pinned = false) {
    const conf = t.conf;
    return el("div.trow" + (t.tid === S.tid ? ".sel" : "") + (pinned ? ".pinned" : ""),
      { "data-tid": t.tid, title: pinned ? "Selected — not in this filter" : null, onclick: () => selectTrack(t.tid).catch(reportError) },
      statusGlyph(t.status),
      el("div", { style: { overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis" } },
        el("span.id", {}, `#${t.tid}`), el("span.sub", {}, `${t.cls === "Car" ? "" : t.cls + " · "}${t.n_frames} fr`)),
      el("div.right", {},
        ["daa", "uav-fallback"].includes(t.seed) && conf !== null && conf !== undefined
          ? [el("span.confbar", { title: `DAA confidence ${conf.toFixed(2)}` }, el("i", { style: { width: `${Math.round(100 * Math.min(1, conf))}%` } })), el("span", {}, conf.toFixed(2))]
          : el("span", {}, { manual: "manual", uav: "UAV", import: "imported" }[t.seed] || "—")));
  }

  let rendered = [];
  function renderList() {
    const rows = visibleTracks();
    rendered = rows.map((r) => r.tid);
    // the selected track stays visible on top even when the filter excludes it (e.g. just added)
    const pinned = S.tid !== null && S.tracks.has(S.tid) && !rows.some((r) => r.tid === S.tid) ? S.tracks.get(S.tid) : null;
    if (!rows.length && !pinned) {
      const seeding = S.jobs.some((j) => j.kind === "seed" && j.state === "running");
      if (S.tracks.size === 0 && seeding) { list.replaceChildren(); return; }      // the banner says it
      const msg = S.tracks.size === 0 ? "No tracks yet. Generate seeds or add a vehicle."
        : { review: "Nothing left to review.", flagged: "No flagged tracks.", done: "No reviewed tracks yet.", deleted: "No deleted tracks." }[S.list.filter] || "Nothing here.";
      list.replaceChildren(el("div.empty", { style: { padding: "36px 16px" } }, icon("cube"), msg));
      return;
    }
    list.replaceChildren(...(pinned ? [row(pinned, true)] : []), ...rows.map((t) => row(t)));
  }

  function renderBanner() {
    const job = S.jobs.find((j) => j.state === "running" && j.kind === "seed");
    if (!job) { banner.replaceChildren(); return; }
    const pct = job.total ? Math.round((100 * job.done) / job.total) : 0;
    banner.replaceChildren(el("div.banner", {}, el("div.spinner"),
      el("div", {}, el("b", {}, job.message || "Seeding"), el("br"),
        `${pct}% · tracks appear here as they finish; you can start reviewing them right away`)));
  }

  function markSelected() {
    for (const n of list.querySelectorAll(".trow.sel")) n.classList.remove("sel");
    const n = list.querySelector(`.trow[data-tid="${S.tid}"]`);
    if (n) { n.classList.add("sel"); n.scrollIntoView({ block: "nearest" }); }
  }

  function updateRow() {                        // a local edit: refresh just this row + counters
    const n = list.querySelector(`.trow[data-tid="${S.tid}"]`);
    const t = S.tracks.get(S.tid);
    if (n && t) { const r = row(t); n.replaceWith(r); }
    renderChips();
  }

  on("project", () => { renderProgress(); renderChips(); renderList(); });
  on("tracks", () => { renderChips(); renderList(); markSelected(); });
  on("tracks-light", updateRow);
  on("stats", renderProgress);
  on("select", () => {
    if (list.querySelector(".trow.pinned") || (S.tid !== null && !list.querySelector(`.trow[data-tid="${S.tid}"]`))) renderList();
    markSelected();
  });
  on("jobs", () => { renderBanner(); if (!S.tracks.size) renderList(); });
  return { focusSearch: () => search.focus(), rendered: () => rendered };
}

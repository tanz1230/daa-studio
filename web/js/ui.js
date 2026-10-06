// Small UI kit: DOM builder, icons, toasts, modals, popovers.

export function el(tag, attrs = {}, ...kids) {
  const [name, ...cls] = tag.split(".");
  const n = document.createElement(name || "div");
  if (cls.length) n.className = cls.join(" ");
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null) continue;
    if (v === false) { if (k in n && typeof n[k] === "boolean") n[k] = false; continue; }   // e.g. spellcheck
    if (k === "class") n.className += (n.className ? " " : "") + v;
    else if (k === "style" && typeof v === "object") Object.assign(n.style, v);
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (k === "html") n.innerHTML = v;
    else if (k in n && typeof v !== "string") n[k] = v;
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat(Infinity)) {
    if (k === null || k === undefined || k === false) continue;
    n.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return n;
}

export const clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); return n; };

// ------------------------------------------------------------------ icons (24px grid, 1.75 stroke)
const P = {
  logo: '<path d="M12 2.5 3.5 7v10l8.5 4.5 8.5-4.5V7z"/><path d="m3.5 7 8.5 4.5L20.5 7M12 11.5v10"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  circle: '<circle cx="12" cy="12" r="7.5"/>',
  pencil: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13.5 6.5 4 4"/>',
  flag: '<path d="M5 21V4M5 4h11l-2 4 2 4H5"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  folder: '<path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2h7a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
  file: '<path d="M14 3.5H7a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8.5z"/><path d="M14 3.5v5h5"/>',
  up: '<path d="M12 19V5M5.5 11.5 12 5l6.5 6.5"/>',
  left: '<path d="m14.5 6-6 6 6 6"/>',
  right: '<path d="m9.5 6 6 6-6 6"/>',
  play: '<path d="M8 5.5v13l10.5-6.5z"/>',
  pause: '<path d="M8 5.5v13M16 5.5v13"/>',
  first: '<path d="M7 5.5v13M18 6l-7.5 6 7.5 6z"/>',
  last: '<path d="M17 5.5v13M6 6l7.5 6L6 18z"/>',
  sparkle: '<path d="M12 3.5 13.8 9l5.7 1.8-5.7 1.9L12 18.5l-1.8-5.8L4.5 10.8 10.2 9z"/>',
  download: '<path d="M12 4v11M6.5 10 12 15.5 17.5 10M4.5 19.5h15"/>',
  undo: '<path d="M8.5 13.5 4 9l4.5-4.5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>',
  redo: '<path d="M15.5 13.5 20 9l-4.5-4.5"/><path d="M20 9H10a6 6 0 0 0 0 12h3"/>',
  keyboard: '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h.01M11 10h.01M15 10h.01M7 14h10"/>',
  crosshair: '<circle cx="12" cy="12" r="7.5"/><path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4"/>',
  layers: '<path d="m12 3.5 8.5 4.5-8.5 4.5L3.5 8z"/><path d="m3.5 12.5 8.5 4.5 8.5-4.5"/>',
  cube: '<path d="M12 2.5 3.5 7v10l8.5 4.5 8.5-4.5V7z"/><path d="m3.5 7 8.5 4.5L20.5 7M12 11.5v10"/>',
  drone: '<circle cx="5.5" cy="5.5" r="2.5"/><circle cx="18.5" cy="5.5" r="2.5"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/><path d="M7.5 7.5 10 10M16.5 7.5 14 10M7.5 16.5 10 14M16.5 16.5 14 14"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/>',
  trash: '<path d="M4.5 7h15M9.5 7V4.5h5V7M6.5 7l1 13h9l1-13"/>',
  rotate: '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3"/><path d="M19.5 4v4.5H15"/>',
  diamond: '<path d="m12 3.5 8.5 8.5-8.5 8.5L3.5 12z"/>',
  alert: '<path d="M12 4 2.5 20h19z"/><path d="M12 10v4.5M12 17.5h.01"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.5M12 7.8h.01"/>',
  more: '<circle cx="5.5" cy="12" r="1.3" fill="currentColor"/><circle cx="12" cy="12" r="1.3" fill="currentColor"/><circle cx="18.5" cy="12" r="1.3" fill="currentColor"/>',
  close: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
  eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.5"/>',
  interp: '<path d="M4 18 20 6"/><path d="m4 18 .01 0M20 6h.01"/><circle cx="4" cy="18" r="1.8"/><circle cx="20" cy="6" r="1.8"/>',
  flip: '<path d="M7 7.5h11l-3-3M17 16.5H6l3 3"/>',
};

export function icon(name, extra = "") {
  const s = document.createElement("span");
  s.className = "ic" + (extra ? " " + extra : "");
  s.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">${P[name] || ""}</svg>`;
  return s;
}

// ------------------------------------------------------------------ status vocabulary (glyph + word)
export const STATUS = {
  unreviewed: { label: "To review", icon: "circle" },
  accepted: { label: "Accepted", icon: "check" },
  edited: { label: "Edited", icon: "pencil" },
  flagged: { label: "Flagged", icon: "flag" },
  deleted: { label: "Deleted", icon: "x" },
  new: { label: "New", icon: "plus" },
};
export const statusGlyph = (s) => el("span.glyph.g-" + s, { title: STATUS[s]?.label }, icon(STATUS[s]?.icon || "circle"));
export const statusBadge = (s) => el("span.badge.st-" + s, {}, icon(STATUS[s]?.icon || "circle"), STATUS[s]?.label || s);

// ------------------------------------------------------------------ toasts
export function toast(msg, { kind = "", action = null, ms = 3200 } = {}) {
  const host = document.getElementById("toasts");
  const t = el("div.toast" + (kind ? "." + kind : ""), {}, msg);
  if (action) t.append(el("button.btn.sm", { onclick: () => { action.fn(); t.remove(); } }, action.label));
  host.append(t);
  setTimeout(() => t.remove(), ms);
  return t;
}

// ------------------------------------------------------------------ modal
export function modal({ title, subtitle, body, footer, wide = false, onClose } = {}) {
  const box = el("div.modal" + (wide ? ".wide" : ""));
  const scrim = el("div.scrim", {}, box);
  const close = () => { scrim.remove(); document.removeEventListener("keydown", esc, true); onClose && onClose(); };
  const esc = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  document.addEventListener("keydown", esc, true);
  scrim.addEventListener("mousedown", (e) => { if (e.target === scrim) close(); });
  const render = ({ title: t = title, subtitle: s = subtitle, body: b = body, footer: f = footer, steps } = {}) => {
    clear(box);
    box.append(el("div.modal-h", {}, el("h3", {}, t), s ? el("p", {}, s) : null));
    if (steps) box.append(el("div.steps", {}, steps.map((on) => el("span" + (on ? ".on" : "")))));
    box.append(el("div.modal-b", {}, b), f ? el("div.modal-f", {}, f) : null);
  };
  render();
  document.body.append(scrim);
  return { close, render, box };
}

export function confirmBox({ title, text, ok = "Confirm", danger = false }) {
  return new Promise((resolve) => {
    const m = modal({
      title, body: el("p", { style: { margin: 0, color: "var(--text-2)" } }, text),
      footer: [el("button.btn", { onclick: () => { resolve(false); m.close(); } }, "Cancel"),
               el("button.btn" + (danger ? ".danger" : ".primary"), { onclick: () => { resolve(true); m.close(); } }, ok)],
      onClose: () => resolve(false),
    });
  });
}

// ------------------------------------------------------------------ popover
export function popover(anchor, items) {
  const r = anchor.getBoundingClientRect();
  const p = el("div.popover", { style: { left: `${Math.round(r.left)}px`, top: `${Math.round(r.bottom + 6)}px` } });
  for (const it of items) {
    if (it === "-") { p.append(el("div.sep")); continue; }
    p.append(el("div.item", { onclick: () => { close(); it.fn(); } }, it.icon ? icon(it.icon) : null, it.label));
  }
  const close = () => { p.remove(); document.removeEventListener("mousedown", out, true); };
  const out = (e) => { if (!p.contains(e.target)) close(); };
  setTimeout(() => document.addEventListener("mousedown", out, true));
  document.body.append(p);
  const pr = p.getBoundingClientRect();
  if (pr.right > innerWidth - 8) p.style.left = `${innerWidth - pr.width - 8}px`;
  return close;
}

export const fmt = {
  m: (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d)),
  pct: (v) => `${Math.round(100 * v)}%`,
  deg: (rad) => ((rad * 180) / Math.PI).toFixed(1),
  when(iso) {                                  // "Today 14:05", "Yesterday 09:12", "3 Oct"
    if (!iso) return "";
    const d = new Date(iso), now = new Date();
    const hm = d.toTimeString().slice(0, 5);
    const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
    const diff = Math.round((day(now) - day(d)) / 86400000);
    if (diff === 0) return `Today ${hm}`;
    if (diff === 1) return `Yesterday ${hm}`;
    return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: d.getFullYear() === now.getFullYear() ? undefined : "numeric" });
  },
  dur(s) {
    if (s === null || s === undefined) return "";
    if (s < 60) return `${Math.max(1, Math.round(s))} s`;
    return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  },
};

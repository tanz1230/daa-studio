// Label operations — the client mirror of studio/labels.py (same semantics, same names).
// A track is plain JSON: { tid, cls, shape: [L, W, H], poses: { "frame": [x, y, z, yaw] }, status,
// flags: { "frame": ["edited", "keyframe"] }, note, seed, version }. Every op returns a new object.

export const clone = (t) => structuredClone(t);
export const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
export const framesOf = (t) => Object.keys(t.poses).map(Number).sort((a, b) => a - b);
export const has = (t, f) => Object.prototype.hasOwnProperty.call(t.poses, String(f));

export function box(t, f) {
  const [x, y, z, yaw] = t.poses[String(f)];
  const [L, W, H] = t.shape;
  return [x, y, z, L, W, H, yaw];
}

export function nearestFrame(t, f) {
  let best = null;
  for (const g of framesOf(t)) if (best === null || Math.abs(g - f) < Math.abs(best - f)) best = g;
  return best;
}

function flag(t, f, name, on = true) {
  const k = String(f);
  const cur = new Set(t.flags[k] || []);
  on ? cur.add(name) : cur.delete(name);
  if (cur.size) t.flags[k] = [...cur].sort(); else delete t.flags[k];
}

function touched(t) {
  if (t.status === "unreviewed" || t.status === "accepted") t.status = "edited";
  return t;
}

const AXIS = { L: 0, W: 1, H: 2 };

// one shared dimension; side=+1 keeps the negative face fixed, -1 the positive one, 0 = about the centre
export function resize(t, axis, size, side = 0, minSize = 0.2) {
  const out = clone(t);
  const i = AXIS[axis];
  size = Math.max(Number(size), minSize);
  const delta = size - t.shape[i];
  out.shape[i] = size;
  if (side) {
    for (const [k, [x, y, z, yaw]] of Object.entries(t.poses)) {
      const s = (side * delta) / 2;
      if (i === 0) out.poses[k] = [x + s * Math.cos(yaw), y + s * Math.sin(yaw), z, yaw];
      else if (i === 1) out.poses[k] = [x - s * Math.sin(yaw), y + s * Math.cos(yaw), z, yaw];
      else out.poses[k] = [x, y, z + s, yaw];
    }
  }
  return touched(out);
}

export function setPose(t, f, { x, y, z, yaw } = {}) {
  const out = clone(t);
  const k = String(f);
  const base = [...(t.poses[k] || t.poses[String(nearestFrame(t, f))])];
  [x, y, z, yaw].forEach((v, i) => { if (v !== undefined && v !== null) base[i] = Number(v); });
  base[3] = wrap(base[3]);
  out.poses[k] = base;
  flag(out, f, "edited");
  return touched(out);
}

export function move(t, frames, dx = 0, dy = 0, dz = 0) {
  const out = clone(t);
  for (const f of frames ?? framesOf(t)) {
    const k = String(f);
    if (!t.poses[k]) continue;
    const [x, y, z, yaw] = t.poses[k];
    out.poses[k] = [x + dx, y + dy, z + dz, yaw];
    if (frames) flag(out, f, "edited");
  }
  return touched(out);
}

export function rotate(t, frames, dyaw) {
  const out = clone(t);
  for (const f of frames ?? framesOf(t)) {
    const k = String(f);
    if (!t.poses[k]) continue;
    const [x, y, z, yaw] = t.poses[k];
    out.poses[k] = [x, y, z, wrap(yaw + dyaw)];
    if (frames) flag(out, f, "edited");
  }
  return touched(out);
}

export const flip = (t, frames) => rotate(t, frames, Math.PI);

// linear centre + heading between keyframes; heading the short way round modulo pi (box axis);
// frames in `fill` that the track lacks are created
export function interpolate(t, f0, f1, fill = []) {
  if (f0 > f1) [f0, f1] = [f1, f0];
  const out = clone(t);
  const [x0, y0, z0, a0] = t.poses[String(f0)];
  const [x1, y1, z1, a1] = t.poses[String(f1)];
  let d = wrap(a1 - a0);
  if (d > Math.PI / 2) d -= Math.PI; else if (d < -Math.PI / 2) d += Math.PI;
  let n = 0;
  for (const f of [...new Set([...framesOf(t), ...fill.map(Number)])].sort((a, b) => a - b)) {
    if (f <= f0 || f >= f1) continue;
    const w = (f - f0) / (f1 - f0);
    out.poses[String(f)] = [x0 + w * (x1 - x0), y0 + w * (y1 - y0), z0 + w * (z1 - z0), wrap(a0 + w * d)];
    flag(out, f, "edited");
    n++;
  }
  return n ? touched(out) : out;
}

// create the frames of `frames` that fall inside the track's gaps, interpolated
export function fillGaps(t, frames) {
  let out = t;
  const fr = framesOf(t);
  for (let i = 1; i < fr.length; i++) {
    const gap = frames.filter((f) => f > fr[i - 1] && f < fr[i]);
    if (gap.length) out = interpolate(out, fr[i - 1], fr[i], gap);
  }
  return out;
}

// copy the box at `src` into every frame of `frames` the track lacks (a parked vehicle)
export function copyTo(t, src, frames) {
  const out = clone(t);
  for (const f of frames) {
    if (has(out, f)) continue;
    out.poses[String(f)] = [...t.poses[String(src)]];
    flag(out, f, "edited");
  }
  return touched(out);
}

export function mark(t, f, name, on = true) { const out = clone(t); flag(out, f, name, on); return out; }

export function setStatus(t, status, note) {
  const out = clone(t);
  out.status = status;
  if (note !== undefined) out.note = note;
  return out;
}

export function setClass(t, cls) { const out = clone(t); out.cls = cls; return touched(out); }

// box corners (8x3): bottom face counter-clockwise, then top
export function corners([cx, cy, cz, L, W, H, yaw]) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const xs = [1, -1, -1, 1], ys = [1, 1, -1, -1];
  const out = [];
  for (const dz of [-H / 2, H / 2]) for (let i = 0; i < 4; i++) {
    const x = (xs[i] * L) / 2, y = (ys[i] * W) / 2;
    out.push([cx + c * x - s * y, cy + s * x + c * y, cz + dz]);
  }
  return out;
}

// map xy -> the box frame (x along the heading)
export function toBox([cx, cy, , , , , yaw], x, y) {
  const c = Math.cos(yaw), s = Math.sin(yaw), dx = x - cx, dy = y - cy;
  return [c * dx + s * dy, -s * dx + c * dy];
}

// shift every frame by an offset given in the BOX frame (removes a systematic offset of the box
// relative to the vehicle, whatever each frame's heading)
export function shiftLocal(t, dxl = 0, dyl = 0, dz = 0) {
  const out = clone(t);
  for (const [k, [x, y, z, yaw]] of Object.entries(t.poses)) {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    out.poses[k] = [x + c * dxl - s * dyl, y + s * dxl + c * dyl, z + dz, yaw];
  }
  return touched(out);
}

export function removeFrame(t, f) {
  const out = clone(t);
  if (Object.keys(out.poses).length > 1) { delete out.poses[String(f)]; delete out.flags[String(f)]; }
  return touched(out);
}

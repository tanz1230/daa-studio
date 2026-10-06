// HTTP client. JSON everywhere except the geometry streams, which are decoded into typed arrays.

export class ApiError extends Error {
  constructor(status, message, data) { super(message); this.status = status; this.data = data; }
}

export const user = {
  get: () => localStorage.getItem("daa.user") || "",
  set: (v) => localStorage.setItem("daa.user", v || ""),
};

async function req(method, path, body, raw = false) {
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: { "Content-Type": "application/json", "X-User": user.get() },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (e) {
    throw new ApiError(0, "The DAA Studio server is not reachable");
  }
  if (raw) {
    if (!res.ok) throw new ApiError(res.status, `HTTP ${res.status}`);
    return res;
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, (data && data.error) || `HTTP ${res.status}`, data);
  return data;
}

const get = (p) => req("GET", p);
const post = (p, b = {}) => req("POST", p, b);
const put = (p, b) => req("PUT", p, b);
const q = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v !== undefined && v !== null)).toString();

export const api = {
  app: () => get("/api/app"),
  createProject: (b) => post("/api/projects", b),
  openProject: (path) => post("/api/projects/open", { path }),
  closeProject: () => post("/api/projects/close"),
  project: () => get("/api/project"),
  updateProject: (b) => put("/api/project", b),
  tracks: () => get("/api/tracks"),
  track: (tid) => get(`/api/tracks/${tid}`),
  saveTrack: (track, expected_version, op) => put(`/api/tracks/${track.tid}`, { track, expected_version, op }),
  addTrack: (track) => post("/api/tracks", { track }),
  history: (tid) => get(`/api/history?${q({ tid })}`),
  frameBoxes: (f) => get(`/api/frames/${f}/boxes`),
  async framePoints(f, agents, ground) {
    const res = await req("GET", `/api/frames/${f}/points?${q({ agents: agents.join(","), ground: ground ? 1 : 0 })}`, undefined, true);
    const buf = await res.arrayBuffer();
    const n = new DataView(buf).getUint32(0, true);
    return { n, xyz: new Float32Array(buf, 4, n * 3), ego: new Uint8Array(buf, 4 + n * 12, n) };
  },
  async aggregate(tid) {
    const res = await req("GET", `/api/tracks/${tid}/aggregate`, undefined, true);
    const buf = await res.arrayBuffer();
    const n = new DataView(buf).getUint32(0, true);
    return { n, xyz: new Float32Array(buf, 4, n * 3), fidx: new Uint16Array(buf, 4 + n * 12, n),
             ego: new Uint8Array(buf, 4 + n * 14, n) };
  },
  uavView: (agent, f, tid, pad, aspect) => get(`/api/uav/${agent}/${f}/view?${q({ tid, pad, aspect: aspect && aspect.toFixed(3) })}`),
  uavImageURL: (agent, f, [x0, y0, w, h], out) => `/api/uav/${agent}/${f}/image?${q({ x0, y0, w, h, out })}`,
  uavLocate: (agent, f, u, v) => get(`/api/uav/${agent}/${f}/locate?${q({ u: u.toFixed(1), v: v.toFixed(1) })}`),
  seed: (b) => post("/api/seed", b),
  jobs: () => get("/api/jobs"),
  job: (id) => get(`/api/jobs/${id}`),
  cancelJob: (id) => post(`/api/jobs/${id}/cancel`),
  exportLabels: (b) => post("/api/export", b),
  fs: (path) => get(`/api/fs?${q({ path })}`),
  suggest: (session) => get(`/api/suggest?${q({ session })}`),
  sessionInfo: (path) => get(`/api/session-info?${q({ path })}`),
  log: (level, message, stack) => post("/api/client-log", { level, message, stack }).catch(() => {}),
};

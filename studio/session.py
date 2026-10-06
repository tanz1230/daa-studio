"""Read-only access to one OpenCOOD session folder of the air-ground release.

    take_<N>/                    e.g. agi_coop/take_3
        0/ 1/ 2/                 Ford, Lexus, roadside LiDAR: <frame>.pcd + <frame>.yaml (lidar_pose, vehicles)
        3/ 4/                    hovering / escorting UAV: <frame>.jpg (3840x2160)
        camera_3.json, camera_4.json    UAV camera models (map -> camera per frame)
        annotation_area.json     the filmed evaluation polygon

Everything returned is in the shared map frame. The session never writes anything: labels live
in a project workspace (project.py) and reach the dataset only through an explicit export.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import poses
from .ground import GroundModel
from .pcdio import read_pcd_xyz

LIDAR_AGENTS = (0, 1, 2)
UAV_AGENTS = (3, 4)
AGENT_NAMES = {0: "Ford", 1: "Lexus", 2: "Roadside", 3: "Hovering UAV", 4: "Escort UAV"}


class _LRU:
    """Tiny thread-safe LRU cache (the server is threaded)."""

    def __init__(self, size: int):
        self.size, self._d, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key, make):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        value = make()
        with self._lock:
            self._d[key] = value
            while len(self._d) > self.size:
                self._d.popitem(last=False)
        return value


class SessionError(ValueError):
    """The folder is not a usable OpenCOOD session."""


class Session:
    def __init__(self, path, pose_convention: str | None = None):
        self.path = Path(path).expanduser().resolve()
        if not (self.path / "0").is_dir():
            raise SessionError(f"{self.path} is not an OpenCOOD session (no '0/' agent folder)")
        self.name = self.path.name
        self.convention = pose_convention or poses.detect_convention(self.path)
        if self.convention is None:
            raise SessionError("could not tell which pose convention this session uses")
        self.frames = sorted(int(p.stem) for p in (self.path / "0").glob("*.yaml"))
        if not self.frames:
            raise SessionError(f"{self.path / '0'} holds no frames")
        self.lidar_agents = [a for a in LIDAR_AGENTS if any((self.path / str(a)).glob("*.pcd"))]
        self.uav_agents = [a for a in UAV_AGENTS
                           if (self.path / f"camera_{a}.json").exists() and (self.path / str(a)).is_dir()]
        self.ground = GroundModel(folders=(self.path, self.path.parent))   # session, dataset root
        self._yaml = _LRU(256)
        self._pts = _LRU(96)
        self._cams = {}

    # ------------------------------------------------------------------ metadata
    def info(self) -> dict:
        return {"name": self.name, "path": str(self.path), "convention": self.convention,
                "frames": len(self.frames), "first": self.frames[0], "last": self.frames[-1],
                "lidar_agents": [{"id": a, "name": AGENT_NAMES[a]} for a in self.lidar_agents],
                "uav_agents": [{"id": a, "name": AGENT_NAMES[a]} for a in self.uav_agents],
                "ground_model": self.ground.ok}

    def has(self, agent: int, frame: int) -> bool:
        return (self.path / str(agent) / f"{frame:06d}.yaml").exists()

    def _doc(self, agent: int, frame: int) -> dict:
        p = self.path / str(agent) / f"{frame:06d}.yaml"
        return self._yaml.get((agent, frame), lambda: poses.load_yaml(p) if p.exists() else {})

    # ------------------------------------------------------------------ geometry
    def pose(self, agent: int, frame: int):
        """(R, t): sensor -> map, or None when the agent has no pose at this frame."""
        lp = self._doc(agent, frame).get("lidar_pose")
        return None if lp is None else poses.pose_R_t(agent, lp, self.convention)

    def sensor_origin(self, agent: int, frame: int):
        p = self.pose(agent, frame)
        return None if p is None else p[1].copy()

    def points(self, agent: int, frame: int, drop_ground: bool = True) -> np.ndarray:
        """(N, 3) float32 map-frame points of one agent's sweep (empty when missing)."""
        def make():
            pcd = self.path / str(agent) / f"{frame:06d}.pcd"
            pose = self.pose(agent, frame)
            if pose is None or not pcd.exists():
                return np.zeros((0, 3), np.float32), np.zeros(0, bool)
            R, t = pose
            pts = (read_pcd_xyz(pcd).astype(np.float64) @ R.T + t).astype(np.float32)
            keep = self.ground.keep_mask(pts) if self.ground.ok else np.ones(len(pts), bool)
            return pts, keep
        pts, keep = self._pts.get((agent, frame), make)
        return pts[keep] if drop_ground else pts

    # ------------------------------------------------------------------ existing labels
    def labels(self, frame: int) -> dict:
        """{vid: (box7, obj_type)} from the session's own annotations (agent 0 YAML)."""
        out = {}
        for vid, v in (self._doc(0, frame).get("vehicles") or {}).items():
            out[int(vid)] = (np.array(poses.box_from_vehicle(v)), str(v.get("obj_type", "Car")))
        return out

    # ------------------------------------------------------------------ UAV
    def image_path(self, uav_agent: int, frame: int):
        p = self.path / str(uav_agent) / f"{frame:06d}.jpg"
        return p if p.exists() else None

    def camera(self, uav_agent: int):
        if uav_agent not in self.uav_agents:
            return None
        if uav_agent not in self._cams:
            from .camera import Camera
            self._cams[uav_agent] = Camera.from_json(self.path / f"camera_{uav_agent}.json")
        return self._cams[uav_agent]

    def annotation_area(self):
        p = self.path / "annotation_area.json"
        if not p.exists():
            return None
        return json.loads(p.read_text()).get("polygon")

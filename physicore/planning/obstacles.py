"""physicore/planning/obstacles.py — collision geometry and obstacle map."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CollisionReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CollisionReport:
    in_collision: bool
    obstacle_name: Optional[str] = None
    obstacle_type: Optional[str] = None
    penetration_depth: float = 0.0
    closest_point: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "in_collision": self.in_collision,
            "obstacle_name": self.obstacle_name,
            "obstacle_type": self.obstacle_type,
            "penetration_depth": round(self.penetration_depth, 6),
            "closest_point": self.closest_point.tolist() if self.closest_point is not None else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal geometry primitives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Sphere:
    name: str
    center: np.ndarray   # (3,)
    radius: float

    def check(self, p: np.ndarray) -> Optional[CollisionReport]:
        diff = p - self.center
        dist = float(np.linalg.norm(diff))
        if dist < self.radius:
            penetration = self.radius - dist
            closest = self.center + diff / (dist + 1e-12) * self.radius
            return CollisionReport(
                in_collision=True,
                obstacle_name=self.name,
                obstacle_type="sphere",
                penetration_depth=penetration,
                closest_point=closest,
            )
        return None

    def nearest_distance(self, p: np.ndarray) -> float:
        return max(0.0, float(np.linalg.norm(p - self.center)) - self.radius)


@dataclass
class _Box:
    name: str
    lo: np.ndarray    # (3,) lower corner
    hi: np.ndarray    # (3,) upper corner

    def check(self, p: np.ndarray) -> Optional[CollisionReport]:
        if np.all(p >= self.lo) and np.all(p <= self.hi):
            # penetration = minimum distance to any face
            dists = np.minimum(p - self.lo, self.hi - p)
            depth = float(np.min(dists))
            axis = int(np.argmin(dists))
            closest = p.copy()
            if p[axis] - self.lo[axis] < self.hi[axis] - p[axis]:
                closest[axis] = self.lo[axis]
            else:
                closest[axis] = self.hi[axis]
            return CollisionReport(
                in_collision=True,
                obstacle_name=self.name,
                obstacle_type="box",
                penetration_depth=depth,
                closest_point=closest,
            )
        return None

    def nearest_distance(self, p: np.ndarray) -> float:
        clamped = np.clip(p, self.lo, self.hi)
        return float(np.linalg.norm(p - clamped))


# ─────────────────────────────────────────────────────────────────────────────
# ObstacleMap
# ─────────────────────────────────────────────────────────────────────────────

class ObstacleMap:
    """
    Holds a collection of static and dynamic obstacles.
    Dynamic obstacles are updated each time check_collision() is called by
    polling their attached PerceptionSource.
    """

    def __init__(self) -> None:
        self._spheres: Dict[str, _Sphere] = {}
        self._boxes: Dict[str, _Box] = {}
        self._dynamic: Dict[str, Any] = {}   # name → PerceptionSource-like

    # ── Construction ──────────────────────────────────────────────────────────

    def add_sphere(self, name: str, center: np.ndarray, radius: float) -> None:
        self._spheres[name] = _Sphere(
            name=name,
            center=np.asarray(center, dtype=float),
            radius=float(radius),
        )

    def add_box(self, name: str, lo: np.ndarray, hi: np.ndarray) -> None:
        self._boxes[name] = _Box(
            name=name,
            lo=np.asarray(lo, dtype=float),
            hi=np.asarray(hi, dtype=float),
        )

    def add_dynamic(self, name: str, source: Any, radius: float = 0.1) -> None:
        """
        source: any object with get_state_observation() → Observation or
                get_state_observation() returning an object with .values[:3] as position.
        """
        self._dynamic[name] = {"source": source, "radius": radius}

    def remove(self, name: str) -> bool:
        removed = False
        if name in self._spheres:
            del self._spheres[name]
            removed = True
        if name in self._boxes:
            del self._boxes[name]
            removed = True
        if name in self._dynamic:
            del self._dynamic[name]
            removed = True
        return removed

    # ── Collision checking ────────────────────────────────────────────────────

    def check_collision(self, p: np.ndarray) -> CollisionReport:
        """Check point p against all obstacles; return first collision found."""
        p = np.asarray(p, dtype=float)

        for sphere in self._spheres.values():
            report = sphere.check(p)
            if report is not None:
                return report

        for box in self._boxes.values():
            report = box.check(p)
            if report is not None:
                return report

        for name, entry in self._dynamic.items():
            try:
                obs = entry["source"].get_state_observation()
                center = np.asarray(obs.values[:3], dtype=float)
                radius = entry["radius"]
                tmp_sphere = _Sphere(name=name, center=center, radius=radius)
                report = tmp_sphere.check(p)
                if report is not None:
                    report.obstacle_type = "dynamic"
                    return report
            except Exception:
                pass

        return CollisionReport(in_collision=False)

    def check_path_clear(self, points: List[np.ndarray], safety_margin: float = 0.0) -> CollisionReport:
        """Check a sequence of points; return the first collision or a clear report."""
        for pt in points:
            p = np.asarray(pt, dtype=float)

            for sphere in self._spheres.values():
                s = _Sphere(
                    name=sphere.name,
                    center=sphere.center,
                    radius=sphere.radius + safety_margin,
                )
                report = s.check(p)
                if report is not None:
                    return report

            for box in self._boxes.values():
                b = _Box(
                    name=box.name,
                    lo=box.lo - safety_margin,
                    hi=box.hi + safety_margin,
                )
                report = b.check(p)
                if report is not None:
                    return report

            for name, entry in self._dynamic.items():
                try:
                    obs = entry["source"].get_state_observation()
                    center = np.asarray(obs.values[:3], dtype=float)
                    radius = entry["radius"] + safety_margin
                    tmp = _Sphere(name=name, center=center, radius=radius)
                    report = tmp.check(p)
                    if report is not None:
                        report.obstacle_type = "dynamic"
                        return report
                except Exception:
                    pass

        return CollisionReport(in_collision=False)

    def nearest_obstacle(self, p: np.ndarray) -> Tuple[Optional[str], float]:
        """Return (name, distance) of the nearest obstacle to point p."""
        p = np.asarray(p, dtype=float)
        best_name: Optional[str] = None
        best_dist = float("inf")

        for sphere in self._spheres.values():
            d = sphere.nearest_distance(p)
            if d < best_dist:
                best_dist = d
                best_name = sphere.name

        for box in self._boxes.values():
            d = box.nearest_distance(p)
            if d < best_dist:
                best_dist = d
                best_name = box.name

        for name, entry in self._dynamic.items():
            try:
                obs = entry["source"].get_state_observation()
                center = np.asarray(obs.values[:3], dtype=float)
                radius = entry["radius"]
                tmp = _Sphere(name=name, center=center, radius=radius)
                d = tmp.nearest_distance(p)
                if d < best_dist:
                    best_dist = d
                    best_name = name
            except Exception:
                pass

        return best_name, best_dist

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"spheres": [], "boxes": [], "dynamic": []}
        for s in self._spheres.values():
            out["spheres"].append({
                "name": s.name,
                "center": s.center.tolist(),
                "radius": s.radius,
            })
        for b in self._boxes.values():
            out["boxes"].append({
                "name": b.name,
                "lo": b.lo.tolist(),
                "hi": b.hi.tolist(),
            })
        for name in self._dynamic:
            out["dynamic"].append({"name": name})
        return out

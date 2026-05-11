"""physicore/perception/interface.py — sensor abstraction layer and EKF fusion."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Observation dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """A single sensor measurement."""
    values: np.ndarray          # shape (obs_dim,)
    timestamp: float            # UNIX seconds
    source: str                 # source name
    confidence: float = 1.0    # [0, 1]
    covariance: Optional[np.ndarray] = None  # (obs_dim, obs_dim) or None → use default R


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class PerceptionSource(ABC):
    """Abstract sensor source.  Subclass and implement get_state_observation()."""

    def __init__(self, name: str, obs_dim: int, state_dim: int):
        self._name = name
        self._obs_dim = obs_dim
        self._state_dim = state_dim
        self._last_ts: float = 0.0

    # ── Required overrides ────────────────────────────────────────────────────

    @abstractmethod
    def get_state_observation(self) -> Observation:
        """Return the most recent measurement."""

    # ── Optional override ─────────────────────────────────────────────────────

    def get_timestamp(self) -> float:
        return self._last_ts

    def is_fresh(self, max_age_s: float = 1.0) -> bool:
        return (time.time() - self._last_ts) < max_age_s

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def H_matrix(self) -> np.ndarray:
        """Observation matrix H mapping state → measurement space."""
        H = np.zeros((self._obs_dim, self._state_dim))
        for i in range(min(self._obs_dim, self._state_dim)):
            H[i, i] = 1.0
        return H

    @property
    def R_matrix(self) -> np.ndarray:
        """Default measurement noise covariance."""
        return np.eye(self._obs_dim) * 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Concrete source adapters
# ─────────────────────────────────────────────────────────────────────────────

class PoseSource(PerceptionSource):
    """6-DOF pose sensor (position xyz + euler angles rpy)."""

    def __init__(self, name: str = "pose", state_dim: int = 13, noise_std: float = 0.005):
        super().__init__(name, obs_dim=6, state_dim=state_dim)
        self._noise_std = noise_std
        self._pose: np.ndarray = np.zeros(6)

    def push(self, pose: np.ndarray) -> None:
        """Inject a 6-element pose vector [x, y, z, roll, pitch, yaw]."""
        self._pose = np.asarray(pose, dtype=float)
        self._last_ts = time.time()

    def get_state_observation(self) -> Observation:
        self._last_ts = time.time()
        noise = np.random.randn(6) * self._noise_std
        return Observation(
            values=self._pose + noise,
            timestamp=self._last_ts,
            source=self._name,
            confidence=1.0,
            covariance=np.eye(6) * self._noise_std ** 2,
        )

    @property
    def R_matrix(self) -> np.ndarray:
        return np.eye(6) * self._noise_std ** 2


class DepthSource(PerceptionSource):
    """Depth/range sensor returning a 3-D point in robot frame."""

    def __init__(self, name: str = "depth", state_dim: int = 13, noise_std: float = 0.02):
        super().__init__(name, obs_dim=3, state_dim=state_dim)
        self._noise_std = noise_std
        self._point: np.ndarray = np.zeros(3)

    def push(self, point: np.ndarray) -> None:
        self._point = np.asarray(point, dtype=float)
        self._last_ts = time.time()

    def get_state_observation(self) -> Observation:
        self._last_ts = time.time()
        noise = np.random.randn(3) * self._noise_std
        return Observation(
            values=self._point + noise,
            timestamp=self._last_ts,
            source=self._name,
            confidence=0.9,
            covariance=np.eye(3) * self._noise_std ** 2,
        )

    @property
    def H_matrix(self) -> np.ndarray:
        H = np.zeros((3, self._state_dim))
        for i in range(3):
            H[i, i] = 1.0
        return H

    @property
    def R_matrix(self) -> np.ndarray:
        return np.eye(3) * self._noise_std ** 2


class MarkerSource(PerceptionSource):
    """Fiducial marker tracker returning 3-D position of detected markers."""

    def __init__(self, name: str = "marker", state_dim: int = 13,
                 n_markers: int = 1, noise_std: float = 0.003):
        super().__init__(name, obs_dim=3 * n_markers, state_dim=state_dim)
        self._noise_std = noise_std
        self._n_markers = n_markers
        self._positions: np.ndarray = np.zeros(3 * n_markers)

    def push(self, positions: np.ndarray) -> None:
        """positions: (n_markers * 3,) flattened array of marker positions."""
        self._positions = np.asarray(positions, dtype=float).ravel()
        self._last_ts = time.time()

    def get_state_observation(self) -> Observation:
        self._last_ts = time.time()
        noise = np.random.randn(self._obs_dim) * self._noise_std
        return Observation(
            values=self._positions + noise,
            timestamp=self._last_ts,
            source=self._name,
            confidence=0.95,
            covariance=np.eye(self._obs_dim) * self._noise_std ** 2,
        )

    @property
    def R_matrix(self) -> np.ndarray:
        return np.eye(self._obs_dim) * self._noise_std ** 2


class JointEncoderSource(PerceptionSource):
    """Joint encoder returning joint positions (and optionally velocities)."""

    def __init__(self, name: str = "encoder", n_joints: int = 6,
                 state_dim: int = 13, noise_std: float = 1e-4,
                 include_velocity: bool = False):
        obs = n_joints * (2 if include_velocity else 1)
        super().__init__(name, obs_dim=obs, state_dim=state_dim)
        self._n_joints = n_joints
        self._noise_std = noise_std
        self._include_velocity = include_velocity
        self._q: np.ndarray = np.zeros(obs)

    def push(self, q: np.ndarray) -> None:
        self._q = np.asarray(q, dtype=float).ravel()
        self._last_ts = time.time()

    def get_state_observation(self) -> Observation:
        self._last_ts = time.time()
        noise = np.random.randn(self._obs_dim) * self._noise_std
        return Observation(
            values=self._q + noise,
            timestamp=self._last_ts,
            source=self._name,
            confidence=0.99,
            covariance=np.eye(self._obs_dim) * self._noise_std ** 2,
        )

    @property
    def R_matrix(self) -> np.ndarray:
        return np.eye(self._obs_dim) * self._noise_std ** 2


class IMUSource(PerceptionSource):
    """6-axis IMU: linear acceleration (3) + angular velocity (3)."""

    def __init__(self, name: str = "imu", state_dim: int = 13,
                 accel_noise: float = 0.01, gyro_noise: float = 0.005):
        super().__init__(name, obs_dim=6, state_dim=state_dim)
        self._accel_noise = accel_noise
        self._gyro_noise = gyro_noise
        self._measurement: np.ndarray = np.zeros(6)

    def push(self, accel: np.ndarray, gyro: np.ndarray) -> None:
        self._measurement = np.concatenate([
            np.asarray(accel, dtype=float).ravel()[:3],
            np.asarray(gyro,  dtype=float).ravel()[:3],
        ])
        self._last_ts = time.time()

    def get_state_observation(self) -> Observation:
        self._last_ts = time.time()
        noise = np.concatenate([
            np.random.randn(3) * self._accel_noise,
            np.random.randn(3) * self._gyro_noise,
        ])
        cov = np.diag([self._accel_noise**2]*3 + [self._gyro_noise**2]*3)
        return Observation(
            values=self._measurement + noise,
            timestamp=self._last_ts,
            source=self._name,
            confidence=0.98,
            covariance=cov,
        )

    @property
    def R_matrix(self) -> np.ndarray:
        return np.diag([self._accel_noise**2]*3 + [self._gyro_noise**2]*3)


# ─────────────────────────────────────────────────────────────────────────────
# PerceptionFusion
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SourceEntry:
    source: PerceptionSource
    registered_with_estimator: bool = False


class PerceptionFusion:
    """
    Aggregates multiple PerceptionSource instances and drives an EKF
    StateEstimator with fused measurements.
    """

    def __init__(self, estimator: Any, max_age_s: float = 1.0, poll_hz: float = 50.0):
        self._estimator = estimator
        self._max_age_s = max_age_s
        self._poll_hz = poll_hz
        self._sources: Dict[str, _SourceEntry] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ── Source management ─────────────────────────────────────────────────────

    def register(self, source: PerceptionSource) -> None:
        """Register a perception source and wire it into the EKF."""
        with self._lock:
            if source.name in self._sources:
                return
            entry = _SourceEntry(source=source)
            self._sources[source.name] = entry
            # Register with the EKF estimator
            try:
                self._estimator.register_sensor(
                    source.name, source.H_matrix, source.R_matrix
                )
                entry.registered_with_estimator = True
            except Exception:
                pass

    def unregister(self, name: str) -> bool:
        with self._lock:
            return bool(self._sources.pop(name, None))

    def source_names(self) -> List[str]:
        with self._lock:
            return list(self._sources.keys())

    # ── Fusion step ───────────────────────────────────────────────────────────

    def fuse(self) -> Dict[str, Any]:
        """
        Poll all fresh sources, call EKF fuse(), return a staleness report.
        Returns dict: {source_name: {"fresh": bool, "age_s": float, "values": list}}
        """
        report: Dict[str, Any] = {}
        obs_map: Dict[str, np.ndarray] = {}
        now = time.time()

        with self._lock:
            sources_snapshot = dict(self._sources)

        for name, entry in sources_snapshot.items():
            src = entry.source
            try:
                obs = src.get_state_observation()
                age = now - obs.timestamp
                fresh = age < self._max_age_s
                report[name] = {
                    "fresh": fresh,
                    "age_s": round(age, 4),
                    "confidence": obs.confidence,
                    "values": obs.values.tolist(),
                }
                if fresh and entry.registered_with_estimator:
                    obs_map[name] = obs.values
            except Exception as exc:
                report[name] = {"fresh": False, "age_s": -1.0, "error": str(exc)}

        if obs_map:
            try:
                self._estimator.fuse(obs_map)
            except Exception:
                pass

        return report

    def staleness_report(self) -> Dict[str, Any]:
        """Return freshness info per source without running an EKF update."""
        now = time.time()
        out: Dict[str, Any] = {}
        with self._lock:
            for name, entry in self._sources.items():
                age = now - entry.source.get_timestamp()
                out[name] = {
                    "fresh": age < self._max_age_s,
                    "age_s": round(age, 4),
                    "last_ts": entry.source.get_timestamp(),
                }
        return out

    # ── Background polling ────────────────────────────────────────────────────

    def start_polling(self) -> None:
        """Start a background thread that calls fuse() at poll_hz."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop_polling(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _poll_loop(self) -> None:
        interval = 1.0 / self._poll_hz
        while self._running:
            t0 = time.monotonic()
            try:
                self.fuse()
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, interval - elapsed)
            time.sleep(sleep_t)

    # ── State access ──────────────────────────────────────────────────────────

    @property
    def state_estimate(self) -> np.ndarray:
        return self._estimator.estimate

    @property
    def uncertainty(self) -> np.ndarray:
        return self._estimator.uncertainty

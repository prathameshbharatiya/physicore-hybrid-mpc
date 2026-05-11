"""physicore/planning/planner.py — trajectory planning, IK solver, and executor."""

from __future__ import annotations

import bisect
import enum
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryPoint + Trajectory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    t: float               # time in seconds from trajectory start
    q: np.ndarray          # joint positions, shape (dof,)
    qd: Optional[np.ndarray] = None   # joint velocities
    qdd: Optional[np.ndarray] = None  # joint accelerations
    ee_pos: Optional[np.ndarray] = None  # EE position at this point, shape (3,)


class Trajectory:
    """Sampled joint-space trajectory with interpolation."""

    def __init__(self, points: List[TrajectoryPoint], trajectory_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        if len(points) < 2:
            raise ValueError("Trajectory requires at least 2 points")
        self.points = sorted(points, key=lambda p: p.t)
        self.trajectory_id = trajectory_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self._times = [p.t for p in self.points]

    @property
    def duration(self) -> float:
        return self.points[-1].t - self.points[0].t

    @property
    def dof(self) -> int:
        return len(self.points[0].q)

    def at(self, t: float) -> TrajectoryPoint:
        """Linearly interpolate the trajectory at time t (clamped to [t0, tf])."""
        t = float(np.clip(t, self._times[0], self._times[-1]))
        idx = bisect.bisect_right(self._times, t) - 1
        idx = max(0, min(idx, len(self.points) - 2))

        p0, p1 = self.points[idx], self.points[idx + 1]
        dt = p1.t - p0.t
        alpha = (t - p0.t) / dt if dt > 1e-12 else 0.0

        q   = p0.q   + alpha * (p1.q   - p0.q)
        qd  = None
        qdd = None
        if p0.qd is not None and p1.qd is not None:
            qd  = p0.qd  + alpha * (p1.qd  - p0.qd)
        if p0.qdd is not None and p1.qdd is not None:
            qdd = p0.qdd + alpha * (p1.qdd - p0.qdd)
        ee = None
        if p0.ee_pos is not None and p1.ee_pos is not None:
            ee = p0.ee_pos + alpha * (p1.ee_pos - p0.ee_pos)
        return TrajectoryPoint(t=t, q=q, qd=qd, qdd=qdd, ee_pos=ee)

    def to_ref_sequence(self, hz: float = 100.0) -> List[TrajectoryPoint]:
        """Sample trajectory at fixed hz for use as a reference signal."""
        n = max(2, int(self.duration * hz) + 1)
        times = np.linspace(self._times[0], self._times[-1], n)
        return [self.at(float(t)) for t in times]

    def visualize_ascii(self, width: int = 60, height: int = 10) -> str:
        """ASCII chart of joint angles over time (first 3 joints)."""
        samples = self.to_ref_sequence(hz=float(width) / max(self.duration, 1e-9))
        cols = min(3, self.dof)
        qs = np.array([[s.q[j] for j in range(cols)] for s in samples])
        lines: List[str] = []
        for j in range(cols):
            col_data = qs[:, j]
            lo, hi = col_data.min(), col_data.max()
            span = hi - lo if (hi - lo) > 1e-9 else 1.0
            chars = ""
            for v in col_data[:width]:
                row = int((v - lo) / span * (height - 1))
                chars += str(min(row, height - 1))
            lines.append(f"q{j}: " + chars)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "duration": round(self.duration, 4),
            "dof": self.dof,
            "n_points": len(self.points),
            "metadata": self.metadata,
            "points": [
                {
                    "t": round(p.t, 6),
                    "q": p.q.tolist(),
                    "qd": p.qd.tolist() if p.qd is not None else None,
                    "ee_pos": p.ee_pos.tolist() if p.ee_pos is not None else None,
                }
                for p in self.points
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# IKSolver
# ─────────────────────────────────────────────────────────────────────────────

class IKSolver:
    """
    Iterative numerical IK using Jacobian pseudoinverse with null-space
    joint-limit avoidance.
    """

    def __init__(self, robot_model: Any, max_iter: int = 200, tol: float = 1e-4,
                 step_size: float = 0.5, null_space_gain: float = 0.3):
        self._model = robot_model
        self._max_iter = max_iter
        self._tol = tol
        self._step_size = step_size
        self._null_space_gain = null_space_gain
        joints = getattr(robot_model, "actuated_joints", [])
        self._q_lo = np.array([getattr(j, "limit_lo", -np.pi) for j in joints])
        self._q_hi = np.array([getattr(j, "limit_hi",  np.pi) for j in joints])

    def solve(self, target_pos: np.ndarray,
              q_init: Optional[np.ndarray] = None,
              target_rot: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, float]:
        """
        Returns (q_solution, converged, final_error).
        target_pos: (3,) EE position in world frame.
        target_rot: (3,3) desired rotation matrix (unused in position-only mode).
        """
        dof = self._model.dof
        q = np.zeros(dof) if q_init is None else np.array(q_init, dtype=float)
        q = np.clip(q, self._q_lo, self._q_hi)
        target = np.asarray(target_pos, dtype=float)

        q_mid = 0.5 * (self._q_lo + self._q_hi)

        for _ in range(self._max_iter):
            try:
                cur_pos = self._model.ee_position(q)
            except Exception:
                break

            dp = target - cur_pos
            err = float(np.linalg.norm(dp))
            if err < self._tol:
                return q, True, err

            try:
                J_full = self._model.jacobian(q)
                J_pos  = J_full[:3, :]
            except Exception:
                break

            # Pseudoinverse
            J_pinv = np.linalg.pinv(J_pos)
            dq_task = J_pinv @ dp * self._step_size

            # Null-space joint-limit avoidance
            null_proj = np.eye(dof) - J_pinv @ J_pos
            grad = -self._null_space_gain * (q - q_mid)
            dq = dq_task + null_proj @ grad

            q = np.clip(q + dq, self._q_lo, self._q_hi)

        try:
            final_err = float(np.linalg.norm(target - self._model.ee_position(q)))
        except Exception:
            final_err = float("inf")

        return q, False, final_err

    def solve_batch(self, targets: List[np.ndarray],
                    q_init: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, bool, float]]:
        """Solve IK for a list of target positions, chaining the previous solution as init."""
        results = []
        q = q_init
        for target in targets:
            q_sol, converged, err = self.solve(target, q_init=q)
            results.append((q_sol, converged, err))
            q = q_sol  # warm-start next step
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Velocity profile helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trapezoid_profile(q0: np.ndarray, q1: np.ndarray,
                       v_max: float = 1.0, a_max: float = 2.0,
                       n_samples: int = 100) -> List[TrajectoryPoint]:
    """Generate joint-space trapezoidal velocity profile between q0 and q1."""
    delta = q1 - q0
    max_delta = float(np.max(np.abs(delta)))
    if max_delta < 1e-9:
        return [
            TrajectoryPoint(t=0.0, q=q0.copy(), qd=np.zeros_like(q0), qdd=np.zeros_like(q0)),
            TrajectoryPoint(t=0.01, q=q1.copy(), qd=np.zeros_like(q0), qdd=np.zeros_like(q0)),
        ]

    # Scale to leading joint
    v = min(v_max, math.sqrt(a_max * max_delta))
    t_ramp = v / a_max
    t_flat = (max_delta - v * t_ramp) / v
    t_flat = max(0.0, t_flat)
    T = 2 * t_ramp + t_flat

    times = np.linspace(0.0, T, n_samples)
    pts: List[TrajectoryPoint] = []
    for t in times:
        if t <= t_ramp:
            s = 0.5 * a_max * t**2 / max_delta
            sd = a_max * t / max_delta
            sdd = a_max / max_delta
        elif t <= t_ramp + t_flat:
            s = (0.5 * a_max * t_ramp**2 + v * (t - t_ramp)) / max_delta
            sd = v / max_delta
            sdd = 0.0
        else:
            tau = t - t_ramp - t_flat
            s = (0.5 * a_max * t_ramp**2 + v * t_flat + v * tau - 0.5 * a_max * tau**2) / max_delta
            sd = (v - a_max * tau) / max_delta
            sdd = -a_max / max_delta

        s = float(np.clip(s, 0.0, 1.0))
        q   = q0 + s * delta
        qd  = sd  * delta
        qdd = sdd * delta
        pts.append(TrajectoryPoint(t=float(t), q=q, qd=qd, qdd=qdd))
    return pts


def _quintic_blend(q0: np.ndarray, qd0: np.ndarray, qdd0: np.ndarray,
                   q1: np.ndarray, qd1: np.ndarray, qdd1: np.ndarray,
                   T: float, n_samples: int = 50) -> List[TrajectoryPoint]:
    """Quintic polynomial segment from (q0,qd0,qdd0) to (q1,qd1,qdd1) over duration T."""
    dof = len(q0)
    pts: List[TrajectoryPoint] = []
    times = np.linspace(0.0, T, n_samples)

    coeffs = []
    for j in range(dof):
        a0 = q0[j]
        a1 = qd0[j]
        a2 = 0.5 * qdd0[j]
        # Solve for a3, a4, a5
        M = np.array([
            [T**3,   T**4,    T**5],
            [3*T**2, 4*T**3,  5*T**4],
            [6*T,    12*T**2, 20*T**3],
        ])
        rhs = np.array([
            q1[j]   - a0 - a1*T - a2*T**2,
            qd1[j]  - a1 - 2*a2*T,
            qdd1[j] - 2*a2,
        ])
        try:
            abc = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            abc = np.zeros(3)
        coeffs.append((a0, a1, a2, abc[0], abc[1], abc[2]))

    for t in times:
        q   = np.array([c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5
                         for c in coeffs])
        qd  = np.array([c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4
                         for c in coeffs])
        qdd = np.array([2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3
                         for c in coeffs])
        pts.append(TrajectoryPoint(t=float(t), q=q, qd=qd, qdd=qdd))
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryPlanner
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryPlanner:
    """
    High-level trajectory generation for joint-space and task-space paths.
    Requires a URDFRobotModel-compatible robot_model.
    """

    def __init__(self, robot_model: Any, v_max: float = 1.0, a_max: float = 2.0):
        self._model = robot_model
        self._v_max = v_max
        self._a_max = a_max
        self._ik = IKSolver(robot_model)
        self._dof = robot_model.dof

    # ── Joint-space trapezoidal ───────────────────────────────────────────────

    def plan_joint_space(self, q_start: np.ndarray, q_goal: np.ndarray,
                         v_max: Optional[float] = None, a_max: Optional[float] = None,
                         obstacle_map: Optional[Any] = None,
                         n_samples: int = 100) -> Trajectory:
        """Trapezoidal joint-space plan from q_start to q_goal."""
        q0 = np.asarray(q_start, dtype=float)
        q1 = np.asarray(q_goal,  dtype=float)
        pts = _trapezoid_profile(q0, q1,
                                 v_max=v_max or self._v_max,
                                 a_max=a_max or self._a_max,
                                 n_samples=n_samples)
        # Compute EE positions
        for p in pts:
            try:
                p.ee_pos = self._model.ee_position(p.q)
            except Exception:
                pass

        traj = Trajectory(pts, metadata={"type": "joint_space"})

        if obstacle_map is not None:
            ee_pts = [p.ee_pos for p in pts if p.ee_pos is not None]
            report = obstacle_map.check_path_clear(ee_pts)
            traj.metadata["collision"] = report.to_dict()

        return traj

    # ── Task-space straight line via IK ──────────────────────────────────────

    def plan_task_space(self, q_start: np.ndarray, target_pos: np.ndarray,
                        n_via: int = 20, obstacle_map: Optional[Any] = None,
                        v_max: Optional[float] = None,
                        a_max: Optional[float] = None) -> Trajectory:
        """Straight-line EE path from current FK position to target_pos."""
        q0 = np.asarray(q_start, dtype=float)
        try:
            p0 = self._model.ee_position(q0)
        except Exception:
            p0 = np.zeros(3)
        pT = np.asarray(target_pos, dtype=float)

        alphas = np.linspace(0.0, 1.0, n_via + 2)
        via_targets = [p0 + a * (pT - p0) for a in alphas]

        q_chain = self._ik.solve_batch(via_targets, q_init=q0)
        q_list  = [res[0] for res in q_chain]

        # Build trapezoidal timing
        all_pts: List[TrajectoryPoint] = []
        t_offset = 0.0
        for i in range(len(q_list) - 1):
            seg = _trapezoid_profile(q_list[i], q_list[i + 1],
                                     v_max=v_max or self._v_max,
                                     a_max=a_max or self._a_max,
                                     n_samples=10)
            for p in seg:
                p.t += t_offset
                try:
                    p.ee_pos = self._model.ee_position(p.q)
                except Exception:
                    pass
                all_pts.append(p)
            t_offset = all_pts[-1].t

        # Deduplicate same-time points
        seen: Dict[float, TrajectoryPoint] = {}
        for p in all_pts:
            seen[round(p.t, 9)] = p
        pts = sorted(seen.values(), key=lambda p: p.t)

        if len(pts) < 2:
            pts = [
                TrajectoryPoint(t=0.0, q=q0, ee_pos=p0),
                TrajectoryPoint(t=0.1, q=q_list[-1], ee_pos=pT),
            ]

        traj = Trajectory(pts, metadata={"type": "task_space", "target": pT.tolist()})

        if obstacle_map is not None:
            ee_pts = [p.ee_pos for p in pts if p.ee_pos is not None]
            report = obstacle_map.check_path_clear(ee_pts)
            traj.metadata["collision"] = report.to_dict()

        return traj

    # ── Waypoints with quintic polynomial blending ────────────────────────────

    def plan_waypoints(self, waypoints: List[np.ndarray],
                       segment_time: float = 1.0,
                       obstacle_map: Optional[Any] = None) -> Trajectory:
        """
        waypoints: list of joint-space configurations.
        Blends segments with quintic polynomials for smooth continuity.
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        wps = [np.asarray(w, dtype=float) for w in waypoints]
        all_pts: List[TrajectoryPoint] = []
        t_offset = 0.0
        n = len(wps)

        for i in range(n - 1):
            q0  = wps[i]
            q1  = wps[i + 1]
            qd0 = np.zeros(self._dof) if i == 0 else 0.5 * (wps[min(i+1, n-1)] - wps[max(i-1, 0)]) / segment_time
            qd1 = np.zeros(self._dof) if i == n - 2 else 0.5 * (wps[min(i+2, n-1)] - wps[i]) / segment_time

            seg = _quintic_blend(
                q0, qd0, np.zeros(self._dof),
                q1, qd1, np.zeros(self._dof),
                T=segment_time,
            )
            for p in seg:
                p.t += t_offset
                try:
                    p.ee_pos = self._model.ee_position(p.q)
                except Exception:
                    pass
                all_pts.append(p)
            t_offset += segment_time

        seen: Dict[float, TrajectoryPoint] = {}
        for p in all_pts:
            seen[round(p.t, 9)] = p
        pts = sorted(seen.values(), key=lambda p: p.t)

        traj = Trajectory(pts, metadata={"type": "waypoints", "n_waypoints": n})

        if obstacle_map is not None:
            ee_pts = [p.ee_pos for p in pts if p.ee_pos is not None]
            report = obstacle_map.check_path_clear(ee_pts)
            traj.metadata["collision"] = report.to_dict()

        return traj

    # ── Circular arc ──────────────────────────────────────────────────────────

    def plan_circular(self, q_start: np.ndarray, center: np.ndarray,
                      normal: np.ndarray, angle_rad: float,
                      n_via: int = 36, obstacle_map: Optional[Any] = None,
                      v_max: Optional[float] = None,
                      a_max: Optional[float] = None) -> Trajectory:
        """
        Arc in task space: EE travels along a circle of given center/normal/angle.
        """
        q0 = np.asarray(q_start, dtype=float)
        try:
            p0 = self._model.ee_position(q0)
        except Exception:
            p0 = np.zeros(3)

        c  = np.asarray(center, dtype=float)
        n  = np.asarray(normal, dtype=float)
        n  = n / (np.linalg.norm(n) + 1e-12)

        # Build orthonormal frame u, v in the arc plane
        r_vec = p0 - c
        r_vec_norm = np.linalg.norm(r_vec)
        if r_vec_norm < 1e-9:
            r_vec = np.array([1.0, 0.0, 0.0])
        u = r_vec / (r_vec_norm + 1e-12)
        v_raw = np.cross(n, u)
        v_norm = np.linalg.norm(v_raw)
        if v_norm < 1e-9:
            v_raw = np.array([0.0, 1.0, 0.0])
            v_norm = 1.0
        v = v_raw / v_norm
        R = r_vec_norm

        thetas = np.linspace(0.0, angle_rad, n_via + 2)
        via_targets = [c + R * (math.cos(th) * u + math.sin(th) * v) for th in thetas]

        q_chain = self._ik.solve_batch(via_targets, q_init=q0)
        q_list  = [res[0] for res in q_chain]

        all_pts: List[TrajectoryPoint] = []
        t_offset = 0.0
        for i in range(len(q_list) - 1):
            seg = _trapezoid_profile(q_list[i], q_list[i + 1],
                                     v_max=v_max or self._v_max,
                                     a_max=a_max or self._a_max,
                                     n_samples=10)
            for p in seg:
                p.t += t_offset
                try:
                    p.ee_pos = self._model.ee_position(p.q)
                except Exception:
                    pass
                all_pts.append(p)
            t_offset = all_pts[-1].t

        seen: Dict[float, TrajectoryPoint] = {}
        for p in all_pts:
            seen[round(p.t, 9)] = p
        pts = sorted(seen.values(), key=lambda p: p.t)

        if len(pts) < 2:
            pts = [
                TrajectoryPoint(t=0.0, q=q0, ee_pos=p0),
                TrajectoryPoint(t=0.1, q=q_list[-1]),
            ]

        traj = Trajectory(pts, metadata={
            "type": "circular",
            "center": c.tolist(),
            "angle_rad": round(angle_rad, 4),
        })

        if obstacle_map is not None:
            ee_pts = [p.ee_pos for p in pts if p.ee_pos is not None]
            report = obstacle_map.check_path_clear(ee_pts)
            traj.metadata["collision"] = report.to_dict()

        return traj


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionResult + ExecutionStatus
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionStatus(str, enum.Enum):
    IDLE      = "idle"
    RUNNING   = "running"
    COMPLETED = "completed"
    ABORTED   = "aborted"
    ERROR     = "error"


@dataclass
class ExecutionResult:
    trajectory_id: str
    status: ExecutionStatus
    elapsed_s: float
    tracking_errors: List[float] = field(default_factory=list)
    final_q: Optional[np.ndarray] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "status": self.status.value,
            "elapsed_s": round(self.elapsed_s, 4),
            "mean_tracking_error": round(float(np.mean(self.tracking_errors)), 6) if self.tracking_errors else 0.0,
            "max_tracking_error":  round(float(np.max(self.tracking_errors)),  6) if self.tracking_errors else 0.0,
            "final_q": self.final_q.tolist() if self.final_q is not None else None,
            "message": self.message,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryExecutor
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryExecutor:
    """
    Executes a Trajectory on a PhysiCore engine, tracking reference vs actual
    joint state.  Supports synchronous and asynchronous (background thread) modes.
    """

    def __init__(self, engine: Any, control_hz: float = 100.0):
        self._engine = engine
        self._hz = control_hz
        self._status = ExecutionStatus.IDLE
        self._result: Optional[ExecutionResult] = None
        self._thread: Optional[threading.Thread] = None
        self._abort_flag = threading.Event()
        self._lock = threading.Lock()

    @property
    def status(self) -> ExecutionStatus:
        with self._lock:
            return self._status

    @property
    def result(self) -> Optional[ExecutionResult]:
        with self._lock:
            return self._result

    def execute(self, trajectory: Trajectory) -> ExecutionResult:
        """Synchronous execution — blocks until complete or aborted."""
        self._abort_flag.clear()
        with self._lock:
            self._status = ExecutionStatus.RUNNING
        result = self._run(trajectory)
        with self._lock:
            self._result = result
        return result

    def execute_async(self, trajectory: Trajectory) -> str:
        """Start execution in a background thread; returns trajectory_id."""
        self._abort_flag.clear()
        with self._lock:
            self._status = ExecutionStatus.RUNNING
            self._result = None
        self._thread = threading.Thread(
            target=self._run_async, args=(trajectory,), daemon=True
        )
        self._thread.start()
        return trajectory.trajectory_id

    def abort(self) -> bool:
        """Signal the executor to stop. Returns True if there was an active execution."""
        if self._status == ExecutionStatus.RUNNING:
            self._abort_flag.set()
            return True
        return False

    def wait(self, timeout: Optional[float] = None) -> Optional[ExecutionResult]:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        with self._lock:
            return self._result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self, trajectory: Trajectory) -> ExecutionResult:
        t_start = time.monotonic()
        errors: List[float] = []
        interval = 1.0 / self._hz

        ref_seq = trajectory.to_ref_sequence(hz=self._hz)
        final_q: Optional[np.ndarray] = None

        for ref_pt in ref_seq:
            if self._abort_flag.is_set():
                elapsed = time.monotonic() - t_start
                return ExecutionResult(
                    trajectory_id=trajectory.trajectory_id,
                    status=ExecutionStatus.ABORTED,
                    elapsed_s=elapsed,
                    tracking_errors=errors,
                    final_q=final_q,
                    message="Aborted by user",
                )

            t_tick = time.monotonic()

            ref_state = ref_pt.q
            try:
                result = self._engine.step(ref_state, ref_state)
                if isinstance(result, dict):
                    actual_q = np.asarray(result.get("state", ref_state), dtype=float)
                else:
                    actual_q = np.asarray(result, dtype=float)
                if len(actual_q) >= len(ref_state):
                    err = float(np.linalg.norm(actual_q[:len(ref_state)] - ref_state))
                else:
                    err = 0.0
                errors.append(err)
                final_q = actual_q
            except Exception:
                errors.append(0.0)
                final_q = ref_state

            elapsed_tick = time.monotonic() - t_tick
            sleep = max(0.0, interval - elapsed_tick)
            if sleep > 0:
                time.sleep(sleep)

        elapsed = time.monotonic() - t_start
        with self._lock:
            self._status = ExecutionStatus.COMPLETED
        return ExecutionResult(
            trajectory_id=trajectory.trajectory_id,
            status=ExecutionStatus.COMPLETED,
            elapsed_s=elapsed,
            tracking_errors=errors,
            final_q=final_q,
            message="Completed",
        )

    def _run_async(self, trajectory: Trajectory) -> None:
        result = self._run(trajectory)
        with self._lock:
            self._result = result
            if self._status == ExecutionStatus.RUNNING:
                self._status = result.status

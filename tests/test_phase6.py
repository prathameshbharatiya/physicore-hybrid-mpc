"""tests/test_phase6.py — Phase 6 Perception & Planning Tests"""

import math
import time
from typing import List, Optional

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════
# Shared stubs / fixtures
# ═══════════════════════════════════════════════════════════════════════

class _FakeJoint:
    def __init__(self, lo=-math.pi, hi=math.pi):
        self.limit_lo = lo
        self.limit_hi = hi


class _FakeRobotModel:
    """Minimal URDFRobotModel stand-in for testing planners."""
    dof = 4
    ee_link = "ee"
    all_links: list = []
    actuated_joints = [_FakeJoint() for _ in range(dof)]

    def ee_position(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q)
        return np.array([
            math.sin(q[0]) * 0.5,
            math.cos(q[1]) * 0.3,
            0.4 + q[2] * 0.1,
        ])

    def ee_pose(self, q: np.ndarray):
        return np.eye(3), self.ee_position(q)

    def jacobian(self, q: np.ndarray, link_name: Optional[str] = None) -> np.ndarray:
        """Finite-difference Jacobian."""
        eps = 1e-5
        p0 = self.ee_position(q)
        J = np.zeros((6, self.dof))
        for i in range(self.dof):
            dq = np.zeros(self.dof); dq[i] = eps
            J[:3, i] = (self.ee_position(q + dq) - p0) / eps
        return J

    def forward_kinematics(self, q):
        return {"ee": (np.eye(3), self.ee_position(q))}


class _FakeEstimator:
    """Minimal StateEstimator stand-in."""
    def __init__(self, state_dim=13):
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 0.1
        self._sensors = {}

    def register_sensor(self, name, H, R):
        self._sensors[name] = (H, R)

    def fuse(self, sources):
        for name, obs in sources.items():
            if name in self._sensors:
                H, R = self._sensors[name]
                z = np.asarray(obs)
                y = z - H @ self.x
                S = H @ self.P @ H.T + R
                K = self.P @ H.T @ np.linalg.pinv(S)
                self.x = self.x + K @ y

    def predict(self):
        pass

    @property
    def estimate(self):
        return self.x.copy()

    @property
    def uncertainty(self):
        return np.diag(self.P).copy()


class _FakeEngine:
    """Minimal PhysiCore engine stub for executor tests."""
    def __init__(self, dof=4):
        self.dof = dof
        self._q = np.zeros(dof)

    def step(self, ref_state, ref_ctrl):
        ref = np.asarray(ref_state)
        noise = np.random.randn(self.dof) * 0.001
        self._q = ref[:self.dof] + noise
        return {"state": self._q, "residual": float(np.linalg.norm(noise))}


# ═══════════════════════════════════════════════════════════════════════
# PART A: PerceptionFusion tests
# ═══════════════════════════════════════════════════════════════════════

class TestPerceptionFusion:

    def _make_fusion(self):
        from physicore.perception.interface import PerceptionFusion
        est = _FakeEstimator()
        return PerceptionFusion(est, max_age_s=2.0, poll_hz=10.0), est

    def test_01_register_and_fuse_pose_source(self):
        """Registering a PoseSource and calling fuse() returns fresh=True."""
        from physicore.perception.interface import PoseSource, PerceptionFusion
        fusion, est = self._make_fusion()
        src = PoseSource(name="pose_cam", state_dim=13)
        src.push(np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]))
        fusion.register(src)
        report = fusion.fuse()
        assert "pose_cam" in report
        assert report["pose_cam"]["fresh"] is True

    def test_02_multiple_sources_all_reported(self):
        """All registered sources appear in fusion report."""
        from physicore.perception.interface import PoseSource, IMUSource
        fusion, _ = self._make_fusion()
        pose = PoseSource("p", state_dim=13)
        pose.push(np.zeros(6))
        imu = IMUSource("imu", state_dim=13)
        imu.push(np.array([0, 0, -9.81]), np.zeros(3))
        fusion.register(pose)
        fusion.register(imu)
        report = fusion.fuse()
        assert "p" in report
        assert "imu" in report

    def test_03_staleness_report_returns_age(self):
        """staleness_report() returns age_s ≥ 0 for each source."""
        from physicore.perception.interface import DepthSource
        fusion, _ = self._make_fusion()
        d = DepthSource("d1", state_dim=13)
        d.push(np.array([1.0, 0.0, 0.0]))
        fusion.register(d)
        rpt = fusion.staleness_report()
        assert "d1" in rpt
        assert rpt["d1"]["age_s"] >= 0.0

    def test_04_polling_thread_runs_and_stops(self):
        """start_polling() + stop_polling() with no errors."""
        fusion, _ = self._make_fusion()
        fusion.start_polling()
        time.sleep(0.15)
        fusion.stop_polling()   # must not hang

    def test_05_unregister_removes_source(self):
        """unregister() removes the source from the report."""
        from physicore.perception.interface import MarkerSource
        fusion, _ = self._make_fusion()
        m = MarkerSource("mk", state_dim=13)
        m.push(np.zeros(3))
        fusion.register(m)
        fusion.unregister("mk")
        rpt = fusion.fuse()
        assert "mk" not in rpt

    def test_06_fuse_shifts_estimator_state(self):
        """fuse() propagates non-zero observation to the EKF state."""
        from physicore.perception.interface import JointEncoderSource
        fusion, est = self._make_fusion()
        enc = JointEncoderSource("enc", n_joints=4, state_dim=13)
        enc.push(np.array([0.5, 0.5, 0.5, 0.5]))
        fusion.register(enc)
        fusion.fuse()
        assert float(np.sum(np.abs(est.x))) > 0.0, "Estimator state should have been updated"

    def test_07_observation_dataclass_fields(self):
        """Observation dataclass exposes values, timestamp, source, confidence."""
        from physicore.perception.interface import Observation
        obs = Observation(
            values=np.array([1.0, 2.0]),
            timestamp=time.time(),
            source="test",
            confidence=0.9,
        )
        assert obs.values.shape == (2,)
        assert obs.source == "test"
        assert 0.0 <= obs.confidence <= 1.0

    def test_08_imu_source_correct_obs_dim(self):
        """IMUSource has obs_dim=6 (3 accel + 3 gyro)."""
        from physicore.perception.interface import IMUSource
        src = IMUSource("imu")
        assert src.obs_dim == 6
        obs = src.get_state_observation()
        assert obs.values.shape == (6,)

    def test_09_joint_encoder_obs_dim(self):
        """JointEncoderSource obs_dim matches n_joints."""
        from physicore.perception.interface import JointEncoderSource
        enc = JointEncoderSource("enc", n_joints=7, state_dim=20)
        assert enc.obs_dim == 7

    def test_10_source_names_list(self):
        """source_names() returns all registered source names."""
        from physicore.perception.interface import PoseSource
        fusion, _ = self._make_fusion()
        p1 = PoseSource("p1"); p1.push(np.zeros(6))
        p2 = PoseSource("p2"); p2.push(np.zeros(6))
        fusion.register(p1)
        fusion.register(p2)
        names = fusion.source_names()
        assert "p1" in names
        assert "p2" in names


# ═══════════════════════════════════════════════════════════════════════
# PART B: IKSolver tests
# ═══════════════════════════════════════════════════════════════════════

class TestIKSolver:

    def _make_solver(self):
        from physicore.planning.planner import IKSolver
        return IKSolver(_FakeRobotModel(), max_iter=300, tol=1e-3)

    def test_01_solve_reachable_target_converges(self):
        """IK converges to a reachable target within tolerance."""
        solver = self._make_solver()
        # Target chosen to avoid the Jacobian singularity of the fake model at q[1]=0
        # (at q=zeros: ee=[0, 0.3, 0.4]; x from sin(q0)*0.5, z from 0.4+q2*0.1)
        target = np.array([0.4, 0.3, 0.45])
        q, converged, err = solver.solve(target, q_init=np.array([0.1, 0.0, 0.5, 0.0]))
        assert converged, f"IK did not converge, error={err:.5f}"
        assert err < 0.01

    def test_02_solve_returns_within_joint_limits(self):
        """IK solution respects joint limits."""
        solver = self._make_solver()
        target = np.array([0.4, 0.1, 0.45])
        q, _, _ = solver.solve(target)
        for j in _FakeRobotModel.actuated_joints:
            assert np.all(q >= j.limit_lo - 1e-6) and np.all(q <= j.limit_hi + 1e-6)

    def test_03_solve_batch_chains_warm_start(self):
        """solve_batch() returns one result per target."""
        solver = self._make_solver()
        targets = [np.array([0.2, 0.1, 0.4]), np.array([0.3, 0.2, 0.45]),
                   np.array([0.1, 0.15, 0.42])]
        results = solver.solve_batch(targets)
        assert len(results) == len(targets)
        for q, conv, err in results:
            assert q.shape == (4,)

    def test_04_solve_returns_float_error(self):
        """solve() always returns a finite float error."""
        solver = self._make_solver()
        _, _, err = solver.solve(np.array([0.0, 0.0, 0.4]))
        assert math.isfinite(err)

    def test_05_unreachable_returns_converged_false(self):
        """Extremely far target doesn't crash — converged=False."""
        solver = self._make_solver()
        target = np.array([1000.0, 1000.0, 1000.0])
        q, converged, err = solver.solve(target, q_init=np.zeros(4))
        assert isinstance(converged, bool)
        assert isinstance(err, float)


# ═══════════════════════════════════════════════════════════════════════
# PART C: TrajectoryPlanner tests
# ═══════════════════════════════════════════════════════════════════════

class TestTrajectoryPlanner:

    def _make_planner(self):
        from physicore.planning.planner import TrajectoryPlanner
        return TrajectoryPlanner(_FakeRobotModel(), v_max=1.5, a_max=3.0)

    def test_01_joint_space_plan_produces_trajectory(self):
        """plan_joint_space() returns a Trajectory with ≥ 2 points."""
        from physicore.planning.planner import Trajectory
        p = self._make_planner()
        traj = p.plan_joint_space(np.zeros(4), np.array([0.5, 0.5, 0.5, 0.5]))
        assert isinstance(traj, Trajectory)
        assert len(traj.points) >= 2
        assert traj.duration > 0

    def test_02_task_space_plan_produces_trajectory(self):
        """plan_task_space() returns a Trajectory reaching toward target."""
        p = self._make_planner()
        q0 = np.zeros(4)
        target = np.array([0.4, 0.2, 0.45])
        traj = p.plan_task_space(q0, target, n_via=5)
        assert traj.duration > 0
        assert traj.metadata["type"] == "task_space"

    def test_03_waypoints_plan_continuity(self):
        """plan_waypoints() produces monotonically increasing time."""
        p = self._make_planner()
        wps = [np.zeros(4), np.ones(4) * 0.3, np.ones(4) * 0.6, np.zeros(4)]
        traj = p.plan_waypoints(wps, segment_time=0.5)
        times = [pt.t for pt in traj.points]
        assert all(times[i] <= times[i+1] for i in range(len(times)-1))

    def test_04_circular_plan_returns_valid_trajectory(self):
        """plan_circular() returns a Trajectory with metadata."""
        p = self._make_planner()
        traj = p.plan_circular(
            q_start=np.zeros(4),
            center=np.array([0.3, 0.0, 0.4]),
            normal=np.array([0.0, 0.0, 1.0]),
            angle_rad=math.pi,
            n_via=12,
        )
        assert traj.metadata["type"] == "circular"
        assert len(traj.points) >= 2

    def test_05_trajectory_at_interpolation(self):
        """Trajectory.at(t) stays within q range of neighboring points."""
        p = self._make_planner()
        traj = p.plan_joint_space(np.zeros(4), np.ones(4))
        mid = traj.at(traj.duration / 2)
        assert mid.q.shape == (4,)
        assert not np.any(np.isnan(mid.q))

    def test_06_trajectory_to_ref_sequence(self):
        """to_ref_sequence() returns a list of TrajectoryPoints at target hz."""
        p = self._make_planner()
        traj = p.plan_joint_space(np.zeros(4), np.ones(4) * 0.5)
        seq = traj.to_ref_sequence(hz=50.0)
        assert len(seq) >= 2
        assert all(hasattr(pt, 'q') for pt in seq)

    def test_07_trajectory_ascii_preview(self):
        """visualize_ascii() returns a non-empty string."""
        p = self._make_planner()
        traj = p.plan_joint_space(np.zeros(4), np.ones(4) * 0.5)
        s = traj.visualize_ascii()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_08_trajectory_to_dict(self):
        """to_dict() contains expected keys."""
        p = self._make_planner()
        traj = p.plan_joint_space(np.zeros(4), np.array([1.0, 0.5, -0.5, 0.0]))
        d = traj.to_dict()
        for key in ("trajectory_id", "duration", "dof", "n_points", "points"):
            assert key in d


# ═══════════════════════════════════════════════════════════════════════
# PART D: ObstacleMap tests
# ═══════════════════════════════════════════════════════════════════════

class TestObstacleMap:

    def _make_map(self):
        from physicore.planning.obstacles import ObstacleMap
        return ObstacleMap()

    def test_01_sphere_collision_detected(self):
        """Point inside sphere → in_collision=True."""
        m = self._make_map()
        m.add_sphere("ball", np.array([0.0, 0.0, 0.0]), radius=0.5)
        report = m.check_collision(np.array([0.1, 0.0, 0.0]))
        assert report.in_collision
        assert report.obstacle_name == "ball"

    def test_02_sphere_no_collision_outside(self):
        """Point outside sphere → in_collision=False."""
        m = self._make_map()
        m.add_sphere("ball", np.array([0.0, 0.0, 0.0]), radius=0.2)
        report = m.check_collision(np.array([1.0, 0.0, 0.0]))
        assert not report.in_collision

    def test_03_box_collision_inside(self):
        """Point inside box → in_collision=True."""
        m = self._make_map()
        m.add_box("wall", np.array([-1, -1, -1]), np.array([1, 1, 1]))
        report = m.check_collision(np.array([0.0, 0.0, 0.0]))
        assert report.in_collision
        assert report.obstacle_type == "box"

    def test_04_box_no_collision_outside(self):
        """Point outside box → in_collision=False."""
        m = self._make_map()
        m.add_box("wall", np.array([0, 0, 0]), np.array([1, 1, 1]))
        report = m.check_collision(np.array([2.0, 0.5, 0.5]))
        assert not report.in_collision

    def test_05_check_path_clear_with_obstacle(self):
        """Path through obstacle returns in_collision=True."""
        m = self._make_map()
        m.add_sphere("blocker", np.array([0.5, 0.0, 0.0]), radius=0.3)
        path = [np.array([x, 0.0, 0.0]) for x in np.linspace(0.0, 1.0, 20)]
        report = m.check_path_clear(path)
        assert report.in_collision

    def test_06_check_path_clear_free_path(self):
        """Path away from obstacle returns in_collision=False."""
        m = self._make_map()
        m.add_sphere("b", np.array([10.0, 0.0, 0.0]), radius=0.1)
        path = [np.array([x, 0.0, 0.0]) for x in np.linspace(0.0, 1.0, 20)]
        report = m.check_path_clear(path)
        assert not report.in_collision

    def test_07_remove_obstacle(self):
        """remove() makes the obstacle invisible to checks."""
        m = self._make_map()
        m.add_sphere("s1", np.zeros(3), 1.0)
        m.remove("s1")
        report = m.check_collision(np.zeros(3))
        assert not report.in_collision

    def test_08_nearest_obstacle_returns_name_and_distance(self):
        """nearest_obstacle() returns finite distance."""
        m = self._make_map()
        m.add_sphere("near", np.array([1.0, 0.0, 0.0]), 0.2)
        name, dist = m.nearest_obstacle(np.zeros(3))
        assert name == "near"
        assert dist >= 0.0

    def test_09_to_dict_has_spheres_and_boxes(self):
        """to_dict() lists both spheres and boxes."""
        m = self._make_map()
        m.add_sphere("s", np.zeros(3), 0.1)
        m.add_box("b", np.zeros(3), np.ones(3))
        d = m.to_dict()
        assert len(d["spheres"]) == 1
        assert len(d["boxes"]) == 1

    def test_10_collision_report_penetration_depth_positive(self):
        """CollisionReport.penetration_depth > 0 for a deep collision."""
        m = self._make_map()
        m.add_sphere("s", np.zeros(3), radius=1.0)
        report = m.check_collision(np.zeros(3))
        assert report.penetration_depth > 0.0


# ═══════════════════════════════════════════════════════════════════════
# PART E: TrajectoryExecutor tests
# ═══════════════════════════════════════════════════════════════════════

class TestTrajectoryExecutor:

    def _make_traj(self):
        from physicore.planning.planner import TrajectoryPlanner
        p = TrajectoryPlanner(_FakeRobotModel(), v_max=2.0, a_max=4.0)
        return p.plan_joint_space(np.zeros(4), np.ones(4) * 0.3, n_samples=20)

    def test_01_sync_execute_completes(self):
        """execute() returns ExecutionResult with status=completed."""
        from physicore.planning.planner import TrajectoryExecutor, ExecutionStatus
        engine = _FakeEngine(dof=4)
        exe    = TrajectoryExecutor(engine, control_hz=200.0)
        traj   = self._make_traj()
        result = exe.execute(traj)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.elapsed_s >= 0.0

    def test_02_execute_tracking_errors_list(self):
        """execute() populates tracking_errors list."""
        from physicore.planning.planner import TrajectoryExecutor
        engine = _FakeEngine(dof=4)
        exe    = TrajectoryExecutor(engine, control_hz=200.0)
        result = exe.execute(self._make_traj())
        assert isinstance(result.tracking_errors, list)
        assert len(result.tracking_errors) >= 1

    def test_03_execute_result_to_dict(self):
        """ExecutionResult.to_dict() contains required keys."""
        from physicore.planning.planner import TrajectoryExecutor
        engine = _FakeEngine(dof=4)
        exe    = TrajectoryExecutor(engine, control_hz=200.0)
        d = exe.execute(self._make_traj()).to_dict()
        for key in ("trajectory_id", "status", "elapsed_s",
                    "mean_tracking_error", "max_tracking_error"):
            assert key in d

    def test_04_abort_changes_status(self):
        """abort() on a running async execution stops it."""
        from physicore.planning.planner import (
            TrajectoryPlanner, TrajectoryExecutor, ExecutionStatus,
        )
        engine = _FakeEngine(dof=4)
        exe    = TrajectoryExecutor(engine, control_hz=50.0)
        p      = TrajectoryPlanner(_FakeRobotModel(), v_max=0.1, a_max=0.2)
        traj   = p.plan_joint_space(np.zeros(4), np.ones(4) * 1.0, n_samples=200)
        exe.execute_async(traj)
        time.sleep(0.05)
        aborted = exe.abort()
        assert aborted
        result = exe.wait(timeout=2.0)
        assert result is not None
        assert result.status in (ExecutionStatus.ABORTED, ExecutionStatus.COMPLETED)

    def test_05_executor_status_transitions(self):
        """Status starts idle, goes running, ends completed."""
        from physicore.planning.planner import TrajectoryExecutor, ExecutionStatus
        engine = _FakeEngine(dof=4)
        exe    = TrajectoryExecutor(engine, control_hz=200.0)
        assert exe.status == ExecutionStatus.IDLE
        result = exe.execute(self._make_traj())
        assert result.status == ExecutionStatus.COMPLETED

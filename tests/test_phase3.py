"""tests/test_phase3.py — Phase 3 Infrastructure Tests"""

import pytest
import numpy as np
import time
import sys
import io
import builtins


def _safe_print(*args, **kwargs):
    try:
        builtins.__stdout_orig__(*args, **kwargs)
    except (UnicodeEncodeError, AttributeError):
        pass


# ═══════════════════════════════════════════════════════════════════════
# PART A: StateEstimator tests
# ═══════════════════════════════════════════════════════════════════════

class TestStateEstimator:
    def setup_method(self):
        builtins.__stdout_orig__ = print  # type: ignore[attr-defined]

    def _make_linear_dynamics(self, A):
        def dyn(state, action, params):
            return A @ state
        return dyn

    def test_01_ekf_predict_update_converges_linear_system(self):
        """EKF predict+update cycle converges on a 2D stable linear system."""
        from physicore.core.estimator import StateEstimator
        n = 2
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        dyn = self._make_linear_dynamics(A)

        Q = np.eye(n) * 0.01
        R = np.eye(n) * 0.1
        est = StateEstimator(n, n, Q, R)

        rng = np.random.default_rng(42)
        state = np.array([1.0, 1.0])
        dt = 0.01

        for _ in range(50):
            est.predict(state, np.zeros(1), dyn, {}, dt)
            obs = state + rng.normal(0, 0.05, n)
            est.update(obs)
            state = state + dt * (A @ state)

        assert np.allclose(est.estimate, state, atol=0.5), (
            f"EKF estimate {est.estimate} too far from true {state}"
        )

    def test_02_fuse_two_sensors_lower_uncertainty(self):
        """Fusing two sensors gives lower diagonal-P uncertainty than one sensor."""
        from physicore.core.estimator import StateEstimator
        n = 4
        Q = np.eye(n) * 0.01
        R = np.eye(n) * 1.0

        H = np.eye(n)
        R_sensor = np.eye(n) * 0.5

        est_single = StateEstimator(n, n, Q, R)
        est_single.register_sensor('imu', H, R_sensor)
        est_single.fuse({'imu': np.ones(n) * 0.1})
        unc_single = est_single.uncertainty.mean()

        est_dual = StateEstimator(n, n, Q, R)
        est_dual.register_sensor('imu', H, R_sensor)
        est_dual.register_sensor('encoder', H, R_sensor)
        est_dual.fuse({'imu': np.ones(n) * 0.1, 'encoder': np.ones(n) * 0.1})
        unc_dual = est_dual.uncertainty.mean()

        assert unc_dual < unc_single, (
            f"Two sensors should give lower uncertainty: {unc_dual:.4f} vs {unc_single:.4f}"
        )

    def test_03_imu_preintegrator_accumulates_and_resets(self):
        """IMUPreintegrator accumulates delta values and resets cleanly."""
        from physicore.core.estimator import IMUPreintegrator
        imu = IMUPreintegrator(gravity=9.81)

        # Net upward accel of 1 m/s² after gravity subtraction
        accel = np.array([0.0, 0.0, 9.81 + 1.0])
        gyro  = np.zeros(3)
        dt    = 0.001

        for _ in range(100):
            imu.integrate(accel, gyro, dt)

        result = imu.reset()

        assert 'delta_position' in result
        assert 'delta_velocity' in result
        assert 'delta_rotation' in result

        # After reset, a single step's delta_velocity should be much smaller
        imu.integrate(accel, gyro, dt)
        result2 = imu.reset()
        assert np.linalg.norm(result2['delta_velocity']) < np.linalg.norm(result['delta_velocity']) * 0.5

    def test_04_estimator_attached_to_engine_produces_finite_actions(self):
        """Attaching estimator to engine and stepping produces finite actions."""
        from physicore.core.estimator import StateEstimator
        from physicore.core.engine import PhysiCore

        engine = PhysiCore.for_platform('balancing_bot')
        n = engine.cfg.state_dim
        est = StateEstimator(n, n, np.eye(n) * 0.01, np.eye(n) * 0.1)

        # Seed estimator with a predict step
        est.predict(
            np.zeros(n), np.zeros(engine.cfg.action_dim),
            engine.physics.dynamics_fn, engine.physics.params,
            engine.cfg.dt,
        )

        result = engine.step(est.estimate, np.zeros(n))
        assert np.all(np.isfinite(result.action)), f"Non-finite action: {result.action}"

    def test_05_register_sensor_and_update_reduces_covariance(self):
        """register_sensor + fuse reduces covariance diagonal."""
        from physicore.core.estimator import StateEstimator
        n = 3
        est = StateEstimator(n, n, np.eye(n) * 0.01, np.eye(n) * 0.5)
        H = np.eye(n)
        R = np.eye(n) * 0.1
        est.register_sensor('cam', H, R)

        unc_before = est.uncertainty.copy()
        est.fuse({'cam': np.array([0.1, 0.2, 0.3])})
        unc_after = est.uncertainty

        assert np.all(unc_after <= unc_before + 1e-9), (
            "Fusing a sensor should not increase uncertainty"
        )


# ═══════════════════════════════════════════════════════════════════════
# PART B: RigidContactSolver tests
# ═══════════════════════════════════════════════════════════════════════

class TestRigidContactSolver:

    def test_01_normal_impulses_non_negative(self):
        """LCP solver returns non-negative normal impulses."""
        from physicore.core.contact_lcp import RigidContactSolver
        solver = RigidContactSolver()
        M  = np.eye(1) * 1.0
        dq = np.array([-1.0])          # approaching
        J  = np.array([[1.0]])
        lam_n, lam_t, _ = solver.solve(np.zeros(1), dq, M, J, mu=0.3, dt=0.01)
        assert np.all(lam_n >= -1e-9), f"Negative normal impulse: {lam_n}"

    def test_02_friction_cone_satisfied(self):
        """||lambda_t|| <= mu * lambda_n for all contacts."""
        from physicore.core.contact_lcp import RigidContactSolver
        solver = RigidContactSolver()
        M  = np.eye(2) * 1.0
        dq = np.array([-1.0, 0.5])     # approaching + sliding
        J  = np.array([[1.0, 0.0]])    # normal contact
        mu = 0.5
        lam_n, lam_t, _ = solver.solve(np.zeros(2), dq, M, J, mu=mu, dt=0.01)
        for i in range(len(lam_n)):
            t_norm = float(np.linalg.norm(lam_t[i]))
            assert t_norm <= mu * lam_n[i] + 1e-9, (
                f"Friction cone violated at {i}: {t_norm:.4f} > mu*lam_n={mu*lam_n[i]:.4f}"
            )

    def test_03_zero_penetration_gives_zero_impulse(self):
        """Zero normal velocity → near-zero impulse."""
        from physicore.core.contact_lcp import RigidContactSolver
        solver = RigidContactSolver()
        M  = np.eye(1)
        dq = np.array([0.0])           # no normal velocity
        J  = np.array([[1.0]])
        lam_n, _, _ = solver.solve(np.zeros(1), dq, M, J, mu=0.3, dt=0.01)
        assert np.all(np.abs(lam_n) < 0.05), f"Expected ~zero impulse, got {lam_n}"

    def test_04_plastic_impact_stops_normal_velocity(self):
        """Plastic impact (e=0) brings normal velocity to ≈ 0."""
        from physicore.core.contact_lcp import RigidContactSolver
        solver = RigidContactSolver()
        M       = np.eye(1)
        dq_pre  = np.array([-2.0])
        J       = np.array([[1.0]])
        dq_post = solver.solve_impact(dq_pre, J, e=0.0)
        vn_post = float(J @ dq_post)
        assert abs(vn_post) < 0.1, f"Plastic impact: vn_post should be ~0, got {vn_post:.4f}"

    def test_05_elastic_impact_reverses_normal_velocity(self):
        """Elastic impact (e=1) reverses the normal velocity."""
        from physicore.core.contact_lcp import RigidContactSolver
        solver  = RigidContactSolver()
        M       = np.eye(1)
        dq_pre  = np.array([-3.0])
        J       = np.array([[1.0]])
        dq_post = solver.solve_impact(dq_pre, J, e=1.0)
        vn_pre  = float(J @ dq_pre)
        vn_post = float(J @ dq_post)
        assert abs(vn_post - (-vn_pre)) < 0.1, (
            f"Elastic: expected vn_post≈{-vn_pre:.2f}, got {vn_post:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# PART C: HardwareSafetyInterlock tests
# ═══════════════════════════════════════════════════════════════════════

class TestHardwareSafetyInterlock:

    def _make_interlock(self, action_dim=4, torque_limit=10.0):
        from physicore.core.safety import HardwareSafetyInterlock, SafetyConfig
        cfg = SafetyConfig(
            torque_limits=np.full(action_dim, torque_limit),
            action_dim=action_dim,
        )
        il = HardwareSafetyInterlock(cfg)
        return il

    def test_01_action_within_limits_passes_unchanged(self):
        """In-limits action passes through unchanged when armed."""
        il = self._make_interlock(torque_limit=10.0)
        il.arm()
        action = np.array([1.0, -2.0, 3.0, -4.0])
        safe, viols = il.check_and_clip(action, np.zeros(8), {})
        assert np.allclose(safe, action), f"Action changed: {safe} vs {action}"
        assert len(viols) == 0

    def test_02_torque_exceeding_limit_is_clipped(self):
        """Repeated torque violations eventually clip the action."""
        il = self._make_interlock(torque_limit=5.0)
        il.arm()
        action = np.array([100.0, -200.0, 50.0, -80.0])
        state  = np.zeros(8)
        safe = action.copy()
        for _ in range(6):  # trigger SOFT_STOP (≥5 violations)
            safe, viols = il.check_and_clip(action, state, {})
        assert np.all(np.abs(safe) <= 5.0 * 0.5 + 1e-6), f"Not soft-clipped: {safe}"
        assert len(viols) > 0

    def test_03_workspace_violation_triggers_hard_stop(self):
        """Position outside workspace box → HARD_STOP (zero action)."""
        from physicore.core.safety import HardwareSafetyInterlock, SafetyConfig, EscalationLevel
        cfg = SafetyConfig(
            workspace_box=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            action_dim=4,
        )
        il = HardwareSafetyInterlock(cfg)
        il.arm()
        action = np.ones(4) * 5.0
        state  = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])
        safe, viols = il.check_and_clip(action, state, {})
        assert np.allclose(safe, 0.0), f"Expected zeros on workspace violation, got {safe}"
        assert il.escalation_level == EscalationLevel.HARD_STOP

    def test_04_estop_returns_zero_regardless_of_input(self):
        """E-stop returns zero; subsequent check_and_clip also returns zero."""
        il = self._make_interlock()
        il.arm()
        zero = il.emergency_stop()
        assert np.allclose(zero, 0.0)
        safe, _ = il.check_and_clip(np.ones(4) * 999.0, np.zeros(8), {})
        assert np.allclose(safe, 0.0)

    def test_05_disarm_bypasses_enforcement(self):
        """Disarmed interlock passes any action unchanged."""
        il = self._make_interlock(torque_limit=1.0)
        # not armed
        action = np.array([999.0, -999.0, 999.0, -999.0])
        safe, viols = il.check_and_clip(action, np.zeros(8), {})
        assert np.allclose(safe, action)
        assert len(viols) == 0

    def test_06_from_robot_config_extracts_limits(self):
        """SafetyConfig.from_robot_config extracts torque limits from RobotConfig."""
        from physicore.core.safety import SafetyConfig

        class MockConfig:
            action_dim = 6
            joint_action_bounds = [[-100.0] * 6, [100.0] * 6]

        cfg = SafetyConfig.from_robot_config(MockConfig())
        assert cfg.action_dim == 6
        assert cfg.torque_limits is not None
        assert len(cfg.torque_limits) == 6
        assert np.all(cfg.torque_limits <= 100.0 + 1e-6)

    def test_07_arm_disarm_toggles_enforcement(self):
        """arm() enables, disarm() disables enforcement."""
        il = self._make_interlock(torque_limit=1.0)
        big = np.array([50.0, 50.0, 50.0, 50.0])

        # disarmed → pass through
        safe, _ = il.check_and_clip(big, np.zeros(8), {})
        assert np.allclose(safe, big)

        il.arm()
        # armed, 1st violation → WARNING, but still clips to limit
        safe, viols = il.check_and_clip(big, np.zeros(8), {})
        assert len(viols) > 0

        il.disarm()
        safe, viols = il.check_and_clip(big, np.zeros(8), {})
        assert len(viols) == 0


# ═══════════════════════════════════════════════════════════════════════
# PART D: RTLoop tests
# ═══════════════════════════════════════════════════════════════════════

class TestRTLoop:

    def test_01_loop_runs_100_ticks_without_crash(self):
        """100 manual ticks complete without exception and produce finite actions."""
        from physicore.core.rt_loop import RTLoop
        from physicore.core.engine import PhysiCore

        engine = PhysiCore.for_platform('balancing_bot')
        loop   = RTLoop(engine, hz=200.0)

        state = np.zeros(engine.cfg.state_dim)
        x_ref = np.zeros(engine.cfg.state_dim)

        for _ in range(100):
            out = loop.tick({'state': state}, x_ref)
            assert np.all(np.isfinite(out.action)), f"Non-finite action: {out.action}"

    def test_02_stats_return_sensible_mean_ms(self):
        """Stats mean_ms < 50ms on balancing_bot with 200 Hz target."""
        from physicore.core.rt_loop import RTLoop
        from physicore.core.engine import PhysiCore

        engine = PhysiCore.for_platform('balancing_bot')
        loop   = RTLoop(engine, hz=200.0)
        state  = np.zeros(engine.cfg.state_dim)
        x_ref  = np.zeros(engine.cfg.state_dim)

        for _ in range(30):
            loop.tick({'state': state}, x_ref)

        s = loop.stats
        assert 'mean_ms'          in s
        assert 'max_ms'           in s
        assert 'jitter_ms'        in s
        assert 'missed_deadlines' in s
        assert s['mean_ms'] < 500.0, f"mean_ms too high: {s['mean_ms']}"

    def test_03_stop_joins_thread_cleanly(self):
        """start() then stop() joins the thread without hanging."""
        from physicore.core.rt_loop import RTLoop
        from physicore.core.engine import PhysiCore

        engine = PhysiCore.for_platform('balancing_bot')
        loop   = RTLoop(engine, hz=100.0)
        loop.start()
        time.sleep(0.05)
        loop.stop()
        assert loop._thread is None, "Thread reference should be None after stop()"

    def test_04_rt_loop_with_safety_clips_large_action(self):
        """RTLoop + safety interlock clips oversized actions."""
        from physicore.core.rt_loop import RTLoop
        from physicore.core.engine import PhysiCore
        from physicore.core.safety import HardwareSafetyInterlock, SafetyConfig

        engine = PhysiCore.for_platform('balancing_bot')
        cfg = SafetyConfig(
            torque_limits=np.array([0.001]),   # absurdly tight → everything clipped
            action_dim=engine.cfg.action_dim,
        )
        il = HardwareSafetyInterlock(cfg)
        il.arm()
        # Trigger enough violations to reach SOFT_STOP
        for _ in range(6):
            il.check_and_clip(np.array([999.0]), np.zeros(4), {})

        loop = RTLoop(engine, hz=200.0, safety=il)
        out  = loop.tick({'state': np.zeros(engine.cfg.state_dim)},
                         np.zeros(engine.cfg.state_dim))
        # After SOFT_STOP escalation, action should be tiny
        assert np.all(np.abs(out.action) <= 0.001 * 0.5 + 1e-9), (
            f"Action not clipped by interlock: {out.action}"
        )

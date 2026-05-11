"""
Example 6: Full PhysiCore Pipeline
=====================================
End-to-end demonstration combining:
  1. Robot loading (built-in platform)
  2. Perception fusion (simulated sensors)
  3. Trajectory planning (joint-space)
  4. Trajectory execution (async)
  5. MPC control loop
  6. Telemetry recording

Run:
    python examples/full_pipeline.py
    physicore run full_pipeline
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

print("=" * 60)
print("  PhysiCore -- Full Pipeline Demo")
print("=" * 60)

# ── Step 1: Engine ─────────────────────────────────────────────────────────────
print("\n[1/6] Initialising PhysiCore engine...")
from physicore import PhysiCore, PLATFORM_DYNAMICS

params = {"mass": 1.0, "friction": 0.15, "inertia": 0.01}
engine = PhysiCore.for_platform("balancing_bot", initial_params=params)
dyn_fn = PLATFORM_DYNAMICS["balancing_bot"][0]
_dt = 1 / engine.cfg.control_hz
print(f"      [OK] Engine: balancing_bot, {engine.cfg.state_dim}D state, dt={_dt:.4f}s")

# ── Step 2: Perception fusion ──────────────────────────────────────────────────
print("\n[2/6] Setting up perception fusion...")
fusion = None
try:
    from physicore.perception.interface import (
        PerceptionFusion, JointEncoderSource, Observation,
    )

    class _SimJointEncoder(JointEncoderSource):
        def __init__(self):
            super().__init__(n_joints=2)

        def get_state_observation(self):
            return Observation(
                values=np.array([0.1, 0.0]),
                source="joint_encoder",
                confidence=0.95,
            )

    try:
        from physicore.core.state_estimator import StateEstimator
        estimator = StateEstimator(state_dim=4, obs_dim=2)
        enc = _SimJointEncoder()
        fusion = PerceptionFusion(estimator)
        fusion.register(enc)
        print("      [OK] Perception fusion: joint encoder registered")
    except Exception as e:
        print(f"      [--] Perception fusion: skipped ({e})")

except ImportError as e:
    print(f"      [--] Perception module unavailable ({e}), skipping")

# ── Step 3: Trajectory planning ────────────────────────────────────────────────
print("\n[3/6] Planning trajectory...")
traj = None
try:
    from physicore.planning.planner import TrajectoryPlanner

    class _FakeArm:
        n_joints = 2
        joint_limits = [(-1.57, 1.57)] * 2

        def forward_kinematics(self, q):
            T = np.eye(4)
            T[0, 3] = 0.3 * np.cos(q[0]) + 0.25 * np.cos(q[0] + q[1])
            T[2, 3] = 0.3 * np.sin(q[0]) + 0.25 * np.sin(q[0] + q[1])
            return T

        def jacobian(self, q):
            J = np.zeros((6, 2))
            J[0, 0] = -0.3 * np.sin(q[0]) - 0.25 * np.sin(q[0] + q[1])
            J[2, 0] =  0.3 * np.cos(q[0]) + 0.25 * np.cos(q[0] + q[1])
            J[0, 1] = -0.25 * np.sin(q[0] + q[1])
            J[2, 1] =  0.25 * np.cos(q[0] + q[1])
            return J

    arm = _FakeArm()
    planner = TrajectoryPlanner(robot=arm)
    q_start = np.array([0.0, 0.5])
    q_goal  = np.array([0.8, -0.3])
    traj = planner.plan_joint_space(q_start, q_goal, duration=2.0)
    n_pts = len(traj.points)
    print(f"      [OK] Joint-space trajectory: {n_pts} points, {traj.duration:.2f} s")
except Exception as e:
    print(f"      [--] Trajectory planning unavailable ({e}), skipping")

# ── Step 4: Trajectory execution ───────────────────────────────────────────────
print("\n[4/6] Executing trajectory (async)...")
if traj is not None:
    try:
        from physicore.planning.planner import TrajectoryExecutor, ExecutionStatus

        class _FakeController:
            def send_joint_command(self, q, qd):
                pass

        executor = TrajectoryExecutor(controller=_FakeController())
        future = executor.execute_async(traj, hz=50)

        deadline = time.time() + 1.0
        while time.time() < deadline and executor.status == ExecutionStatus.RUNNING:
            time.sleep(0.05)

        if future.running():
            executor.abort()
        print(f"      [OK] Execution: status={executor.status.value}")
    except Exception as e:
        print(f"      [--] Executor unavailable ({e}), skipping")
else:
    print("      [--] Skipped (no trajectory)")

# ── Step 5: MPC control loop ──────────────────────────────────────────────────
print("\n[5/6] Running MPC control loop (50 steps)...")
x = np.array([0.15, 0.0, 0.0, 0.0])
x_ref = np.zeros(4)
N = 50
residuals = []

for t in range(N):
    step = engine.step(x, x_ref)
    residuals.append(step.residual_norm)
    xdot = dyn_fn(x, step.action, params)
    x = x + _dt * xdot

print(f"      [OK] MPC: {N} steps, final angle={x[0]:.4f} rad, avg_residual={np.mean(residuals):.6f}")

# ── Step 6: Telemetry ─────────────────────────────────────────────────────────
print("\n[6/6] Recording telemetry...")
try:
    from physicore.api.telemetry_store import get_telemetry_store

    store = get_telemetry_store()
    session_id = store.create_session("bot-001", "balancing_bot")

    for i, r in enumerate(residuals):
        store.record_step(session_id, {
            "step": i, "residual_norm": r, "angle": float(x[0]),
        })

    sessions = store.list_sessions()
    print(f"      [OK] Telemetry: session {session_id[:8]}..., {len(residuals)} steps recorded")
    print(f"      [OK] Total sessions in store: {len(sessions)}")
except Exception as e:
    print(f"      [--] Telemetry unavailable ({e})")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'-'*60}")
print("  PhysiCore Full Pipeline Complete")
print(f"  Engine steps     : {N}")
print(f"  Final state      : angle={x[0]:+.4f} rad, pos={x[2]:+.4f} m")
print(f"  Avg residual     : {np.mean(residuals):.6f}")
print(f"  Trajectory       : {'planned' if traj else 'skipped'}")
print(f"  Perception       : {'active' if fusion else 'skipped'}")
print(f"{'-'*60}")
print("\nFull pipeline demo complete.")

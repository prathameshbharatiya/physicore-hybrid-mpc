"""
Example 3: Load Any URDF
=========================
Demonstrates loading a robot from a URDF file (or generating a synthetic one),
inspecting joints via RobotConfig, and running forward kinematics.

Run:
    python examples/load_any_urdf.py
    python examples/load_any_urdf.py --urdf /path/to/robot.urdf
    physicore run load_any_urdf
"""

import sys
import argparse
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ── Synthetic URDF for demo purposes ─────────────────────────────────────────
_DEMO_URDF = """\
<?xml version="1.0"?>
<robot name="demo_arm">
  <link name="base_link">
    <visual><geometry><box size="0.1 0.1 0.05"/></geometry></visual>
  </link>
  <link name="link1">
    <visual><geometry><cylinder radius="0.03" length="0.3"/></geometry></visual>
  </link>
  <link name="link2">
    <visual><geometry><cylinder radius="0.025" length="0.25"/></geometry></visual>
  </link>
  <link name="link3">
    <visual><geometry><cylinder radius="0.02" length="0.2"/></geometry></visual>
  </link>
  <link name="end_effector"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.025" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="2.0"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="80" velocity="2.0"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/><child link="link3"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="60" velocity="2.0"/>
  </joint>
  <joint name="ee_joint" type="fixed">
    <parent link="link3"/><child link="end_effector"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>
</robot>
"""


def main():
    parser = argparse.ArgumentParser(description="PhysiCore URDF Loader Example")
    parser.add_argument("--urdf", default=None, help="Path to URDF file (default: synthetic demo)")
    args = parser.parse_args()

    print("=" * 60)
    print("  PhysiCore -- Load Any URDF")
    print("=" * 60)

    # ── 1. Prepare URDF ───────────────────────────────────────────────────────
    if args.urdf:
        urdf_path = args.urdf
        print(f"\nLoading: {urdf_path}")
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
        tmp.write(_DEMO_URDF)
        tmp.flush()
        urdf_path = tmp.name
        print(f"\nNo --urdf given. Using synthetic 3-DOF arm.")
        print(f"Temp file: {urdf_path}")

    # ── 2. Load robot ─────────────────────────────────────────────────────────
    try:
        from physicore.core.urdf_loader import load_robot, build_robot_model
        engine, config = load_robot(urdf_path)  # returns (PhysiCore, RobotConfig)
    except Exception as e:
        print(f"\n[ERROR] load_robot failed: {e}")
        print("Tip: make sure the URDF file is well-formed.")
        sys.exit(1)

    # ── 3. Inspect via RobotConfig ────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"  Robot name  : {config.name}")
    print(f"  Platform    : {config.platform}")
    print(f"  DOF         : {config.dof}")
    print(f"  Mass        : {config.mass:.3f} kg")
    print(f"  Control hz  : {config.control_hz:.0f}")

    if config.joint_names:
        print(f"\n  {'Joint':<20} {'Type':<12} {'Lo':>8} {'Hi':>8}")
        print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8}")
        for i, name in enumerate(config.joint_names):
            jtype = config.joint_types[i] if i < len(config.joint_types) else "?"
            lo = config.joint_limits_lo[i] if i < len(config.joint_limits_lo) else -3.14
            hi = config.joint_limits_hi[i] if i < len(config.joint_limits_hi) else 3.14
            print(f"  {name:<20} {jtype:<12} {lo:>8.3f} {hi:>8.3f}")

    # ── 4. Build numerical robot model ────────────────────────────────────────
    try:
        robot_model = build_robot_model(urdf_path)
        n = robot_model.n_joints
        print(f"\n{'-'*60}")
        print(f"  Numerical model : {n} DOF")

        # Forward kinematics at zero config
        q_zero = np.zeros(n)
        fk_zero = robot_model.forward_kinematics(q_zero)
        print(f"  FK (q=0) ee pos : {fk_zero[:3, 3].round(4)}")

        # Jacobian at zero config
        J = robot_model.jacobian(q_zero)
        print(f"  Jacobian shape  : {J.shape}")
        print(f"  Jacobian rank   : {np.linalg.matrix_rank(J)}")

        # Random configuration
        np.random.seed(42)
        lo = np.array(config.joint_limits_lo[:n]) if len(config.joint_limits_lo) >= n else np.full(n, -1.57)
        hi = np.array(config.joint_limits_hi[:n]) if len(config.joint_limits_hi) >= n else np.full(n,  1.57)
        q_rand = lo + (hi - lo) * np.random.rand(n)
        fk_rand = robot_model.forward_kinematics(q_rand)
        print(f"\n  Random config   : {q_rand.round(3)}")
        print(f"  FK (random) ee  : {fk_rand[:3, 3].round(4)}")

    except Exception as e:
        print(f"\n  build_robot_model: {e}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("  URDF loaded and inspected successfully.")
    if not args.urdf:
        try:
            Path(urdf_path).unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows: file still open by loader, cleaned up at process exit


if __name__ == "__main__":
    main()

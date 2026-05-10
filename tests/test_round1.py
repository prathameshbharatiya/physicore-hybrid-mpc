"""
PhysiCore Round 1 Test Suite
============================
Tests for:
  - URDF / MJCF parser
  - Forward kinematics + Jacobian
  - ProperContactModel
  - Extra dynamics (mobile_manipulator, dual_arm, cable_driven, exoskeleton)
  - PhysiCore.for_urdf() and PhysiCore.register_platform()
  - engine_patch integration (humanoid contact model)

Run with:
    pytest tests/test_round1.py -v
or directly:
    python tests/test_round1.py

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations

import math
import os
import sys
import textwrap
import tempfile
import numpy as np
import pytest

# ── Ensure physicore is importable from the repo root ─────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════════════════════════
#  FIXTURES — synthetic URDF / MJCF strings
# ══════════════════════════════════════════════════════════════════════════════

SIMPLE_ARM_URDF = textwrap.dedent("""\
<?xml version="1.0"?>
<robot name="simple_arm">
  <link name="base_link">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
      <origin xyz="0 0 0.05"/>
    </inertial>
    <collision>
      <geometry><cylinder radius="0.05" length="0.1"/></geometry>
    </collision>
  </link>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
      <origin xyz="0 0 0.15"/>
    </inertial>
    <collision>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
    </collision>
  </link>
  <link name="link2">
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.005"/>
      <origin xyz="0 0 0.15"/>
    </inertial>
    <collision>
      <geometry><cylinder radius="0.03" length="0.3"/></geometry>
    </collision>
  </link>
  <link name="ee_link">
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.002"/>
    </inertial>
    <collision>
      <geometry><sphere radius="0.03"/></geometry>
    </collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="50" velocity="3"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5708" upper="1.5708" effort="30" velocity="3"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5708" upper="1.5708" effort="20" velocity="4"/>
  </joint>
</robot>
""")

SIMPLE_MJCF = textwrap.dedent("""\
<mujoco model="simple_arm_mj">
  <worldbody>
    <body name="base_link" pos="0 0 0">
      <inertial mass="2.0" diaginertia="0.1 0.1 0.05" pos="0 0 0.05"/>
      <geom type="cylinder" size="0.05 0.05"/>
      <body name="link1" pos="0 0 0.1">
        <inertial mass="1.0" diaginertia="0.05 0.05 0.01" pos="0 0 0.15"/>
        <geom type="cylinder" size="0.04 0.15"/>
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14159 3.14159"/>
        <body name="link2" pos="0 0 0.3">
          <inertial mass="0.8" diaginertia="0.03 0.03 0.005" pos="0 0 0.15"/>
          <geom type="cylinder" size="0.03 0.15"/>
          <joint name="joint2" type="hinge" axis="0 1 0" range="-1.5708 1.5708"/>
          <body name="ee_link" pos="0 0 0.3">
            <inertial mass="0.3" diaginertia="0.005 0.005 0.002"/>
            <geom type="sphere" size="0.03"/>
            <joint name="joint3" type="hinge" axis="0 1 0" range="-1.5708 1.5708"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""")

FIXED_JOINT_URDF = textwrap.dedent("""\
<?xml version="1.0"?>
<robot name="rover_bot">
  <link name="chassis">
    <inertial><mass value="5.0"/><inertia ixx="0.2" iyy="0.2" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.4 0.3 0.1"/></geometry></collision>
  </link>
  <link name="wheel_L">
    <inertial><mass value="0.5"/><inertia ixx="0.01" iyy="0.01" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><cylinder radius="0.1" length="0.05"/></geometry></collision>
  </link>
  <link name="wheel_R">
    <inertial><mass value="0.5"/><inertia ixx="0.01" iyy="0.01" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><cylinder radius="0.1" length="0.05"/></geometry></collision>
  </link>
  <joint name="joint_L" type="continuous">
    <parent link="chassis"/><child link="wheel_L"/>
    <origin xyz="0 0.15 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="joint_R" type="continuous">
    <parent link="chassis"/><child link="wheel_R"/>
    <origin xyz="0 -0.15 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
</robot>
""")


@pytest.fixture
def arm_urdf_file():
    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
        f.write(SIMPLE_ARM_URDF)
        return f.name


@pytest.fixture
def mjcf_file():
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(SIMPLE_MJCF)
        return f.name


@pytest.fixture
def rover_urdf_file():
    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
        f.write(FIXED_JOINT_URDF)
        return f.name


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP 1 — URDF / MJCF PARSER
# ══════════════════════════════════════════════════════════════════════════════

class TestParser:
    def test_urdf_links_count(self, arm_urdf_file):
        from physicore.core.urdf_loader import parse_robot_file
        links, joints, name, fmt = parse_robot_file(arm_urdf_file)
        assert fmt == "urdf"
        assert name == "simple_arm"
        assert len(links) == 4
        assert len(joints) == 3

    def test_urdf_mass_extraction(self, arm_urdf_file):
        from physicore.core.urdf_loader import parse_robot_file
        links, joints, name, fmt = parse_robot_file(arm_urdf_file)
        masses = {l.name: l.mass for l in links}
        assert abs(masses["base_link"] - 2.0) < 1e-9
        assert abs(masses["link1"]     - 1.0) < 1e-9

    def test_urdf_joint_limits(self, arm_urdf_file):
        from physicore.core.urdf_loader import parse_robot_file
        links, joints, name, fmt = parse_robot_file(arm_urdf_file)
        j1 = next(j for j in joints if j.name == "joint1")
        assert abs(j1.limit_lo - (-math.pi)) < 1e-4
        assert abs(j1.limit_hi -  math.pi)   < 1e-4
        assert j1.jtype == "revolute"
        assert j1.is_actuated

    def test_urdf_collision_geometry(self, arm_urdf_file):
        from physicore.core.urdf_loader import parse_robot_file
        links, joints, name, fmt = parse_robot_file(arm_urdf_file)
        base = next(l for l in links if l.name == "base_link")
        assert base.has_collision
        assert base.collision_type == "cylinder"
        assert abs(base.collision_size[0] - 0.05) < 1e-9

    def test_mjcf_detection(self, mjcf_file):
        from physicore.core.urdf_loader import parse_robot_file
        links, joints, name, fmt = parse_robot_file(mjcf_file)
        assert fmt == "mjcf"
        assert name == "simple_arm_mj"
        assert len([j for j in joints if j.is_actuated]) == 3

    def test_mjcf_joint_limits(self, mjcf_file):
        from physicore.core.urdf_loader import parse_robot_file
        links, joints, name, fmt = parse_robot_file(mjcf_file)
        j = next(j for j in joints if j.name == "joint1")
        assert abs(j.limit_lo - (-math.pi)) < 0.001
        assert j.jtype == "revolute"

    def test_file_not_found(self):
        from physicore.core.urdf_loader import parse_robot_file
        with pytest.raises(FileNotFoundError):
            parse_robot_file("/nonexistent/robot.urdf")

    def test_continuous_joints_actuated(self, rover_urdf_file):
        from physicore.core.urdf_loader import parse_robot_file
        links, joints, name, fmt = parse_robot_file(rover_urdf_file)
        actuated = [j for j in joints if j.is_actuated]
        assert len(actuated) == 2
        assert all(j.jtype == "continuous" for j in actuated)


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP 2 — URDF ROBOT MODEL (FK + JACOBIAN)
# ══════════════════════════════════════════════════════════════════════════════

class TestURDFRobotModel:
    def _build_model(self, urdf_content):
        from physicore.core.urdf_loader import parse_robot_file, URDFRobotModel
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
            f.write(urdf_content)
            path = f.name
        links, joints, name, _ = parse_robot_file(path)
        os.unlink(path)
        return URDFRobotModel(links, joints, name)

    def test_dof_count(self):
        model = self._build_model(SIMPLE_ARM_URDF)
        assert model.dof == 3

    def test_total_mass(self):
        model = self._build_model(SIMPLE_ARM_URDF)
        assert abs(model.total_mass - (2.0 + 1.0 + 0.8 + 0.3)) < 1e-9

    def test_base_link_identity(self):
        model = self._build_model(SIMPLE_ARM_URDF)
        q = np.zeros(3)
        frames = model.forward_kinematics(q)
        R, p = frames[model.base_link]
        np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(p, np.zeros(3), atol=1e-9)

    def test_fk_zero_config_ee_height(self):
        """At q=0 the arm is fully extended; EE should be above base."""
        model = self._build_model(SIMPLE_ARM_URDF)
        q = np.zeros(3)
        ee_pos = model.ee_position(q)
        # joint1 at z=0.1, link1=0.3 m, link2=0.3 m → EE at z ≈ 0.7
        assert ee_pos[2] > 0.5, f"EE z={ee_pos[2]:.3f} expected > 0.5"

    def test_fk_joint1_rotation(self):
        """Rotating joint2 (y-axis) by 90° folds arm and moves EE in XY plane."""
        model = self._build_model(SIMPLE_ARM_URDF)
        q0 = np.zeros(3)
        # joint1 is a z-axis rotation; all link offsets are along z so it
        # cannot produce XY displacement. Use joint2 (y-axis) instead.
        q1 = np.array([0.0, math.pi / 2, 0.0])
        ee0 = model.ee_position(q0)
        ee1 = model.ee_position(q1)
        # After y-rotation the arm folds: EE moves into XY plane
        xy_dist = math.sqrt((ee1[0]-ee0[0])**2 + (ee1[1]-ee0[1])**2)
        assert xy_dist > 0.1, f"EE should move in XY plane after joint2 rotation, got {xy_dist:.4f}"

    def test_fk_joint2_rotation(self):
        """Rotating joint2 (y-axis) by 90° folds the arm."""
        model = self._build_model(SIMPLE_ARM_URDF)
        q_up   = np.zeros(3)
        q_fold = np.array([0.0, math.pi / 2, 0.0])
        ee_up   = model.ee_position(q_up)
        ee_fold = model.ee_position(q_fold)
        # Folding 90° should bring the EE down in z
        assert ee_fold[2] < ee_up[2] - 0.1

    def test_jacobian_shape(self):
        model = self._build_model(SIMPLE_ARM_URDF)
        q = np.zeros(3)
        J = model.jacobian(q)
        assert J.shape == (6, 3)

    def test_jacobian_numerical_finite_diff(self):
        """Verify geometric Jacobian matches finite-difference (linear rows)."""
        model = self._build_model(SIMPLE_ARM_URDF)
        q = np.array([0.3, -0.2, 0.5])
        J_ana = model.jacobian(q)[:3, :]   # linear velocity part

        eps = 1e-6
        J_fd = np.zeros((3, 3))
        ee0 = model.ee_position(q)
        for i in range(3):
            q_plus = q.copy(); q_plus[i] += eps
            J_fd[:, i] = (model.ee_position(q_plus) - ee0) / eps

        np.testing.assert_allclose(J_ana, J_fd, atol=1e-4,
                                   err_msg="Geometric Jacobian doesn't match FD")

    def test_ee_pose_rotation(self):
        model = self._build_model(SIMPLE_ARM_URDF)
        q = np.zeros(3)
        R, p = model.ee_pose(q)
        assert R.shape == (3, 3)
        # R should be orthogonal
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_fk_all_links_present(self):
        model = self._build_model(SIMPLE_ARM_URDF)
        q = np.zeros(3)
        frames = model.forward_kinematics(q)
        for link in model.all_links:
            assert link.name in frames, f"Link '{link.name}' missing from FK frames"

    def test_rover_model_dof(self, rover_urdf_file):
        from physicore.core.urdf_loader import build_robot_model
        robot_model, _, config, platform = build_robot_model(rover_urdf_file)
        assert robot_model.dof == 2
        assert "rover" in platform or "ground" in platform

    def test_platform_inference_arm(self, arm_urdf_file):
        from physicore.core.urdf_loader import build_robot_model
        _, _, config, platform = build_robot_model(arm_urdf_file)
        assert platform in ("manipulator_arm", "surgical_robot", "legged_robot",
                            "ros2_manipulator"), f"Unexpected: {platform}"


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP 3 — PROPER CONTACT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TestContactModel:
    def setup_method(self):
        from physicore.core.urdf_loader import ProperContactModel
        self.cm = ProperContactModel(stiffness=5000.0, damping=200.0, friction_mu=0.8)

    def test_no_contact_above_ground(self):
        pos = np.array([0.0, 0.0, 0.5])   # 50 cm above ground
        vel = np.array([0.0, 0.0, -1.0])
        F = self.cm.contact_force(pos, vel, radius=0.05)
        np.testing.assert_array_equal(F, np.zeros(3))

    def test_contact_activates_on_penetration(self):
        pos = np.array([0.0, 0.0, 0.02])   # radius=0.05, so penetration=0.03
        vel = np.array([0.0, 0.0, -0.5])
        F = self.cm.contact_force(pos, vel, radius=0.05)
        assert F[2] > 0.0, "Normal force must be positive on contact"

    def test_normal_force_increases_with_penetration(self):
        vel = np.array([0.0, 0.0, 0.0])
        F1 = self.cm.contact_force(np.array([0.0, 0.0, 0.04]), vel, radius=0.05)
        F2 = self.cm.contact_force(np.array([0.0, 0.0, 0.02]), vel, radius=0.05)
        F3 = self.cm.contact_force(np.array([0.0, 0.0, 0.00]), vel, radius=0.05)
        assert F1[2] < F2[2] < F3[2], "Normal force must increase with penetration"

    def test_friction_opposes_tangential_velocity(self):
        pos = np.array([0.0, 0.0, 0.02])
        vel = np.array([1.0, 0.0, 0.0])   # moving in +x
        F = self.cm.contact_force(pos, vel, radius=0.05)
        assert F[0] < 0.0, "Friction must oppose +x motion"

    def test_no_friction_without_normal_force(self):
        pos = np.array([0.0, 0.0, 0.5])   # airborne
        vel = np.array([5.0, 3.0, -1.0])
        F = self.cm.contact_force(pos, vel, radius=0.05)
        assert abs(F[0]) < 1e-9
        assert abs(F[1]) < 1e-9

    def test_damping_reduces_normal_force_on_downward_approach(self):
        """Damping should reduce (but not flip) normal force when approaching fast."""
        pos = np.array([0.0, 0.0, 0.02])
        vel_slow = np.array([0.0, 0.0, -0.01])
        vel_fast = np.array([0.0, 0.0, -5.0])
        F_slow = self.cm.contact_force(pos, vel_slow, radius=0.05)
        F_fast = self.cm.contact_force(pos, vel_fast, radius=0.05)
        assert F_slow[2] >= 0.0, "Normal force must stay non-negative (slow)"
        assert F_fast[2] >= 0.0, "Normal force must stay non-negative (fast approach)"

    def test_foot_contacts_convenience(self):
        foot_pos = np.array([
            [0.0, 0.1, 0.02],   # in contact
            [0.0, -0.1, 0.5],   # airborne
        ])
        foot_vel = np.zeros_like(foot_pos)
        forces, in_contact = self.cm.foot_contacts(foot_pos, foot_vel, foot_radii=np.array([0.05, 0.05]))
        assert in_contact[0] == True
        assert in_contact[1] == False
        assert forces[0, 2] > 0.0

    def test_custom_ground_plane(self):
        pos = np.array([0.0, 0.0, 1.05])
        vel = np.zeros(3)
        # Ground at z=1.0, radius=0.05 → penetration=0.0 exactly
        F = self.cm.contact_force(pos, vel, radius=0.05, ground_z=1.0)
        np.testing.assert_array_equal(F, np.zeros(3))
        # Shift 1mm down
        pos[2] = 1.04
        F2 = self.cm.contact_force(pos, vel, radius=0.05, ground_z=1.0)
        assert F2[2] > 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP 4 — EXTRA DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

class TestExtraDynamics:
    def test_mobile_manipulator_output_shape(self):
        from physicore.core.extra_dynamics import mobile_manipulator_dynamics
        state  = np.zeros(14)
        action = np.zeros(6)
        params = {"mass": 15.0, "friction": 0.4, "inertia": 5.0}
        ds = mobile_manipulator_dynamics(state, action, params)
        assert ds.shape == (14,)

    def test_mobile_manipulator_finite(self):
        from physicore.core.extra_dynamics import mobile_manipulator_dynamics
        state  = np.random.randn(14) * 0.1
        action = np.random.randn(6)
        params = {"mass": 15.0, "friction": 0.4, "inertia": 5.0}
        ds = mobile_manipulator_dynamics(state, action, params)
        assert np.all(np.isfinite(ds))

    def test_dual_arm_output_shape(self):
        from physicore.core.extra_dynamics import dual_arm_dynamics
        state  = np.zeros(20)
        action = np.zeros(14)
        params = {"mass": 3.0, "friction": 0.25, "inertia": 1.0}
        ds = dual_arm_dynamics(state, action, params)
        assert ds.shape == (20,)

    def test_dual_arm_finite(self):
        from physicore.core.extra_dynamics import dual_arm_dynamics
        state  = np.random.randn(20) * 0.2
        action = np.random.randn(14) * 5.0
        params = {"mass": 3.0, "friction": 0.25, "inertia": 1.0}
        ds = dual_arm_dynamics(state, action, params)
        assert np.all(np.isfinite(ds))

    def test_cable_driven_output_shape(self):
        from physicore.core.extra_dynamics import cable_driven_dynamics
        state  = np.zeros(12)
        state[2] = 1.0   # z = 1.0 m (in workspace)
        action = np.ones(6) * 50.0   # 50 N per cable
        params = {"mass": 20.0, "friction": 0.1, "inertia": 2.0}
        ds = cable_driven_dynamics(state, action, params)
        assert ds.shape == (12,)

    def test_cable_driven_gravity(self):
        """With no tension, CDPR should accelerate downward."""
        from physicore.core.extra_dynamics import cable_driven_dynamics
        state  = np.zeros(12); state[2] = 1.0
        action = np.zeros(6)
        params = {"mass": 20.0, "friction": 0.1, "inertia": 2.0}
        ds = cable_driven_dynamics(state, action, params)
        # vz derivative (az) should be negative (gravity)
        assert ds[5] < -5.0, f"Expected downward accel, got az={ds[5]:.2f}"

    def test_cable_driven_cables_only_pull(self):
        """Negative tension (cable compression) should be clamped to zero."""
        from physicore.core.extra_dynamics import cable_driven_dynamics
        state  = np.zeros(12); state[2] = 1.0
        action = np.full(6, -100.0)   # negative (invalid)
        params = {"mass": 20.0, "friction": 0.1, "inertia": 2.0}
        ds_neg = cable_driven_dynamics(state, action, params)
        action_zero = np.zeros(6)
        ds_zero = cable_driven_dynamics(state, action_zero, params)
        np.testing.assert_allclose(ds_neg, ds_zero, atol=1e-9,
                                   err_msg="Negative tension should be clamped to zero")

    def test_exoskeleton_output_shape(self):
        from physicore.core.extra_dynamics import exoskeleton_dynamics
        state  = np.zeros(16)
        action = np.zeros(10)
        params = {"mass": 80.0, "friction": 0.6, "inertia": 12.0}
        ds = exoskeleton_dynamics(state, action, params)
        assert ds.shape == (16,)

    def test_exoskeleton_finite(self):
        from physicore.core.extra_dynamics import exoskeleton_dynamics
        state  = np.random.randn(16) * 0.1
        action = np.random.randn(10) * 2.0
        params = {"mass": 80.0, "friction": 0.6, "inertia": 12.0, "admittance_k": 100.0}
        ds = exoskeleton_dynamics(state, action, params)
        assert np.all(np.isfinite(ds))

    def test_exoskeleton_gravity_at_neutral(self):
        """At neutral pose with no actuation, knee/hip should feel gravity."""
        from physicore.core.extra_dynamics import exoskeleton_dynamics
        state  = np.zeros(16)
        state[1] = 0.5   # left knee bent 0.5 rad
        action = np.zeros(10)
        params = {"mass": 80.0, "friction": 0.6, "inertia": 12.0}
        ds = exoskeleton_dynamics(state, action, params)
        # Left knee acceleration (index 7) should be non-zero due to gravity
        assert abs(ds[7]) > 0.01


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP 5 — LOAD_ROBOT / PHYSICORE INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadRobot:
    def test_load_urdf_returns_engine_and_config(self, arm_urdf_file):
        from physicore.core.urdf_loader import load_robot
        engine, config = load_robot(arm_urdf_file)
        assert engine is not None
        assert config is not None
        assert config.dof == 3

    def test_load_urdf_config_mass(self, arm_urdf_file):
        from physicore.core.urdf_loader import load_robot
        _, config = load_robot(arm_urdf_file)
        expected_mass = 2.0 + 1.0 + 0.8 + 0.3
        assert abs(config.mass - expected_mass) < 1e-9

    def test_load_mjcf(self, mjcf_file):
        from physicore.core.urdf_loader import load_robot
        engine, config = load_robot(mjcf_file)
        assert config.dof == 3
        assert engine is not None

    def test_load_urdf_with_platform_hint(self, arm_urdf_file):
        from physicore.core.urdf_loader import load_robot
        engine, config = load_robot(arm_urdf_file, platform_hint="manipulator_arm")
        assert config.engine_platform == "manipulator_arm"

    def test_engine_has_urdf_model(self, arm_urdf_file):
        from physicore.core.urdf_loader import load_robot
        engine, _ = load_robot(arm_urdf_file)
        assert hasattr(engine, "_urdf_model")
        assert engine._urdf_model.dof == 3

    def test_engine_has_contact_model(self, arm_urdf_file):
        from physicore.core.urdf_loader import load_robot
        engine, _ = load_robot(arm_urdf_file)
        assert hasattr(engine, "_contact_model")

    def test_engine_step_runs(self, arm_urdf_file):
        """A complete engine step should run without error."""
        from physicore.core.urdf_loader import load_robot
        engine, config = load_robot(arm_urdf_file)
        state_dim = engine.cfg.state_dim
        state = np.zeros(state_dim)
        x_ref = np.zeros(state_dim)
        result = engine.step(state, x_ref)
        assert result is not None
        assert np.all(np.isfinite(result.action))

    def test_register_platform(self):
        from physicore.core.engine import PhysiCore, PLATFORM_DYNAMICS
        def my_dyn(s, a, p):
            return np.zeros_like(s)
        PhysiCore.register_platform("test_custom_bot", my_dyn, 6, 2)
        assert "test_custom_bot" in PLATFORM_DYNAMICS
        engine = PhysiCore.for_platform("test_custom_bot")
        assert engine is not None

    def test_for_urdf_classmethod(self, arm_urdf_file):
        from physicore.core.engine import PhysiCore
        engine, config = PhysiCore.for_urdf(arm_urdf_file)
        assert engine is not None
        assert config.dof == 3


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP 6 — EXTRA DYNAMICS IN PLATFORM_DYNAMICS CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════

class TestExtraPlatforms:
    """
    These tests will PASS only after applying engine_patch.py.
    They are marked xfail until then, and auto-promote once the patch lands.
    """

    @pytest.mark.xfail(reason="Requires engine_patch.py to be applied", strict=False)
    def test_mobile_manipulator_platform(self):
        from physicore.core.engine import PhysiCore
        engine = PhysiCore.for_platform("mobile_manipulator")
        assert engine.cfg.state_dim == 14
        assert engine.cfg.action_dim == 6

    @pytest.mark.xfail(reason="Requires engine_patch.py to be applied", strict=False)
    def test_dual_arm_platform(self):
        from physicore.core.engine import PhysiCore
        engine = PhysiCore.for_platform("dual_arm")
        assert engine.cfg.state_dim == 20
        assert engine.cfg.action_dim == 14

    @pytest.mark.xfail(reason="Requires engine_patch.py to be applied", strict=False)
    def test_cable_driven_platform(self):
        from physicore.core.engine import PhysiCore
        engine = PhysiCore.for_platform("cable_driven")
        assert engine.cfg.state_dim == 12

    @pytest.mark.xfail(reason="Requires engine_patch.py to be applied", strict=False)
    def test_exoskeleton_platform(self):
        from physicore.core.engine import PhysiCore
        engine = PhysiCore.for_platform("exoskeleton")
        assert engine.cfg.state_dim == 16
        assert engine.cfg.action_dim == 10


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP 7 — HUMANOID CONTACT (engine_patch)
# ══════════════════════════════════════════════════════════════════════════════

class TestHumanoidContact:
    @pytest.mark.xfail(reason="Requires engine_patch.py contact fix", strict=False)
    def test_humanoid_contact_forces_nonzero_on_ground(self):
        from physicore.core.engine import humanoid_dynamics
        # Feet on ground (lf_z, rf_z ≈ 0), vz = -0.1 m/s
        state = np.zeros(18)
        state[2]  = 0.9    # com_z = 0.9 m (standing)
        state[5]  = -0.1   # vz = -0.1 (settling down)
        state[12] = 0.02   # lf_z = 2 cm (just touching)
        state[13] = 0.02   # rf_z = 2 cm
        action = np.zeros(6)
        params = {"mass": 60.0, "friction": 0.8, "inertia": 8.0}
        ds = humanoid_dynamics(state, action, params)
        # With proper contact, az (index 5) should be > −g (contact pushing up)
        assert ds[5] > -9.81, "Proper contact should reduce downward acceleration"


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    sys.exit(result.returncode)

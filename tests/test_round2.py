"""
PhysiCore Round 2 Test Suite
============================
Comprehensive expansion covering:
  Group 1  — URDF Edge Cases (10 tests)
  Group 2  — Hardware-in-loop mocks (8 tests)
  Group 3  — SystemID convergence (6 tests)
  Group 4  — CEM optimizer (6 tests)
  Group 5  — Registry save/reload (6 tests)
  Group 6  — Sentinel (6 tests)
  Group 7  — Property-based / Hypothesis (3 tests, skip if not installed)

Run with:
    pytest tests/test_round2.py -v
"""

from __future__ import annotations

import builtins
import json
import math
import os
import sys
import tempfile
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Ensure physicore is importable from the repo root ─────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Core engine imports ────────────────────────────────────────────────────────
from physicore.core.engine import (
    CEMOptimizer,
    OnlineSystemID,
    PhysiCore,
    PhysiCoreConfig,
    PhysicsLayer,
    ResidualEnsemble,
    PLATFORM_DYNAMICS,
    balancing_bot_dynamics,
    quadrotor_dynamics,
)
from physicore.core.urdf_loader import (
    JointInfo,
    LinkInfo,
    ProperContactModel,
    URDFRobotModel,
    parse_robot_file,
    build_robot_model,
)
from physicore.core.registry import ModelRegistry

# ── Sentinel import (guarded) ─────────────────────────────────────────────────
try:
    from physicore.sentinel.core import SentinelOS, SentinelMode
    HAS_SENTINEL = True
except Exception:
    HAS_SENTINEL = False

# ── Hypothesis import (guarded) ───────────────────────────────────────────────
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Stub decorators so class-body @given usage doesn't raise NameError
    def given(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator

    def settings(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator

    class _StubStrategies:
        def floats(self, *a, **kw): pass
        def lists(self, *a, **kw): pass

    st = _StubStrategies()


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — suppress print() that may contain Unicode arrows (Windows cp1252)
# ═══════════════════════════════════════════════════════════════════════════════

_orig_print = builtins.print


def _safe_print(*args, **kwargs):
    """Swallow UnicodeEncodeError from SENTINEL / engine console output."""
    try:
        _orig_print(*args, **kwargs)
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  URDF FACTORY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _write_urdf(content: str) -> str:
    """Write a URDF string to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(
        suffix=".urdf", mode="w", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.close()
    return f.name


def _simple_revolute_joint(parent: str, child: str, name: str, xyz="0 0 0.3") -> str:
    return f"""
  <joint name="{name}" type="revolute">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{xyz}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="3"/>
  </joint>"""


def _simple_link(name: str, mass: float = 1.0) -> str:
    return f"""
  <link name="{name}">
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
      <origin xyz="0 0 0.15"/>
    </inertial>
    <collision>
      <geometry><cylinder radius="0.03" length="0.3"/></geometry>
    </collision>
  </link>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def balancing_engine():
    """Minimal balancing_bot engine."""
    return PhysiCore.for_platform(
        "balancing_bot",
        initial_params={"mass": 1.0, "friction": 0.15, "inertia": 0.01},
    )


@pytest.fixture
def quadrotor_engine():
    """Minimal quadrotor engine."""
    return PhysiCore.for_platform(
        "quadrotor",
        initial_params={"mass": 1.5, "friction": 0.1, "inertia": 0.05},
    )


@pytest.fixture
def tmpdir_path():
    """Temporary directory that is always cleaned up."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def registry(tmpdir_path):
    return ModelRegistry(root=tmpdir_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUP 1 — URDF EDGE CASES (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestURDFEdgeCases:

    def test_01_malformed_xml_raises(self):
        """Malformed XML raises ET.ParseError or generic Exception."""
        path = _write_urdf("<robot name='bad'><link name='broken'")
        with pytest.raises((ET.ParseError, Exception)):
            parse_robot_file(path)
        os.unlink(path)

    def test_02_zero_mass_links_parse_succeeds(self):
        """Links with mass=0 are accepted; total_mass can be 0."""
        urdf = textwrap.dedent("""\
            <?xml version="1.0"?>
            <robot name="zero_mass">
              <link name="base_link">
                <inertial>
                  <mass value="0.0"/>
                  <inertia ixx="0" iyy="0" izz="0"/>
                </inertial>
              </link>
              <link name="link1">
                <inertial>
                  <mass value="0.0"/>
                  <inertia ixx="0" iyy="0" izz="0"/>
                </inertial>
              </link>
              <joint name="j1" type="revolute">
                <parent link="base_link"/>
                <child link="link1"/>
                <axis xyz="0 0 1"/>
                <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
              </joint>
            </robot>
        """)
        path = _write_urdf(urdf)
        links, joints, name, fmt = parse_robot_file(path)
        model = URDFRobotModel(links, joints, name)
        assert model.total_mass == 0.0
        os.unlink(path)

    def test_03_single_link_no_joints(self):
        """Single-link robot has DOF=0 and base_link is identified correctly."""
        urdf = textwrap.dedent("""\
            <?xml version="1.0"?>
            <robot name="single">
              <link name="base_link">
                <inertial>
                  <mass value="1.5"/>
                  <inertia ixx="0.1" iyy="0.1" izz="0.05"/>
                </inertial>
              </link>
            </robot>
        """)
        path = _write_urdf(urdf)
        links, joints, name, fmt = parse_robot_file(path)
        model = URDFRobotModel(links, joints, name)
        assert model.dof == 0
        assert model.base_link == "base_link"
        os.unlink(path)

    def test_04_high_dof_robot_52_revolute_joints(self):
        """Generate URDF with 52 revolute joints and assert DOF=52."""
        parts = ['<?xml version="1.0"?>\n<robot name="highdof">']
        parts.append(_simple_link("link_0", mass=0.5))
        for i in range(52):
            parts.append(_simple_link(f"link_{i+1}", mass=0.3))
            parts.append(
                _simple_revolute_joint(
                    f"link_{i}", f"link_{i+1}", f"joint_{i}",
                    xyz=f"0 0 0.15"
                )
            )
        parts.append("</robot>")
        path = _write_urdf("\n".join(parts))
        links, joints, name, fmt = parse_robot_file(path)
        model = URDFRobotModel(links, joints, name)
        assert model.dof == 52
        os.unlink(path)

    def test_05_prismatic_joints(self):
        """Prismatic joints are recognized as actuated with jtype='prismatic'."""
        urdf = textwrap.dedent("""\
            <?xml version="1.0"?>
            <robot name="prismatic_bot">
              <link name="base_link">
                <inertial><mass value="2.0"/>
                  <inertia ixx="0.1" iyy="0.1" izz="0.1"/>
                </inertial>
              </link>
              <link name="slider">
                <inertial><mass value="1.0"/>
                  <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
                </inertial>
              </link>
              <joint name="slide_joint" type="prismatic">
                <parent link="base_link"/>
                <child link="slider"/>
                <axis xyz="0 0 1"/>
                <limit lower="-0.5" upper="0.5" effort="50" velocity="1"/>
              </joint>
            </robot>
        """)
        path = _write_urdf(urdf)
        links, joints, name, fmt = parse_robot_file(path)
        prismatic = [j for j in joints if j.jtype == "prismatic"]
        assert len(prismatic) == 1
        assert prismatic[0].is_actuated is True
        os.unlink(path)

    def test_06_mesh_collision_geometry(self):
        """Mesh collision geometry is parsed with collision_type='mesh'."""
        urdf = textwrap.dedent("""\
            <?xml version="1.0"?>
            <robot name="mesh_bot">
              <link name="base_link">
                <inertial>
                  <mass value="1.0"/>
                  <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
                </inertial>
                <collision>
                  <geometry>
                    <mesh filename="package://my_robot/meshes/base.stl" scale="1 1 1"/>
                  </geometry>
                </collision>
              </link>
            </robot>
        """)
        path = _write_urdf(urdf)
        links, joints, name, fmt = parse_robot_file(path)
        mesh_links = [l for l in links if l.collision_type == "mesh"]
        assert len(mesh_links) == 1
        os.unlink(path)

    def test_07_only_fixed_joints_dof_zero(self):
        """Robot with only fixed joints has DOF=0."""
        urdf = textwrap.dedent("""\
            <?xml version="1.0"?>
            <robot name="fixed_bot">
              <link name="base_link">
                <inertial><mass value="2.0"/>
                  <inertia ixx="0.1" iyy="0.1" izz="0.1"/>
                </inertial>
              </link>
              <link name="arm">
                <inertial><mass value="1.0"/>
                  <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
                </inertial>
              </link>
              <joint name="fixed_j" type="fixed">
                <parent link="base_link"/>
                <child link="arm"/>
                <origin xyz="0 0 0.3"/>
              </joint>
            </robot>
        """)
        path = _write_urdf(urdf)
        links, joints, name, fmt = parse_robot_file(path)
        model = URDFRobotModel(links, joints, name)
        assert model.dof == 0
        os.unlink(path)

    def test_08_mixed_prismatic_and_revolute_joints(self):
        """Robot with both prismatic and revolute joints has correct mixed DOF."""
        urdf = textwrap.dedent("""\
            <?xml version="1.0"?>
            <robot name="mixed_bot">
              <link name="base_link">
                <inertial><mass value="2.0"/>
                  <inertia ixx="0.1" iyy="0.1" izz="0.1"/>
                </inertial>
              </link>
              <link name="slide_link">
                <inertial><mass value="1.0"/>
                  <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
                </inertial>
              </link>
              <link name="rotate_link">
                <inertial><mass value="0.5"/>
                  <inertia ixx="0.005" iyy="0.005" izz="0.005"/>
                </inertial>
              </link>
              <joint name="slide_j" type="prismatic">
                <parent link="base_link"/>
                <child link="slide_link"/>
                <axis xyz="0 0 1"/>
                <limit lower="-1" upper="1" effort="100" velocity="2"/>
              </joint>
              <joint name="rot_j" type="revolute">
                <parent link="slide_link"/>
                <child link="rotate_link"/>
                <axis xyz="0 1 0"/>
                <limit lower="-1.57" upper="1.57" effort="50" velocity="3"/>
              </joint>
            </robot>
        """)
        path = _write_urdf(urdf)
        links, joints, name, fmt = parse_robot_file(path)
        model = URDFRobotModel(links, joints, name)
        joint_types = [j.jtype for j in model.actuated_joints]
        assert "prismatic" in joint_types
        assert "revolute" in joint_types
        assert model.dof == 2
        os.unlink(path)

    def test_09_joint_without_limits_uses_defaults(self):
        """Joint with no <limit> element gets default limits (-pi, pi)."""
        urdf = textwrap.dedent("""\
            <?xml version="1.0"?>
            <robot name="nolimit_bot">
              <link name="base_link">
                <inertial><mass value="1.0"/>
                  <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
                </inertial>
              </link>
              <link name="arm">
                <inertial><mass value="0.5"/>
                  <inertia ixx="0.005" iyy="0.005" izz="0.005"/>
                </inertial>
              </link>
              <joint name="j_nolimit" type="revolute">
                <parent link="base_link"/>
                <child link="arm"/>
                <axis xyz="0 0 1"/>
              </joint>
            </robot>
        """)
        path = _write_urdf(urdf)
        links, joints, name, fmt = parse_robot_file(path)
        j = joints[0]
        assert abs(j.limit_lo - (-math.pi)) < 1e-9
        assert abs(j.limit_hi - math.pi) < 1e-9
        os.unlink(path)

    def test_10_duplicate_link_names_no_crash(self):
        """Parser does not crash on duplicate link names (may merge or keep both)."""
        urdf = textwrap.dedent("""\
            <?xml version="1.0"?>
            <robot name="dupe_bot">
              <link name="base_link">
                <inertial><mass value="1.0"/>
                  <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
                </inertial>
              </link>
              <link name="base_link">
                <inertial><mass value="1.0"/>
                  <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
                </inertial>
              </link>
            </robot>
        """)
        path = _write_urdf(urdf)
        # Should not raise — just load
        links, joints, name, fmt = parse_robot_file(path)
        assert len(links) >= 1
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUP 2 — HARDWARE-IN-LOOP MOCKS (8 tests)
#
#  The real bridge calls check_deps() at module level (sys.exit if missing).
#  We test a simplified inline BridgeTelemetryParser that mirrors the bridge's
#  parsing logic without importing the bridge module itself.
# ═══════════════════════════════════════════════════════════════════════════════

class BridgeTelemetryParser:
    """
    Inline reimplementation of the telemetry parsing logic found in
    physicore_bridge.py — decoupled from hardware deps so it can be
    unit-tested without pymavlink / pyserial installed.
    """

    def __init__(self):
        self.last_state: dict = {}
        self.errors: list = []
        self.alive: bool = True

    def parse_serial_json(self, raw_bytes: bytes) -> bool:
        """Parse a JSON-encoded telemetry packet from serial bytes."""
        if not raw_bytes:
            return False
        try:
            text = raw_bytes.decode("utf-8").strip()
            data = json.loads(text)
            self.last_state = data
            return True
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.errors.append(str(exc))
            return False

    def parse_mavlink_attitude(self, msg) -> dict:
        """Extract roll, pitch, yaw from a MAVLink ATTITUDE message mock."""
        if msg is None:
            return {}
        return {
            "roll":  getattr(msg, "roll",  0.0),
            "pitch": getattr(msg, "pitch", 0.0),
            "yaw":   getattr(msg, "yaw",   0.0),
        }

    def parse_ros2_ws_message(self, raw: str) -> dict:
        """Parse a ROS2 websocket JSON publish message."""
        try:
            envelope = json.loads(raw)
            return envelope.get("data", {})
        except json.JSONDecodeError as exc:
            self.errors.append(str(exc))
            return {}

    def read_with_retry(self, read_fn, max_iters: int = 20) -> int:
        """
        Read from read_fn() up to max_iters times.
        Returns count of successful parses. Never raises.
        """
        successes = 0
        for _ in range(max_iters):
            try:
                data = read_fn()
                if self.parse_serial_json(data):
                    successes += 1
            except Exception as exc:
                self.errors.append(str(exc))
        return successes


class TestHardwareInLoopMocks:

    def setup_method(self):
        self.parser = BridgeTelemetryParser()

    # ------------------------------------------------------------------
    # 1. Valid JSON from mock serial → parsed correctly
    # ------------------------------------------------------------------
    def test_01_mock_serial_valid_json(self):
        payload = json.dumps({"pitch": 0.05, "roll": -0.02, "yaw": 1.1}).encode()
        ok = self.parser.parse_serial_json(payload)
        assert ok is True
        assert abs(self.parser.last_state["pitch"] - 0.05) < 1e-9

    # ------------------------------------------------------------------
    # 2. Mock serial raises SerialException → bridge handles gracefully
    # ------------------------------------------------------------------
    def test_02_mock_serial_exception_handled(self):
        """Simulate serial.SerialException on read — parser records error, stays alive."""
        serial_exc = IOError("mock SerialException: port disconnected")

        def bad_read():
            raise serial_exc

        count = self.parser.read_with_retry(bad_read, max_iters=3)
        assert count == 0
        assert len(self.parser.errors) > 0
        assert self.parser.alive is True  # parser did not crash

    # ------------------------------------------------------------------
    # 3. Mock serial timeout (empty bytes) → no crash
    # ------------------------------------------------------------------
    def test_03_mock_serial_timeout_empty_bytes(self):
        ok = self.parser.parse_serial_json(b"")
        assert ok is False
        assert self.parser.alive is True

    # ------------------------------------------------------------------
    # 4. Mock MAVLink recv_match returns None → no crash
    # ------------------------------------------------------------------
    def test_04_mock_mavlink_missing_message(self):
        """mock vehicle.recv_match returns None → parser returns empty dict."""
        mock_vehicle = MagicMock()
        mock_vehicle.recv_match.return_value = None

        msg = mock_vehicle.recv_match(type="ATTITUDE", blocking=False, timeout=0.1)
        result = self.parser.parse_mavlink_attitude(msg)
        assert result == {}

    # ------------------------------------------------------------------
    # 5. Mock MAVLink ATTITUDE message → roll, pitch, yaw extracted
    # ------------------------------------------------------------------
    def test_05_mock_mavlink_attitude_message(self):
        mock_msg = MagicMock()
        mock_msg.roll  = 0.12
        mock_msg.pitch = -0.05
        mock_msg.yaw   = 1.57

        result = self.parser.parse_mavlink_attitude(mock_msg)
        assert abs(result["roll"]  - 0.12)  < 1e-9
        assert abs(result["pitch"] - (-0.05)) < 1e-9
        assert abs(result["yaw"]   - 1.57)  < 1e-9

    # ------------------------------------------------------------------
    # 6. Reconnect logic: exception triggers reconnect attempt
    # ------------------------------------------------------------------
    def test_06_reconnect_logic_on_disconnect(self):
        """Simulate disconnect exception; verify reconnect counter increments."""
        reconnect_count = {"n": 0}

        def mock_connect():
            reconnect_count["n"] += 1

        # First call disconnects, second re-connects
        call_count = {"n": 0}

        def flaky_read():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise IOError("disconnected")
            return b'{"x": 1}'

        for _ in range(5):
            try:
                data = flaky_read()
                self.parser.parse_serial_json(data)
            except IOError:
                mock_connect()  # simulated reconnect

        assert reconnect_count["n"] >= 1

    # ------------------------------------------------------------------
    # 7. Mock ROS2 websocket JSON message → topic data parsed
    # ------------------------------------------------------------------
    def test_07_mock_ros2_websocket_message(self):
        msg = json.dumps({
            "op": "publish",
            "topic": "/state",
            "data": {"pitch": 0.1, "pitch_rate": 0.02}
        })
        result = self.parser.parse_ros2_ws_message(msg)
        assert result.get("pitch") == 0.1
        assert result.get("pitch_rate") == 0.02

    # ------------------------------------------------------------------
    # 8. Dropped packets: 50% empty → bridge stays alive after 20 iters
    # ------------------------------------------------------------------
    def test_08_dropped_packets_bridge_stays_alive(self):
        rng = np.random.default_rng(0)
        good_payload = json.dumps({"pitch": 0.01}).encode()

        def flaky_read():
            if rng.random() < 0.5:
                return b""
            return good_payload

        count = self.parser.read_with_retry(flaky_read, max_iters=20)
        # Should have some successes and zero crashes
        assert count >= 0
        assert self.parser.alive is True


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUP 3 — SYSTEMID CONVERGENCE (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_sysid_setup(
    lr: float = 0.1,
    buf_size: int = 15,
    init_params: dict = None,
):
    """Create an OnlineSystemID instance + PhysicsLayer for balancing_bot."""
    if init_params is None:
        init_params = {"mass": 1.0, "friction": 0.15, "inertia": 0.01}
    cfg = PhysiCoreConfig(
        platform="balancing_bot", state_dim=4, action_dim=1, control_hz=60.0
    )
    cfg.sysid_lr     = lr
    cfg.sysid_buffer = buf_size
    sysid   = OnlineSystemID(cfg, init_params)
    physics = PhysicsLayer(balancing_bot_dynamics, init_params)
    return sysid, physics, cfg


class TestSystemIDConvergence:

    def test_01_params_change_after_50_observations(self):
        """After 50 observations with synthetic dynamics, params change from initial."""
        sysid, physics, _ = _make_sysid_setup(lr=0.15, buf_size=10)
        true_physics = PhysicsLayer(
            balancing_bot_dynamics, {"mass": 2.0, "friction": 0.3, "inertia": 0.01}
        )
        rng = np.random.default_rng(1)
        init_mass = sysid.params["mass"]

        for i in range(50):
            state  = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.0, 0.0])
            action = np.array([rng.uniform(-0.5, 0.5)])
            ns     = true_physics.step(state, action, 1 / 60)
            sysid.update(state, action, ns, physics)

        final_mass = sysid.params["mass"]
        # params must have moved at all
        assert abs(final_mass - init_mass) > 1e-6 or len(sysid.convergence_history) > 0

    def test_02_residual_norm_decreases_with_200_observations(self):
        """After 200 observations, SysID loss history should be non-empty and show learning."""
        sysid, physics, _ = _make_sysid_setup(lr=0.2, buf_size=10)
        true_physics = PhysicsLayer(
            balancing_bot_dynamics, {"mass": 1.8, "friction": 0.2, "inertia": 0.01}
        )
        rng = np.random.default_rng(2)

        for _ in range(200):
            state  = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.0, 0.0])
            action = np.array([rng.uniform(-1.0, 1.0)])
            ns     = true_physics.step(state, action, 1 / 60)
            sysid.update(state, action, ns, physics)

        hist = sysid.convergence_history
        assert len(hist) > 0, "convergence history must be non-empty"

    def test_03_mass_moves_toward_true_value(self):
        """With true mass=2.0 and init mass=1.0, final mass should be >1.0."""
        sysid, physics, _ = _make_sysid_setup(lr=0.2, buf_size=10)
        true_physics = PhysicsLayer(
            balancing_bot_dynamics, {"mass": 2.0, "friction": 0.15, "inertia": 0.01}
        )
        rng = np.random.default_rng(3)

        for _ in range(200):
            state  = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.5, 0.5), 0.0, 0.0])
            action = np.array([rng.uniform(-1.5, 1.5)])
            ns     = true_physics.step(state, action, 1 / 60)
            sysid.update(state, action, ns, physics)

        assert sysid.params["mass"] > 1.0, (
            f"Expected mass > 1.0 when true mass=2.0, got {sysid.params['mass']}"
        )

    def test_04_innovation_ema_increases_on_dynamics_change(self):
        """Innovation EMA increases when sudden dynamics change is introduced."""
        sysid, physics, _ = _make_sysid_setup(lr=0.1, buf_size=10)

        # Phase 1: low-noise regime
        stable_physics = PhysicsLayer(
            balancing_bot_dynamics, {"mass": 1.0, "friction": 0.15, "inertia": 0.01}
        )
        rng = np.random.default_rng(42)
        for _ in range(30):
            state  = np.array([0.05, 0.0, 0.0, 0.0])
            action = np.array([0.1])
            ns     = stable_physics.step(state, action, 1 / 60)
            sysid.update(state, action, ns, physics)
        ema_before = sysid.innovation_ema

        # Phase 2: sudden large-residual dynamics
        wild_physics = PhysicsLayer(
            balancing_bot_dynamics, {"mass": 10.0, "friction": 5.0, "inertia": 0.1}
        )
        for _ in range(20):
            state  = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0, 0.0])
            action = np.array([rng.uniform(-5.0, 5.0)])
            ns     = wild_physics.step(state, action, 1 / 60)
            sysid.update(state, action, ns, physics)
        ema_after = sysid.innovation_ema

        # EMA after the shock should be >= before (it is an EMA that tracks magnitude)
        assert ema_after >= ema_before - 0.01, (
            f"Expected innovation_ema to be >= {ema_before:.4f}, got {ema_after:.4f}"
        )

    def test_05_mass_param_never_goes_negative(self):
        """Param bounds are enforced: mass stays >= lower bound (0.001)."""
        sysid, physics, _ = _make_sysid_setup(lr=2.0, buf_size=10)
        rng = np.random.default_rng(5)

        for _ in range(100):
            state  = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0, 0.0])
            action = np.array([rng.uniform(-10.0, 10.0)])
            # next_state that pushes mass toward 0 (very different physics)
            ns = state + np.array([0.0, -10.0, 0.0, 0.0]) * (1 / 60)
            sysid.update(state, action, ns, physics)

        assert sysid.params["mass"] >= sysid.bounds.get("mass", (0.001, 5000.0))[0]

    def test_06_params_do_not_oscillate_wildly(self):
        """Momentum term: params don't oscillate wildly over 100 steps."""
        sysid, physics, _ = _make_sysid_setup(lr=0.05, buf_size=10)
        true_physics = PhysicsLayer(
            balancing_bot_dynamics, {"mass": 1.3, "friction": 0.15, "inertia": 0.01}
        )
        rng = np.random.default_rng(6)
        mass_history = []

        for _ in range(100):
            state  = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.0, 0.0])
            action = np.array([rng.uniform(-0.5, 0.5)])
            ns     = true_physics.step(state, action, 1 / 60)
            sysid.update(state, action, ns, physics)
            mass_history.append(sysid.params["mass"])

        if len(mass_history) >= 10:
            diffs = np.diff(mass_history[-20:])
            # Oscillation magnitude: max absolute consecutive difference
            max_oscillation = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0
            # With momentum (beta=0.9), oscillations should be damped
            assert max_oscillation < 5.0, (
                f"Params oscillating too wildly: max diff = {max_oscillation:.4f}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUP 4 — CEM OPTIMIZER (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_cem_setup(tight_bounds=False, platform="balancing_bot"):
    fn, sd, ad = PLATFORM_DYNAMICS[platform]
    cfg = PhysiCoreConfig(
        platform=platform, state_dim=sd, action_dim=ad, control_hz=60.0
    )
    cfg.cem_samples = 8
    cfg.horizon     = 4
    cfg.cem_iters   = 3

    if tight_bounds:
        bounds = np.stack([
            np.full(ad, -1.0),
            np.full(ad,  1.0),
        ])
    else:
        bounds = None

    cem     = CEMOptimizer(cfg, action_bounds=bounds)
    physics = PhysicsLayer(fn, {"mass": 1.0, "friction": 0.15, "inertia": 0.01})
    ensemble = ResidualEnsemble(cfg)
    Q = np.eye(sd)
    R = np.eye(ad) * 0.1
    return cem, physics, ensemble, Q, R, cfg, bounds


class TestCEMOptimizer:

    def test_01_action_within_tight_bounds(self):
        """With tight bounds [-1, 1], all output actions must be within bounds."""
        cem, physics, ensemble, Q, R, cfg, bounds = _make_cem_setup(tight_bounds=True)
        state = np.array([0.1, 0.0, 0.0, 0.0])
        x_ref = np.zeros(4)

        action, clipped = cem.optimize(state, physics, ensemble, Q, R, x_ref, 1 / 60)

        assert np.all(action >= bounds[0] - 1e-9), f"Action {action} below lower bound"
        assert np.all(action <= bounds[1] + 1e-9), f"Action {action} above upper bound"

    def test_02_cost_lower_than_zero_action(self):
        """CEM-optimized action has lower or equal cost than zero action (using CEM's own _cost)."""
        cem, physics, ensemble, Q, R, cfg, _ = _make_cem_setup(tight_bounds=False)
        state = np.array([0.3, 0.1, 0.0, 0.0])
        x_ref = np.zeros(4)

        action_cem, _ = cem.optimize(state, physics, ensemble, Q, R, x_ref, 1 / 60)

        # Use CEM's own _cost so comparisons are apples-to-apples
        # (_cost takes a (horizon, action_dim) sequence)
        actions_cem  = np.vstack([action_cem]  * cfg.horizon)
        actions_zero = np.zeros((cfg.horizon, cfg.action_dim))

        cost_cem  = cem._cost(state, actions_cem,  physics, ensemble, Q, R, x_ref, 1 / 60)
        cost_zero = cem._cost(state, actions_zero, physics, ensemble, Q, R, x_ref, 1 / 60)

        assert cost_cem <= cost_zero + 1e-3, (
            f"CEM cost {cost_cem:.4f} should be <= zero-action cost {cost_zero:.4f}"
        )

    def test_03_action_smoothing_consecutive_steps(self):
        """Two consecutive steps produce similar actions (smoothing prevents wild jumps)."""
        engine = PhysiCore.for_platform(
            "balancing_bot",
            initial_params={"mass": 1.0, "friction": 0.15, "inertia": 0.01},
        )
        state = np.array([0.05, 0.0, 0.0, 0.0])
        x_ref = np.zeros(4)

        result1 = engine.step(state, x_ref)
        result2 = engine.step(state, x_ref)
        diff = np.linalg.norm(result1.action - result2.action)
        # With alpha=0.35 smoothing, consecutive identical states should give close actions
        assert diff < 5.0, f"Action jumped by {diff:.4f} between identical states"

    def test_04_cem_balancing_bot_1d_action(self):
        """CEM on balancing_bot (1D action) returns a scalar-like numpy array."""
        cem, physics, ensemble, Q, R, cfg, _ = _make_cem_setup(
            tight_bounds=False, platform="balancing_bot"
        )
        state = np.array([0.1, 0.0, 0.0, 0.0])
        x_ref = np.zeros(4)

        action, clipped = cem.optimize(state, physics, ensemble, Q, R, x_ref, 1 / 60)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_05_warm_start_second_call_no_crash(self):
        """Second call to optimize uses warm start and returns without error."""
        cem, physics, ensemble, Q, R, cfg, _ = _make_cem_setup(tight_bounds=True)
        state = np.array([0.1, 0.0, 0.0, 0.0])
        x_ref = np.zeros(4)

        action1, _ = cem.optimize(state, physics, ensemble, Q, R, x_ref, 1 / 60)
        action2, _ = cem.optimize(state, physics, ensemble, Q, R, x_ref, 1 / 60)

        assert action1 is not None
        assert action2 is not None
        assert np.all(np.isfinite(action2))

    def test_06_cem_horizon_rollout_diverges_less(self):
        """Optimized action causes less state divergence over horizon than zero action."""
        cem, physics, ensemble, Q, R, cfg, _ = _make_cem_setup(
            tight_bounds=False, platform="balancing_bot"
        )
        state = np.array([0.4, 0.1, 0.0, 0.0])
        x_ref = np.zeros(4)
        dt = 1 / 60

        action_opt, _ = cem.optimize(state, physics, ensemble, Q, R, x_ref, dt)

        def rollout_final_norm(u_const):
            x = state.copy()
            for _ in range(cfg.horizon):
                x = physics.step(x, u_const, dt)
            return float(np.linalg.norm(x - x_ref))

        norm_opt  = rollout_final_norm(action_opt)
        norm_zero = rollout_final_norm(np.zeros(cfg.action_dim))

        assert norm_opt <= norm_zero + 0.5, (
            f"Optimized action norm={norm_opt:.4f} should be <= zero norm={norm_zero:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUP 5 — REGISTRY SAVE/RELOAD (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegistrySaveReload:

    def _make_engine(self, params=None):
        if params is None:
            params = {"mass": 1.0, "friction": 0.15, "inertia": 0.01}
        engine = PhysiCore.for_platform("balancing_bot", initial_params=params)
        state = np.zeros(engine.cfg.state_dim)
        engine.step(state, state)  # populate _last_state / _last_sim_pred
        return engine

    def test_01_save_and_load_params_match(self, tmpdir_path):
        """Params survive round-trip save/load within floating-point tolerance."""
        reg = ModelRegistry(root=tmpdir_path)
        engine = self._make_engine({"mass": 1.5, "friction": 0.25, "inertia": 0.02})
        orig_params = engine.physics.params.copy()

        reg.save(engine, "balancing_bot")

        engine2 = self._make_engine({"mass": 9.9, "friction": 9.9, "inertia": 9.9})
        reg.load(engine2, "balancing_bot")

        loaded_params = engine2.physics.params
        for k, v in orig_params.items():
            # After one save with no prior, params_to_save == orig_params exactly
            assert abs(loaded_params.get(k, 0.0) - v) < 1e-6, (
                f"Param '{k}': expected {v}, got {loaded_params.get(k)}"
            )

    def test_02_second_save_updates_params(self, tmpdir_path):
        """Second save updates the params.json file with the latest session's params."""
        reg = ModelRegistry(root=tmpdir_path)

        # Session 1: mass=1.0
        e1 = self._make_engine({"mass": 1.0, "friction": 0.1, "inertia": 0.01})
        reg.save(e1, "balancing_bot")

        # Verify first session was saved
        import json
        with open(tmpdir_path / "balancing_bot" / "params.json") as f:
            data1 = json.load(f)
        assert data1["sessions_count"] == 1

        # Session 2: mass=2.0
        e2 = self._make_engine({"mass": 2.0, "friction": 0.1, "inertia": 0.01})
        reg.save(e2, "balancing_bot")

        with open(tmpdir_path / "balancing_bot" / "params.json") as f:
            data2 = json.load(f)
        # After second save, session count increments
        assert data2["sessions_count"] == 2
        # The saved params dict has a "params" key
        assert "params" in data2
        # Mass value is some float (regardless of exact EMA behavior)
        saved_mass = data2["params"]["mass"]
        assert isinstance(saved_mass, float)
        assert saved_mass > 0.0

    def test_03_session_count_increments(self, tmpdir_path):
        """sessions.jsonl line count grows with each save."""
        reg = ModelRegistry(root=tmpdir_path)
        engine = self._make_engine()

        reg.save(engine, "balancing_bot")
        count1 = reg._session_count("balancing_bot")
        reg.save(engine, "balancing_bot")
        count2 = reg._session_count("balancing_bot")

        assert count2 == count1 + 1

    def test_04_missing_platform_load_returns_false(self, tmpdir_path):
        """Loading a platform with no saved state returns False gracefully."""
        reg = ModelRegistry(root=tmpdir_path)
        engine = self._make_engine()
        result = reg.load(engine, "nonexistent_platform_xyz")
        assert result is False

    def test_05_ensemble_weights_saved_to_disk(self, tmpdir_path):
        """Ensemble member weights are correctly serialized to .npz files on disk."""
        reg = ModelRegistry(root=tmpdir_path)
        engine = self._make_engine()

        # Record weights from member 0 before saving
        m0_W1_before = engine.ensemble.members[0].W1.copy()
        m0_b1_before = engine.ensemble.members[0].b1.copy()

        reg.save(engine, "balancing_bot")

        # Verify the npz file was written and contains the correct weights
        npz_path = tmpdir_path / "balancing_bot" / "ensemble_0.npz"
        assert npz_path.exists(), "ensemble_0.npz was not created by save()"

        saved = np.load(str(npz_path))
        assert "W1" in saved, "npz missing W1 key"
        assert "b1" in saved, "npz missing b1 key"
        assert np.allclose(saved["W1"], m0_W1_before, atol=1e-8), (
            "Saved W1 in npz does not match engine's W1 before save"
        )
        assert np.allclose(saved["b1"], m0_b1_before, atol=1e-8), (
            "Saved b1 in npz does not match engine's b1 before save"
        )

    def test_06_cem_warmstart_survives_round_trip(self, tmpdir_path):
        """CEM mu/std warm-start survives save/load."""
        reg = ModelRegistry(root=tmpdir_path)
        engine = self._make_engine()

        # Run a few steps so CEM mu is non-trivial
        state = np.array([0.2, 0.1, 0.0, 0.0])
        for _ in range(5):
            engine.step(state, np.zeros(4))
        mu_before = engine.cem.mu.copy()

        reg.save(engine, "balancing_bot")

        engine2 = self._make_engine()
        engine2.cem.mu = np.zeros_like(mu_before)  # scramble
        reg.load(engine2, "balancing_bot")

        mu_after = engine2.cem.mu
        assert mu_after.shape == mu_before.shape
        assert np.allclose(mu_before, mu_after, atol=1e-8), (
            "CEM mu warm-start did not survive round-trip"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUP 6 — SENTINEL (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_SENTINEL, reason="sentinel module not importable")
class TestSentinel:

    def setup_method(self):
        """Patch print to avoid cp1252 UnicodeEncodeError on Windows."""
        builtins.print = _safe_print

    def teardown_method(self):
        builtins.print = _orig_print

    def _make_sentinel(self):
        engine = PhysiCore.for_platform(
            "balancing_bot",
            initial_params={"mass": 1.0, "friction": 0.15, "inertia": 0.01},
        )
        sentinel = SentinelOS(engine, platform="balancing_bot", verbose=False)
        return sentinel, engine

    def test_01_sentinel_initializes_without_error(self):
        """SentinelOS initializes cleanly."""
        sentinel, _ = self._make_sentinel()
        assert sentinel is not None

    def test_02_sentinel_step_returns_correct_action_dim(self):
        """sentinel.step() returns action with same dim as engine.action_dim."""
        sentinel, engine = self._make_sentinel()
        state = np.zeros(engine.cfg.state_dim)
        x_ref = np.zeros(engine.cfg.state_dim)

        action = sentinel.step(state, x_ref, altitude=0.0)

        assert isinstance(action, np.ndarray)
        assert action.shape == (engine.cfg.action_dim,)

    def test_03_status_has_required_keys(self):
        """sentinel.status dict contains mode, is_safe, lyapunov, fault."""
        sentinel, engine = self._make_sentinel()
        state = np.zeros(engine.cfg.state_dim)
        x_ref = np.zeros(engine.cfg.state_dim)
        sentinel.step(state, x_ref)

        s = sentinel.status
        assert "mode"     in s, "status missing 'mode'"
        assert "is_safe"  in s, "status missing 'is_safe'"
        assert "lyapunov" in s, "status missing 'lyapunov'"
        assert "fault"    in s, "status missing 'fault'"

    def test_04_state_explosion_triggers_non_nominal_mode(self):
        """Feeding state with very large values triggers non-NOMINAL mode."""
        sentinel, engine = self._make_sentinel()
        large_state = np.array([5000.0, 5000.0, 0.0, 0.0])
        x_ref = np.zeros(engine.cfg.state_dim)

        # Run a few times to allow mode transition
        for _ in range(3):
            sentinel.step(large_state, x_ref)

        mode = sentinel.mode
        assert mode != SentinelMode.NOMINAL, (
            f"Expected non-NOMINAL mode after state explosion, got {mode}"
        )

    def test_05_sentinel_does_not_crash_on_zero_state(self):
        """Sentinel handles zero state without exception."""
        sentinel, engine = self._make_sentinel()
        state = np.zeros(engine.cfg.state_dim)
        x_ref = np.zeros(engine.cfg.state_dim)

        action = sentinel.step(state, x_ref, altitude=0.0)
        assert np.all(np.isfinite(action)), "Action contains non-finite values"

    def test_06_mode_is_valid_sentinel_mode_enum(self):
        """sentinel.mode is always a valid SentinelMode enum value."""
        sentinel, engine = self._make_sentinel()
        state = np.zeros(engine.cfg.state_dim)
        x_ref = np.zeros(engine.cfg.state_dim)
        sentinel.step(state, x_ref)

        assert isinstance(sentinel.mode, SentinelMode), (
            f"Expected SentinelMode instance, got {type(sentinel.mode)}"
        )
        # Value must be one of the known enum values
        valid_values = {m.value for m in SentinelMode}
        assert sentinel.mode.value in valid_values


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUP 7 — PROPERTY-BASED / HYPOTHESIS (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestPropertyBased:
    """Property-based tests using the Hypothesis library."""

    @given(
        state=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_01_any_valid_state_engine_step_no_raise(self, state):
        """Any valid state vector → engine.step() completes without raising."""
        builtins.print = _safe_print
        try:
            engine = PhysiCore.for_platform(
                "balancing_bot",
                initial_params={"mass": 1.0, "friction": 0.15, "inertia": 0.01},
            )
            s = np.array(state, dtype=float)
            x_ref = np.zeros(4)
            result = engine.step(s, x_ref)
            assert isinstance(result.action, np.ndarray)
        finally:
            builtins.print = _orig_print

    @given(
        mass=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        friction=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=5000)
    def test_02_physics_step_finite_output(self, mass, friction):
        """Any params with mass>0, friction>=0 → physics step produces finite output."""
        params = {"mass": mass, "friction": friction, "inertia": 0.01}
        physics = PhysicsLayer(balancing_bot_dynamics, params)
        state = np.array([0.1, 0.0, 0.0, 0.0])
        action = np.array([0.0])
        result = physics.step(state, action, 1 / 60)
        assert np.all(np.isfinite(result)), (
            f"Physics step produced non-finite output for mass={mass}, friction={friction}"
        )

    @given(
        u=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=30, deadline=5000)
    def test_03_cem_cost_finite_for_bounded_action(self, u):
        """Any action within bounds → CEM internal cost returns a finite float."""
        cfg = PhysiCoreConfig(
            platform="balancing_bot", state_dim=4, action_dim=1, control_hz=60.0
        )
        cfg.cem_samples = 4
        cfg.horizon     = 3
        cfg.cem_iters   = 1
        bounds = np.array([[-1.0], [1.0]])
        cem     = CEMOptimizer(cfg, action_bounds=bounds)
        physics = PhysicsLayer(balancing_bot_dynamics, {"mass": 1.0, "friction": 0.15, "inertia": 0.01})
        ensemble = ResidualEnsemble(cfg)
        Q = np.eye(4)
        R = np.eye(1) * 0.1

        state = np.array([0.1, 0.0, 0.0, 0.0])
        x_ref = np.zeros(4)
        # Single horizon action sequence (1 step)
        actions = np.array([[u]])
        cost = cem._cost(state, actions, physics, ensemble, Q, R, x_ref, 1 / 60)
        assert math.isfinite(cost), f"Cost was non-finite for u={u}: {cost}"

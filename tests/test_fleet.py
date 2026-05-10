"""
PhysiCore Fleet Manager Test Suite
===================================
Tests for FleetManager, FleetRobotSpec, FleetHealth.

Run with:
    pytest tests/test_fleet.py -v

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Shared URDF fixture ──────────────────────────────────────────────────────

MINI_ARM_URDF = textwrap.dedent("""\
<?xml version="1.0"?>
<robot name="mini_arm">
  <link name="base"><inertial><mass value="1.0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
    <origin xyz="0 0 0.05"/>
  </inertial></link>
  <link name="link1"><inertial><mass value="0.5"/>
    <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
    <origin xyz="0 0 0.1"/>
  </inertial></link>
  <link name="ee"><inertial><mass value="0.2"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
  </inertial></link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="3"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="ee"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="5" velocity="3"/>
  </joint>
</robot>
""")


@pytest.fixture
def arm_urdf():
    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
        f.write(MINI_ARM_URDF)
        return f.name


@pytest.fixture
def fleet():
    from physicore.core.fleet import FleetManager
    fm = FleetManager()
    yield fm
    fm.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  ADD / REMOVE
# ══════════════════════════════════════════════════════════════════════════════

class TestFleetAddRemove:
    def test_add_from_urdf(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        assert "arm1" in fleet
        assert len(fleet) == 1

    def test_add_from_config(self, fleet):
        from physicore.core.robot_config import RobotConfig
        cfg = RobotConfig(platform="balancing_bot", mass=1.2)
        fleet.add_from_config("bot1", cfg)
        assert "bot1" in fleet

    def test_duplicate_id_raises(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        with pytest.raises(ValueError):
            fleet.add_from_urdf("arm1", arm_urdf)

    def test_remove_robot(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        fleet.remove("arm1")
        assert "arm1" not in fleet
        assert len(fleet) == 0

    def test_remove_nonexistent_raises(self, fleet):
        with pytest.raises(KeyError):
            fleet.remove("ghost")

    def test_clear(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        fleet.add_from_urdf("arm2", arm_urdf)
        fleet.clear()
        assert len(fleet) == 0

    def test_list_robots(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        fleet.add_from_urdf("arm2", arm_urdf)
        names = fleet.list_robots()
        assert set(names) == {"arm1", "arm2"}

    def test_add_from_spec_urdf(self, arm_urdf):
        from physicore.core.fleet import FleetManager, FleetRobotSpec
        fm = FleetManager()
        spec = FleetRobotSpec(robot_id="spec_arm", urdf_path=arm_urdf, tags=["test"])
        fm.add_from_spec(spec)
        assert "spec_arm" in fm
        fm.clear()

    def test_add_fleet_parallel(self, arm_urdf):
        from physicore.core.fleet import FleetManager, FleetRobotSpec
        fm = FleetManager()
        specs = [FleetRobotSpec(robot_id=f"r{i}", urdf_path=arm_urdf) for i in range(4)]
        fm.add_fleet(specs)
        assert len(fm) == 4
        fm.clear()

    def test_context_manager(self, arm_urdf):
        from physicore.core.fleet import FleetManager
        with FleetManager() as fm:
            fm.add_from_urdf("arm1", arm_urdf)
            assert len(fm) == 1
        assert len(fm) == 0

    def test_tags_stored(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf, tags=["lab_A", "test"])
        diag = fleet.diagnostics("arm1")
        assert "lab_A" in diag["tags"]


# ══════════════════════════════════════════════════════════════════════════════
#  STEPPING
# ══════════════════════════════════════════════════════════════════════════════

class TestFleetStepping:
    def test_single_step(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        slot = fleet._robots["arm1"]
        dim  = slot.engine.cfg.state_dim
        state = np.zeros(dim)
        x_ref = np.zeros(dim)
        result = fleet.step("arm1", state, x_ref)
        assert result is not None
        assert np.all(np.isfinite(result.action))

    def test_step_increments_counter(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        slot = fleet._robots["arm1"]
        dim  = slot.engine.cfg.state_dim
        for _ in range(5):
            fleet.step("arm1", np.zeros(dim), np.zeros(dim))
        assert fleet._robots["arm1"].step_count == 5

    def test_step_unknown_robot_raises(self, fleet):
        with pytest.raises(KeyError):
            fleet.step("ghost", np.zeros(4), np.zeros(4))

    def test_broadcast_step(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        fleet.add_from_urdf("arm2", arm_urdf)
        states = {}
        refs   = {}
        for rid in ["arm1", "arm2"]:
            dim = fleet._robots[rid].engine.cfg.state_dim
            states[rid] = np.zeros(dim)
            refs[rid]   = np.zeros(dim)
        results = fleet.broadcast_step(states, refs)
        assert set(results.keys()) == {"arm1", "arm2"}
        for r in results.values():
            assert not isinstance(r, Exception)

    def test_broadcast_parallel_vs_serial_same_shape(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        dim = fleet._robots["arm1"].engine.cfg.state_dim
        states = {"arm1": np.zeros(dim)}
        refs   = {"arm1": np.zeros(dim)}
        r_par = fleet.broadcast_step(states, refs, parallel=True)
        r_ser = fleet.broadcast_step(states, refs, parallel=False)
        assert r_par["arm1"].action.shape == r_ser["arm1"].action.shape

    def test_broadcast_skips_missing_states(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        fleet.add_from_urdf("arm2", arm_urdf)
        dim = fleet._robots["arm1"].engine.cfg.state_dim
        # Only provide state for arm1
        results = fleet.broadcast_step(
            {"arm1": np.zeros(dim)},
            {"arm1": np.zeros(dim)},
        )
        assert "arm1" in results
        assert "arm2" not in results

    def test_thread_safety(self, fleet, arm_urdf):
        """Multiple threads stepping the same robot concurrently should not crash."""
        fleet.add_from_urdf("arm1", arm_urdf)
        dim = fleet._robots["arm1"].engine.cfg.state_dim
        errors = []

        def _step():
            try:
                fleet.step("arm1", np.zeros(dim), np.zeros(dim))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_step) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == [], f"Concurrent step errors: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTICS & HEALTH
# ══════════════════════════════════════════════════════════════════════════════

class TestFleetDiagnostics:
    def test_health_all_healthy_at_start(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        h = fleet.health()
        assert h.robot_count == 1
        # Before any steps residual/unc = 0 → healthy
        assert h.healthy == 1

    def test_health_per_robot_keys(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        h = fleet.health()
        per = h.per_robot["arm1"]
        for key in ("status", "residual", "uncertainty", "loop_ms", "steps", "errors"):
            assert key in per

    def test_health_summary_string(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        summary = fleet.health().summary()
        assert "arm1" in summary
        assert "Fleet" in summary

    def test_diagnostics_robot_id(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        d = fleet.diagnostics("arm1")
        assert d["robot_id"] == "arm1"

    def test_diagnostics_step_count_updates(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        dim = fleet._robots["arm1"].engine.cfg.state_dim
        for _ in range(3):
            fleet.step("arm1", np.zeros(dim), np.zeros(dim))
        d = fleet.diagnostics("arm1")
        assert d["step_count"] == 3

    def test_params_returns_dict(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        p = fleet.params("arm1")
        assert isinstance(p, dict)
        assert "mass" in p

    def test_set_params(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        fleet.set_params("arm1", mass=9.9)
        p = fleet.params("arm1")
        assert abs(p.get("mass", 0) - 9.9) < 1e-6

    def test_multi_robot_health_count(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        fleet.add_from_urdf("arm2", arm_urdf)
        fleet.add_from_urdf("arm3", arm_urdf)
        h = fleet.health()
        assert h.robot_count == 3
        assert h.healthy + h.degraded + h.critical == 3

    def test_repr(self, fleet, arm_urdf):
        fleet.add_from_urdf("arm1", arm_urdf)
        r = repr(fleet)
        assert "FleetManager" in r
        assert "arm1" in r

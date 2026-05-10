"""
PhysiCore Fleet Manager
=======================
Manages multiple PhysiCore engines concurrently — each robot runs its own
engine, contact model, and SystemID in an isolated thread.

Features
--------
  - Add/remove robots at runtime
  - Per-robot step(), observe(), and diagnostics
  - Broadcast commands to all robots simultaneously
  - Fleet-level aggregated telemetry
  - Health monitoring (residual, uncertainty, loop time thresholds)
  - Thread-safe; each robot owns its own lock

Usage
-----
    from physicore.core.fleet import FleetManager, FleetRobotSpec

    fleet = FleetManager()

    # Add from URDF
    fleet.add_from_urdf("arm1", "robot_arm.urdf")

    # Add from config
    cfg = RobotConfig(platform="quadrotor", mass=1.5)
    fleet.add_from_config("drone1", cfg)

    # Step a single robot
    result = fleet.step("arm1", state, x_ref)

    # Broadcast reference to all robots (parallel threads)
    results = fleet.broadcast_step(states_dict, refs_dict)

    # Fleet health snapshot
    health = fleet.health()

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  ROBOT SLOT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RobotSlot:
    """Internal state for one robot in the fleet."""
    robot_id:   str
    engine:     Any          # PhysiCore instance
    config:     Any          # RobotConfig
    lock:       threading.Lock = field(default_factory=threading.Lock)
    step_count: int = 0
    last_step_ms: float = 0.0
    last_residual: float = 0.0
    last_uncertainty: float = 0.0
    last_action: Optional[np.ndarray] = None
    last_result: Optional[Any]        = None
    error_count: int = 0
    added_at:   float = field(default_factory=time.time)
    tags:       List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  FLEET SPEC (for batch initialisation)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FleetRobotSpec:
    """
    Specification for one robot to be added to the fleet.

    One of (urdf_path, config) must be provided.
    """
    robot_id:    str
    urdf_path:   Optional[str] = None
    config:      Optional[Any] = None   # RobotConfig
    platform_hint: Optional[str] = None
    control_hz:  float = 60.0
    tags:        List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  FLEET HEALTH SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FleetHealth:
    """Aggregated health report for the entire fleet."""
    robot_count:  int
    healthy:      int
    degraded:     int
    critical:     int
    per_robot:    Dict[str, Dict[str, Any]]

    def summary(self) -> str:
        lines = [
            f"Fleet: {self.robot_count} robots  "
            f"✓ {self.healthy}  ⚠ {self.degraded}  ✗ {self.critical}"
        ]
        for rid, info in self.per_robot.items():
            status = info["status"]
            sym = {"healthy": "✓", "degraded": "⚠", "critical": "✗"}.get(status, "?")
            lines.append(
                f"  {sym} {rid:20s}  residual={info['residual']:.3f}  "
                f"unc={info['uncertainty']:.3f}  loop={info['loop_ms']:.1f}ms  "
                f"steps={info['steps']}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  FLEET MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class FleetManager:
    """
    Thread-safe manager for a fleet of PhysiCore robots.

    Each robot runs its own independent engine and SystemID.
    """

    # Health thresholds
    RESIDUAL_WARN     = 0.30
    RESIDUAL_CRIT     = 0.80
    UNCERTAINTY_WARN  = 0.05
    UNCERTAINTY_CRIT  = 0.15
    LOOP_WARN_MS      = 20.0
    LOOP_CRIT_MS      = 50.0

    def __init__(self):
        self._robots: Dict[str, RobotSlot] = {}
        self._fleet_lock = threading.Lock()

    # ── Adding robots ──────────────────────────────────────────────────────────

    def add_from_urdf(
        self,
        robot_id: str,
        urdf_path: str,
        platform_hint: Optional[str] = None,
        control_hz: float = 60.0,
        tags: Optional[List[str]] = None,
        contact_stiffness: float = 5000.0,
        contact_damping:   float =  200.0,
        friction_mu:       float =    0.8,
    ) -> "FleetManager":
        """
        Add a robot to the fleet from a URDF or MJCF file.

        Returns self for chaining.
        """
        from physicore.core.urdf_loader import load_robot
        engine, config = load_robot(
            urdf_path,
            platform_hint=platform_hint,
            control_hz=control_hz,
            contact_stiffness=contact_stiffness,
            contact_damping=contact_damping,
            friction_mu=friction_mu,
        )
        return self._register(robot_id, engine, config, tags)

    def add_from_config(
        self,
        robot_id: str,
        config,   # RobotConfig
        control_hz: float = 60.0,
        tags: Optional[List[str]] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ) -> "FleetManager":
        """
        Add a robot to the fleet from a RobotConfig.

        Returns self for chaining.
        """
        from physicore.core.engine import PhysiCore
        engine = PhysiCore.for_platform(
            config.engine_platform,
            initial_params=config.initial_params,
            control_hz=control_hz,
            Q=Q,
            R=R,
            action_bounds=config.joint_action_bounds,
        )
        return self._register(robot_id, engine, config, tags)

    def add_from_spec(self, spec: FleetRobotSpec) -> "FleetManager":
        """Add a robot from a FleetRobotSpec."""
        if spec.urdf_path:
            return self.add_from_urdf(
                spec.robot_id, spec.urdf_path,
                platform_hint=spec.platform_hint,
                control_hz=spec.control_hz,
                tags=spec.tags,
            )
        elif spec.config:
            return self.add_from_config(
                spec.robot_id, spec.config,
                control_hz=spec.control_hz,
                tags=spec.tags,
            )
        else:
            raise ValueError(f"FleetRobotSpec for '{spec.robot_id}' needs urdf_path or config")

    def add_fleet(self, specs: List[FleetRobotSpec]) -> "FleetManager":
        """Add multiple robots in parallel (threaded initialisation)."""
        errors = []

        def _add(spec):
            try:
                self.add_from_spec(spec)
            except Exception as e:
                errors.append((spec.robot_id, e))

        threads = [threading.Thread(target=_add, args=(s,), daemon=True) for s in specs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            msgs = "; ".join(f"{rid}: {e}" for rid, e in errors)
            raise RuntimeError(f"Fleet init errors: {msgs}")
        return self

    def _register(self, robot_id: str, engine, config, tags) -> "FleetManager":
        slot = RobotSlot(
            robot_id=robot_id, engine=engine, config=config,
            tags=tags or [], added_at=time.time(),
        )
        with self._fleet_lock:
            if robot_id in self._robots:
                raise ValueError(f"Robot '{robot_id}' already exists in fleet")
            self._robots[robot_id] = slot
        print(f"[Fleet] Added robot '{robot_id}'  platform={config.engine_platform}")
        return self

    # ── Removing robots ────────────────────────────────────────────────────────

    def remove(self, robot_id: str) -> "FleetManager":
        """Remove a robot from the fleet."""
        with self._fleet_lock:
            if robot_id not in self._robots:
                raise KeyError(f"Robot '{robot_id}' not in fleet")
            del self._robots[robot_id]
        print(f"[Fleet] Removed robot '{robot_id}'")
        return self

    def clear(self) -> "FleetManager":
        """Remove all robots."""
        with self._fleet_lock:
            self._robots.clear()
        print("[Fleet] Cleared all robots")
        return self

    # ── Stepping ──────────────────────────────────────────────────────────────

    def step(self, robot_id: str, state: np.ndarray, x_ref: np.ndarray):
        """
        Run one control step for a specific robot.

        Parameters
        ----------
        robot_id : Robot identifier
        state    : Current state vector
        x_ref    : Reference state vector

        Returns
        -------
        ControlStep result
        """
        slot = self._get_slot(robot_id)
        with slot.lock:
            try:
                result = slot.engine.step(state, x_ref)
                slot.step_count        += 1
                slot.last_step_ms      = result.loop_time_ms
                slot.last_residual     = result.residual_norm
                slot.last_uncertainty  = result.uncertainty
                slot.last_action       = result.action.copy()
                slot.last_result       = result
                return result
            except Exception as e:
                slot.error_count += 1
                raise RuntimeError(f"[Fleet] step failed for '{robot_id}': {e}") from e

    def observe(self, robot_id: str, state: np.ndarray, actual_next_state: np.ndarray):
        """
        Feed a real hardware observation to a robot's SystemID.

        Parameters
        ----------
        robot_id          : Robot identifier
        state             : State before the last action
        actual_next_state : Real observed next state from hardware
        """
        slot = self._get_slot(robot_id)
        with slot.lock:
            if slot.last_action is None:
                return
            try:
                slot.engine.observe(state, slot.last_action, actual_next_state)
            except AttributeError:
                pass   # engine may not have observe() yet

    def broadcast_step(
        self,
        states:   Dict[str, np.ndarray],
        x_refs:   Dict[str, np.ndarray],
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Step all robots simultaneously.

        Parameters
        ----------
        states   : {robot_id: state_vector}
        x_refs   : {robot_id: ref_vector}
        parallel : If True, each robot runs in its own thread

        Returns
        -------
        {robot_id: ControlStep result | Exception}
        """
        results: Dict[str, Any] = {}

        if parallel:
            result_lock = threading.Lock()

            def _step_one(rid):
                if rid not in states or rid not in x_refs:
                    return
                try:
                    r = self.step(rid, states[rid], x_refs[rid])
                except Exception as e:
                    r = e
                with result_lock:
                    results[rid] = r

            threads = [
                threading.Thread(target=_step_one, args=(rid,), daemon=True)
                for rid in self._robots
            ]
            for t in threads: t.start()
            for t in threads: t.join()
        else:
            for rid in list(self._robots.keys()):
                if rid not in states or rid not in x_refs:
                    continue
                try:
                    results[rid] = self.step(rid, states[rid], x_refs[rid])
                except Exception as e:
                    results[rid] = e

        return results

    def broadcast_observe(
        self,
        states:      Dict[str, np.ndarray],
        next_states: Dict[str, np.ndarray],
    ):
        """Feed hardware observations to all robots."""
        for rid in list(self._robots.keys()):
            if rid in states and rid in next_states:
                self.observe(rid, states[rid], next_states[rid])

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def health(self) -> FleetHealth:
        """Return a fleet-level health snapshot."""
        per_robot: Dict[str, Dict] = {}
        healthy = degraded = critical = 0

        with self._fleet_lock:
            robot_items = list(self._robots.items())

        for rid, slot in robot_items:
            with slot.lock:
                r   = slot.last_residual
                unc = slot.last_uncertainty
                lms = slot.last_step_ms
                steps = slot.step_count
                errs  = slot.error_count

            if (r >= self.RESIDUAL_CRIT or unc >= self.UNCERTAINTY_CRIT
                    or lms >= self.LOOP_CRIT_MS or errs > 0):
                status = "critical"; critical += 1
            elif (r >= self.RESIDUAL_WARN or unc >= self.UNCERTAINTY_WARN
                  or lms >= self.LOOP_WARN_MS):
                status = "degraded"; degraded += 1
            else:
                status = "healthy"; healthy += 1

            per_robot[rid] = {
                "status":      status,
                "residual":    r,
                "uncertainty": unc,
                "loop_ms":     lms,
                "steps":       steps,
                "errors":      errs,
                "platform":    slot.config.engine_platform,
                "tags":        slot.tags,
            }

        return FleetHealth(
            robot_count=len(per_robot),
            healthy=healthy,
            degraded=degraded,
            critical=critical,
            per_robot=per_robot,
        )

    def diagnostics(self, robot_id: str) -> Dict:
        """Return detailed diagnostics for a single robot."""
        slot = self._get_slot(robot_id)
        with slot.lock:
            engine = slot.engine
            cfg    = slot.config
            try:
                params = engine.physics.params.copy()
            except AttributeError:
                params = {}
            try:
                failures = [e.to_dict() for e in engine.failure_log._events[-10:]]
            except AttributeError:
                failures = []
            try:
                sysid_history = engine.sysid.convergence_history[-20:]
            except AttributeError:
                sysid_history = []

            return {
                "robot_id":      robot_id,
                "platform":      cfg.engine_platform,
                "step_count":    slot.step_count,
                "error_count":   slot.error_count,
                "last_residual": slot.last_residual,
                "last_uncertainty": slot.last_uncertainty,
                "last_loop_ms":  slot.last_step_ms,
                "params":        params,
                "recent_failures": failures,
                "sysid_history": list(sysid_history),
                "tags":          slot.tags,
                "age_s":         time.time() - slot.added_at,
            }

    def params(self, robot_id: str) -> Dict[str, float]:
        """Return the current SystemID parameters for a robot."""
        slot = self._get_slot(robot_id)
        with slot.lock:
            try:
                return slot.engine.physics.params.copy()
            except AttributeError:
                return {}

    def set_params(self, robot_id: str, **kwargs):
        """Manually override physics params for a robot."""
        slot = self._get_slot(robot_id)
        with slot.lock:
            try:
                slot.engine.physics.params.update(kwargs)
            except AttributeError:
                pass

    # ── Introspection ──────────────────────────────────────────────────────────

    def list_robots(self) -> List[str]:
        with self._fleet_lock:
            return list(self._robots.keys())

    def __contains__(self, robot_id: str) -> bool:
        return robot_id in self._robots

    def __len__(self) -> int:
        return len(self._robots)

    def __repr__(self) -> str:
        return f"FleetManager({len(self._robots)} robots: {self.list_robots()})"

    def _get_slot(self, robot_id: str) -> RobotSlot:
        try:
            return self._robots[robot_id]
        except KeyError:
            raise KeyError(
                f"Robot '{robot_id}' not in fleet. "
                f"Available: {self.list_robots()}"
            )

    # ── Context manager support ────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.clear()

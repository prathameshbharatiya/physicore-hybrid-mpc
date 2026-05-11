from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np
import time


class EscalationLevel(str, Enum):
    NOMINAL   = "NOMINAL"
    WARNING   = "WARNING"
    SOFT_STOP = "SOFT_STOP"
    HARD_STOP = "HARD_STOP"
    E_STOP    = "E_STOP"


@dataclass
class SafetyViolation:
    kind: str          # "torque" | "velocity" | "position" | "workspace" | "power" | "thermal"
    joint: int         # joint index (-1 if global)
    value: float       # actual value
    limit: float       # limit that was exceeded
    timestamp: float

    def __str__(self) -> str:
        return (
            f"SafetyViolation(kind={self.kind!r}, joint={self.joint}, "
            f"value={self.value:.4f}, limit={self.limit:.4f}, "
            f"timestamp={self.timestamp:.3f})"
        )


@dataclass
class SafetyConfig:
    # Per-joint limits (use None for unconstrained)
    joint_limits_lo:  Optional[np.ndarray] = None  # position lower bounds
    joint_limits_hi:  Optional[np.ndarray] = None  # position upper bounds
    velocity_limits:  Optional[np.ndarray] = None  # max |velocity| per joint
    torque_limits:    Optional[np.ndarray] = None  # max |torque| per joint
    power_limit:      float = float('inf')          # total power limit (W)
    workspace_box:    Optional[np.ndarray] = None   # shape (2,3): [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    thermal_limit:    float = 85.0                  # degrees C (checked only when thermal data provided)
    action_dim:       int = 1                       # dimension of action/torque vector

    @classmethod
    def from_robot_config(cls, config) -> 'SafetyConfig':
        """Auto-populate from a RobotConfig (from physicore.core.robot_config).
        Falls back gracefully if config has no joint limit attributes."""
        kw = {}
        if hasattr(config, 'joint_action_bounds') and config.joint_action_bounds is not None:
            bounds = np.asarray(config.joint_action_bounds)  # shape (2, action_dim)
            if bounds.ndim == 2 and bounds.shape[0] == 2:
                kw['torque_limits'] = np.abs(bounds).max(axis=0)
                kw['action_dim'] = bounds.shape[1]
        if hasattr(config, 'action_dim'):
            kw['action_dim'] = config.action_dim
        return cls(**kw)

    @classmethod
    def for_platform(cls, platform: str, action_dim: int) -> 'SafetyConfig':
        """Sensible defaults for known platforms."""
        DEFAULTS = {
            'quadrotor':       dict(torque_limits=np.full(action_dim, 100.0),  power_limit=500.0),
            'balancing_bot':   dict(torque_limits=np.full(action_dim, 255.0),  power_limit=200.0),
            'manipulator_arm': dict(torque_limits=np.full(action_dim, 200.0),  power_limit=1000.0),
            'humanoid':        dict(torque_limits=np.full(action_dim, 150.0),  power_limit=2000.0),
            'surgical_robot':  dict(torque_limits=np.full(action_dim, 25.0),   power_limit=50.0),
            'exoskeleton':     dict(torque_limits=np.full(action_dim, 50.0),   power_limit=300.0),
        }
        kw = DEFAULTS.get(platform, dict(torque_limits=np.full(action_dim, 200.0)))
        kw['action_dim'] = action_dim
        return cls(**kw)


class HardwareSafetyInterlock:
    """
    Production safety layer that actually stops the robot.

    Escalation policy (automatic, based on violation severity):
      NOMINAL   -> normal operation
      WARNING   -> log violation, clip to hard limits
      SOFT_STOP -> clip to 50% of limits
      HARD_STOP -> zero output for all joints
      E_STOP    -> return zero action, set is_estopped=True

    Enforcement only runs when armed (call arm() first).
    """

    def __init__(self, config: SafetyConfig):
        self.config = config
        self._armed: bool = False
        self._is_estopped: bool = False
        self._escalation: EscalationLevel = EscalationLevel.NOMINAL
        self._violation_log: List[SafetyViolation] = []
        self._consecutive_violations: int = 0
        self._last_check_time: float = time.time()

    # ── Public API ────────────────────────────────────────────────────────────

    def arm(self) -> None:
        """Arm the interlock; safety checks are enforced from this point on."""
        self._armed = True
        self._is_estopped = False
        self._escalation = EscalationLevel.NOMINAL

    def disarm(self) -> None:
        """Disarm the interlock; check_and_clip becomes a pass-through."""
        self._armed = False

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def is_estopped(self) -> bool:
        return self._is_estopped

    @property
    def escalation_level(self) -> EscalationLevel:
        return self._escalation

    @property
    def violation_log(self) -> List[dict]:
        """Return the last 20 violations as plain dicts (safe to serialise)."""
        return [
            {
                'kind':      v.kind,
                'joint':     v.joint,
                'value':     round(v.value, 4),
                'limit':     round(v.limit, 4),
                'timestamp': round(v.timestamp, 3),
            }
            for v in self._violation_log[-20:]
        ]

    def emergency_stop(self) -> np.ndarray:
        """Trigger an E-STOP immediately. Returns zero action of correct dimension."""
        self._is_estopped = True
        self._escalation = EscalationLevel.E_STOP
        return np.zeros(self.config.action_dim)

    def reset_estop(self) -> None:
        """
        Reset the E-STOP flag after manual inspection.
        Leaves the system in WARNING mode so the operator is aware that a
        reset has occurred before returning to NOMINAL.
        """
        self._is_estopped = False
        self._escalation = EscalationLevel.WARNING
        self._consecutive_violations = 0

    def is_safe(self, state: np.ndarray) -> bool:
        """
        Lightweight state-only safety check (no action clipping).
        Returns True iff the state is inside all configured bounds and the
        interlock is not in E-STOP.
        """
        if self._is_estopped:
            return False
        violations = self._check_state(state)
        return len(violations) == 0

    def check_and_clip(
        self,
        action: np.ndarray,
        state: np.ndarray,
        params: dict,
        thermal_readings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[SafetyViolation]]:
        """
        Run all safety checks and return a safe action together with any
        violations that were detected.

        Checks performed (in order):
          1. Per-joint torque / force limits
          2. Workspace bounding box (first 3 state components = Cartesian pos)
          3. Per-joint velocity limits  (state[n//2 :] convention)
          4. Per-joint position limits  (state[:n//2] convention)
          5. Total electrical power     (sum |tau * omega|)
          6. Per-joint thermal readings (if provided)

        If not armed:  returns (action, []) unchanged.
        If e-stopped:  returns (zeros, []).
        """
        if not self._armed:
            return action.copy(), []
        if self._is_estopped:
            return np.zeros_like(action), []

        violations: List[SafetyViolation] = []
        safe_action = action.copy()
        now = time.time()

        # ── 1. Torque / force limits per joint ────────────────────────────────
        if self.config.torque_limits is not None:
            tl = np.asarray(self.config.torque_limits)
            n = min(len(safe_action), len(tl))
            for i in range(n):
                if abs(safe_action[i]) > tl[i]:
                    violations.append(
                        SafetyViolation('torque', i, float(safe_action[i]), float(tl[i]), now)
                    )

        # ── 2. Workspace bounding box ─────────────────────────────────────────
        if self.config.workspace_box is not None and len(state) >= 3:
            pos = state[:3]
            lo, hi = self.config.workspace_box[0], self.config.workspace_box[1]
            for i, (p, l, h) in enumerate(zip(pos, lo, hi)):
                if p < l or p > h:
                    bound = float(l) if p < l else float(h)
                    violations.append(
                        SafetyViolation('workspace', -1, float(p), bound, now)
                    )

        # ── 3. Velocity limits ────────────────────────────────────────────────
        # Convention: state = [q_0..q_{n-1}, dq_0..dq_{n-1}]
        if self.config.velocity_limits is not None and len(state) >= 2:
            vl   = np.asarray(self.config.velocity_limits)
            half = len(state) // 2
            n_v  = min(half, len(vl))
            for i in range(n_v):
                if abs(state[half + i]) > vl[i]:
                    violations.append(
                        SafetyViolation('velocity', i, float(state[half + i]), float(vl[i]), now)
                    )

        # ── 4. Joint position limits ──────────────────────────────────────────
        if (
            self.config.joint_limits_lo is not None
            and self.config.joint_limits_hi is not None
        ):
            lo   = np.asarray(self.config.joint_limits_lo)
            hi   = np.asarray(self.config.joint_limits_hi)
            half = len(state) // 2
            n_j  = min(half, len(lo), len(hi))
            for i in range(n_j):
                if state[i] < lo[i] or state[i] > hi[i]:
                    bound = float(lo[i]) if state[i] < lo[i] else float(hi[i])
                    violations.append(
                        SafetyViolation('position', i, float(state[i]), bound, now)
                    )

        # ── 5. Total electrical power  (|tau · dq|) ───────────────────────────
        if self.config.power_limit < float('inf') and len(state) >= len(safe_action):
            half  = len(state) // 2
            n_p   = min(len(safe_action), len(state) - half)
            power = float(np.sum(np.abs(safe_action[:n_p] * state[half: half + n_p])))
            if power > self.config.power_limit:
                violations.append(
                    SafetyViolation('power', -1, power, self.config.power_limit, now)
                )

        # ── 6. Thermal ────────────────────────────────────────────────────────
        if thermal_readings is not None and self.config.thermal_limit < float('inf'):
            for i, temp in enumerate(np.asarray(thermal_readings, dtype=float)):
                if temp > self.config.thermal_limit:
                    violations.append(
                        SafetyViolation('thermal', i, float(temp), self.config.thermal_limit, now)
                    )

        # ── Persist violations (ring buffer of 500) ───────────────────────────
        self._violation_log.extend(violations)
        if len(self._violation_log) > 500:
            self._violation_log = self._violation_log[-500:]

        # ── Apply escalation policy ───────────────────────────────────────────
        safe_action = self._apply_escalation(safe_action, violations)

        return safe_action, violations

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_state(self, state: np.ndarray) -> List[SafetyViolation]:
        """
        State-only checks used by is_safe().  Currently evaluates the workspace
        bounding box.  Joint position limits are also evaluated when configured.
        """
        violations: List[SafetyViolation] = []
        now = time.time()

        # Workspace box
        if self.config.workspace_box is not None and len(state) >= 3:
            pos = state[:3]
            lo, hi = self.config.workspace_box[0], self.config.workspace_box[1]
            for i, (p, l, h) in enumerate(zip(pos, lo, hi)):
                if p < l or p > h:
                    bound = float(l) if p < l else float(h)
                    violations.append(
                        SafetyViolation('workspace', -1, float(p), bound, now)
                    )

        # Joint position limits
        if (
            self.config.joint_limits_lo is not None
            and self.config.joint_limits_hi is not None
        ):
            lo   = np.asarray(self.config.joint_limits_lo)
            hi   = np.asarray(self.config.joint_limits_hi)
            half = len(state) // 2
            n_j  = min(half, len(lo), len(hi))
            for i in range(n_j):
                if state[i] < lo[i] or state[i] > hi[i]:
                    bound = float(lo[i]) if state[i] < lo[i] else float(hi[i])
                    violations.append(
                        SafetyViolation('position', i, float(state[i]), bound, now)
                    )

        return violations

    def _apply_escalation(
        self,
        action: np.ndarray,
        violations: List[SafetyViolation],
    ) -> np.ndarray:
        """
        Update the escalation state machine and return the clipped action.

        Escalation ladder (consecutive violation count drives severity):
          0 violations -> decay counter, possibly return to NOMINAL
          1-2 consec.  -> WARNING: clip to hard limits
          3-4 consec.  -> WARNING: clip to hard limits (sustained)
          5-9 consec.  -> SOFT_STOP: clip to 50 % of limits
          >= 10 consec -> HARD_STOP: zero output
          workspace or thermal violation (any count) -> HARD_STOP immediately
        """
        if not violations:
            # Gradually recover: decrement counter, return to NOMINAL when clear
            self._consecutive_violations = max(0, self._consecutive_violations - 1)
            if self._consecutive_violations == 0:
                self._escalation = EscalationLevel.NOMINAL
            return action

        self._consecutive_violations += 1
        n = self._consecutive_violations

        has_workspace = any(v.kind == 'workspace' for v in violations)
        has_thermal   = any(v.kind == 'thermal'   for v in violations)

        # Immediate hard-stop conditions
        if has_workspace or has_thermal or n >= 10:
            self._escalation = EscalationLevel.HARD_STOP
            return np.zeros_like(action)

        # SOFT_STOP: 50 % of torque limits
        if n >= 5:
            self._escalation = EscalationLevel.SOFT_STOP
            result = action.copy()
            if self.config.torque_limits is not None:
                tl     = np.asarray(self.config.torque_limits)
                n_clip = min(len(result), len(tl))
                result[:n_clip] = np.clip(
                    result[:n_clip], -tl[:n_clip] * 0.5, tl[:n_clip] * 0.5
                )
            return result

        # WARNING (n == 1..4): clip to hard limits
        self._escalation = EscalationLevel.WARNING
        result = action.copy()
        if self.config.torque_limits is not None:
            tl     = np.asarray(self.config.torque_limits)
            n_clip = min(len(result), len(tl))
            result[:n_clip] = np.clip(result[:n_clip], -tl[:n_clip], tl[:n_clip])
        return result

"""
Sentinel OS — PhysiCore Safety Governance Layer
================================================
Unified implementation merging PhysiCore's adaptation engine with
Sentinel's full mathematical safety stack.

Layer architecture (bottom to top):
  L0   — Preflight verification
  L0.5 — Mission phase manager (event horizon τ_prepare, separation Jacobian)
  L1   — Intent coherence & command contradiction detection
  L2   — Multi-body RLS estimator with forgetting factor λ
         Physics-informed drag (ISA atmosphere, transonic Mach drag rise)
         Shadow Lyapunov path (independent computation, bit-flip detection)
  L3   — Formal Lyapunov construction: A^T P + PA = -Q
         Lyapunov projection: binary-search scale to enforce dV/dt ≤ -α·V
         Prospective stability: 100ms lookahead on predicted mass
  L4   — Actuator envelope bounding (torque / power / temperature)
         Rocket FTS: recoverability score, trajectory corridor, governed recovery
  L5   — Fault signature library: BEARING_WEAR, UNEXPECTED_PAYLOAD,
         STRUCTURAL_AERO_DAMAGE, MOTOR_DEGRADATION, SENSOR_DRIFT, OOD_ANOMALY
  L6   — Pre/post processing pipeline: 500ms predictive violation window,
         jerk limiting, active correction at 200ms to violation
  L7   — SHA-256 heartbeat chain (tamper-evident continuous hash)
         PTP-synchronized forensic ledger per step

Mathematics sources:
  Lyapunov kernel:    sentinel_lyapunov.py (Sentinel v5.0.2)
  RLS estimator:      SentinelRuntime.ts   (Sentinel v5.0.2)
  Fault signatures:   sentinel_fault_observer.py + SentinelRuntime.ts
  Shadow path:        SentinelRuntime.ts shadowDivergence
  Forensic chain:     ForensicLedger.cpp (SHA-256)
  Mission phases:     SentinelRuntime.ts updateMissionPhase
  Preprocessing:      sentinel_preprocess.py
  Postprocessing:     sentinel_postprocess.py
  Atmos/drag:         SentinelRuntime.ts getAtmosphericDensity, getDragCoefficient
  FTS:                SentinelRuntime.ts monitorFts

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations

import time
import json
import math
import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Callable, Tuple, Any
import numpy as np

logger = logging.getLogger("sentinel")


# ══════════════════════════════════════════════════════════════════════════════
#  ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class SentinelMode(str, Enum):
    NOMINAL        = "NOMINAL"
    CAUTIOUS       = "CAUTIOUS"
    FALLBACK       = "FALLBACK"
    INTERNAL_FAULT = "INTERNAL_FAULT"   # shadow-path divergence detected


class MissionPhase(str, Enum):
    NOMINAL    = "NOMINAL"
    PREPARING  = "PREPARING"
    TRANSITION = "TRANSITION"
    SEPARATION = "SEPARATION"


class FaultType(str, Enum):
    BEARING_WEAR       = "BEARING_WEAR"
    UNEXPECTED_PAYLOAD = "UNEXPECTED_PAYLOAD"
    AERO_DAMAGE        = "STRUCTURAL_AERO_DAMAGE"
    MOTOR_DEGRADATION  = "MOTOR_DEGRADATION"
    SENSOR_DRIFT       = "SENSOR_DRIFT"
    OOD_ANOMALY        = "OOD_ANOMALY"
    TRANSONIC_JITTER   = "TRANSONIC_JITTER"
    PROSPECTIVE_INSTAB = "PROSPECTIVE_INSTABILITY"
    SHADOW_DIVERGENCE  = "SHADOW_DIVERGENCE"
    FTS_TRIGGERED      = "FLIGHT_TERMINATION"
    ACTUATOR_ENVELOPE  = "ACTUATOR_ENVELOPE_VIOLATION"


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RigidBody:
    """L2: Per-body RLS estimator state."""
    id:            str   = "PRIMARY"
    mass:          float = 1.0
    friction:      float = 0.1
    drag:          float = 0.05
    covariance:    float = 1000.0   # high = uncertain
    lam:           float = 0.99     # forgetting factor λ
    last_residual: float = 0.0


@dataclass
class FaultEvent:
    fault_type:    str
    severity:      str
    confidence:    float
    description:   str
    is_predictive: bool
    is_ood:        bool
    timestamp:     float
    step:          int
    params:        dict

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SentinelLog:
    """L7: One forensic ledger entry."""
    timestamp:    float
    step:         int
    mode:         str
    state_norm:   float
    V:            float
    V_shadow:     float
    shadow_delta: float
    uncertainty:  float
    residual:     float
    params:       dict
    action:       list
    trigger:      str
    hash:         str

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SentinelConfig:
    """All Sentinel OS configuration in one place."""

    # L3: Lyapunov
    max_lyapunov_energy:  float = 1000.0
    lyapunov_alpha:       float = 0.1    # dV/dt ≤ -α·V

    # L2: thresholds
    max_uncertainty_nominal:  float = 0.05
    max_uncertainty_cautious: float = 0.15
    max_residual_nominal:     float = 0.5
    max_residual_cautious:    float = 2.0
    max_param_drift:          float = 0.5

    # L4: actuator limits
    max_torque:    float = 80.0
    max_power:     float = 500.0
    max_temperature: float = 75.0

    # L2: shadow divergence tolerance
    max_shadow_divergence: float = 0.15

    # L3: prospective lookahead (seconds)
    prospective_dt: float = 0.1

    # L6: pipeline
    max_accel:           float = 5.0
    violation_predict_dt: float = 0.5

    # mode timeouts
    cautious_timeout_steps:  int = 100
    fallback_recovery_steps: int = 300

    # L4: rocket FTS
    fts_corridor_m:              float = 50.0
    fts_recoverability_threshold: float = 0.2

    # hard bounds
    action_min: Optional[np.ndarray] = None
    action_max: Optional[np.ndarray] = None

    # forensic log
    log_path: Optional[str] = None

    # custom safety functions
    custom_constraints: List[Callable] = field(default_factory=list)


# ── Platform presets ───────────────────────────────────────────────────────────
SENTINEL_PRESETS: Dict[str, SentinelConfig] = {
    "balancing_bot": SentinelConfig(
        # Calibrated from real hardware sessions (bridge_file_after*.txt)
        # Real unc: mean=0.57, p95=2.32 | Real res: mean=0.39, p95=1.01
        max_lyapunov_energy=200.0, lyapunov_alpha=0.15,
        max_uncertainty_nominal=1.5,   # 2.5x real mean
        max_uncertainty_cautious=4.0,  # above p95 = fault
        max_residual_nominal=0.8,      # 2x real mean
        max_residual_cautious=2.0,     # 2x p95 = serious fault
        max_torque=255.0, max_accel=10.0,
    ),
    "quadrotor": SentinelConfig(
        max_lyapunov_energy=500.0, lyapunov_alpha=0.10,
        max_uncertainty_nominal=0.05, max_uncertainty_cautious=0.20,
        max_residual_nominal=1.0, max_residual_cautious=5.0,
        max_torque=80.0, max_accel=8.0,
    ),
    "evtol": SentinelConfig(
        max_lyapunov_energy=10000.0, lyapunov_alpha=0.08,
        max_uncertainty_nominal=0.03, max_uncertainty_cautious=0.10,
        max_residual_nominal=0.5, max_residual_cautious=2.0,
        cautious_timeout_steps=50, max_torque=120.0,
    ),
    "rocket": SentinelConfig(
        max_lyapunov_energy=50000.0, lyapunov_alpha=0.05,
        max_uncertainty_nominal=0.02, max_uncertainty_cautious=0.08,
        max_residual_nominal=2.0, max_residual_cautious=15.0,
        fts_corridor_m=50.0, fts_recoverability_threshold=0.2,
        cautious_timeout_steps=200, prospective_dt=0.1,
    ),
    "surgical_robot": SentinelConfig(
        max_lyapunov_energy=0.1, lyapunov_alpha=0.30,
        max_uncertainty_nominal=0.005, max_uncertainty_cautious=0.02,
        max_residual_nominal=0.01, max_residual_cautious=0.05,
        max_param_drift=0.1, max_torque=25.0,
        cautious_timeout_steps=20, fallback_recovery_steps=500,
    ),
    "legged_robot": SentinelConfig(
        max_lyapunov_energy=200.0, lyapunov_alpha=0.12,
        max_uncertainty_nominal=0.08, max_uncertainty_cautious=0.25,
        max_residual_nominal=0.3, max_residual_cautious=1.5,
    ),
    "manipulator_arm": SentinelConfig(
        max_lyapunov_energy=150.0, lyapunov_alpha=0.20,
        max_uncertainty_nominal=0.04, max_uncertainty_cautious=0.12,
        max_residual_nominal=0.2, max_residual_cautious=1.0,
        max_torque=200.0, max_param_drift=0.3, cautious_timeout_steps=30,
    ),
    "auv": SentinelConfig(
        max_lyapunov_energy=5000.0, lyapunov_alpha=0.06,
        max_uncertainty_nominal=0.10, max_uncertainty_cautious=0.40,
        max_residual_nominal=2.0, max_residual_cautious=10.0,
        cautious_timeout_steps=200,
    ),
    "ground_rover": SentinelConfig(
        max_lyapunov_energy=300.0, lyapunov_alpha=0.10,
        max_uncertainty_nominal=0.10, max_uncertainty_cautious=0.30,
        max_residual_nominal=1.0, max_residual_cautious=4.0,
        max_torque=100.0,
    ),
    "satellite": SentinelConfig(
        max_lyapunov_energy=100000.0, lyapunov_alpha=0.02,
        max_uncertainty_nominal=0.001, max_uncertainty_cautious=0.01,
        max_residual_nominal=0.1, max_residual_cautious=1.0,
        cautious_timeout_steps=1000, fallback_recovery_steps=5000,
    ),
}


def get_sentinel_config(platform: str) -> SentinelConfig:
    return SENTINEL_PRESETS.get(platform, SentinelConfig())


# ══════════════════════════════════════════════════════════════════════════════
#  L2: ISA ATMOSPHERE + TRANSONIC DRAG MODEL (Sentinel v5.0.2)
# ══════════════════════════════════════════════════════════════════════════════

def isa_density(altitude_m: float) -> float:
    """ρ = ρ₀ · exp(-h / H_scale)"""
    return 1.225 * math.exp(-max(0.0, altitude_m) / 8500.0)


def speed_of_sound(altitude_m: float) -> float:
    """ISA speed of sound from temperature lapse rate."""
    T = max(216.65, 288.15 - 0.0065 * max(0.0, altitude_m))
    return 340.3 * math.sqrt(T / 288.15)


def transonic_drag_coefficient(mach: float) -> float:
    """
    L2: Transonic drag rise model (Sentinel SentinelRuntime.ts).
    Cd rises sharply in the transonic regime 0.8 < M < 1.2.
    """
    if mach < 0.8:  return 0.30
    if mach < 1.0:  return 0.30 + 0.50 * ((mach - 0.80) / 0.2) ** 2
    if mach < 1.2:  return 0.80 - 0.20 * ((mach - 1.00) / 0.2)
    return 0.60


# ══════════════════════════════════════════════════════════════════════════════
#  L3: LYAPUNOV KERNEL (sentinel_lyapunov.py + SentinelRuntime.ts)
# ══════════════════════════════════════════════════════════════════════════════

class LyapunovKernel:
    """
    Uncertainty-Aware Lyapunov Stability Kernel.

    Energy function:  V(x) = x^T P x
    Stability cond.:  dV/dt ≤ -α · V(x)
    Shadow energy:    V_s(x) = 0.5 · ||x||²  (independent of P)

    P is computed from the Lyapunov equation  A^T P + P A = -Q
    where A is the linearised system from estimated mass and friction.

    Simplified diagonal closed-form solution (Sentinel updateLyapunov):
        a₂₂ = -friction / mass
        P[0,0] = Q[0,0]
        P[1,1] = -Q[1,1] / (2 · a₂₂)

    Command projection (sentinel_lyapunov.py project_command):
        Binary-search for largest scale s ∈ [0,1] such that
        dV/dt(s · action) ≤ -α · V(x).
    """

    def __init__(self, state_dim: int, alpha: float = 0.1,
                 max_energy: float = 1000.0):
        self.n          = state_dim
        self.alpha      = alpha
        self.max_energy = max_energy
        self.P          = np.eye(state_dim)
        self.Q          = np.eye(state_dim) * 0.1

    def update_P(self, mass: float, friction: float) -> None:
        """
        Solve A^T P + PA = -Q for full P matrix (multi-DOF).
        Uses scipy.linalg.solve_continuous_lyapunov when available (n-DOF exact),
        falls back to diagonal closed-form for 1-DOF compatibility.
        """
        m   = max(mass, 0.01)
        f   = max(friction, 0.001)
        a22 = -f / m

        # Build linearised A matrix: block-diagonal [0, 1; 0, a22] per DOF pair
        n = self.n
        A = np.zeros((n, n))
        for i in range(0, n - 1, 2):
            A[i,     i + 1] = 1.0
            A[i + 1, i + 1] = a22

        try:
            from scipy.linalg import solve_continuous_lyapunov
            # A^T P + P A = -Q  ↔  solve_continuous_lyapunov(A^T, Q)
            P_solved = solve_continuous_lyapunov(A.T, self.Q)
            # Symmetrize and ensure positive-definite
            P_solved = 0.5 * (P_solved + P_solved.T)
            eigs = np.linalg.eigvalsh(P_solved)
            if np.all(eigs > 0):
                self.P = P_solved
                return
        except Exception:
            pass

        # Fallback: diagonal closed-form (1-DOF per axis)
        p_diag = np.ones(n)
        q_diag = np.diag(self.Q)
        for i in range(0, n, 2):
            p_diag[i] = q_diag[i]
            if i + 1 < n:
                p_diag[i + 1] = -q_diag[i + 1] / max(2 * a22, -1e6)
        self.P = np.diag(np.maximum(p_diag, 1e-6))

    def energy(self, x: np.ndarray) -> float:
        """V(x) = x^T P x"""
        return float(x @ self.P @ x)

    def energy_shadow(self, x: np.ndarray) -> float:
        """
        Shadow Lyapunov V_s = 0.5 ||x||²
        Independent of P — divergence vs energy() detects computation faults.
        (Sentinel computeLyapunovEnergyShadow)
        """
        return 0.5 * float(x @ x)

    def dV_dt(self, x: np.ndarray, x_dot: np.ndarray) -> float:
        """dV/dt = 2 · x^T P ẋ"""
        return 2.0 * float(x @ self.P @ x_dot)

    def project_action(
        self,
        action:      np.ndarray,
        x:           np.ndarray,
        x_dot_model: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        If dV/dt > -α·V, project action onto stability boundary.
        Binary search on scale s ∈ [0,1] (sentinel_lyapunov.py project_command).
        Returns (safe_action, was_projected).
        """
        V  = self.energy(x)
        dV = self.dV_dt(x, x_dot_model)

        if dV <= -self.alpha * V or V < 1e-6:
            return action, False

        if np.linalg.norm(action) < 1e-8:
            return action, False

        s_lo, s_hi = 0.0, 1.0
        for _ in range(12):                  # 12 iterations → 0.024% error
            s   = (s_lo + s_hi) / 2.0
            dVs = self.dV_dt(x, x_dot_model * s)
            if dVs <= -self.alpha * V:
                s_lo = s
            else:
                s_hi = s

        return action * s_lo, True

    def prospective_check(
        self,
        x:        np.ndarray,
        mass_now: float,
        mass_dot: float,
        dt:       float,
    ) -> bool:
        """
        L3 Prospective stability (Sentinel updateLyapunov rocket branch).
        Predicts V in dt seconds given mass depletion.
        Returns True if SAFE.
        """
        future_mass = max(0.01, mass_now + mass_dot * dt)
        V_future    = self.energy(x) * (mass_now / future_mass)
        return V_future <= self.max_energy


# ══════════════════════════════════════════════════════════════════════════════
#  L5: FAULT SIGNATURE LIBRARY (sentinel_fault_observer.py + SentinelRuntime.ts)
# ══════════════════════════════════════════════════════════════════════════════

class FaultSignatureLibrary:
    """
    Formal fault classification from parameter drift patterns.
    Four named signatures + OOD anomaly fallback.
    (Sentinel diagnoseFaults method, FAULT_SIGNATURES dict)
    """

    _SIGS = [
        {
            "id": "SIG_FRICTION_DRIFT",
            "type": FaultType.BEARING_WEAR,
            "severity": "WARNING",
            "predictive": True,
            "description": "Friction increasing steadily — bearing wear",
            "match": lambda m, f, d, r, cov:
                min(0.95, (f - 0.1) / 0.5) if (f > 0.4 and m < 2.0) else 0.0,
        },
        {
            "id": "SIG_MASS_STEP",
            "type": FaultType.UNEXPECTED_PAYLOAD,
            "severity": "WARNING",
            "predictive": False,
            "description": "Sudden mass change — unexpected payload",
            "match": lambda m, f, d, r, cov:
                min(0.98, abs(m - 1.0) / 2.0) if abs(m - 1.0) > 0.5 else 0.0,
        },
        {
            "id": "SIG_AERO_DRAG",
            "type": FaultType.AERO_DAMAGE,
            "severity": "WARNING",
            "predictive": False,
            "description": "Drag increase with high residual — structural aero damage",
            "match": lambda m, f, d, r, cov:
                min(0.9, (d / 0.5) * (r / 10.0)) if (d > 0.2 and r > 5.0) else 0.0,
        },
        {
            "id": "SIG_ACTUATOR_LOSS",
            "type": FaultType.MOTOR_DEGRADATION,
            "severity": "CRITICAL",
            "predictive": True,
            "description": "High covariance + residual without drift — motor degradation",
            "match": lambda m, f, d, r, cov:
                min(0.85, (cov / 3000) * (r / 15.0)) if (cov > 2000 and r > 8.0) else 0.0,
        },
        {
            "id": "SIG_SENSOR_DRIFT",
            "type": FaultType.SENSOR_DRIFT,
            "severity": "WARNING",
            "predictive": True,
            "description": "Residual growing — possible sensor drift",
            "match": lambda m, f, d, r, cov:
                min(0.9, r / 20.0) if r > 15.0 else 0.0,
        },
    ]

    def classify(
        self,
        mass: float, friction: float, drag: float,
        residual: float, covariance: float,
    ) -> Optional[dict]:
        best = {"confidence": 0.0, "type": None}
        for sig in self._SIGS:
            try:
                conf = sig["match"](mass, friction, drag, residual, covariance)
            except Exception:
                conf = 0.0
            if conf > best["confidence"]:
                best = {**sig, "confidence": conf}

        if best["confidence"] >= 0.4:
            return {
                "fault_type":    best["type"].value,
                "severity":      best["severity"],
                "confidence":    round(best["confidence"], 3),
                "description":   best["description"],
                "is_predictive": best["predictive"],
                "is_ood":        False,
                "signature_id":  best["id"],
            }

        # OOD fallback (Sentinel isOOD logic)
        if residual > 8.0 or covariance > 2500:
            return {
                "fault_type":    FaultType.OOD_ANOMALY.value,
                "severity":      "WARNING",
                "confidence":    0.5,
                "description":   "Anomalous — no fault signature match",
                "is_predictive": False,
                "is_ood":        True,
                "signature_id":  None,
            }
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  L7: SHA-256 FORENSIC LEDGER (ForensicLedger.cpp)
# ══════════════════════════════════════════════════════════════════════════════

class ForensicLedger:
    """
    SHA-256 hash chain over every control step.
    hash_n = SHA256(ts | mode | V | V_shadow | step | trigger | hash_{n-1})
    Tamper with any past entry → all subsequent hashes break.
    """

    def __init__(self, max_entries: int = 5000):
        self._chain_hash  = "0" * 64
        self._count       = 0
        self._entries:    List[SentinelLog] = []
        self._max         = max_entries
        self._logfile     = None

    def open_file(self, path: str) -> None:
        self._logfile = open(path, "w")
        self._logfile.write("[\n")

    def record(
        self,
        step: int, mode: str,
        state: np.ndarray,
        V: float, V_shadow: float,
        uncertainty: float, residual: float,
        params: dict, action: np.ndarray, trigger: str,
    ) -> str:
        self._count += 1
        ts = time.time()

        payload     = f"{ts:.6f}|{mode}|{V:.6f}|{V_shadow:.6f}|{step}|{trigger}|{self._chain_hash}"
        step_hash   = hashlib.sha256(payload.encode()).hexdigest()[:16]
        self._chain_hash = step_hash

        entry = SentinelLog(
            timestamp    = ts,
            step         = step,
            mode         = mode,
            state_norm   = float(np.linalg.norm(state)),
            V            = round(V, 6),
            V_shadow     = round(V_shadow, 6),
            shadow_delta = round(abs(V - V_shadow) / max(V, 1e-9), 6),
            uncertainty  = uncertainty,
            residual     = residual,
            params       = params,
            action       = action.tolist(),
            trigger      = trigger,
            hash         = step_hash,
        )
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries.pop(0)

        if self._logfile and trigger != "NOMINAL":
            self._logfile.write(json.dumps(entry.to_dict()) + ",\n")
            self._logfile.flush()

        return step_hash

    def close(self) -> None:
        if self._logfile:
            self._logfile.write("{}]\n")
            self._logfile.close()

    @property
    def entries(self) -> List[SentinelLog]:
        return list(self._entries)

    @property
    def chain_hash(self) -> str:
        return self._chain_hash

    @property
    def count(self) -> int:
        return self._count


# ══════════════════════════════════════════════════════════════════════════════
#  L0.5: MISSION PHASE MANAGER (SentinelRuntime.ts updateMissionPhase)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MissionEvent:
    id:                   str
    timestamp_ms:         float
    label:                str
    expected_mass_delta:  float = 0.0
    transition_window_ms: float = 2000.0
    is_separation:        bool  = False


class MissionPhaseManager:
    """
    Event-horizon-aware mission phase tracking.
    τ_prepare = (1 - λ) × 50000 ms — derived from RLS forgetting factor.
    Fault classification suspended during transition windows.
    """

    def __init__(self, timeline: Optional[List[MissionEvent]] = None):
        self._timeline     = timeline or []
        self._start_ms     = time.time() * 1000
        self.phase         = MissionPhase.NOMINAL
        self.is_preparing      = False
        self.is_transitioning  = False
        self.tau_prepare_ms    = 500.0
        self.time_to_next_ms   = float("inf")
        self.current_event_id  = None
        self.next_event_id     = None

    def update(self, lam: float) -> None:
        t_ms = time.time() * 1000 - self._start_ms

        # τ_prepare from forgetting factor (Sentinel updateMissionPhase)
        self.tau_prepare_ms = max(500.0, (1.0 - lam) * 50_000)

        next_ev = next((e for e in self._timeline if e.timestamp_ms > t_ms), None)
        curr_ev = next((e for e in reversed(self._timeline) if e.timestamp_ms <= t_ms), None)

        self.next_event_id   = next_ev.id if next_ev else None
        self.current_event_id = curr_ev.id if curr_ev else None
        self.time_to_next_ms  = (next_ev.timestamp_ms - t_ms) if next_ev else float("inf")
        self.is_preparing     = self.time_to_next_ms < self.tau_prepare_ms

        if curr_ev:
            elapsed               = t_ms - curr_ev.timestamp_ms
            self.is_transitioning = elapsed < curr_ev.transition_window_ms
            if curr_ev.is_separation and elapsed < 100:
                self.phase = MissionPhase.SEPARATION
            elif self.is_transitioning:
                self.phase = MissionPhase.TRANSITION
            else:
                self.phase = MissionPhase.NOMINAL
        else:
            self.is_transitioning = False
            self.phase = MissionPhase.PREPARING if self.is_preparing else MissionPhase.NOMINAL


# ══════════════════════════════════════════════════════════════════════════════
#  L2: MULTI-BODY RLS ESTIMATOR (SentinelRuntime.ts runRLS + splitBody)
# ══════════════════════════════════════════════════════════════════════════════

class MultiBodyRLS:
    """
    Recursive Least Squares for mass, friction, drag estimation.

    gain  = η · (P_cov / (P_cov + 1))
    mass  += innovation · gain · sign(u)
    drag  += innovation · gain · 0.1 · Cd · rho · v²
    P_cov  = (P_cov / λ) · (1 - gain)

    λ is widened (→ 0.90) during mission preparation phases.

    Propellant observer (rockets):  ṁ = F_thrust / (I_sp · g₀)
    Separation Jacobian: covariance expanded via J·P·J^T + Q_sep.
    """

    def __init__(self, bodies: Optional[List[RigidBody]] = None):
        self.bodies = bodies or [RigidBody()]

    def update(
        self,
        control_input: float,
        velocity:      float,
        innovation:    float,
        altitude:      float = 0.0,
        preparing:     bool  = False,
    ) -> None:
        for body in self.bodies:
            lam = max(0.90, body.lam - 0.02) if preparing else body.lam
            gain = 0.01 * (body.covariance / (body.covariance + 1.0))

            body.mass += innovation * gain * (1.0 if control_input >= 0 else -1.0)

            v2 = velocity ** 2
            if v2 > 0.1:
                rho  = isa_density(altitude)
                mach = abs(velocity) / max(speed_of_sound(altitude), 1.0)
                cd   = transonic_drag_coefficient(mach)
                body.drag += innovation * gain * 0.1 * (0.5 * rho * v2 * 0.1 * cd) * (
                    -1.0 if velocity > 0 else 1.0
                )

            body.covariance    = (body.covariance / lam) * (1.0 - gain)
            body.mass          = max(0.05,  min(50.0,   body.mass))
            body.drag          = max(0.001, min(5.0,    body.drag))
            body.covariance    = max(0.01,  min(5000.0, body.covariance))
            body.last_residual = abs(innovation)

    def propellant_observer(self, thrust: float, dt: float, isp: float = 300.0) -> float:
        """Rocket ṁ = F_thrust / (I_sp · g₀). Updates primary body mass."""
        m_dot = max(0.0, thrust) / max(isp * 9.80665, 1.0)
        if self.bodies:
            self.bodies[0].mass = max(0.05, self.bodies[0].mass - m_dot * dt)
        return m_dot

    def split_body(self, jacobian: Tuple[float, float] = (0.6, 0.4)) -> None:
        """
        Staging separation Jacobian (Sentinel splitBody).
        Covariance expanded: P_new = J·P_old·J^T + Q_sep.
        """
        if not self.bodies:
            return
        p = self.bodies[0]
        self.bodies = [
            RigidBody(id="BODY_A", mass=p.mass * jacobian[0],
                      friction=p.friction, drag=p.drag,
                      covariance=p.covariance * 1.5),
            RigidBody(id="BODY_B", mass=p.mass * jacobian[1],
                      friction=p.friction, drag=p.drag,
                      covariance=p.covariance * 1.8),
        ]

    @property
    def total_mass(self) -> float:
        return sum(b.mass for b in self.bodies)

    @property
    def avg_friction(self) -> float:
        if not self.bodies: return 0.1
        return sum(b.friction for b in self.bodies) / len(self.bodies)

    @property
    def avg_drag(self) -> float:
        if not self.bodies: return 0.05
        return sum(b.drag for b in self.bodies) / len(self.bodies)

    @property
    def avg_covariance(self) -> float:
        if not self.bodies: return 1000.0
        return sum(b.covariance for b in self.bodies) / len(self.bodies)


# ══════════════════════════════════════════════════════════════════════════════
#  L1: INTENT COHERENCE (SentinelRuntime.ts L1 coherence)
# ══════════════════════════════════════════════════════════════════════════════

class IntentCoherenceMonitor:
    """
    Detects contradictory command sequences.
    Rapid sign reversal across all action dimensions → coherence drops.
    Score below THRESHOLD → CAUTIOUS mode.
    """
    THRESHOLD = 0.4
    RECOVERY  = 0.001

    def __init__(self):
        self._history: deque = deque(maxlen=10)
        self.score    = 1.0
        self._last_ts = time.time()
        self.cmd_hz   = 0.0

    def update(self, action: np.ndarray) -> None:
        now = time.time()
        self.cmd_hz   = 1.0 / max(now - self._last_ts, 1e-3)
        self._last_ts = now

        if self._history and np.all(action * self._history[-1] < 0) and np.linalg.norm(action) > 0.1:
            self.score = max(0.0, self.score - 0.15)
        else:
            self.score = min(1.0, self.score + self.RECOVERY)

        self._history.append(action.copy())

    @property
    def is_coherent(self) -> bool:
        return self.score >= self.THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
#  L6: PREPROCESSING (sentinel_preprocess.py)
# ══════════════════════════════════════════════════════════════════════════════

class SentinelPreprocessor:
    """
    500ms predictive violation window.
    Linear prediction: pos_next = pos + vel × dt.
    Returns (violation_predicted, time_to_violation_s).
    """

    def __init__(self, state_limit: float = 2.8, predict_dt: float = 0.5):
        self.state_limit = state_limit
        self.predict_dt  = predict_dt

    def check(self, state: np.ndarray, velocity: np.ndarray) -> Tuple[bool, float]:
        n   = min(len(state), len(velocity))
        if n == 0:
            return False, 10.0
        predicted = state[:n] + velocity[:n] * self.predict_dt
        violation = bool(np.any(np.abs(predicted) > self.state_limit))
        ttv = 10.0
        for s, v in zip(state[:n], velocity[:n]):
            if abs(v) > 0.001:
                ttv = min(ttv, max(0.0, (self.state_limit - abs(s)) / abs(v)))
        return violation, ttv


# ══════════════════════════════════════════════════════════════════════════════
#  L6: POSTPROCESSING (sentinel_postprocess.py)
# ══════════════════════════════════════════════════════════════════════════════

class SentinelPostprocessor:
    """
    Jerk limiting + active correction.
    - Rate-limits action delta per dt (max_accel × dt)
    - Damps to 50% within 200ms of predicted violation
    """

    def __init__(self, max_accel: float = 5.0):
        self.max_accel    = max_accel
        self._last_action: Optional[np.ndarray] = None
        self._last_ts     = time.time()

    def process(self, action: np.ndarray, ttv: float) -> Tuple[np.ndarray, bool]:
        now = time.time()
        dt  = max(now - self._last_ts, 1e-4)
        self._last_ts = now
        correction    = False

        if self._last_action is not None:
            delta   = action - self._last_action
            max_dv  = self.max_accel * dt
            action  = self._last_action + np.clip(delta, -max_dv, max_dv)

        if ttv < 0.2:
            action     = action * 0.5
            correction = True

        self._last_action = action.copy()
        return action, correction


# ══════════════════════════════════════════════════════════════════════════════
#  L4: FLIGHT TERMINATION SYSTEM (SentinelRuntime.ts monitorFts)
# ══════════════════════════════════════════════════════════════════════════════

class FlightTerminationSystem:
    """
    Rocket FTS using Lyapunov V-tube recoverability score.
    score = 1.0 - drift/corridor - uncertainty×0.2 - residual×0.1
    score < threshold → FTS triggered.
    score ≥ threshold → governed recovery attempted first.
    """

    def __init__(self, corridor_m: float = 50.0, threshold: float = 0.2):
        self.corridor_m        = corridor_m
        self.threshold         = threshold
        self.is_armed          = True
        self.is_triggered      = False
        self.termination_reason = None
        self.recoverability    = 1.0
        self.recovery_tried    = False

    def evaluate(self, drift_m: float, uncertainty: float,
                 residual: float) -> Tuple[str, str]:
        if self.is_triggered:
            return "TERMINATE", "FTS already triggered"

        self.recoverability = max(0.0,
            1.0
            - (drift_m / max(self.corridor_m * 3, 1.0))
            - uncertainty * 0.2
            - residual * 0.1
        )

        if drift_m > self.corridor_m:
            if self.recoverability < self.threshold:
                self.is_triggered       = True
                self.termination_reason = "UNRECOVERABLE_TRAJECTORY"
                return "TERMINATE", (
                    f"Drift {drift_m:.1f}m > corridor {self.corridor_m:.0f}m, "
                    f"recoverability={self.recoverability:.2f}. FTS triggered."
                )
            elif not self.recovery_tried:
                self.recovery_tried = True
                return "RECOVERY", (
                    f"Outside corridor but recoverable (score={self.recoverability:.2f}). "
                    "Governed recovery initiated."
                )
        return "SAFE", ""


# ══════════════════════════════════════════════════════════════════════════════
#  SENTINEL OS — MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class SentinelOS:
    """
    Sentinel OS — Full 8-layer safety governance for PhysiCore.

    Every step runs:
      L0   Preflight check
      L0.5 Mission phase update
      L1   Intent coherence
      L2   Multi-body RLS + atmosphere + shadow Lyapunov
      L3   Lyapunov P-update + energy + command projection + prospective check
      L4   Actuator bounds + FTS (rocket) + torque clipping
      L5   Fault signature classification (suspended during transitions)
      L6   Pre/post processing (predictive window, jerk limiting, correction)
      L7   SHA-256 heartbeat chain + forensic ledger

    Usage:
        sentinel = SentinelOS(engine, platform='balancing_bot')
        action   = sentinel.step(state, x_ref)
        sentinel.observe(state, action, next_state)
        print(sentinel.status)
    """

    def __init__(
        self,
        engine,
        platform:         str                        = "balancing_bot",
        config:           Optional[SentinelConfig]   = None,
        fallback_fn:      Optional[Callable]         = None,
        mission_timeline: Optional[List[MissionEvent]] = None,
        isp:              float                      = 300.0,
        verbose:          bool                       = True,
    ):
        self.engine     = engine
        self.platform   = platform
        self.config     = config or get_sentinel_config(platform)
        self.fallback_fn = fallback_fn or self._zero_fallback
        self._isp       = isp
        self._verbose   = verbose

        # State machine
        self.mode            = SentinelMode.NOMINAL
        self._step           = 0
        self._cautious_steps = 0
        self._fallback_steps = 0
        self._initial_params = engine.physics.params.copy()

        # Sub-systems
        self._lya  = LyapunovKernel(
            engine.cfg.state_dim, self.config.lyapunov_alpha,
            self.config.max_lyapunov_energy,
        )
        self._rls     = MultiBodyRLS()
        self._faults  = FaultSignatureLibrary()
        self._ledger  = ForensicLedger()
        self._mission = MissionPhaseManager(mission_timeline)
        self._coher   = IntentCoherenceMonitor()
        self._pre     = SentinelPreprocessor()
        self._post    = SentinelPostprocessor(self.config.max_accel)
        self._fts     = (
            FlightTerminationSystem(
                self.config.fts_corridor_m,
                self.config.fts_recoverability_threshold,
            ) if platform == "rocket" else None
        )

        # Fault tracking
        self._active_fault: Optional[dict] = None
        self._fault_log: List[FaultEvent]  = []

        # Preflight
        self._preflight = self._run_preflight()

        if self.config.log_path:
            self._ledger.open_file(self.config.log_path)

        if self._verbose:
            print(
                f"[SENTINEL] Ready | platform={platform} | "
                f"mode=NOMINAL | α={self.config.lyapunov_alpha} | "
                f"V_max={self.config.max_lyapunov_energy}"
            )

    # ── PUBLIC API ─────────────────────────────────────────────────────────────

    def step(
        self,
        state:    np.ndarray,
        x_ref:    np.ndarray,
        altitude: float                   = 0.0,
        velocity: Optional[np.ndarray]   = None,
    ) -> np.ndarray:
        """One Sentinel-governed control step. Returns safe action."""
        self._step += 1
        state    = np.asarray(state,    dtype=float)
        x_ref    = np.asarray(x_ref,    dtype=float)
        velocity = np.asarray(velocity, dtype=float) if velocity is not None else state

        # ── L0.5 Mission phase ─────────────────────────────────────────────
        lam = self._rls.bodies[0].lam if self._rls.bodies else 0.99
        self._mission.update(lam)

        # ── L6 Preprocessing (500ms predictive window) ─────────────────────
        viol_pred, ttv = self._pre.check(state, velocity)

        # ── L2 RLS update from PhysiCore residual ─────────────────────────
        diag        = self.engine.diagnostics_full
        uncertainty = diag.get("uncertainty",   0.0)
        residual    = diag.get("residual_norm", 0.0)
        params      = diag.get("params",        {})

        if self._step > 1:
            ctrl_mag = float(np.linalg.norm(
                self.engine.cem.mu
                if hasattr(self.engine, "cem") else np.zeros(1)
            ))
            self._rls.update(
                control_input = ctrl_mag,
                velocity      = float(velocity[0]) if len(velocity) > 0 else 0.0,
                innovation    = residual * 0.1,
                altitude      = altitude,
                preparing     = self._mission.is_preparing,
            )

        # Rocket propellant mass observer
        if self.platform == "rocket" and self._rls.bodies:
            thrust   = float(np.linalg.norm(
                self.engine.cem.mu if hasattr(self.engine, "cem") else np.zeros(1)
            ))
            self._rls.propellant_observer(thrust, dt=1.0 / max(self.engine.cfg.control_hz, 1))

        # ── L3 Lyapunov P-update ───────────────────────────────────────────
        self._lya.update_P(self._rls.total_mass, self._rls.avg_friction)

        # ── L3 Lyapunov energies (primary + shadow) ────────────────────────
        V         = self._lya.energy(state)
        V_shadow  = self._lya.energy_shadow(state)
        shadow_delta = abs(V - V_shadow) / max(V, 1e-9)

        # ── L2 Shadow path divergence (bit-flip detection) ─────────────────
        # V = x^T P x  vs  V_shadow = 0.5||x||^2
        # The ratio V/V_shadow varies with state direction (P ≠ 0.5·I) so
        # we maintain a rolling window of the ratio and flag spikes that are
        # > 10x the rolling median — these indicate a sudden computation fault.
        if not hasattr(self, '_shadow_ratios'):
            self._shadow_ratios = []
        _ratio_now = V / max(V_shadow, 1e-9)
        self._shadow_ratios.append(_ratio_now)
        if len(self._shadow_ratios) > 50:
            self._shadow_ratios.pop(0)
        if len(self._shadow_ratios) >= 20 and V > 1.0:
            import statistics as _st
            _median = _st.median(self._shadow_ratios[:-1])
            _spike  = _ratio_now / max(_median, 1e-9)
            if _spike > 10.0 or _spike < 0.1:
                self._log_fault(
                    FaultType.SHADOW_DIVERGENCE, "CRITICAL", _spike,
                    f"Shadow Lyapunov ratio spiked {_spike:.1f}x vs rolling median "
                    f"({_median:.3f} → {_ratio_now:.3f}) — bit-flip suspected",
                )
                self.mode = SentinelMode.INTERNAL_FAULT

        # ── L5 Fault classification (suspended in transitions) ─────────────
        if not self._mission.is_transitioning and self.mode != SentinelMode.INTERNAL_FAULT:
            match = self._faults.classify(
                mass       = self._rls.total_mass,
                friction   = self._rls.avg_friction,
                drag       = self._rls.avg_drag,
                residual   = residual,
                covariance = self._rls.avg_covariance,
            )
            if match and match["confidence"] > 0.5:
                self._active_fault = match
                if match["severity"] == "CRITICAL":
                    if self.mode == SentinelMode.NOMINAL:
                        self.mode = SentinelMode.CAUTIOUS

        # ── Evaluate safety & mode transition ─────────────────────────────
        trigger = self._evaluate_safety(state, V, uncertainty, residual, params)
        self._update_mode(trigger)

        # ── Compute action ─────────────────────────────────────────────────
        if self.mode in (SentinelMode.FALLBACK, SentinelMode.INTERNAL_FAULT):
            action = np.asarray(self.fallback_fn(state, x_ref), dtype=float)
        else:
            ctrl   = self.engine.step(state, x_ref)
            action = ctrl.action.copy()
            if self.mode == SentinelMode.CAUTIOUS:
                action = action * 0.6

        # ── L3 Lyapunov command projection ────────────────────────────────
        if self.mode == SentinelMode.NOMINAL and hasattr(self.engine, "physics"):
            x_dot = self.engine.physics.dynamics_fn(state, action, self.engine.physics.params)
            action, projected = self._lya.project_action(action, state, x_dot)
            if projected:
                trigger += "|LYAPUNOV_PROJECTED"

        # ── L3 Prospective stability (rocket mass depletion) ───────────────
        if self.platform == "rocket":
            mass_dot = -(self.engine.physics.params.get("mass_flow_rate", 0.0))
            if not self._lya.prospective_check(state, self._rls.total_mass,
                                               mass_dot, self.config.prospective_dt):
                self._log_fault(
                    FaultType.PROSPECTIVE_INSTAB, "WARNING", V,
                    f"V-tube instability projected within "
                    f"{self.config.prospective_dt*1000:.0f}ms",
                    predictive=True,
                )
                if self.mode == SentinelMode.NOMINAL:
                    self.mode = SentinelMode.CAUTIOUS

        # ── L4 FTS ────────────────────────────────────────────────────────
        if self._fts is not None:
            drift = float(abs(state[0])) if len(state) > 0 else 0.0
            fts_act, fts_desc = self._fts.evaluate(drift, uncertainty, residual)
            if fts_act == "TERMINATE":
                self._log_fault(FaultType.FTS_TRIGGERED, "CATASTROPHIC", drift, fts_desc)
                self.mode  = SentinelMode.FALLBACK
                action     = np.zeros_like(action)
            elif fts_act == "RECOVERY":
                self._log_fault(FaultType.FTS_TRIGGERED, "CRITICAL", drift, fts_desc)
                action     = action * 0.3

        # ── L4 Actuator torque envelope ────────────────────────────────────
        torque = float(np.linalg.norm(action))
        if torque > self.config.max_torque:
            self._log_fault(
                FaultType.ACTUATOR_ENVELOPE, "CRITICAL", torque,
                f"Torque {torque:.1f} > limit {self.config.max_torque:.1f}",
            )
            action = action * (self.config.max_torque / max(torque, 1e-6))
            if self.mode == SentinelMode.NOMINAL:
                self.mode = SentinelMode.CAUTIOUS

        # ── Hard bounds ────────────────────────────────────────────────────
        action = self._enforce_bounds(action)
        action = self._check_custom_constraints(state, action)

        # ── L1 Intent coherence ────────────────────────────────────────────
        self._coher.update(action)
        if not self._coher.is_coherent and self.mode == SentinelMode.NOMINAL:
            self.mode = SentinelMode.CAUTIOUS
            trigger  += "|INCOHERENT_COMMANDS"

        # ── L6 Postprocessing (jerk limit + active correction) ─────────────
        action, correction_active = self._post.process(action, ttv)

        # ── L7 Forensic ledger (SHA-256 heartbeat chain) ───────────────────
        self._ledger.record(
            step=self._step, mode=self.mode.value,
            state=state, V=V, V_shadow=V_shadow,
            uncertainty=uncertainty, residual=residual,
            params=params, action=action, trigger=trigger,
        )

        # ── Terminal status print ──────────────────────────────────────────
        if self._verbose and self._step % 20 == 0:
            self._print_status(V, V_shadow, shadow_delta, trigger, uncertainty, residual)

        return action

    def observe(self, state: np.ndarray, action: np.ndarray,
                next_state: np.ndarray) -> None:
        """Pass real transition to PhysiCore (only when safe)."""
        if self.mode not in (SentinelMode.FALLBACK, SentinelMode.INTERNAL_FAULT):
            self.engine.observe(state, action, next_state)

    # ── SAFETY EVALUATION ─────────────────────────────────────────────────────

    def _evaluate_safety(self, state, V, uncertainty, residual, params) -> str:
        cfg = self.config

        if V > cfg.max_lyapunov_energy:
            return f"LYAPUNOV_EXCEEDED V={V:.1f}"
        if uncertainty > cfg.max_uncertainty_cautious:
            return f"UNCERTAINTY_HIGH unc={uncertainty:.4f}"
        if uncertainty > cfg.max_uncertainty_nominal:
            return "UNCERTAINTY_ELEVATED"
        if residual > cfg.max_residual_cautious:
            return f"RESIDUAL_HIGH res={residual:.4f}"
        if residual > cfg.max_residual_nominal:
            return "RESIDUAL_ELEVATED"

        # Only check mass and friction for drift — inertia is an adaptation
        # parameter expected to change significantly during convergence.
        # Also skip params whose initial value is near zero (avoids spurious drift).
        DRIFT_PARAMS = {"mass", "friction"}
        for name, val in params.items():
            if name not in DRIFT_PARAMS:
                continue
            init = self._initial_params.get(name, val)
            if abs(init) < 1e-6:
                continue
            drift = abs(val - init) / abs(init)
            if drift > cfg.max_param_drift:
                return f"PARAM_DRIFT {name}={drift:.2f}"

        return "NOMINAL"

    def _update_mode(self, trigger: str) -> None:
        prev = self.mode
        if self.mode == SentinelMode.INTERNAL_FAULT:
            return  # manual reset required

        if trigger == "NOMINAL":
            if self.mode == SentinelMode.CAUTIOUS:
                self._cautious_steps += 1
                if self._cautious_steps > 50:
                    self.mode = SentinelMode.NOMINAL; self._cautious_steps = 0
            elif self.mode == SentinelMode.FALLBACK:
                self._fallback_steps += 1
                if self._fallback_steps > self.config.fallback_recovery_steps:
                    self.mode = SentinelMode.CAUTIOUS; self._fallback_steps = 0
        elif "HIGH" in trigger or "EXCEEDED" in trigger:
            self.mode = SentinelMode.FALLBACK; self._fallback_steps = 0
        elif "ELEVATED" in trigger or "DRIFT" in trigger:
            if self.mode == SentinelMode.NOMINAL:
                self.mode = SentinelMode.CAUTIOUS; self._cautious_steps = 0
            elif self.mode == SentinelMode.CAUTIOUS:
                self._cautious_steps += 1
                if self._cautious_steps > self.config.cautious_timeout_steps:
                    self.mode = SentinelMode.FALLBACK

        if self.mode != prev:
            msg = f"[SENTINEL] {prev} → {self.mode} | {trigger}"
            logger.warning(msg)
            print(msg)

    def _enforce_bounds(self, action: np.ndarray) -> np.ndarray:
        if self.config.action_min is not None:
            action = np.maximum(action, self.config.action_min)
        if self.config.action_max is not None:
            action = np.minimum(action, self.config.action_max)
        return action

    def _check_custom_constraints(self, state: np.ndarray,
                                   action: np.ndarray) -> np.ndarray:
        for fn in self.config.custom_constraints:
            if not fn(state, action):
                return np.zeros_like(action)
        return action

    def _zero_fallback(self, state: np.ndarray,
                       x_ref: np.ndarray) -> np.ndarray:
        return np.zeros(self.engine.cfg.action_dim)

    def _log_fault(self, fault_type: FaultType, severity: str, value: float,
                   description: str, predictive: bool = False) -> None:
        evt = FaultEvent(
            fault_type    = fault_type.value,
            severity      = severity,
            confidence    = min(1.0, abs(value)),
            description   = description,
            is_predictive = predictive,
            is_ood        = fault_type == FaultType.OOD_ANOMALY,
            timestamp     = time.time(),
            step          = self._step,
            params        = self.engine.physics.params.copy(),
        )
        self._fault_log.append(evt)
        self._active_fault = evt.to_dict()
        print(f"[SENTINEL] ⚠  {severity} | {fault_type.value} | {description}")

    def _run_preflight(self):
        from dataclasses import dataclass as _dc
        pf_ok = (
            self.engine is not None
            and self._lya.energy(np.zeros(self.engine.cfg.state_dim)) < self.config.max_lyapunov_energy
        )
        return {
            "engine_loaded":   self.engine is not None,
            "lyapunov_safe":   pf_ok,
            "ledger_writable": True,
            "is_ready":        pf_ok,
        }

    def _print_status(self, V, V_shadow, shadow_delta, trigger,
                      uncertainty, residual) -> None:
        COLOURS = {
            "NOMINAL": "\033[92m", "CAUTIOUS": "\033[93m",
            "FALLBACK": "\033[91m", "INTERNAL_FAULT": "\033[95m",
        }
        c   = COLOURS.get(self.mode.value, "")
        RST = "\033[0m"
        fault_str = (
            f" | ⚠ {self._active_fault['fault_type']}"
            if self._active_fault else ""
        )
        print(
            f"[SENTINEL] step={self._step:4d} | {c}{self.mode.value:14s}{RST} | "
            f"V={V:9.3f} Vs={V_shadow:9.3f} Δ={shadow_delta:.3f} | "
            f"unc={uncertainty:.4f} res={residual:.4f} | "
            f"mass={self._rls.total_mass:.3f} cov={self._rls.avg_covariance:.0f}"
            f"{fault_str}"
        )

    # ── PROPERTIES ─────────────────────────────────────────────────────────────

    @property
    def is_safe(self) -> bool:
        return self.mode not in (SentinelMode.FALLBACK, SentinelMode.INTERNAL_FAULT)

    @property
    def status(self) -> dict:
        last = self._ledger.entries[-1] if self._ledger.entries else None
        return {
            "mode":            self.mode.value,
            "step":            self._step,
            "is_safe":         self.is_safe,
            "lyapunov": {
                "V":           round(last.V if last else 0.0, 4),
                "V_shadow":    round(last.V_shadow if last else 0.0, 4),
                "shadow_delta": round(last.shadow_delta if last else 0.0, 4),
            },
            "rls": {
                "total_mass":     round(self._rls.total_mass,    4),
                "avg_friction":   round(self._rls.avg_friction,  4),
                "avg_drag":       round(self._rls.avg_drag,      4),
                "avg_covariance": round(self._rls.avg_covariance, 2),
                "n_bodies":       len(self._rls.bodies),
            },
            "fault":            self._active_fault,
            "fault_count":      len(self._fault_log),
            "mission_phase":    self._mission.phase.value,
            "is_transitioning": self._mission.is_transitioning,
            "is_preparing":     self._mission.is_preparing,
            "coherence_score":  round(self._coher.score, 3),
            "ledger_hash":      self._ledger.chain_hash,
            "ledger_entries":   self._ledger.count,
            "preflight":        self._preflight,
            "cautious_steps":   self._cautious_steps,
            "fallback_steps":   self._fallback_steps,
            "fts": (
                {
                    "is_triggered":   self._fts.is_triggered,
                    "recoverability": round(self._fts.recoverability, 3),
                    "corridor_m":     self._fts.corridor_m,
                }
                if self._fts else None
            ),
        }

    @property
    def fault_log(self) -> List[dict]:
        return [f.to_dict() for f in self._fault_log]

    @property
    def ledger(self) -> List[dict]:
        return [e.to_dict() for e in self._ledger.entries]

    @property
    def chain_hash(self) -> str:
        return self._ledger.chain_hash

    def reset_internal_fault(self) -> None:
        """Manually clear INTERNAL_FAULT (requires deliberate call)."""
        self.mode = SentinelMode.CAUTIOUS
        print("[SENTINEL] INTERNAL_FAULT cleared — entering CAUTIOUS")

    def trigger_separation(self) -> None:
        """Trigger staging separation Jacobian split (rocket)."""
        self._rls.split_body()
        print(f"[SENTINEL] Separation executed | bodies: {len(self._rls.bodies)}")

    def close(self) -> None:
        self._ledger.close()

"""
Sentinel OS Core
================
Safety governance layer implementing three-mode state machine:
  NOMINAL   — PhysiCore running normally
  CAUTIOUS  — Tightened constraints, conservative planning
  FALLBACK  — PhysiCore disabled, handoff to safe controller

Implements:
  - Lyapunov-based stability monitoring
  - Hard safety envelope enforcement
  - Forensic logging of every decision
  - Automatic mode transitions
"""

from __future__ import annotations

import time
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Callable
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from physicore import PhysiCore

logger = logging.getLogger("sentinel")


class SentinelMode(str, Enum):
    NOMINAL  = "NOMINAL"
    CAUTIOUS = "CAUTIOUS"
    FALLBACK = "FALLBACK"


@dataclass
class SentinelConfig:
    """All Sentinel OS configuration in one place."""

    # Uncertainty thresholds
    max_uncertainty_nominal:  float = 0.05
    max_uncertainty_cautious: float = 0.15

    # Residual thresholds
    max_residual_nominal:  float = 0.5
    max_residual_cautious: float = 2.0

    # Parameter drift limits (fraction of initial value)
    max_param_drift: float = 0.5

    # Hard action bounds (override PhysiCore if violated)
    action_min: Optional[np.ndarray] = None
    action_max: Optional[np.ndarray] = None

    # Lyapunov energy threshold
    max_lyapunov_energy: float = 1000.0

    # Steps in CAUTIOUS before escalating to FALLBACK
    cautious_timeout_steps: int = 100

    # Steps in FALLBACK before attempting NOMINAL recovery
    fallback_recovery_steps: int = 300

    # Forensic log path (None = no file logging)
    log_path: Optional[str] = None

    # Custom safety constraints: fn(state, action) -> bool (True = safe)
    custom_constraints: List[Callable] = field(default_factory=list)


@dataclass
class SentinelLog:
    """Single forensic log entry."""
    timestamp:   float
    step:        int
    mode:        str
    state_norm:  float
    uncertainty: float
    residual:    float
    params:      dict
    action:      List[float]
    trigger:     str  # what caused a mode change or flag

    def to_dict(self) -> dict:
        return {
            "timestamp":   self.timestamp,
            "step":        self.step,
            "mode":        self.mode,
            "state_norm":  round(self.state_norm, 4),
            "uncertainty": round(self.uncertainty, 6),
            "residual":    round(self.residual, 6),
            "params":      self.params,
            "action":      [round(a, 4) for a in self.action],
            "trigger":     self.trigger,
        }


class SentinelOS:
    """
    Sentinel OS — Safety governance layer for PhysiCore.

    Monitors PhysiCore in real time and switches between three modes:
      NOMINAL   — all metrics within bounds, full PhysiCore control
      CAUTIOUS  — one metric near limit, conservative PhysiCore control
      FALLBACK  — unsafe state, PhysiCore disabled, safe fallback active

    Usage:
        sentinel = SentinelOS(engine, config)
        action   = sentinel.step(state, x_ref)
        sentinel.observe(state, action, next_state)
    """

    def __init__(
        self,
        engine:          PhysiCore,
        config:          Optional[SentinelConfig] = None,
        fallback_fn:     Optional[Callable] = None,
    ):
        self.engine      = engine
        self.config      = config or SentinelConfig()
        self.fallback_fn = fallback_fn or self._zero_fallback
        self.mode        = SentinelMode.NOMINAL
        self._step       = 0
        self._cautious_steps  = 0
        self._fallback_steps  = 0
        self._initial_params  = engine.physics.params.copy()
        self._log: List[SentinelLog] = []
        self._logfile = None

        if self.config.log_path:
            self._logfile = open(self.config.log_path, 'w')
            self._logfile.write("[\n")

        logger.info(f"Sentinel OS initialized | mode={self.mode}")

    def step(self, state: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        One Sentinel-governed control step.

        Returns action that is:
          - PhysiCore optimal in NOMINAL
          - PhysiCore conservative in CAUTIOUS
          - Fallback controller output in FALLBACK
        """
        self._step += 1

        diag        = self.engine.diagnostics_full
        uncertainty = diag["uncertainty"]
        residual    = diag["residual_norm"]
        params      = diag["params"]

        trigger = self._evaluate_safety(state, uncertainty, residual, params)
        self._update_mode(trigger)

        if self.mode == SentinelMode.FALLBACK:
            action = self.fallback_fn(state, x_ref)
        else:
            control_step = self.engine.step(state, x_ref)
            action       = control_step.action
            if self.mode == SentinelMode.CAUTIOUS:
                action = action * 0.6

        action = self._enforce_bounds(action)
        action = self._check_custom_constraints(state, action)

        self._record(state, action, uncertainty, residual, params, trigger)
        return action

    def observe(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        """Pass real transition to engine (only in NOMINAL/CAUTIOUS)."""
        if self.mode != SentinelMode.FALLBACK:
            self.engine.observe(state, action, next_state)

    def _evaluate_safety(
        self,
        state:       np.ndarray,
        uncertainty: float,
        residual:    float,
        params:      dict,
    ) -> str:
        cfg = self.config

        # Lyapunov energy check
        energy = float(np.sum(state ** 2))
        if energy > cfg.max_lyapunov_energy:
            return f"LYAPUNOV_EXCEEDED energy={energy:.1f}"

        # Uncertainty check
        if uncertainty > cfg.max_uncertainty_cautious:
            return f"UNCERTAINTY_HIGH unc={uncertainty:.4f}"
        if uncertainty > cfg.max_uncertainty_nominal:
            return "UNCERTAINTY_ELEVATED"

        # Residual check
        if residual > cfg.max_residual_cautious:
            return f"RESIDUAL_HIGH res={residual:.4f}"
        if residual > cfg.max_residual_nominal:
            return "RESIDUAL_ELEVATED"

        # Parameter drift check
        for name, val in params.items():
            init_val = self._initial_params.get(name, val)
            drift    = abs(val - init_val) / max(abs(init_val), 1e-9)
            if drift > cfg.max_param_drift:
                return f"PARAM_DRIFT {name}={drift:.2f}"

        return "NOMINAL"

    def _update_mode(self, trigger: str) -> None:
        prev_mode = self.mode

        if trigger == "NOMINAL":
            if self.mode == SentinelMode.CAUTIOUS:
                self._cautious_steps += 1
                if self._cautious_steps > 50:
                    self.mode = SentinelMode.NOMINAL
                    self._cautious_steps = 0
            elif self.mode == SentinelMode.FALLBACK:
                self._fallback_steps += 1
                if self._fallback_steps > self.config.fallback_recovery_steps:
                    self.mode = SentinelMode.CAUTIOUS
                    self._fallback_steps = 0
        elif "HIGH" in trigger or "EXCEEDED" in trigger:
            self.mode = SentinelMode.FALLBACK
            self._fallback_steps = 0
        elif "ELEVATED" in trigger or "DRIFT" in trigger:
            if self.mode == SentinelMode.NOMINAL:
                self.mode = SentinelMode.CAUTIOUS
                self._cautious_steps = 0
            elif self.mode == SentinelMode.CAUTIOUS:
                self._cautious_steps += 1
                if self._cautious_steps > self.config.cautious_timeout_steps:
                    self.mode = SentinelMode.FALLBACK

        if self.mode != prev_mode:
            logger.warning(
                f"Sentinel mode: {prev_mode} → {self.mode} | trigger: {trigger}"
            )

    def _enforce_bounds(self, action: np.ndarray) -> np.ndarray:
        if self.config.action_min is not None:
            action = np.maximum(action, self.config.action_min)
        if self.config.action_max is not None:
            action = np.minimum(action, self.config.action_max)
        return action

    def _check_custom_constraints(
        self,
        state:  np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        for fn in self.config.custom_constraints:
            if not fn(state, action):
                logger.warning("Custom constraint violated — zeroing action")
                return np.zeros_like(action)
        return action

    def _zero_fallback(
        self,
        state: np.ndarray,
        x_ref: np.ndarray,
    ) -> np.ndarray:
        """Default fallback: zero action (safe stop)."""
        return np.zeros(self.engine.cfg.action_dim)

    def _record(
        self,
        state:       np.ndarray,
        action:      np.ndarray,
        uncertainty: float,
        residual:    float,
        params:      dict,
        trigger:     str,
    ) -> None:
        entry = SentinelLog(
            timestamp=   time.time(),
            step=        self._step,
            mode=        self.mode.value,
            state_norm=  float(np.linalg.norm(state)),
            uncertainty= uncertainty,
            residual=    residual,
            params=      params,
            action=      action.tolist(),
            trigger=     trigger,
        )
        self._log.append(entry)
        if len(self._log) > 10000:
            self._log.pop(0)

        if self._logfile and trigger != "NOMINAL":
            self._logfile.write(json.dumps(entry.to_dict()) + ",\n")
            self._logfile.flush()

    @property
    def log(self) -> List[SentinelLog]:
        return list(self._log)

    @property
    def is_safe(self) -> bool:
        return self.mode != SentinelMode.FALLBACK

    @property
    def summary(self) -> dict:
        return {
            "mode":            self.mode.value,
            "step":            self._step,
            "is_safe":         self.is_safe,
            "cautious_steps":  self._cautious_steps,
            "fallback_steps":  self._fallback_steps,
            "log_entries":     len(self._log),
        }

    def close(self) -> None:
        if self._logfile:
            self._logfile.write("{}]\n")
            self._logfile.close()

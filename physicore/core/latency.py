"""
PhysiCore Latency Compensation — Smith Predictor
==================================================
Compensates for round-trip communication latency in real-time control.

Without compensation at 20ms latency + 60Hz:
  MPC plans for x(t), action arrives at x(t+1.2) — wrong state.
  Result: oscillations, overshoot, instability.

With Smith Predictor:
  Propagate x(t) forward by L steps using physics model.
  MPC plans for x(t+L) — action arrives exactly when predicted.
  Result: ~95% latency effect eliminated.

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations
import time
import collections
import numpy as np
from typing import Callable, Optional, Dict, Deque, Tuple


class LatencyEstimator:
    """
    Online RTT estimator with rolling window.

    Usage:
        est = LatencyEstimator()
        t0  = est.send_ping()
        ...
        est.record_pong(t0)
        ms  = est.latency_ms
    """

    def __init__(self, window: int = 50):
        self._window  = window
        self._rtts:   Deque[float] = collections.deque(maxlen=window)
        self._ema_ms  = 20.0
        self._alpha   = 0.1

    def send_ping(self) -> float:
        return time.perf_counter()

    def record_pong(self, t_sent: float):
        rtt_ms = (time.perf_counter() - t_sent) * 1000
        if 0 < rtt_ms < 2000:
            self._rtts.append(rtt_ms)
            self._ema_ms = (1 - self._alpha) * self._ema_ms + self._alpha * rtt_ms

    def record_rtt(self, rtt_ms: float):
        if 0 < rtt_ms < 2000:
            self._rtts.append(rtt_ms)
            self._ema_ms = (1 - self._alpha) * self._ema_ms + self._alpha * rtt_ms

    @property
    def latency_ms(self) -> float:
        """One-way latency estimate (RTT / 2)."""
        return self._ema_ms / 2.0

    @property
    def rtt_ms(self) -> float:
        return self._ema_ms

    @property
    def median_rtt_ms(self) -> float:
        if not self._rtts:
            return self._ema_ms
        return float(np.median(list(self._rtts)))

    @property
    def p95_rtt_ms(self) -> float:
        if not self._rtts:
            return self._ema_ms * 2
        return float(np.percentile(list(self._rtts), 95))

    @property
    def n_samples(self) -> int:
        return len(self._rtts)

    def to_dict(self) -> dict:
        return {
            "latency_ms":    round(self.latency_ms, 2),
            "rtt_ema_ms":    round(self._ema_ms, 2),
            "rtt_median_ms": round(self.median_rtt_ms, 2),
            "rtt_p95_ms":    round(self.p95_rtt_ms, 2),
            "n_samples":     self.n_samples,
        }


class SmithPredictor:
    """
    Smith Predictor for MPC latency compensation.

    Steps each control cycle:
      1. record(state, action)           — push into history buffer
      2. state_for_mpc = compensate(...) — propagate state forward by L steps
      3. Pass state_for_mpc to engine.step() instead of raw state
    """

    def __init__(self, dynamics_fn: Callable, initial_params: Dict,
                 control_hz: float = 60.0, max_latency_ms: float = 200.0):
        self.dynamics_fn    = dynamics_fn
        self.params         = initial_params.copy()
        self.dt             = 1.0 / control_hz
        self.control_hz     = control_hz
        self.estimator      = LatencyEstimator()

        max_steps = int(max_latency_ms / 1000.0 * control_hz) + 5
        self._history: Deque[Tuple[float, np.ndarray, Optional[np.ndarray]]] = \
            collections.deque(maxlen=max_steps)

        self._manual_latency_ms: Optional[float] = None
        self._compensation_steps  = 0
        self._total_compensations = 0

    def update_params(self, new_params: Dict):
        self.params = new_params.copy()

    def update_latency(self, latency_ms: float):
        """Manually override latency estimate."""
        self._manual_latency_ms = float(latency_ms)

    def record(self, state: np.ndarray, action: Optional[np.ndarray] = None):
        """Push current state and last action into history. Call every step."""
        self._history.append((
            time.perf_counter(),
            state.copy(),
            action.copy() if action is not None else None,
        ))

    def compensate(self, current_state: np.ndarray,
                   last_action: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return latency-compensated state for MPC planning.
        Returns current_state unchanged if latency < 0.5 steps or buffer empty.
        """
        latency_ms = self._manual_latency_ms or self.estimator.latency_ms
        L_steps    = latency_ms / 1000.0 * self.control_hz

        if L_steps < 0.5 or len(self._history) < 2:
            return current_state

        L_int = min(int(round(L_steps)), len(self._history) - 1)
        if L_int == 0:
            return current_state

        hist_list = list(self._history)
        idx       = max(0, len(hist_list) - 1 - L_int)
        _, hist_state, _ = hist_list[idx]

        x = hist_state.copy()
        for i in range(idx, len(hist_list)):
            _, _, hist_action = hist_list[i]
            u = hist_action if hist_action is not None else \
                (last_action if last_action is not None else np.zeros(1))
            k1 = self.dynamics_fn(x, u, self.params)
            k2 = self.dynamics_fn(x + self.dt*k1/2, u, self.params)
            k3 = self.dynamics_fn(x + self.dt*k2/2, u, self.params)
            k4 = self.dynamics_fn(x + self.dt*k3,   u, self.params)
            x  = x + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
            if len(x) == 13:
                q = x[6:10]
                n = np.linalg.norm(q)
                if n > 1e-10:
                    x[6:10] = q / n

        self._total_compensations += 1
        self._compensation_steps   = L_int
        return x

    def ping_hardware(self) -> float:
        return self.estimator.send_ping()

    def pong_hardware(self, t_sent: float):
        self.estimator.record_pong(t_sent)

    @property
    def status(self) -> dict:
        return {
            "latency_ms":          round(self._manual_latency_ms or self.estimator.latency_ms, 2),
            "compensation_steps":  self._compensation_steps,
            "total_compensations": self._total_compensations,
            "buffer_size":         len(self._history),
            "estimator":           self.estimator.to_dict(),
        }

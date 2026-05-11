"""
Energy Monitor Plugin — tracks power consumption and cumulative energy.

Power model:
  P(t) = ||action(t)||² * resistance_factor
  E(t) = ∫ P dt  (trapezoidal)
  efficiency = useful_work / E  (simplified as velocity_proxy / power ratio)
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import List

import numpy as np

from physicore.extensions import PhysiCoreExtension, ExtensionMeta


_RESISTANCE = 0.5          # Ω — nominal motor resistance factor
_MAX_ENERGY  = 5000.0      # J — gauge max for display


class EnergyMonitorPlugin(PhysiCoreExtension):
    meta = ExtensionMeta(
        name        = "Energy Monitor",
        version     = "1.0.0",
        description = "Tracks instantaneous power, cumulative energy, and efficiency.",
        author      = "PhysiCore Examples",
        hooks       = ["post_step"],
    )

    def __init__(self):
        self._power_history: deque = deque(maxlen=200)
        self._total_energy:  float = 0.0
        self._last_power:    float = 0.0
        self._last_time:     float = time.monotonic()
        self._step_count:    int   = 0

    def setup(self, engine) -> None:
        self._power_history.clear()
        self._total_energy = 0.0
        self._last_time    = time.monotonic()
        self._step_count   = 0

    def teardown(self) -> None:
        self._power_history.clear()

    def post_step(self, step, engine) -> None:
        self._step_count += 1
        now = time.monotonic()
        dt  = max(now - self._last_time, 1e-6)
        self._last_time = now

        action = np.asarray(getattr(step, "action", [0.0]))
        power  = float(np.dot(action, action)) * _RESISTANCE

        # Trapezoidal integration
        self._total_energy += 0.5 * (power + self._last_power) * dt
        self._last_power    = power

        state     = np.asarray(getattr(step, "state", [0.0]))
        vel_proxy = float(np.linalg.norm(state[:max(len(state)//2, 1)]))

        self._power_history.append({
            "time":       round(now, 4),
            "power":      round(power, 4),
            "energy":     round(self._total_energy, 4),
            "vel_proxy":  round(vel_proxy, 4),
        })

    def _efficiency_score(self) -> float:
        if not self._power_history or self._total_energy < 1e-6:
            return 0.0
        recent = list(self._power_history)[-20:]
        avg_power = sum(r["power"] for r in recent) / max(len(recent), 1)
        avg_vel   = sum(r["vel_proxy"] for r in recent) / max(len(recent), 1)
        if avg_power < 1e-6:
            return 0.0
        raw = min(avg_vel / (avg_power + 1e-6), 1.0)
        return round(raw, 4)

    def get_panel_data(self, panel_id: str):
        if panel_id == "power_timeline":
            points = list(self._power_history)[-50:]
            return {
                "series": [
                    {
                        "name":   "power_W",
                        "color":  "#f59e0b",
                        "points": [{"time": p["time"], "value": p["power"]} for p in points],
                    }
                ]
            }

        if panel_id == "energy_gauge":
            pct = min(self._total_energy / _MAX_ENERGY, 1.0)
            return {
                "value":   round(self._total_energy, 2),
                "max":     _MAX_ENERGY,
                "percent": round(pct * 100.0, 1),
                "unit":    "J",
                "color":   "#06b6d4" if pct < 0.6 else ("#f59e0b" if pct < 0.85 else "#ef4444"),
            }

        if panel_id == "efficiency_value":
            eff = self._efficiency_score()
            return {
                "value": eff,
                "unit":  "",
                "label": "efficiency",
                "color": "#22c55e" if eff > 0.6 else ("#f59e0b" if eff > 0.3 else "#ef4444"),
            }

        return {}


main = EnergyMonitorPlugin

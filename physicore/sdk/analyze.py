"""
PhysiCore Analyzer
==================
Post-flight analysis tools.
Given a flight log, compute what PhysiCore estimates and where the gap was.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from physicore import PhysiCore, PLATFORM_DYNAMICS


@dataclass
class FlightAnalysis:
    residuals:      List[float]
    param_drift:    List[Dict[str, float]]
    uncertainty:    List[float]
    gap_closed_pct: float
    dominant_gap:   str
    recommendations: List[str]

    def summary(self) -> str:
        lines = [
            "PhysiCore Flight Analysis",
            f"  Gap closed:      {self.gap_closed_pct:.1f}%",
            f"  Dominant gap:    {self.dominant_gap}",
            f"  Recommendations:",
        ]
        for r in self.recommendations:
            lines.append(f"    - {r}")
        return "\n".join(lines)


class PhysicoreAnalyzer:
    """
    Analyze flight logs to understand the sim-to-real gap.

    Usage:
        analyzer = PhysicoreAnalyzer("quadrotor")
        analysis = analyzer.analyze(
            states=real_flight_states,
            actions=commanded_actions,
            initial_params={"mass": 1.5, "friction": 0.1}
        )
        print(analysis.summary())
    """

    def __init__(self, platform: str):
        if platform not in PLATFORM_DYNAMICS:
            raise ValueError(f"Unknown platform '{platform}'")
        self.platform    = platform
        dynamics_fn, state_dim, action_dim = PLATFORM_DYNAMICS[platform]
        self.dynamics_fn = dynamics_fn
        self.state_dim   = state_dim
        self.action_dim  = action_dim

    def analyze(
        self,
        states:         np.ndarray,
        actions:        np.ndarray,
        initial_params: Dict[str, float],
    ) -> FlightAnalysis:
        """
        Replay a flight log through PhysiCore and compute gap metrics.

        Args:
            states:         (N+1, state_dim) real flight states
            actions:        (N, action_dim)  commanded actions
            initial_params: Starting parameter estimates

        Returns:
            FlightAnalysis with residuals, param drift, and recommendations
        """
        engine = PhysiCore.for_platform(
            platform=self.platform,
            initial_params=initial_params,
        )

        residuals, uncertainties, param_history = [], [], []
        n = min(len(states) - 1, len(actions))

        for i in range(n):
            state      = states[i]
            action     = actions[i]
            next_state = states[i + 1]

            step = engine.step(state, state)
            engine.observe(state, action, next_state)

            d = engine.diagnostics_full
            residuals.append(d["residual_norm"])
            uncertainties.append(d["uncertainty"])
            param_history.append(d["params"].copy())

        gap_init    = float(np.mean(residuals[:max(1, n//5)]))
        gap_final   = float(np.mean(residuals[max(0, n - n//5):]))
        gap_closed  = (gap_init - gap_final) / max(gap_init, 1e-9) * 100

        param_changes = {}
        if param_history:
            for key in param_history[0]:
                init_val  = param_history[0][key]
                final_val = param_history[-1][key]
                param_changes[key] = abs(final_val - init_val) / max(abs(init_val), 1e-9)

        dominant = max(param_changes, key=param_changes.get) if param_changes else "unknown"

        recommendations = self._generate_recommendations(
            param_changes, gap_closed, float(np.mean(uncertainties))
        )

        return FlightAnalysis(
            residuals=residuals,
            param_drift=param_history,
            uncertainty=uncertainties,
            gap_closed_pct=max(0.0, gap_closed),
            dominant_gap=dominant,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        param_changes: Dict[str, float],
        gap_closed_pct: float,
        mean_uncertainty: float,
    ) -> List[str]:
        recs = []
        if gap_closed_pct < 20:
            recs.append("Gap closed less than 20% — consider more flight data for system ID to converge")
        if param_changes.get("mass", 0) > 0.15:
            recs.append("Mass estimate drifted >15% — check payload attachment or fuel consumption model")
        if param_changes.get("friction", 0) > 0.20:
            recs.append("Friction drifted >20% — terrain/surface variation detected, consider terrain-adaptive mode")
        if mean_uncertainty > 0.05:
            recs.append("High epistemic uncertainty — model is operating outside training distribution. Add more diverse flight data.")
        if not recs:
            recs.append("PhysiCore adaptation nominal — model converging well to real hardware")
        return recs

"""
Terrain Classifier Plugin — classifies terrain from state dynamics.

Heuristic classifier based on:
  - high-frequency velocity variance → rough terrain
  - low velocity + high control effort → soft/sand terrain
  - low effort + smooth motion → pavement
  - large oscillation in state → gravel/rocky
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from physicore.extensions import PhysiCoreExtension, ExtensionMeta


TERRAIN_LABELS = ["pavement", "gravel", "sand", "mud", "grass"]


class TerrainClassifierPlugin(PhysiCoreExtension):
    meta = ExtensionMeta(
        name        = "Terrain Classifier",
        version     = "1.0.0",
        description = "Classifies terrain type from state dynamics.",
        author      = "PhysiCore Examples",
        hooks       = ["post_step"],
    )

    def __init__(self):
        self._vel_buffer:    deque = deque(maxlen=30)
        self._effort_buffer: deque = deque(maxlen=30)
        self._confidences:   list  = [0.2, 0.2, 0.2, 0.2, 0.2]
        self._label:         str   = "unknown"
        self._step_count:    int   = 0

    def setup(self, engine) -> None:
        self._vel_buffer.clear()
        self._effort_buffer.clear()
        self._step_count = 0

    def teardown(self) -> None:
        self._vel_buffer.clear()
        self._effort_buffer.clear()

    def post_step(self, step, engine) -> None:
        self._step_count += 1

        # Extract velocity proxy: L2 norm of action as effort proxy
        action = np.asarray(getattr(step, "action", [0.0]))
        effort = float(np.linalg.norm(action))

        state = np.asarray(getattr(step, "state", [0.0]))
        vel_proxy = float(np.linalg.norm(state)) if len(state) > 0 else 0.0

        self._vel_buffer.append(vel_proxy)
        self._effort_buffer.append(effort)

        if len(self._vel_buffer) < 5:
            return

        vel_arr    = np.array(self._vel_buffer)
        effort_arr = np.array(self._effort_buffer)

        vel_var    = float(np.var(vel_arr))
        mean_eff   = float(np.mean(effort_arr))
        mean_vel   = float(np.mean(np.abs(vel_arr)))

        # Heuristic scoring (higher = more likely)
        scores = np.zeros(5)
        # pavement: low variance, low effort, moderate speed
        scores[0] = 1.0 / (1.0 + vel_var * 10.0 + mean_eff * 0.5)
        # gravel: moderate variance
        scores[1] = math.exp(-abs(vel_var - 0.1) * 20.0)
        # sand: high effort, low speed
        scores[2] = mean_eff * 0.3 / (1.0 + mean_vel * 2.0)
        # mud: high effort + high variance
        scores[3] = (mean_eff * 0.2 + vel_var * 5.0) / 3.0
        # grass: low effort, moderate variance
        scores[4] = 1.0 / (1.0 + mean_eff * 2.0 + abs(vel_var - 0.05) * 10.0)

        # Softmax
        scores = np.clip(scores, 0.0, None)
        total  = scores.sum()
        if total > 0:
            scores /= total
        else:
            scores = np.ones(5) / 5.0

        # EMA smoothing
        alpha = 0.3
        self._confidences = [
            alpha * float(scores[i]) + (1.0 - alpha) * self._confidences[i]
            for i in range(5)
        ]

        best_idx    = int(np.argmax(self._confidences))
        self._label = TERRAIN_LABELS[best_idx]

    def get_panel_data(self, panel_id: str):
        if panel_id == "terrain_confidence":
            return {
                "labels": TERRAIN_LABELS,
                "values": [round(c, 4) for c in self._confidences],
                "colors": ["#06b6d4", "#f59e0b", "#84cc16", "#8b5cf6", "#ec4899"],
            }
        if panel_id == "terrain_label":
            best_idx = int(np.argmax(self._confidences))
            return {
                "value":      self._label,
                "confidence": round(self._confidences[best_idx], 3),
                "unit":       "",
            }
        return {}


main = TerrainClassifierPlugin

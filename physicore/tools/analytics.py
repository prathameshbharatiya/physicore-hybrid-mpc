"""
physicore/tools/analytics.py — Session comparison and fleet analytics

Builds on physicore/data/telemetry_store.py for time-series data and
physicore/core/registry.py for session records.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from physicore.data.telemetry_store import TelemetryStore


# ── ComparisonReport ──────────────────────────────────────────────────────────

@dataclass
class ComparisonReport:
    winner:                  str
    session_a:               str
    session_b:               str
    residual_improvement_pct: float
    convergence_speedup:     float
    param_deltas:            Dict[str, float]
    summary_text:            str

    def to_dict(self) -> dict:
        return {
            "winner":                   self.winner,
            "session_a":                self.session_a,
            "session_b":                self.session_b,
            "residual_improvement_pct": self.residual_improvement_pct,
            "convergence_speedup":      self.convergence_speedup,
            "param_deltas":             self.param_deltas,
            "summary_text":             self.summary_text,
        }


@dataclass
class FleetAnalyticsReport:
    robot_ids:            List[str]
    total_sessions:       int
    mean_convergence_pct: float
    best_robot:           str
    worst_robot:          str
    per_robot:            Dict[str, dict]
    summary_text:         str

    def to_dict(self) -> dict:
        return {
            "robot_ids":            self.robot_ids,
            "total_sessions":       self.total_sessions,
            "mean_convergence_pct": self.mean_convergence_pct,
            "best_robot":           self.best_robot,
            "worst_robot":          self.worst_robot,
            "per_robot":            self.per_robot,
            "summary_text":         self.summary_text,
        }


# ── SessionAnalytics ──────────────────────────────────────────────────────────

class SessionAnalytics:
    """
    Compute analytical metrics over session telemetry.

    Requires a TelemetryStore and (optionally) the ModelRegistry for
    accessing session metadata and prior distributions.
    """

    def __init__(
        self,
        store:            TelemetryStore,
        registry_root:    Optional[Path] = None,
    ):
        self._store = store
        self._registry_root = registry_root

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_series(self, session_id: str, key: str) -> np.ndarray:
        data = self._store.query_session(session_id, keys=[key])
        pts  = data.get(key, [])
        return np.array([p["value"] for p in pts], dtype=float)

    def _get_registry_session(self, session_id: str) -> Optional[dict]:
        """Load a session record from the registry JSONL (by session_id)."""
        if self._registry_root is None:
            try:
                from physicore.core.registry import _REGISTRY_ROOT
                self._registry_root = _REGISTRY_ROOT
            except ImportError:
                return None

        root = Path(self._registry_root)
        for platform_dir in root.iterdir():
            if not platform_dir.is_dir():
                continue
            sessions_file = platform_dir / "sessions.jsonl"
            if not sessions_file.exists():
                continue
            for line in sessions_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("session_id") == session_id:
                        return rec
                except json.JSONDecodeError:
                    pass
        return None

    # ── convergence_rate ──────────────────────────────────────────────────────

    def convergence_rate(self, session_id: str) -> float:
        """
        Fit an exponential decay y = A * exp(-t/tau) + C to the residual series.
        Returns tau (the time constant). Smaller tau = faster convergence.
        Falls back to a simple linear estimate if optimisation fails.
        """
        vals = self._get_series(session_id, "residual")
        if len(vals) < 4:
            # Fallback: use registry convergence_pct
            rec = self._get_registry_session(session_id)
            if rec:
                pct = rec.get("convergence_pct", 0.0)
                return max(1.0, 100.0 / (pct + 1e-6))
            return 0.0

        vals = np.clip(vals, 0, None)
        n    = len(vals)
        xs   = np.arange(n, dtype=float)

        # Simple exponential fit via log-linear regression
        eps     = 1e-9
        log_v   = np.log(vals + eps)
        A_mat   = np.column_stack([xs, np.ones_like(xs)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_v, rcond=None)
            slope  = coeffs[0]
            if slope < -eps:
                tau = -1.0 / slope
            else:
                tau = float(n)
        except Exception:
            tau = float(n)

        return max(float(tau), 0.1)

    # ── parameter_drift ───────────────────────────────────────────────────────

    def parameter_drift(self, session_id: str) -> Dict[str, float]:
        """
        Per-parameter drift from initial (first 5%) to final (last 5%) value.
        Returns {param_name: delta} for each parameter tracked in the session.
        """
        data = self._store.query_session(session_id)
        result: Dict[str, float] = {}

        for key, pts in data.items():
            if not key.startswith("param_"):
                continue
            vals = [p["value"] for p in pts]
            if len(vals) < 2:
                continue
            n_slice  = max(1, len(vals) // 20)
            initial  = float(np.mean(vals[:n_slice]))
            final    = float(np.mean(vals[-n_slice:]))
            result[key[6:]] = round(final - initial, 6)  # strip "param_" prefix

        # If no param_ keys, try to get from registry
        if not result:
            rec = self._get_registry_session(session_id)
            if rec:
                fp = rec.get("final_params", {})
                for k in fp:
                    result[k] = 0.0  # no drift info available

        return result

    # ── compare ───────────────────────────────────────────────────────────────

    def compare(self, session_id_a: str, session_id_b: str) -> ComparisonReport:
        """Statistical comparison of two sessions."""
        res_a = self._get_series(session_id_a, "residual")
        res_b = self._get_series(session_id_b, "residual")

        def _final_residual(vals: np.ndarray) -> float:
            if len(vals) == 0:
                return float("inf")
            n = max(1, len(vals) // 10)
            return float(np.mean(vals[-n:]))

        def _initial_residual(vals: np.ndarray) -> float:
            if len(vals) == 0:
                return 1.0
            n = max(1, len(vals) // 10)
            return float(np.mean(vals[:n]))

        # Fallback to registry if no telemetry
        if len(res_a) == 0:
            rec = self._get_registry_session(session_id_a)
            final_a   = rec.get("final_params", {}).get("residual", 1.0) if rec else 1.0
            conv_a    = rec.get("convergence_pct", 0.0) if rec else 0.0
            init_a    = 1.0
        else:
            init_a    = _initial_residual(res_a)
            final_a   = _final_residual(res_a)
            conv_a    = max(0.0, (init_a - final_a) / max(init_a, 1e-9) * 100)

        if len(res_b) == 0:
            rec = self._get_registry_session(session_id_b)
            final_b   = rec.get("final_params", {}).get("residual", 1.0) if rec else 1.0
            conv_b    = rec.get("convergence_pct", 0.0) if rec else 0.0
            init_b    = 1.0
        else:
            init_b    = _initial_residual(res_b)
            final_b   = _final_residual(res_b)
            conv_b    = max(0.0, (init_b - final_b) / max(init_b, 1e-9) * 100)

        winner              = session_id_a if final_a <= final_b else session_id_b
        improvement_pct     = (final_b - final_a) / max(abs(final_b), 1e-9) * 100
        convergence_speedup = (conv_a - conv_b) / max(conv_b, 1e-9) if conv_b > 0 else 0.0

        # Param deltas from registry
        rec_a = self._get_registry_session(session_id_a) or {}
        rec_b = self._get_registry_session(session_id_b) or {}
        pa = rec_a.get("final_params", {})
        pb = rec_b.get("final_params", {})
        param_deltas = {
            k: round(pb.get(k, 0.0) - pa.get(k, 0.0), 6)
            for k in set(list(pa.keys()) + list(pb.keys()))
        }

        if winner == session_id_a:
            summary = (
                f"Session {session_id_a[:8]} outperformed {session_id_b[:8]}: "
                f"final residual {final_a:.4f} vs {final_b:.4f} "
                f"({abs(improvement_pct):.1f}% improvement). "
                f"Convergence: {conv_a:.1f}% vs {conv_b:.1f}%."
            )
        else:
            summary = (
                f"Session {session_id_b[:8]} outperformed {session_id_a[:8]}: "
                f"final residual {final_b:.4f} vs {final_a:.4f} "
                f"({abs(improvement_pct):.1f}% improvement). "
                f"Convergence: {conv_b:.1f}% vs {conv_a:.1f}%."
            )

        return ComparisonReport(
            winner                   = winner,
            session_a                = session_id_a,
            session_b                = session_id_b,
            residual_improvement_pct = round(improvement_pct, 2),
            convergence_speedup      = round(convergence_speedup, 4),
            param_deltas             = param_deltas,
            summary_text             = summary,
        )

    # ── fleet_summary ─────────────────────────────────────────────────────────

    def fleet_summary(self, robot_ids: List[str]) -> FleetAnalyticsReport:
        """Aggregate analytics across a fleet of robots."""
        per_robot: Dict[str, dict] = {}
        all_conv:  List[float]     = []

        for rid in robot_ids:
            sessions = self._store.sessions(robot_id=rid)
            if not sessions:
                per_robot[rid] = {
                    "session_count":    0,
                    "mean_convergence": 0.0,
                    "last_step_count":  0,
                }
                all_conv.append(0.0)
                continue

            conv_vals: List[float] = []
            for s in sessions:
                # Try registry
                rec = self._get_registry_session(s.session_id)
                if rec:
                    conv_vals.append(rec.get("convergence_pct", 0.0))

            mean_conv = float(np.mean(conv_vals)) if conv_vals else 0.0
            all_conv.append(mean_conv)
            per_robot[rid] = {
                "session_count":    len(sessions),
                "mean_convergence": round(mean_conv, 2),
                "last_step_count":  sessions[0].step_count if sessions else 0,
            }

        mean_global = float(np.mean(all_conv)) if all_conv else 0.0
        best_robot  = max(per_robot, key=lambda r: per_robot[r]["mean_convergence"],
                          default="none")
        worst_robot = min(per_robot, key=lambda r: per_robot[r]["mean_convergence"],
                          default="none")

        summary = (
            f"Fleet of {len(robot_ids)} robot(s). "
            f"Mean convergence: {mean_global:.1f}%. "
            f"Best: {best_robot} ({per_robot.get(best_robot, {}).get('mean_convergence', 0):.1f}%), "
            f"Worst: {worst_robot} ({per_robot.get(worst_robot, {}).get('mean_convergence', 0):.1f}%)."
        )

        return FleetAnalyticsReport(
            robot_ids            = robot_ids,
            total_sessions       = sum(v["session_count"] for v in per_robot.values()),
            mean_convergence_pct = round(mean_global, 2),
            best_robot           = best_robot,
            worst_robot          = worst_robot,
            per_robot            = per_robot,
            summary_text         = summary,
        )

    # ── anomaly_score ─────────────────────────────────────────────────────────

    def anomaly_score(self, session_id: str) -> float:
        """
        Mahalanobis distance of this session's final params from the platform prior.
        Higher score = more anomalous. Score in [0, ∞).
        """
        rec = self._get_registry_session(session_id)
        if rec is None:
            return 0.0

        platform = rec.get("platform", "")
        fp       = rec.get("final_params", {})
        if not fp or not platform:
            return 0.0

        try:
            from physicore.core.transfer import PlatformPrior
            from physicore.core.registry import _REGISTRY_ROOT
            prior_file = _REGISTRY_ROOT / platform / "transfer_prior.json"
            prior      = PlatformPrior.load(platform, prior_file)

            if prior.n_sessions < 2:
                return 0.0

            mu   = prior.map_estimate()
            unc  = prior.uncertainty()
            keys = [k for k in fp if k in mu]

            if not keys:
                return 0.0

            diffs  = np.array([(fp[k] - mu[k]) / max(unc.get(k, 1.0), 1e-9)
                                for k in keys])
            dist   = float(np.sqrt(np.dot(diffs, diffs)))
            return round(dist, 4)

        except Exception:
            return 0.0

"""
PhysiCore Analyzer
==================
Post-flight analysis tools.

Two entry points:
  1. PhysicoreAnalyzer.analyze()     — replay a flight log through PhysiCore
  2. PhysicoreAnalyzer.from_insight_log() — load Insight 2.0 / CSV log files directly

Supports:
  - Insight 2.0 format (Ti, A_X, A_Y, A_Z, R_X, R_Y, R_Z, H, Vin, Temp, Pressure)
  - Generic CSV with headers
  - PhysiCore session JSON logs
  - Any tab/comma delimited file with altitude + acceleration columns
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from physicore import PhysiCore, PLATFORM_DYNAMICS


@dataclass
class FlightAnalysis:
    residuals:       List[float]
    param_drift:     List[Dict[str, float]]
    uncertainty:     List[float]
    gap_closed_pct:  float
    dominant_gap:    str
    recommendations: List[str]
    # Extra: raw flight stats
    apogee_m:        float = 0.0
    max_accel_g:     float = 0.0
    descent_rate_mps: float = 0.0
    flight_duration_s: float = 0.0
    imu_saturated:   bool  = False
    estimated_cd:    float = 0.0

    def summary(self) -> str:
        lines = [
            "PhysiCore Flight Analysis",
            f"  Apogee:          {self.apogee_m:.1f} m AGL",
            f"  Max accel:       {self.max_accel_g:.1f} g {'(IMU SATURATED — real peak was higher)' if self.imu_saturated else ''}",
            f"  Descent rate:    {self.descent_rate_mps:.1f} m/s",
            f"  Flight duration: {self.flight_duration_s:.1f} s",
            f"  Estimated Cd:    {self.estimated_cd:.3f}",
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

    Usage (from code):
        analyzer = PhysicoreAnalyzer("rocket")
        analysis = analyzer.analyze(states, actions, initial_params)
        print(analysis.summary())

    Usage (from log file):
        analyzer = PhysicoreAnalyzer("rocket")
        analysis = analyzer.from_log_file("flight_data.txt", initial_mass_kg=0.5)
        print(analysis.summary())
    """

    def __init__(self, platform: str):
        if platform not in PLATFORM_DYNAMICS:
            raise ValueError(f"Unknown platform '{platform}'. Available: {sorted(PLATFORM_DYNAMICS.keys())}")
        self.platform    = platform
        fn, state_dim, action_dim = PLATFORM_DYNAMICS[platform]
        self.dynamics_fn  = fn
        self.state_dim    = state_dim
        self.action_dim   = action_dim

    # ── Main analysis entry point ──────────────────────────────────────────────

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

        gap_init   = float(np.mean(residuals[:max(1, n // 5)]))
        gap_final  = float(np.mean(residuals[max(0, n - n // 5):]))
        gap_closed = (gap_init - gap_final) / max(gap_init, 1e-9) * 100

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

    # ── Log file ingestion ─────────────────────────────────────────────────────

    @classmethod
    def from_log_file(
        cls,
        filepath: Union[str, Path],
        platform: str = "rocket",
        initial_mass_kg: float = 0.5,
        dry_mass_kg:     float = 0.3,
        imu_max_g:       float = 16.0,
    ) -> "FlightAnalysis":
        """
        Load a flight log file and run analysis directly.

        Supports:
          - Insight 2.0 format (.txt): Ti, A_X, A_Y, A_Z, R_X, R_Y, R_Z, H, Vin, Temp, Pressure
          - Generic CSV with column headers
          - PhysiCore session JSON

        Args:
            filepath:        Path to log file
            platform:        PhysiCore platform name (default: "rocket")
            initial_mass_kg: Total mass at launch (kg)
            dry_mass_kg:     Dry mass after propellant burnt (kg)
            imu_max_g:       IMU range for saturation detection (default 16g for MPU-6050)

        Returns:
            FlightAnalysis with full stats
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {filepath}")

        ext = path.suffix.lower()

        if ext == ".json":
            rows = cls._load_json_log(path)
        else:
            rows = cls._load_csv_log(path)

        return cls._analyze_rows(rows, platform, initial_mass_kg, dry_mass_kg, imu_max_g)

    @staticmethod
    def _load_csv_log(path: Path) -> List[Dict]:
        """
        Load CSV/TXT log. Auto-detects:
          - Insight 2.0: Ti, A_X, A_Y, A_Z, R_X, R_Y, R_Z, H, Vin, Temp, Pressure
          - Generic: any CSV with recognizable column names
        Timestamps in milliseconds are auto-detected and converted to seconds.
        """
        rows = []
        with open(path, encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            raw_headers = next(reader)
            headers = [h.strip().rstrip(":").lower() for h in raw_headers]

        # Detect Insight 2.0 format
        insight_map = {
            "ti": "timestamp_ms", "a_x": "accel_x", "a_y": "accel_y", "a_z": "accel_z",
            "r_x": "gyro_x", "r_y": "gyro_y", "r_z": "gyro_z",
            "h": "altitude", "vin": "battery", "temp": "temperature", "pressure": "pressure",
        }
        # Generic column name mapping
        generic_map = {
            "ax": "accel_x", "ay": "accel_y", "az": "accel_z",
            "gx": "gyro_x",  "gy": "gyro_y",  "gz": "gyro_z",
            "alt": "altitude", "h": "altitude", "height": "altitude",
            "t": "timestamp_ms", "time": "timestamp_ms", "timestamp": "timestamp_ms",
            "temp": "temperature", "pres": "pressure", "press": "pressure",
        }
        combined_map = {**insight_map, **generic_map}
        mapped_headers = [combined_map.get(h, h) for h in headers]

        with open(path, encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                try:
                    vals = [float(x) for x in row if x.strip()]
                    if len(vals) < 4:
                        continue
                    d = {mapped_headers[i]: vals[i] for i in range(min(len(mapped_headers), len(vals)))}
                    rows.append(d)
                except (ValueError, IndexError):
                    continue

        if not rows:
            raise ValueError(f"No numeric data found in {path}")

        # Detect timestamp units
        if "timestamp_ms" in rows[0]:
            t0 = rows[0]["timestamp_ms"]
            total_span = rows[-1]["timestamp_ms"] - t0
            # Insight 2.0: timestamps in ms. If span > 10000, it's ms (10+ seconds)
            if total_span > 10000:
                for r in rows:
                    r["timestamp_s"] = (r["timestamp_ms"] - t0) / 1000.0
            else:
                for r in rows:
                    r["timestamp_s"] = r["timestamp_ms"] - t0

        return rows

    @staticmethod
    def _load_json_log(path: Path) -> List[Dict]:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "steps" in data:
            return data["steps"]
        raise ValueError("JSON log must be a list or dict with 'steps' key")

    @classmethod
    def _analyze_rows(
        cls,
        rows:            List[Dict],
        platform:        str,
        initial_mass_kg: float,
        dry_mass_kg:     float,
        imu_max_g:       float,
    ) -> FlightAnalysis:
        """Core analysis from parsed rows."""
        from physicore.core.sensor_filter import AltitudeKalmanFilter, IMUSaturationDetector

        if not rows:
            raise ValueError("No data rows to analyze")

        # ── Extract time axis ──────────────────────────────────────────────────
        if "timestamp_s" in rows[0]:
            times = [r["timestamp_s"] for r in rows]
        else:
            times = list(range(len(rows)))
        duration = times[-1] - times[0] if len(times) > 1 else 0.0
        sample_hz = len(rows) / max(duration, 1.0)

        # ── Extract sensor columns ─────────────────────────────────────────────
        alt_raw  = np.array([r.get("altitude", 0.0) for r in rows])
        ax_raw   = np.array([r.get("accel_x", 0.0)  for r in rows])
        ay_raw   = np.array([r.get("accel_y", 0.0)  for r in rows])
        az_raw   = np.array([r.get("accel_z", 9.81) for r in rows])
        pressure = np.array([r.get("pressure", None) for r in rows])
        temp_c   = np.array([r.get("temperature", 20.0) for r in rows])

        # ── Saturation detection ───────────────────────────────────────────────
        sat_detector = IMUSaturationDetector(max_g=imu_max_g)
        imu_saturated = False
        az_clean = az_raw.copy()
        for i in range(len(rows)):
            is_sat, _, _, az_c = sat_detector.check(ax_raw[i], ay_raw[i], az_raw[i])
            az_clean[i] = az_c
            if is_sat:
                imu_saturated = True

        # ── Altitude Kalman filter ─────────────────────────────────────────────
        kf = AltitudeKalmanFilter()
        alt_filtered = np.zeros(len(rows))
        vz_filtered  = np.zeros(len(rows))
        for i in range(len(rows)):
            dt = (times[i] - times[i-1]) if i > 0 else 1.0/sample_hz
            vert_accel = az_clean[i] - 9.81  # net vertical accel
            alt_filtered[i], vz_filtered[i] = kf.update(alt_raw[i], vert_accel, dt)

        # ── Ground reference ───────────────────────────────────────────────────
        n_ground = max(1, int(sample_hz * 2))   # first 2 seconds = ground
        ground_alt = float(np.mean(alt_filtered[:n_ground]))
        alt_agl    = alt_filtered - ground_alt

        # ── Key stats ──────────────────────────────────────────────────────────
        apogee_m      = float(np.max(alt_agl))
        apogee_idx    = int(np.argmax(alt_agl))
        accel_mag_g   = np.sqrt(ax_raw**2 + ay_raw**2 + az_raw**2) / 9.81
        max_accel_g   = float(np.max(accel_mag_g))

        # Descent rate (stable window after apogee)
        post_apogee   = vz_filtered[apogee_idx:]
        stable_window = post_apogee[len(post_apogee)//3:]  # last 2/3 of descent
        descent_rate  = float(abs(np.mean(stable_window))) if len(stable_window) > 10 else 0.0

        # ── Cd estimation from coast phase ─────────────────────────────────────
        # During coast, drag = -m * net_accel (gravity subtracted)
        # Simplified: Cd ≈ 2 * m * deceleration / (rho * v² * A)
        # We use a rough estimate from the altitude trajectory
        estimated_cd = 0.0
        coast_start = min(apogee_idx, len(rows) - 10)
        coast_section = alt_agl[n_ground:coast_start]
        if len(coast_section) > 10:
            vz_coast = vz_filtered[n_ground:coast_start]
            avg_vz   = float(np.mean(np.abs(vz_coast))) + 0.1
            avg_decel = float(np.mean(np.abs(np.diff(vz_coast)))) * sample_hz
            rho      = 1.225 * math.exp(-apogee_m / 2 / 8500)  # mid-point density
            dia      = 0.08  # assume 80mm diameter (typical 1:9 scale Lambda)
            area     = math.pi * (dia / 2) ** 2
            if avg_vz > 1.0 and area > 0 and rho > 0:
                estimated_cd = 2 * initial_mass_kg * avg_decel / (rho * avg_vz**2 * area + 1e-9)
                estimated_cd = min(max(estimated_cd, 0.1), 3.0)  # clamp to sane range

        # ── Build recommendations ──────────────────────────────────────────────
        recommendations = []
        if imu_saturated:
            recommendations.append(
                f"IMU saturated at ±{imu_max_g}g during powered phase. "
                "Real peak acceleration was higher. "
                "Use a ±100g IMU (ADXL375, ICM-42688-P) to capture the full thrust curve."
            )
        if descent_rate > 15:
            recommendations.append(
                f"Descent rate {descent_rate:.1f} m/s is high — possible parachute failure or partial deployment. "
                "Verify ejection charge and parachute pack."
            )
        elif descent_rate > 0 and descent_rate < 3:
            recommendations.append(
                f"Descent rate {descent_rate:.1f} m/s is very slow — main chute may be oversized."
            )
        if estimated_cd > 0:
            recommendations.append(
                f"Estimated Cd ≈ {estimated_cd:.3f} from coast-phase deceleration. "
                "Compare against your CFD/simulation value to validate aero model."
            )
        if apogee_m > 0:
            recommendations.append(
                f"Apogee {apogee_m:.0f}m AGL. "
                "Feed this session to PhysiCore live on next flight for real-time adaptation."
            )
        if not recommendations:
            recommendations.append("Flight data nominal. No anomalies detected.")

        # ── Minimal PhysiCore replay for gap metrics ───────────────────────────
        # Build simplified state vectors from filtered data
        analyzer = cls(platform)
        residuals, param_drift, uncertainty = [], [], []

        try:
            engine = PhysiCore.for_platform(
                platform=platform,
                initial_params={"mass": initial_mass_kg, "friction": 0.45, "inertia": 220.0},
            )
            mass = initial_mass_kg
            for i in range(min(len(rows) - 1, 500)):  # cap at 500 steps
                dt = (times[i+1] - times[i]) if i+1 < len(times) else 1.0/sample_hz
                vz = float(vz_filtered[i])
                alt = float(alt_agl[i])
                state = np.array([0, alt, 0, vz, mass, 0.0])
                state = state[:engine.cfg.state_dim]
                if len(state) < engine.cfg.state_dim:
                    state = np.pad(state, (0, engine.cfg.state_dim - len(state)))
                x_ref = np.zeros(engine.cfg.state_dim)
                step  = engine.step(state, x_ref)
                d     = engine.diagnostics_full
                residuals.append(d["residual_norm"])
                uncertainty.append(d["uncertainty"])
                param_drift.append(d["params"].copy())
        except Exception:
            residuals  = [0.0]
            uncertainty = [0.0]
            param_drift = [{"mass": initial_mass_kg}]

        gap_init   = float(np.mean(residuals[:max(1, len(residuals)//5)]))
        gap_final  = float(np.mean(residuals[max(0, len(residuals)-len(residuals)//5):]))
        gap_closed = max(0.0, (gap_init - gap_final) / max(gap_init, 1e-9) * 100)

        return FlightAnalysis(
            residuals=residuals,
            param_drift=param_drift,
            uncertainty=uncertainty,
            gap_closed_pct=gap_closed,
            dominant_gap="drag_coefficient" if estimated_cd > 0 else "mass",
            recommendations=recommendations,
            apogee_m=apogee_m,
            max_accel_g=max_accel_g,
            descent_rate_mps=descent_rate,
            flight_duration_s=duration,
            imu_saturated=imu_saturated,
            estimated_cd=estimated_cd,
        )

    def _generate_recommendations(
        self,
        param_changes:    Dict[str, float],
        gap_closed_pct:   float,
        mean_uncertainty: float,
    ) -> List[str]:
        recs = []
        if gap_closed_pct < 20:
            recs.append("Gap closed less than 20% — consider more flight data for SysID to converge")
        if param_changes.get("mass", 0) > 0.15:
            recs.append("Mass estimate drifted >15% — check payload attachment or propellant consumption model")
        if param_changes.get("friction", 0) > 0.20:
            recs.append("Drag drifted >20% — surface or atmospheric variation detected")
        if mean_uncertainty > 0.05:
            recs.append("High epistemic uncertainty — model operating outside training distribution. Add more flight data.")
        if not recs:
            recs.append("PhysiCore adaptation nominal — model converging well to real hardware")
        return recs

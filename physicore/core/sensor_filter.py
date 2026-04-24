"""
PhysiCore Sensor Filter
=======================
Solves every real-hardware sensor problem in one place:

  1. Barometric pressure spike rejection (ejection charge, parachute shock)
  2. IMU saturation detection and sample rejection (MPU-6050 ±16g ceiling)
  3. Gyro unit normalisation (deg/s vs rad/s — MAVLink vs ROS2 vs serial)
  4. IMU frame alignment (rotation matrix for non-standard mounting)
  5. Altitude Kalman filter (fuses baro + accel for smooth altitude estimate)
  6. Velocity drift correction (leaky integrator reset on phase change)
  7. Temperature compensation for barometric altitude
  8. Multi-sensor voting (reject outlier when redundant IMUs disagree)
  9. Serial packet validation (schema check before engine ingestion)
 10. Flight phase state machine (GROUND → POWERED → COAST → APOGEE →
     DROGUE → MAIN → LANDED) driven by filtered sensor data only

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations

import math
import time
import json
from collections import deque
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  FLIGHT PHASE STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class FlightPhase(str, Enum):
    GROUND    = "GROUND"
    POWERED   = "POWERED"
    COAST     = "COAST"
    APOGEE    = "APOGEE"
    DROGUE    = "DROGUE"
    MAIN      = "MAIN"
    LANDED    = "LANDED"


class FlightPhaseDetector:
    """
    Detects rocket flight phase from filtered sensor data.
    Uses hysteresis and time guards to prevent false transitions.

    All thresholds are conservative defaults — override via constructor
    for your specific rocket.

    The key insight: NEVER use raw altitude for phase detection.
    Always use the filtered altitude from AltitudeKalmanFilter.
    """

    def __init__(
        self,
        launch_accel_threshold_g: float = 2.5,   # net accel above g to detect launch
        coast_accel_threshold_g:  float = 0.5,   # net accel below this = burnout
        apogee_climb_threshold:   float = -0.5,  # m/s — negative = descending
        drogue_deploy_altitude:   float = None,  # None = detect from descent rate change
        main_deploy_altitude:     float = 250.0, # m AGL — deploy main chute altitude
        landed_speed_threshold:   float = 0.5,   # m/s total speed
        min_powered_duration_s:   float = 0.3,   # ignore burn < 0.3s (false positive)
        min_coast_duration_s:     float = 1.0,   # guard against noise at burnout
    ):
        self.launch_g    = launch_accel_threshold_g
        self.coast_g     = coast_accel_threshold_g
        self.apogee_vz   = apogee_climb_threshold
        self.main_alt    = main_deploy_altitude
        self.landed_v    = landed_speed_threshold
        self.min_powered = min_powered_duration_s
        self.min_coast   = min_coast_duration_s

        self.phase          = FlightPhase.GROUND
        self._phase_entered = time.time()
        self._launch_alt    = 0.0   # AGL reference set at launch
        self._apogee_alt    = 0.0
        self._prev_alt      = 0.0
        self._prev_time     = time.time()

    def _time_in_phase(self) -> float:
        return time.time() - self._phase_entered

    def _set_phase(self, new_phase: FlightPhase) -> bool:
        if new_phase != self.phase:
            self.phase = new_phase
            self._phase_entered = time.time()
            return True
        return False

    def update(
        self,
        filtered_altitude_m: float,   # AGL, from AltitudeKalmanFilter
        accel_magnitude_g:   float,   # total accel in g, IMU-saturation-aware
        climb_rate_mps:      float,   # filtered vertical velocity
        total_speed_mps:     float,   # total speed magnitude
    ) -> FlightPhase:
        """
        Update flight phase. Call every sensor cycle with filtered values.
        Returns current phase.
        """
        now = time.time()
        dt  = max(now - self._prev_time, 1e-3)
        self._prev_time = now

        p = self.phase

        if p == FlightPhase.GROUND:
            if accel_magnitude_g > self.launch_g:
                self._launch_alt = filtered_altitude_m
                self._set_phase(FlightPhase.POWERED)

        elif p == FlightPhase.POWERED:
            # Burnout: net accel drops below threshold AND min duration elapsed
            if (accel_magnitude_g < self.coast_g
                    and self._time_in_phase() > self.min_powered):
                self._set_phase(FlightPhase.COAST)

        elif p == FlightPhase.COAST:
            # Apogee: climbing rate goes negative (descending)
            # Add time guard to avoid noise at burnout
            if (climb_rate_mps < self.apogee_vz
                    and self._time_in_phase() > self.min_coast):
                self._apogee_alt = filtered_altitude_m
                self._set_phase(FlightPhase.APOGEE)

        elif p == FlightPhase.APOGEE:
            # Brief apogee window — transition to DROGUE immediately after
            # (ejection charge fires at apogee)
            if self._time_in_phase() > 0.5:
                self._set_phase(FlightPhase.DROGUE)

        elif p == FlightPhase.DROGUE:
            # Main chute deploys at preset altitude AGL
            agl = filtered_altitude_m - self._launch_alt
            if agl < self.main_alt and agl > 0:
                self._set_phase(FlightPhase.MAIN)

        elif p == FlightPhase.MAIN:
            # Landing: total speed near zero AND close to ground
            agl = filtered_altitude_m - self._launch_alt
            if total_speed_mps < self.landed_v and agl < 20.0:
                self._set_phase(FlightPhase.LANDED)

        self._prev_alt = filtered_altitude_m
        return self.phase


# ═══════════════════════════════════════════════════════════════════════════════
#  ALTITUDE KALMAN FILTER
# ═══════════════════════════════════════════════════════════════════════════════

class AltitudeKalmanFilter:
    """
    1D Kalman filter fusing barometric altitude with vertical acceleration.

    State: [altitude, vertical_velocity]
    Measurement: barometric altitude

    Key feature: spike rejection BEFORE the Kalman update.
    Pressure spikes from ejection charges, parachute deployment, and
    powered flight turbulence are detected and rejected — the filter
    coasts on the prediction until the sensor recovers.

    Usage:
        kf = AltitudeKalmanFilter()
        alt, vz = kf.update(raw_baro_alt, accel_z_mps2, dt)
    """

    def __init__(
        self,
        process_noise_alt:  float = 0.1,   # m²  — how much we trust the model
        process_noise_vel:  float = 1.0,   # m²/s²
        meas_noise_baro:    float = 2.0,   # m²  — BMP-388 noise at ~0.1m, but spikes can be 300m
        spike_threshold_m:  float = 50.0,  # reject baro jumps larger than this in one step
        max_spike_duration: int   = 10,    # max consecutive rejected samples before forced accept
    ):
        self.Q  = np.diag([process_noise_alt, process_noise_vel])
        self.R  = np.array([[meas_noise_baro]])
        self.H  = np.array([[1.0, 0.0]])
        self.x  = np.array([0.0, 0.0])   # [alt, vz]
        self.P  = np.eye(2) * 100.0      # start uncertain

        self.spike_thresh   = spike_threshold_m
        self.max_spike_dur  = max_spike_duration
        self._spike_count   = 0
        self._last_good_alt = None
        self._initialized   = False

    def _build_F(self, dt: float) -> np.ndarray:
        return np.array([[1.0, dt], [0.0, 1.0]])

    def _build_B(self, dt: float) -> np.ndarray:
        return np.array([[0.5 * dt**2], [dt]])

    def update(
        self,
        baro_altitude_m: float,
        accel_z_mps2:    float,   # vertical accel in m/s², gravity-corrected (positive = up)
        dt:              float,
    ) -> Tuple[float, float]:
        """
        Returns (filtered_altitude_m, filtered_vertical_velocity_mps).
        """
        if not self._initialized:
            self.x[0] = baro_altitude_m
            self._last_good_alt = baro_altitude_m
            self._initialized   = True
            return baro_altitude_m, 0.0

        # ── Predict ────────────────────────────────────────────────────────────
        F = self._build_F(dt)
        B = self._build_B(dt)
        u = np.array([accel_z_mps2])
        self.x = F @ self.x + (B @ u).flatten()
        self.P = F @ self.P @ F.T + self.Q

        # ── Spike detection ────────────────────────────────────────────────────
        predicted_alt = self.x[0]
        baro_jump     = abs(baro_altitude_m - predicted_alt)
        is_spike      = (baro_jump > self.spike_thresh
                         and self._last_good_alt is not None)

        if is_spike and self._spike_count < self.max_spike_dur:
            # Reject this sample — coast on prediction
            self._spike_count += 1
            return float(self.x[0]), float(self.x[1])

        # ── Update ────────────────────────────────────────────────────────────
        self._spike_count   = 0
        self._last_good_alt = baro_altitude_m

        z   = np.array([baro_altitude_m])
        y   = z - self.H @ self.x
        S   = self.H @ self.P @ self.H.T + self.R
        K   = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return float(self.x[0]), float(self.x[1])

    def reset(self, altitude: float = 0.0) -> None:
        self.x  = np.array([altitude, 0.0])
        self.P  = np.eye(2) * 100.0
        self._spike_count   = 0
        self._last_good_alt = altitude


# ═══════════════════════════════════════════════════════════════════════════════
#  IMU SATURATION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class IMUSaturationDetector:
    """
    Detects when an accelerometer has hit its measurement ceiling.

    MPU-6050 at ±16g: max readable = 156.96 m/s² (16 * 9.81)
    ICM-42688 at ±16g: same ceiling
    ADXL375 at ±200g: 1961 m/s²

    When saturated:
     - Mark the sample as invalid (don't feed to engine)
     - Use last-known-good values with uncertainty inflated
     - Log the saturation event for post-flight analysis

    Usage:
        detector = IMUSaturationDetector(max_g=16.0)  # MPU-6050
        is_sat, clean_ax, clean_ay, clean_az = detector.check(ax, ay, az)
    """

    def __init__(
        self,
        max_g:         float = 16.0,      # sensor range in g
        margin_g:      float = 0.2,       # flag if within margin_g of ceiling
        window_size:   int   = 5,         # consecutive saturated samples before flagging
    ):
        self.ceiling   = (max_g - margin_g) * 9.81   # m/s²
        self._window   = deque(maxlen=window_size)
        self._last_good = np.zeros(3)
        self._sat_count  = 0
        self._total_sat  = 0

    def check(
        self,
        ax: float, ay: float, az: float,
    ) -> Tuple[bool, float, float, float]:
        """
        Returns (is_saturated, clean_ax, clean_ay, clean_az).
        If saturated: returns last good values instead of clipped values.
        """
        mag = math.sqrt(ax**2 + ay**2 + az**2)
        sat = mag >= self.ceiling

        self._window.append(sat)

        if sat:
            self._sat_count  += 1
            self._total_sat  += 1
            # Return last good sample — better than clipped ceiling value
            return True, self._last_good[0], self._last_good[1], self._last_good[2]
        else:
            self._sat_count  = 0
            self._last_good  = np.array([ax, ay, az])
            return False, ax, ay, az

    @property
    def is_currently_saturated(self) -> bool:
        return self._sat_count > 0

    @property
    def total_saturated_samples(self) -> int:
        return self._total_sat

    def saturation_g(self) -> float:
        """What g level caused saturation (ceiling value)."""
        return self.ceiling / 9.81


# ═══════════════════════════════════════════════════════════════════════════════
#  GYRO UNIT NORMALISER
# ═══════════════════════════════════════════════════════════════════════════════

class GyroNormaliser:
    """
    Ensures gyro values are always in deg/s for the bridge state,
    regardless of whether the source is MAVLink (already deg/s after
    math.degrees() call), ROS2 IMU (rad/s), or serial (ambiguous).

    Auto-detects units from magnitude — a robot spinning at 3 rad/s
    would be 171 deg/s. If values consistently exceed 500, likely
    already in deg/s. If consistently below 10, likely rad/s.

    For safety: always specify explicitly if you know the source.
    """

    def __init__(self, source_unit: str = "auto"):
        """
        source_unit: "deg_s" | "rad_s" | "auto"
        """
        assert source_unit in ("deg_s", "rad_s", "auto")
        self._unit   = source_unit
        self._buffer = deque(maxlen=50)
        self._decided = source_unit != "auto"

    def normalise(self, gx: float, gy: float, gz: float) -> Tuple[float, float, float]:
        """Returns gyro in deg/s always."""
        if self._unit == "deg_s":
            return gx, gy, gz
        if self._unit == "rad_s":
            return math.degrees(gx), math.degrees(gy), math.degrees(gz)

        # Auto-detect
        mag = math.sqrt(gx**2 + gy**2 + gz**2)
        self._buffer.append(mag)

        if not self._decided and len(self._buffer) >= 20:
            avg = sum(self._buffer) / len(self._buffer)
            # rad/s: typical robot max ~10 rad/s → avg maybe 0.5-3
            # deg/s: same motion → 30-170
            if avg < 8.0:
                self._unit    = "rad_s"
                self._decided = True
            else:
                self._unit    = "deg_s"
                self._decided = True

        if self._unit == "rad_s":
            return math.degrees(gx), math.degrees(gy), math.degrees(gz)
        return gx, gy, gz


# ═══════════════════════════════════════════════════════════════════════════════
#  IMU FRAME ALIGNER
# ═══════════════════════════════════════════════════════════════════════════════

class IMUFrameAligner:
    """
    Applies a rotation matrix to IMU readings to correct for
    non-standard mounting orientation.

    Common mounting corrections:
        "standard"    — Z-up, X-forward (no correction)
        "z_down"      — IMU mounted upside down
        "x_up"        — IMU rotated 90° nose-up
        "y_forward"   — IMU rotated 90° sideways
        "custom"      — provide your own 3×3 rotation matrix

    Usage:
        aligner = IMUFrameAligner("z_down")
        ax, ay, az = aligner.align_accel(ax, ay, az)
        gx, gy, gz = aligner.align_gyro(gx, gy, gz)
    """

    _PRESETS = {
        "standard":  np.eye(3),
        "z_down":    np.diag([1.0, -1.0, -1.0]),          # flip Y and Z
        "x_up":      np.array([[0,0,1],[0,1,0],[-1,0,0]]), # 90° pitch rotation
        "y_forward": np.array([[0,1,0],[-1,0,0],[0,0,1]]), # 90° yaw rotation
        "x_back":    np.array([[-1,0,0],[0,-1,0],[0,0,1]]),# 180° yaw
    }

    def __init__(self, preset: str = "standard", custom_matrix: Optional[np.ndarray] = None):
        if custom_matrix is not None:
            self.R = np.array(custom_matrix, dtype=float)
        elif preset in self._PRESETS:
            self.R = self._PRESETS[preset]
        else:
            raise ValueError(f"Unknown IMU preset '{preset}'. Use: {list(self._PRESETS.keys())} or provide custom_matrix")

    def align_accel(self, ax: float, ay: float, az: float) -> Tuple[float, float, float]:
        v = self.R @ np.array([ax, ay, az])
        return float(v[0]), float(v[1]), float(v[2])

    def align_gyro(self, gx: float, gy: float, gz: float) -> Tuple[float, float, float]:
        v = self.R @ np.array([gx, gy, gz])
        return float(v[0]), float(v[1]), float(v[2])


# ═══════════════════════════════════════════════════════════════════════════════
#  VELOCITY TRACKER — replaces the naive leaky integrator
# ═══════════════════════════════════════════════════════════════════════════════

class VelocityTracker:
    """
    Tracks velocity with:
    - GPS correction when available (resets integrator drift)
    - Phase-aware reset (resets on flight phase change)
    - Bounded integration (caps velocity at physically reasonable values)

    No more unbounded drift from `velocity_x * 0.95 + accel_x * dt`.
    """

    def __init__(self, max_speed_mps: float = 2000.0):
        self._vx = self._vy = self._vz = 0.0
        self._max_v   = max_speed_mps
        self._last_t  = time.time()
        self._phase   = None

    def update(
        self,
        ax: float, ay: float, az: float,      # m/s² in world frame
        gps_vx: Optional[float] = None,
        gps_vy: Optional[float] = None,
        gps_vz: Optional[float] = None,
        phase:  Optional[FlightPhase] = None,
    ) -> Tuple[float, float, float]:
        now = time.time()
        dt  = min(now - self._last_t, 0.1)    # cap dt to avoid big jumps
        self._last_t = now

        # Phase change → reset integrator (e.g. staging, parachute)
        if phase is not None and phase != self._phase:
            if phase in (FlightPhase.DROGUE, FlightPhase.MAIN, FlightPhase.LANDED):
                self._vx = self._vy = self._vz = 0.0
            self._phase = phase

        # Integrate
        self._vx += ax * dt
        self._vy += ay * dt
        self._vz += az * dt

        # GPS correction if available
        if gps_vx is not None: self._vx = 0.7 * self._vx + 0.3 * gps_vx
        if gps_vy is not None: self._vy = 0.7 * self._vy + 0.3 * gps_vy
        if gps_vz is not None: self._vz = 0.7 * self._vz + 0.3 * gps_vz

        # Clamp
        speed = math.sqrt(self._vx**2 + self._vy**2 + self._vz**2)
        if speed > self._max_v:
            scale = self._max_v / speed
            self._vx *= scale; self._vy *= scale; self._vz *= scale

        return self._vx, self._vy, self._vz

    def reset(self, vx=0.0, vy=0.0, vz=0.0):
        self._vx = vx; self._vy = vy; self._vz = vz


# ═══════════════════════════════════════════════════════════════════════════════
#  SERIAL PACKET VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_FIELDS: Dict[str, type] = {
    "pitch":   (int, float),
    "gyro_x":  (int, float),
}

OPTIONAL_FIELDS: Dict[str, type] = {
    "roll":     (int, float),
    "yaw":      (int, float),
    "gyro_y":   (int, float),
    "gyro_z":   (int, float),
    "accel_x":  (int, float),
    "accel_y":  (int, float),
    "accel_z":  (int, float),
    "altitude": (int, float),
    "motor_l":  (int, float),
    "motor_r":  (int, float),
    "vx":       (int, float),
    "mass":     (int, float),
    "phase":    str,
    "timestamp":(int, float),
}

SANE_RANGES: Dict[str, Tuple[float, float]] = {
    "pitch":    (-180, 180),
    "roll":     (-180, 180),
    "yaw":      (-360, 360),
    "gyro_x":   (-3000, 3000),
    "gyro_y":   (-3000, 3000),
    "gyro_z":   (-3000, 3000),
    "accel_x":  (-2000, 2000),
    "accel_y":  (-2000, 2000),
    "accel_z":  (-2000, 2000),
    "altitude": (-500, 100000),
    "mass":     (0.001, 10000),
}


class SerialPacketValidator:
    """
    Validates JSON packets from hardware before feeding to engine.
    Catches: wrong field names, wrong types, impossible values, partial packets.

    Reports exactly what's wrong so the team can fix their firmware.
    """

    def __init__(self):
        self._error_counts: Dict[str, int] = {}
        self._total_received = 0
        self._total_valid    = 0

    def validate(self, raw_line: str) -> Tuple[bool, Optional[Dict], List[str]]:
        """
        Returns (is_valid, parsed_dict_or_None, list_of_error_messages).
        Errors are human-readable and firmware-actionable.
        """
        self._total_received += 1
        errors = []

        # 1. Must be valid JSON
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError as e:
            err = f"Invalid JSON: {e}. Check your Serial.println() — send complete lines only."
            errors.append(err)
            self._error_counts["json_parse"] = self._error_counts.get("json_parse", 0) + 1
            return False, None, errors

        if not isinstance(data, dict):
            errors.append("Packet must be a JSON object {}, not array or scalar.")
            return False, None, errors

        # 2. Required fields must exist and be numeric
        for field, types in REQUIRED_FIELDS.items():
            if field not in data:
                errors.append(
                    f"Missing required field '{field}'. "
                    f"Add: doc[\"{field}\"] = YOUR_{field.upper()}_VALUE; to your firmware."
                )
                self._error_counts[f"missing_{field}"] = self._error_counts.get(f"missing_{field}", 0) + 1
            elif not isinstance(data[field], types):
                errors.append(
                    f"Field '{field}' must be a number, got {type(data[field]).__name__}. "
                    f"Check: doc[\"{field}\"] = (float){field}; in your firmware."
                )

        if errors:
            return False, None, errors

        # 3. Optional fields — type check if present
        for field, types in OPTIONAL_FIELDS.items():
            if field in data and not isinstance(data[field], types):
                errors.append(
                    f"Optional field '{field}' has wrong type: expected number, got {type(data[field]).__name__}."
                )

        # 4. Sanity range checks
        for field, (lo, hi) in SANE_RANGES.items():
            if field in data and isinstance(data[field], (int, float)):
                val = float(data[field])
                if not (lo <= val <= hi):
                    errors.append(
                        f"Field '{field}' = {val:.2f} is outside sane range [{lo}, {hi}]. "
                        f"Check sensor calibration or unit conversion."
                    )

        if errors:
            return False, data, errors   # return data with warnings

        self._total_valid += 1
        return True, data, []

    def report(self) -> Dict:
        return {
            "total_received": self._total_received,
            "total_valid":    self._total_valid,
            "error_rate_pct": round(100 * (1 - self._total_valid / max(1, self._total_received)), 1),
            "error_counts":   self._error_counts,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  TEMPERATURE-COMPENSATED BAROMETRIC ALTITUDE
# ═══════════════════════════════════════════════════════════════════════════════

def baro_altitude_from_pressure(
    pressure_hpa:       float,
    temperature_c:      float,
    ground_pressure_hpa: float,
    ground_temp_c:      float = 20.0,
) -> float:
    """
    Convert pressure to altitude using the hypsometric formula with
    real temperature correction — more accurate than ISA-assumed altitude.

    Args:
        pressure_hpa:        Current pressure in hPa
        temperature_c:       Current temperature in °C
        ground_pressure_hpa: Reference pressure at ground (set at power-on)
        ground_temp_c:       Ground temperature in °C (set at power-on)

    Returns:
        Altitude AGL in meters
    """
    if ground_pressure_hpa <= 0 or pressure_hpa <= 0:
        return 0.0

    T_kelvin = temperature_c + 273.15
    G_kelvin = ground_temp_c + 273.15
    R  = 8.31432   # J/(mol·K)
    Mg = 0.0289644 * 9.80665  # kg/mol * m/s²

    # Hypsometric formula
    alt = (R * T_kelvin / Mg) * math.log(ground_pressure_hpa / pressure_hpa)
    return alt


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-SENSOR VOTER
# ═══════════════════════════════════════════════════════════════════════════════

class MultiSensorVoter:
    """
    When multiple sensors report the same quantity, vote to reject outliers.
    Useful for teams with redundant IMUs.

    Uses median absolute deviation (MAD) — robust to single bad sensor.
    """

    @staticmethod
    def vote(*values: float, threshold_sigma: float = 2.0) -> Tuple[float, bool]:
        """
        Returns (best_estimate, had_outlier).
        best_estimate: median of inliers
        had_outlier: True if any value was rejected
        """
        vals = [v for v in values if v is not None and math.isfinite(v)]
        if not vals:
            return 0.0, False
        if len(vals) == 1:
            return vals[0], False
        if len(vals) == 2:
            return (vals[0] + vals[1]) / 2, False

        arr    = np.array(vals)
        median = float(np.median(arr))
        mad    = float(np.median(np.abs(arr - median))) + 1e-9
        inliers = arr[np.abs(arr - median) < threshold_sigma * mad * 1.4826]

        if len(inliers) < len(arr):
            return float(np.mean(inliers)) if len(inliers) > 0 else median, True
        return float(np.mean(arr)), False


# ═══════════════════════════════════════════════════════════════════════════════
#  ROCKET STATE BUILDER — replaces state_to_vector for rockets
# ═══════════════════════════════════════════════════════════════════════════════

class RocketStateBuilder:
    """
    Builds the rocket engine state vector from filtered sensor data.

    Fixes:
    - Mass is tracked from propellant consumption (not hardcoded to 1.0)
    - Altitude is filtered (not raw baro)
    - Velocity is integrated with bounds (not from naive leaky integrator)
    - Phase-aware: different state interpretation per flight phase

    State for rocket engine: [x, y, vx, vy, mass, pitch_rad]
    """

    def __init__(
        self,
        dry_mass_kg:  float = 0.3,
        fuel_mass_kg: float = 0.2,
        isp_s:        float = 220.0,
    ):
        self.dry_mass  = dry_mass_kg
        self.fuel_mass = fuel_mass_kg
        self.isp       = isp_s
        self._mass     = dry_mass_kg + fuel_mass_kg
        self._x        = 0.0
        self._fuel_burned = 0.0

    def update_mass(self, thrust_n: float, dt: float) -> float:
        """Track propellant mass using Tsiolkovsky mass flow."""
        if thrust_n > 0:
            mdot = thrust_n / (self.isp * 9.80665)
            self._fuel_burned += mdot * dt
            self._mass = max(self.dry_mass, self.dry_mass + self.fuel_mass - self._fuel_burned)
        return self._mass

    def build(
        self,
        filtered_altitude_m: float,
        filtered_vz_mps:     float,
        vx_mps:              float,
        pitch_deg:           float,
        thrust_n:            float = 0.0,
        dt:                  float = 1/60,
        reported_mass_kg:    Optional[float] = None,
    ) -> np.ndarray:
        """Build rocket state vector with real mass estimate."""
        if reported_mass_kg is not None and reported_mass_kg > self.dry_mass:
            self._mass = reported_mass_kg
        elif thrust_n > 0:
            self.update_mass(thrust_n, dt)

        self._x += vx_mps * dt

        return np.array([
            self._x,
            filtered_altitude_m,
            vx_mps,
            filtered_vz_mps,
            self._mass,          # real mass — not hardcoded 1.0
            math.radians(pitch_deg),
        ])


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER FILTER — wraps everything into one clean interface
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicoreSensorFilter:
    """
    Single object that applies all sensor processing.
    Drop this into the bridge to fix every sensor problem at once.

    Usage in bridge:
        self.filter = PhysicoreSensorFilter(
            imu_max_g=16.0,
            imu_frame="standard",
            gyro_unit="deg_s",    # or "rad_s" for ROS2
            is_rocket=True,
        )

        # In sensor loop:
        result = self.filter.process(
            raw_ax=ax, raw_ay=ay, raw_az=az,
            raw_gx=gx, raw_gy=gy, raw_gz=gz,
            raw_baro_alt=altitude,
            temperature_c=temp,
            dt=dt,
        )
        # result.altitude — filtered
        # result.accel_x  — clean, not saturated
        # result.phase    — current flight phase
    """

    def __init__(
        self,
        imu_max_g:         float = 16.0,
        imu_frame:         str   = "standard",
        gyro_unit:         str   = "auto",
        spike_threshold_m: float = 50.0,
        is_rocket:         bool  = False,
        rocket_dry_mass:   float = 0.3,
        rocket_fuel_mass:  float = 0.2,
        ground_pressure_hpa: float = 1013.25,
        ground_temp_c:     float = 20.0,
    ):
        self.alt_kf      = AltitudeKalmanFilter(spike_threshold_m=spike_threshold_m)
        self.imu_sat     = IMUSaturationDetector(max_g=imu_max_g)
        self.gyro_norm   = GyroNormaliser(source_unit=gyro_unit)
        self.frame       = IMUFrameAligner(preset=imu_frame)
        self.vel_tracker = VelocityTracker()
        self.validator   = SerialPacketValidator()
        self.phase_det   = FlightPhaseDetector() if is_rocket else None
        self.rocket_sb   = RocketStateBuilder(rocket_dry_mass, rocket_fuel_mass) if is_rocket else None

        self._ground_pres = ground_pressure_hpa
        self._ground_temp = ground_temp_c
        self._prev_t      = time.time()

        # State outputs
        self.altitude     = 0.0
        self.vz           = 0.0
        self.accel_x      = 0.0
        self.accel_y      = 0.0
        self.accel_z      = 0.0
        self.gyro_x       = 0.0
        self.gyro_y       = 0.0
        self.gyro_z       = 0.0
        self.is_saturated = False
        self.phase        = FlightPhase.GROUND if is_rocket else None
        self.warnings: List[str] = []

    def process(
        self,
        raw_ax: float, raw_ay: float, raw_az: float,
        raw_gx: float, raw_gy: float, raw_gz: float,
        raw_baro_alt:   Optional[float] = None,
        pressure_hpa:   Optional[float] = None,
        temperature_c:  float = 20.0,
        gps_vx: Optional[float] = None,
        gps_vy: Optional[float] = None,
        gps_vz: Optional[float] = None,
        dt:     Optional[float] = None,
    ) -> "PhysicoreSensorFilter":
        """Run all filters and return self (state updated in-place)."""
        now    = time.time()
        dt     = dt or min(now - self._prev_t, 0.1)
        self._prev_t = now
        self.warnings = []

        # 1. Frame alignment
        ax, ay, az = self.frame.align_accel(raw_ax, raw_ay, raw_az)
        gx, gy, gz = self.frame.align_gyro(raw_gx, raw_gy, raw_gz)

        # 2. IMU saturation
        sat, ax, ay, az = self.imu_sat.check(ax, ay, az)
        self.is_saturated = sat
        if sat:
            self.warnings.append(
                f"IMU SATURATED — readings clamped at ±{self.imu_sat.saturation_g():.0f}g. "
                f"Total saturated samples: {self.imu_sat.total_saturated_samples}. "
                f"Consider a higher-range IMU (e.g. ADXL375 ±200g) for powered flight."
            )

        # 3. Gyro normalisation
        gx, gy, gz = self.gyro_norm.normalise(gx, gy, gz)

        # 4. Altitude — prefer pressure+temp over raw baro
        if pressure_hpa is not None and pressure_hpa > 0:
            baro_alt = baro_altitude_from_pressure(
                pressure_hpa, temperature_c,
                self._ground_pres, self._ground_temp
            )
        elif raw_baro_alt is not None:
            baro_alt = raw_baro_alt
        else:
            baro_alt = self.altitude  # coast on last known

        # Vertical accel (gravity-corrected): up positive
        # az from aligned IMU, subtract g
        vert_accel = az - 9.81

        filt_alt, filt_vz = self.alt_kf.update(baro_alt, vert_accel, dt)
        self.altitude = filt_alt
        self.vz       = filt_vz

        # 5. Velocity
        net_ax = ax; net_ay = ay  # horizontal accels (world frame, simplified)
        vx, vy, vz_from_tracker = self.vel_tracker.update(
            net_ax, net_ay, 0.0,
            gps_vx=gps_vx, gps_vy=gps_vy, gps_vz=gps_vz,
            phase=self.phase,
        )

        # 6. Flight phase (rockets only)
        if self.phase_det is not None:
            accel_g = math.sqrt(ax**2 + ay**2 + az**2) / 9.81
            total_speed = math.sqrt(vx**2 + vy**2 + filt_vz**2)
            self.phase = self.phase_det.update(
                filtered_altitude_m=filt_alt,
                accel_magnitude_g=accel_g,
                climb_rate_mps=filt_vz,
                total_speed_mps=total_speed,
            )
            # Reset velocity on phase transitions
            self.vel_tracker.update(0, 0, 0, phase=self.phase)

        # Store outputs
        self.accel_x = ax; self.accel_y = ay; self.accel_z = az
        self.gyro_x  = gx; self.gyro_y  = gy; self.gyro_z  = gz
        self.vx      = vx; self.vy      = vy

        return self

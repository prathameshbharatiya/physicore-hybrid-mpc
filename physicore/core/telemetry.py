import os
"""
PhysiCore Telemetry — The Data Flywheel
========================================
Every team that opts in makes PhysiCore smarter for everyone.

What gets collected (ONLY if opt_in_telemetry: true in config):
  - Platform type (balancing_bot, quadrotor, etc)
  - Hardware config (IMU type, motor driver, MCU — NO serial numbers)
  - SystemID convergence trajectory (mass/friction/inertia over time)
  - Residual norm history (how much the sim was wrong)
  - Session duration and step count
  - Final converged parameters

What NEVER gets collected:
  - IP addresses
  - Control commands sent to hardware
  - Raw sensor data
  - Any personally identifiable information
  - Location data

The aggregated data feeds a platform prior — so the 101st lab to run
PhysiCore on a UR5 arm starts with 100 labs worth of prior knowledge,
not random weights.

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations

import json
import time
import hashlib
import threading
import platform as _platform
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


# ── Local telemetry buffer ─────────────────────────────────────────────────────
# All telemetry is stored locally first. Upload happens in background.
_TELEM_ROOT = Path.home() / ".physicore" / "telemetry"
_TELEM_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class TelemetryPacket:
    """One session's worth of telemetry data."""

    # Identity — NO personal info
    packet_id:        str
    platform:         str
    hardware_class:   str        # e.g. "arduino_mpu6050_l298n" — hardware type only
    physicore_version:str        = "2.1.0"
    python_version:   str        = ""
    os_type:          str        = ""   # "Windows", "Darwin", "Linux" only

    # Session stats
    session_duration_s:   float  = 0.0
    total_steps:          int    = 0
    control_hz:           float  = 60.0

    # SystemID convergence trajectory
    # List of dicts: [{step, mass, friction, inertia}, ...]
    # Sampled every 100 steps to keep size small
    param_trajectory:     List   = None

    # Residual convergence
    # List of floats: residual_norm sampled every 100 steps
    residual_trajectory:  List   = None

    # Final converged params
    final_params:         Dict   = None

    # Quality metrics
    convergence_pct:      float  = 0.0   # how much residual dropped
    innovation_ema_final: float  = 0.0

    # Timestamp (UTC unix — no timezone info)
    timestamp:            float  = 0.0

    def __post_init__(self):
        if self.param_trajectory    is None: self.param_trajectory    = []
        if self.residual_trajectory is None: self.residual_trajectory = []
        if self.final_params        is None: self.final_params        = {}
        if not self.python_version:
            import sys
            self.python_version = sys.version.split()[0]
        if not self.os_type:
            self.os_type = _platform.system()
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self):
        return asdict(self)

    def size_kb(self):
        return len(json.dumps(self.to_dict())) / 1024


class TelemetryCollector:
    """
    Collects telemetry during a PhysiCore session.
    Runs in the background — zero impact on control loop timing.
    """

    def __init__(self, platform: str, hardware_class: str, control_hz: float = 60.0):
        self.platform       = platform
        self.hardware_class = hardware_class
        self.control_hz     = control_hz
        self._param_traj:    List[Dict] = []
        self._residual_traj: List[float] = []
        self._start_time    = time.time()
        self._sample_every  = 100   # sample every N steps

    def record(self, step: int, params: Dict, residual_norm: float):
        """Call every step — lightweight, just appends to lists."""
        if step % self._sample_every == 0:
            self._param_traj.append({
                "step":     step,
                "mass":     round(params.get("mass", 0), 4),
                "friction": round(params.get("friction", 0), 4),
                "inertia":  round(params.get("inertia", 0), 6),
            })
            self._residual_traj.append(round(residual_norm, 6))

    def build_packet(self, engine) -> TelemetryPacket:
        """Build the final telemetry packet at session end."""
        hist = engine.sysid.convergence_history
        if len(hist) >= 2 and hist[0] > 0:
            convergence_pct = max(0.0, (hist[0] - hist[-1]) / hist[0] * 100)
        else:
            convergence_pct = 0.0

        packet_id = hashlib.sha256(
            f"{self.platform}-{time.time()}-{id(engine)}".encode()
        ).hexdigest()[:16]

        return TelemetryPacket(
            packet_id=            packet_id,
            platform=             self.platform,
            hardware_class=       self.hardware_class,
            session_duration_s=   time.time() - self._start_time,
            total_steps=          engine._step_count,
            control_hz=           self.control_hz,
            param_trajectory=     self._param_traj,
            residual_trajectory=  self._residual_traj,
            final_params=         engine.physics.params.copy(),
            convergence_pct=      convergence_pct,
            innovation_ema_final= engine.sysid.innovation_ema,
        )


class TelemetryManager:
    """
    Manages opt-in telemetry for PhysiCore.

    Flow:
        1. User sets opt_in_telemetry: true in their robot config
        2. TelemetryCollector records adaptation trajectory in background
        3. At session end, packet is saved locally
        4. Background thread attempts upload to PhysiCore telemetry endpoint
        5. If upload fails (no internet, etc), packet stays local for next time
        6. Aggregated data improves platform priors for all users
    """

    # Live endpoint — receives opt-in telemetry from all PhysiCore deployments
    # Set PHYSICORE_TELEMETRY_URL to override (for self-hosted setups)
    ENDPOINT = os.environ.get(
        "PHYSICORE_TELEMETRY_URL",
        "https://api.physicore.ai/telemetry/ingest"
    )

    def __init__(self, enabled: bool = False):
        self.enabled    = enabled
        self._collector: Optional[TelemetryCollector] = None
        self._queue_dir = _TELEM_ROOT / "queue"
        self._sent_dir  = _TELEM_ROOT / "sent"
        self._queue_dir.mkdir(exist_ok=True)
        self._sent_dir.mkdir(exist_ok=True)

    def start_session(self, platform: str, hardware_class: str, control_hz: float = 60.0):
        """Call when a hardware session starts."""
        if not self.enabled:
            return
        self._collector = TelemetryCollector(platform, hardware_class, control_hz)
        print(f"[TELEMETRY] Session started — opt-in data collection active")
        print(f"  Platform: {platform} | Hardware: {hardware_class}")
        print(f"  Data collected: convergence trajectory only. No raw sensor data.")
        print(f"  Disable with: opt_in_telemetry: false in your config file")

    def record(self, step: int, params: Dict, residual_norm: float):
        """Call every engine step — zero-cost if disabled."""
        if self.enabled and self._collector:
            self._collector.record(step, params, residual_norm)

    def end_session(self, engine):
        """Call when session ends — saves packet and attempts upload."""
        if not self.enabled or not self._collector:
            return
        if engine._step_count < 30:
            print(f"[TELEMETRY] Session too short ({engine._step_count} steps) — not recording")
            return

        packet = self._collector.build_packet(engine)
        self._save_local(packet)

        # Upload in background — never blocks the main process
        t = threading.Thread(target=self._upload_queue, daemon=True)
        t.start()

        print(f"[TELEMETRY] Session saved ({engine._step_count} steps, {packet.convergence_pct:.1f}% convergence)")
        print(f"  Thank you for contributing to the PhysiCore data flywheel.")

    def _save_local(self, packet: TelemetryPacket):
        """Save packet to local queue."""
        path = self._queue_dir / f"{packet.packet_id}.json"
        with open(path, "w") as f:
            json.dump(packet.to_dict(), f, indent=2)

    def _upload_queue(self):
        """Attempt to upload all queued packets. Runs in background thread."""
        try:
            import urllib.request
            import urllib.error
        except ImportError:
            return

        for path in self._queue_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)

                req = urllib.request.Request(
                    self.ENDPOINT,
                    data=json.dumps(data).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent":   f"PhysiCore/{data.get('physicore_version','2.1.0')}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        # Move to sent
                        sent_path = self._sent_dir / path.name
                        path.rename(sent_path)
            except Exception:
                # Upload failed — packet stays in queue for next time
                pass

    def pending_count(self) -> int:
        return len(list(self._queue_dir.glob("*.json")))

    def sent_count(self) -> int:
        return len(list(self._sent_dir.glob("*.json")))

    @property
    def status(self) -> dict:
        return {
            "enabled":       self.enabled,
            "pending":       self.pending_count(),
            "sent":          self.sent_count(),
            "session_active":self._collector is not None,
        }


# ── Consent prompt ─────────────────────────────────────────────────────────────

CONSENT_TEXT = """
┌─────────────────────────────────────────────────────────┐
│          PhysiCore Data Flywheel — Opt-In               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  You can help make PhysiCore smarter for everyone.      │
│                                                         │
│  When you run PhysiCore on real hardware, it learns     │
│  your robot's real physics. If you opt in, that         │
│  learning gets aggregated (anonymously) to improve      │
│  the starting point for every future user.              │
│                                                         │
│  What gets shared:                                      │
│    ✓ Platform type (balancing_bot, drone, etc)          │
│    ✓ How fast mass/friction estimates converge          │
│    ✓ Final converged parameter values                   │
│                                                         │
│  What NEVER gets shared:                                │
│    ✗ IP address                                         │
│    ✗ Raw sensor data                                    │
│    ✗ Control commands                                   │
│    ✗ Any personal information                           │
│                                                         │
│  To opt in: set opt_in_telemetry: true in your config   │
│  To opt out: set opt_in_telemetry: false (default)      │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""


# ── Singleton ─────────────────────────────────────────────────────────────────
_manager: Optional[TelemetryManager] = None

def get_telemetry(enabled: bool = False) -> TelemetryManager:
    global _manager
    if _manager is None:
        _manager = TelemetryManager(enabled=enabled)
    return _manager
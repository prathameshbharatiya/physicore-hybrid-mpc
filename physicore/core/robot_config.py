"""
PhysiCore Hardware Config
=========================
One YAML file describes your robot. One command connects it.

Instead of:
    python physicore/bridge/physicore_bridge.py \\
        --platform balancing_bot_arduino \\
        --connection COM3 \\
        --baud 115200

You write a YAML file once:
    # my_robot.yaml
    name: My Balancing Bot
    platform: balancing_bot
    connection: COM3
    baud: 115200
    mass: 1.2
    imu: MPU6050
    motor_driver: L298N

And then just:
    physicore run --config my_robot.yaml

The config system also manages per-robot registry namespacing,
so two balancing bots at different labs don't share params.

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations

import os
import re
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


# ── Platform aliases ───────────────────────────────────────────────────────────
PLATFORM_ALIASES = {
    # Balancing bot variants
    "balancing_bot":          "balancing_bot_arduino",
    "self_balancing":         "balancing_bot_arduino",
    "inverted_pendulum":      "balancing_bot_arduino",
    "segway":                 "balancing_bot_arduino",
    # Drones
    "px4":                    "px4_quadrotor",
    "pixhawk":                "px4_quadrotor",
    "ardupilot":              "ardupilot_quadrotor",
    "ardupilot_quad":         "ardupilot_quadrotor",
    "ardupilot_plane":        "ardupilot_plane",
    "drone":                  "px4_quadrotor",
    "quadrotor":              "px4_quadrotor",
    "evtol":                  "evtol",
    # ROS2
    "ros2_arm":               "ros2_manipulator",
    "arm":                    "ros2_manipulator",
    "manipulator":            "ros2_manipulator",
    "humanoid":               "ros2_legged",
    "legged":                 "ros2_legged",
    "biped":                  "ros2_legged",
    "quadruped":              "ros2_legged",
    "ugv":                    "ros2_ground_rover",
    "ground_rover":           "ros2_ground_rover",
    "rover":                  "ros2_ground_rover",
    "auv":                    "ros2_auv",
    "underwater":             "ros2_auv",
    "surgical":               "ros2_surgical",
    # Serial
    "rocket":                 "custom_rocket_fc",
    "sounding_rocket":        "custom_rocket_fc",
    "satellite":              "satellite_serial",
    "spacecraft":             "satellite_serial",
}

# ── Engine platform mapping ────────────────────────────────────────────────────
ENGINE_PLATFORM = {
    "balancing_bot_arduino":  "balancing_bot",
    "px4_quadrotor":          "quadrotor",
    "ardupilot_quadrotor":    "quadrotor",
    "ardupilot_plane":        "fixed_wing",
    "evtol":                  "evtol",
    "ros2_manipulator":       "manipulator_arm",
    "ros2_legged":            "legged_robot",
    "ros2_ground_rover":      "ground_rover",
    "ros2_auv":               "auv",
    "ros2_surgical":          "surgical_robot",
    "custom_rocket_fc":       "rocket",
    "satellite_serial":       "satellite",
}


@dataclass
class RobotConfig:
    """
    Complete description of a robot for PhysiCore.
    Load from YAML or build programmatically.
    """

    # Required
    platform:    str                  # e.g. "balancing_bot", "px4", "ros2_arm"

    # Identity
    name:        str         = "My Robot"
    description: str         = ""
    version:     str         = "1.0"

    # Connection
    connection:  str         = "auto"  # COM3, /dev/ttyUSB0, udp:14550, ros2
    baud:        int         = 115200
    timeout:     float       = 2.0

    # Physical parameters (starting point for SystemID)
    mass:        float       = 1.0    # kg
    friction:    float       = 0.15   # dimensionless
    inertia:     float       = 0.01   # kg*m^2

    # Hardware info (metadata only — for registry namespacing)
    imu:         str         = "MPU6050"
    motor_driver:str         = "L298N"
    mcu:         str         = "Arduino Uno"

    # PhysiCore behaviour
    control_hz:  float       = 60.0
    wind:        float       = 0.0    # 0=calm, 0.5=moderate, 1.0=severe

    # Registry
    use_registry:    bool    = True   # load/save learned params
    opt_in_telemetry:bool    = False  # contribute to platform prior

    # Safety
    sentinel_enabled:bool    = True
    max_torque:      float   = 2.5    # N*m for serial platforms

    # Extra platform-specific params
    extra:       Dict[str, Any] = field(default_factory=dict)

    # ── Derived ───────────────────────────────────────────────────────────────

    @property
    def bridge_platform(self) -> str:
        """The --platform argument for physicore_bridge.py"""
        raw = self.platform.lower().replace(" ", "_").replace("-", "_")
        return PLATFORM_ALIASES.get(raw, raw)

    @property
    def engine_platform(self) -> str:
        """The platform name for PhysiCore.for_platform()"""
        return ENGINE_PLATFORM.get(self.bridge_platform, self.bridge_platform)

    @property
    def initial_params(self) -> Dict[str, float]:
        return {"mass": self.mass, "friction": self.friction, "inertia": self.inertia}

    @property
    def registry_key(self) -> str:
        """Unique key for this robot in the registry — platform + hardware combo."""
        hw = f"{self.imu}_{self.motor_driver}_{self.mcu}".lower()
        hw = re.sub(r'[^a-z0-9_]', '_', hw)
        return f"{self.engine_platform}__{hw}"

    @property
    def is_serial(self) -> bool:
        return self.bridge_platform in (
            "balancing_bot_arduino", "custom_rocket_fc",
            "satellite_serial", "ground_rover_serial"
        )

    @property
    def is_mavlink(self) -> bool:
        return self.bridge_platform in (
            "px4_quadrotor", "ardupilot_quadrotor",
            "ardupilot_plane", "evtol"
        )

    @property
    def is_ros2(self) -> bool:
        return self.bridge_platform in (
            "ros2_manipulator", "ros2_legged",
            "ros2_ground_rover", "ros2_auv", "ros2_surgical"
        )

    @property
    def resolved_connection(self) -> str:
        """Auto-resolve connection string based on platform and OS."""
        if self.connection != "auto":
            return self.connection
        if self.is_mavlink:
            return "udp:14550"
        if self.is_ros2:
            return "ros2"
        # Serial — auto-detect OS
        import platform as _p
        sys = _p.system()
        if sys == "Windows": return "COM3"
        if sys == "Darwin":  return "/dev/cu.usbserial-0001"
        return "/dev/ttyUSB0"

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "extra"} | self.extra

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_yaml())
        print(f"[CONFIG] Saved robot config to {path}")

    # ── Class methods ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "RobotConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "RobotConfig":
        """Load config from dict."""
        known = {
            "platform", "name", "description", "version",
            "connection", "baud", "timeout",
            "mass", "friction", "inertia",
            "imu", "motor_driver", "mcu",
            "control_hz", "wind",
            "use_registry", "opt_in_telemetry",
            "sentinel_enabled", "max_torque",
        }
        base = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**base, extra=extra)

    @classmethod
    def from_args(cls, args) -> "RobotConfig":
        """Build config from argparse args (bridge CLI)."""
        return cls(
            platform=   args.platform or "balancing_bot",
            connection= args.connection,
            baud=       args.baud,
            mass=       getattr(args, "mass", 1.0),
        )

    @classmethod
    def balancing_bot(cls, connection="auto", mass=1.0, **kw) -> "RobotConfig":
        return cls(platform="balancing_bot", connection=connection,
                   mass=mass, imu="MPU6050", motor_driver="L298N", **kw)

    @classmethod
    def px4_drone(cls, connection="udp:14550", mass=1.5, **kw) -> "RobotConfig":
        return cls(platform="px4", connection=connection, mass=mass, **kw)

    @classmethod
    def ros2_arm(cls, dof=6, mass=2.0, **kw) -> "RobotConfig":
        return cls(platform="ros2_arm", mass=mass,
                   extra={"dof": dof}, **kw)

    @classmethod
    def rocket(cls, connection="auto", dry_mass=5.0, **kw) -> "RobotConfig":
        return cls(platform="rocket", connection=connection,
                   mass=dry_mass, **kw)


# ── Template YAML files ────────────────────────────────────────────────────────

TEMPLATES = {
    "balancing_bot": """\
# PhysiCore — Balancing Bot Config
# Run with: python physicore/bridge/physicore_bridge.py --config this_file.yaml

name: My Balancing Bot
platform: balancing_bot

# Connection — replace COM3 with your actual port
# Windows: check Device Manager → Ports (COM & LPT)
# Mac:     ls /dev/cu.*
# Linux:   ls /dev/ttyUSB*
connection: COM3
baud: 115200

# Physical parameters — PhysiCore starts here, learns the real values
mass: 1.0       # kg — your best estimate
friction: 0.15  # dimensionless
inertia: 0.01   # kg*m^2 (moment of inertia about pitch axis)

# Hardware
imu: MPU6050
motor_driver: L298N
mcu: Arduino Uno

# PhysiCore settings
control_hz: 60.0
use_registry: true          # save/load learned params between sessions
opt_in_telemetry: false     # set true to contribute to platform prior
sentinel_enabled: true
max_torque: 2.5             # DO NOT change — PhysiCore outputs ±2.5 N·m
""",

    "px4_drone": """\
# PhysiCore — PX4 Drone Config
name: My Drone
platform: px4

connection: udp:14550       # or: /dev/ttyACM0 for USB
mass: 1.5                   # kg with battery
friction: 0.1
inertia: 0.05

control_hz: 60.0
use_registry: true
opt_in_telemetry: false
""",

    "rocket": """\
# PhysiCore — Sounding Rocket Config
name: My Rocket
platform: rocket

connection: auto            # auto-detects OS serial port
baud: 115200
mass: 5.0                   # kg dry mass
friction: 0.45              # Cd
inertia: 220                # Isp (s)

control_hz: 60.0
use_registry: true
opt_in_telemetry: false
""",

    "ros2_arm": """\
# PhysiCore — ROS2 Robot Arm Config
name: My Robot Arm
platform: ros2_arm

connection: ros2
mass: 2.0                   # end-effector + max payload (kg)
friction: 0.3               # joint friction
inertia: 0.1

# ROS2 settings
ros2_distro: humble
joint_topic: /joint_states
dof: 6

control_hz: 60.0
use_registry: true
opt_in_telemetry: false
""",
}


def create_template(platform: str, output_path: str):
    """Write a template config file for the given platform."""
    key = PLATFORM_ALIASES.get(platform.lower().replace("-","_"), platform).split("_")[0]
    template = TEMPLATES.get(key) or TEMPLATES.get(platform) or TEMPLATES["balancing_bot"]
    with open(output_path, "w") as f:
        f.write(template)
    print(f"[CONFIG] Created template config: {output_path}")
    print(f"  Edit the file and run: python physicore/bridge/physicore_bridge.py --config {output_path}")
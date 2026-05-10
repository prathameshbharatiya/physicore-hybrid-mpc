"""
PhysiCore — Hybrid Uncertainty-Aware Sim-to-Real Engine
"""

from .core.engine import (
    PhysiCore, PhysiCoreConfig, ControlStep,
    PhysicsLayer, ResidualEnsemble, CEMOptimizer, OnlineSystemID,
    quadrotor_dynamics, fixed_wing_dynamics, evtol_dynamics,
    manipulator_arm_dynamics, surgical_robot_dynamics, legged_robot_dynamics,
    balancing_bot_dynamics, rocket_dynamics, ground_rover_dynamics,
    rover_dynamics, auv_dynamics, satellite_dynamics,
    PLATFORM_DYNAMICS,
)

from .core.performance_config import configure, PerformanceConfig, PROFILES, from_env

__version__ = "1.3.0"
__author__  = "Prathamesh Shirbhate"
__all__ = [
    "PhysiCore", "PhysiCoreConfig", "ControlStep",
    "PhysicsLayer", "ResidualEnsemble", "CEMOptimizer", "OnlineSystemID",
    "quadrotor_dynamics", "fixed_wing_dynamics", "evtol_dynamics",
    "manipulator_arm_dynamics", "surgical_robot_dynamics", "legged_robot_dynamics",
    "balancing_bot_dynamics", "rocket_dynamics", "ground_rover_dynamics",
    "rover_dynamics", "auv_dynamics", "satellite_dynamics",
    "PLATFORM_DYNAMICS",
    # Performance
    "configure", "PerformanceConfig", "PROFILES", "from_env",
]

# ── Round 1 additions ──────────────────────────────────────────────────────────
from .core.urdf_loader import (
    load_robot,
    build_robot_model,
    parse_robot_file,
    URDFRobotModel,
    ProperContactModel,
    JointInfo,
    LinkInfo,
)
from .core.extra_dynamics import (
    mobile_manipulator_dynamics,
    dual_arm_dynamics,
    cable_driven_dynamics,
    exoskeleton_dynamics,
)
from .core.fleet import FleetManager, FleetRobotSpec, FleetHealth
__all__ += [
    # URDF loader
    "load_robot", "build_robot_model", "parse_robot_file",
    "URDFRobotModel", "ProperContactModel", "JointInfo", "LinkInfo",
    # Extra dynamics
    "mobile_manipulator_dynamics", "dual_arm_dynamics",
    "cable_driven_dynamics", "exoskeleton_dynamics",
    # Fleet
    "FleetManager", "FleetRobotSpec", "FleetHealth",
]

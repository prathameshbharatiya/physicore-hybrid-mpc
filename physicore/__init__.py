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

__version__ = "1.2.0"
__author__  = "Prathamesh Shirbhate"
__all__ = [
    "PhysiCore", "PhysiCoreConfig", "ControlStep",
    "PhysicsLayer", "ResidualEnsemble", "CEMOptimizer", "OnlineSystemID",
    "quadrotor_dynamics", "fixed_wing_dynamics", "evtol_dynamics",
    "manipulator_arm_dynamics", "surgical_robot_dynamics", "legged_robot_dynamics",
    "balancing_bot_dynamics", "rocket_dynamics", "ground_rover_dynamics",
    "rover_dynamics", "auv_dynamics", "satellite_dynamics",
    "PLATFORM_DYNAMICS",
]


# ── Three-line SDK ────────────────────────────────────────────────────────────
from physicore.__init__connect import connect, PhysicoreRobot

__all__ += ['connect', 'PhysicoreRobot']

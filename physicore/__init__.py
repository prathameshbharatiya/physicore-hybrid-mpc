"""
PhysiCore Core Engine
=====================
Hybrid physics engine combining classical mechanics with online adaptation.
"""

from .core.engine import (
    PhysiCore,
    PhysiCoreConfig,
    PLATFORM_DYNAMICS,
    PhysicoreSimulator
)
from .sentinel import (
    SentinelOS,
    SentinelConfig,
    SentinelMode,
    SentinelLog
)

__all__ = [
    "PhysiCore",
    "PhysiCoreConfig",
    "PLATFORM_DYNAMICS",
    "PhysicoreSimulator",
    "SentinelOS",
    "SentinelConfig",
    "SentinelMode",
    "SentinelLog"
]

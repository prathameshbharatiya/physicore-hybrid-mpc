"""
Sentinel OS
===========
Safety governance layer for PhysiCore.
Sits above the PhysiCore engine and enforces safety constraints.

Architecture:
    NOMINAL → CAUTIOUS → FALLBACK

Usage:
    from physicore.sentinel import SentinelOS, SentinelConfig

    sentinel = SentinelOS(engine, config=SentinelConfig(
        max_uncertainty=0.05,
        max_residual=0.5,
        param_bounds={"mass": (0.5, 5.0)}
    ))

    while True:
        state  = robot.get_state()
        action = sentinel.step(state, x_ref)
        robot.apply(action)
        sentinel.observe(state, action, robot.get_state())

        if sentinel.mode == "FALLBACK":
            print("Sentinel: unsafe state detected, fallback active")
"""

from .core import SentinelOS, SentinelConfig, SentinelMode, SentinelLog

__all__ = ["SentinelOS", "SentinelConfig", "SentinelMode", "SentinelLog"]

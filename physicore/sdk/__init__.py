"""
PhysiCore SDK
=============
High-level SDK for integrating PhysiCore into any robotics stack.

Usage:
    from physicore.sdk import PhysicoreClient

    client = PhysicoreClient()
    client.configure("quadrotor", {"mass": 1.5, "friction": 0.1})

    while True:
        state  = robot.get_state()
        action = client.step(state, x_ref)
        robot.apply(action)
        client.observe(state, action, robot.get_state())
"""

from .client import PhysicoreClient
from .simulate import PhysicoreSimulator
from .analyze  import PhysicoreAnalyzer

__all__ = ["PhysicoreClient", "PhysicoreSimulator", "PhysicoreAnalyzer"]

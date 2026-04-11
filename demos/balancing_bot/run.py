"""
PhysiCore Demo: Self-Balancing Robot
=====================================
Demonstrates PhysiCore on an inverted pendulum balancing robot.
Shows how PhysiCore adapts when the real robot's mass and
moment of inertia differ from the simulation model.

Run:
  python demos/balancing_bot/run.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import logging
from physicore import PhysiCore
from physicore.sdk import PhysicoreSimulator

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("demo.balancing_bot")


def run_demo(n_steps: int = 400):
    logger.info("PhysiCore — Balancing Bot Demo")
    logger.info("Nominal mass: 1.0 kg | True mass: 1.3 kg (30% mismatch)")
    logger.info("-" * 50)

    sim = PhysicoreSimulator(
        platform="balancing_bot",
        initial_params={"mass": 1.0, "friction": 0.15, "inertia": 0.01},
    )

    initial_state = np.array([0.05, 0.0, 0.0, 0.0])  # slight initial lean
    x_ref         = np.array([0.0,  0.0, 0.0, 0.0])  # upright, stationary

    result = sim.run(
        initial_state=initial_state,
        x_ref=x_ref,
        n_steps=n_steps,
        true_params={"mass": 1.3, "friction": 0.18, "inertia": 0.013},
        verbose=True,
        log_every=80,
    )

    print()
    print(result.summary())
    val = sim.validate(initial_state, x_ref, {"mass": 1.3, "friction": 0.18, "inertia": 0.013})
    print(f"\n  L2 improvement: {val['improvement_pct']:.1f}%")
    print(f"  Final mass estimate: {val['final_params'].get('mass', 0):.3f} kg (true: 1.300)")


if __name__ == "__main__":
    run_demo()

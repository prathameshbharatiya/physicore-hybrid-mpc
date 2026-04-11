"""
PhysiCore Demo: Ground Rover
=============================
Demonstrates PhysiCore on a differential drive ground robot.
Shows terrain friction adaptation.

Run:
  python demos/ground_rover/run.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import logging
from physicore.sdk import PhysicoreSimulator

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("demo.ground_rover")


def run_demo(n_steps: int = 300):
    logger.info("PhysiCore — Ground Rover Demo")
    logger.info("Nominal friction: 0.5 | True friction: 0.8 (carpet vs tile)")
    logger.info("-" * 50)

    sim = PhysicoreSimulator(
        platform="ground_rover",
        initial_params={"mass": 5.0, "friction": 0.5, "inertia": 0.1},
    )

    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_ref         = np.array([5.0, 5.0, 0.785, 0.0, 0.0, 0.0])

    result = sim.run(
        initial_state=initial_state,
        x_ref=x_ref,
        n_steps=n_steps,
        true_params={"mass": 5.2, "friction": 0.8, "inertia": 0.12},
        verbose=True,
        log_every=60,
    )

    print()
    print(result.summary())


if __name__ == "__main__":
    run_demo()

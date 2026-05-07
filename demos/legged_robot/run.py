"""
PhysiCore Demo: Legged Robot (Quadruped)
=========================================
Run: python demos/legged_robot/run.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import logging
from physicore.sdk import PhysicoreSimulator

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

def run():
    sim = PhysicoreSimulator("legged_robot", {"mass":30.0,"friction":0.7,"inertia":0.5})
    initial = np.zeros(12)
    initial[2] = 0.4
    x_ref = np.array([5,0,0.4,0,0,0,0,0,0,0,0,0], dtype=float)
    result = sim.run(
        initial_state=initial, x_ref=x_ref,
        n_steps=300, true_params={"mass":32.0,"friction":0.85,"inertia":0.55},
        verbose=True, log_every=60
    )
    print(result.summary())

if __name__ == "__main__":
    run()

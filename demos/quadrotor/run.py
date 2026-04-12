"""
PhysiCore Demo: Quadrotor
=========================
Run: python demos/quadrotor/run.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import logging
from physicore.sdk import PhysicoreSimulator

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

def run():
    sim = PhysicoreSimulator("quadrotor", {"mass":1.5,"friction":0.1,"inertia":0.02})
    result = sim.run(
        initial_state=np.array([0,0,5,0,0,0,0,0,0,0,0,0], dtype=float),
        x_ref=np.array([0,0,10,0,0,0,0,0,0,0,0,0], dtype=float),
        n_steps=300, true_params={"mass":1.8,"friction":0.15,"inertia":0.022},
        verbose=True, log_every=60
    )
    print(result.summary())

if __name__ == "__main__":
    run()

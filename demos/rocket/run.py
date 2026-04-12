"""
PhysiCore Demo: Sounding Rocket
================================
Run: python demos/rocket/run.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import logging
from physicore.sdk import PhysicoreSimulator

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

def run():
    sim = PhysicoreSimulator("rocket", {"mass":0.15,"friction":0.45,"inertia":220.0})
    initial = np.array([0,0,0,0,5.0,0.02], dtype=float)
    x_ref   = np.array([0,3000,0,150,0,0], dtype=float)
    result  = sim.run(
        initial_state=initial, x_ref=x_ref,
        n_steps=160, true_params={"mass":0.15,"friction":0.52,"inertia":209.0},
        verbose=True, log_every=40
    )
    print(result.summary())

if __name__ == "__main__":
    run()

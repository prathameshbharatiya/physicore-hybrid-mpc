"""
PhysiCore Demo: Manipulator Arm
================================
Run: python demos/manipulator_arm/run.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import logging
from physicore.sdk import PhysicoreSimulator

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

def run():
    sim = PhysicoreSimulator("manipulator_arm", {"mass":2.0,"friction":0.3,"inertia":0.1})
    initial = np.zeros(12)
    initial[:6] = [0.2,-0.3,0.8,0.1,0.3,0.1]
    x_ref = np.zeros(12)
    x_ref[:6] = [0.0,-0.5,1.0,0.0,0.5,0.0]
    result = sim.run(
        initial_state=initial, x_ref=x_ref,
        n_steps=400, true_params={"mass":3.5,"friction":0.35,"inertia":0.12},
        verbose=True, log_every=80
    )
    print(result.summary())

if __name__ == "__main__":
    run()

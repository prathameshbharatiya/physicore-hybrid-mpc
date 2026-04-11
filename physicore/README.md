# PhysiCore Engine

This directory contains the core multiphysics engine for the PhysiCore AI platform.

## Contents

- `core/engine.py`: Python implementation of the physics dynamics and integration.
- `physicore-bridge/`: Hardware bridge for connecting real robots to the simulator.

## Usage (Python Engine)

```python
import numpy as np
from core.engine import PhysicoreSimulator

# Initialize a quadrotor simulator
sim = PhysicoreSimulator(platform="quadrotor", params={"mass": 1.5, "friction": 0.1})

# Simulation step
action = np.array([15.0, 0.0, 0.0, 0.0]) # Thrust, roll, pitch, yaw
state = sim.step(action, dt=0.01)

print(f"New state: {state}")
```

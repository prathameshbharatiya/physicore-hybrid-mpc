"""
Example 1: Balancing Bot Simulation
====================================
Runs 200 MPC steps on a simulated balancing bot and prints a summary.
No hardware required — pure simulation.

Run:
    python examples/balancing_bot_sim.py
    physicore run balancing_bot_sim
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from physicore import PhysiCore, PLATFORM_DYNAMICS

print("=" * 60)
print("  PhysiCore -- Balancing Bot Simulation")
print("=" * 60)

# ── 1. Build engine ───────────────────────────────────────────────────────────
params = {"mass": 1.0, "friction": 0.15, "inertia": 0.01}
engine = PhysiCore.for_platform("balancing_bot", initial_params=params, control_hz=50)
dyn_fn = PLATFORM_DYNAMICS["balancing_bot"][0]   # (fn, state_dim, action_dim)
dt = 1 / engine.cfg.control_hz

print(f"\nPlatform   : balancing_bot")
print(f"State dim  : {engine.cfg.state_dim}")
print(f"Action dim : {engine.cfg.action_dim}")
print(f"Horizon    : {engine.cfg.horizon}")
print(f"dt         : {dt:.4f} s")

# ── 2. Initial state: slightly tilted ─────────────────────────────────────────
#   state = [angle, angular_vel, x_pos, x_vel]
x = np.array([0.15, 0.0, 0.0, 0.0])
x_ref = np.zeros(4)
print(f"\nInitial state: angle={x[0]:.3f} rad  pos={x[2]:.3f} m")

# ── 3. Simulate ───────────────────────────────────────────────────────────────
N = 200
history = {"angle": [], "pos": [], "residual": [], "action": []}

print("\nRunning 200 simulation steps...")
for t in range(N):
    step = engine.step(x, x_ref)
    history["angle"].append(float(x[0]))
    history["pos"].append(float(x[2]))
    history["residual"].append(float(step.residual_norm))
    history["action"].append(float(step.action[0]))

    xdot = dyn_fn(x, step.action, params)
    x = x + dt * xdot

# ── 4. Report ─────────────────────────────────────────────────────────────────
angles = np.array(history["angle"])
residuals = np.array(history["residual"])

print(f"\n{'-'*60}")
print(f"  Steps completed : {N}")
print(f"  Final angle     : {angles[-1]:.4f} rad ({np.degrees(angles[-1]):.2f} deg)")
print(f"  Max angle       : {np.max(np.abs(angles)):.4f} rad")
print(f"  Avg residual    : {residuals.mean():.6f}")
print(f"  Max residual    : {residuals.max():.6f}")
print(f"  Converged       : {abs(angles[-1]) < 0.05}")
print(f"{'-'*60}")

# ── 5. ASCII trajectory ───────────────────────────────────────────────────────
print("\nAngle trajectory (positive = leaning right):\n")
width = 50
for i in range(0, N, N // 20):
    a = angles[i]
    bar_len = int(min(abs(a) / 0.2 * (width // 2), width // 2))
    if a >= 0:
        bar = " " * (width // 2) + "#" * bar_len
    else:
        bar = " " * (width // 2 - bar_len) + "#" * bar_len
    print(f"  t={i:3d} |{bar}| {a:+.3f}")

print("\nSimulation complete.")

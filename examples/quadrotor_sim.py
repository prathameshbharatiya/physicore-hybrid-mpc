"""
Example 2: Quadrotor Simulation
=================================
Demonstrates full 3D quadrotor flight simulation with MPC control,
online system ID, and residual learning.

Run:
    python examples/quadrotor_sim.py
    physicore run quadrotor_sim
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from physicore import PhysiCore, PLATFORM_DYNAMICS

print("=" * 60)
print("  PhysiCore -- Quadrotor Flight Simulation")
print("=" * 60)

# ── 1. Build engine ───────────────────────────────────────────────────────────
params = {
    "mass": 1.5,
    "arm_length": 0.25,
    "k_thrust": 3.16e-6,
    "k_torque": 7.94e-9,
    "Ixx": 0.0213, "Iyy": 0.0213, "Izz": 0.04,
}
engine = PhysiCore.for_platform("quadrotor", initial_params=params, control_hz=50)
dyn_fn = PLATFORM_DYNAMICS["quadrotor"][0]   # (fn, state_dim, action_dim)
dt = 1 / engine.cfg.control_hz

print(f"\nPlatform   : quadrotor")
print(f"State dim  : {engine.cfg.state_dim}  (pos x3, vel x3, euler x3, omega x3)")
print(f"Action dim : {engine.cfg.action_dim}  (4 motor thrusts)")
print(f"Mass       : {params['mass']} kg")
print(f"dt         : {dt:.4f} s")

# ── 2. Hover from 0.5 m below target ─────────────────────────────────────────
# state = [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
x0 = np.zeros(engine.cfg.state_dim)
x0[2] = -0.5  # start 0.5 m below origin
x = x0.copy()
x_ref = np.zeros(engine.cfg.state_dim)   # hover at origin

hover_thrust = params["mass"] * 9.81 / 4

print(f"\nInitial position : ({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}) m")
print(f"Target position  : (0.00, 0.00, 0.00) m")
print(f"Hover thrust/mot : {hover_thrust:.3f} N")

# ── 3. Simulate ───────────────────────────────────────────────────────────────
N = 150
history = {
    "z": [], "vz": [], "roll": [], "pitch": [],
    "residual": [],
}

print(f"\nRunning {N} steps ({N * dt:.1f} s simulated)...")
for t in range(N):
    step = engine.step(x, x_ref)
    history["z"].append(float(x[2]))
    history["vz"].append(float(x[5]))
    history["roll"].append(float(np.degrees(x[6])) if engine.cfg.state_dim > 6 else 0.0)
    history["pitch"].append(float(np.degrees(x[7])) if engine.cfg.state_dim > 7 else 0.0)
    history["residual"].append(float(step.residual_norm))

    xdot = dyn_fn(x, step.action, params)
    x = x + dt * xdot

# ── 4. Report ─────────────────────────────────────────────────────────────────
z_arr = np.array(history["z"])
res_arr = np.array(history["residual"])
roll_arr = np.array(history["roll"])

print(f"\n{'-'*60}")
print(f"  Steps              : {N}")
print(f"  Final altitude     : {z_arr[-1]:.4f} m (target 0.0)")
print(f"  Altitude error     : {abs(z_arr[-1]):.4f} m")
print(f"  Max roll           : {np.max(np.abs(roll_arr)):.2f} deg")
print(f"  Avg residual norm  : {res_arr.mean():.6f}")
print(f"  Residual (final)   : {res_arr[-1]:.6f}")
print(f"{'-'*60}")

# ── 5. Altitude chart ─────────────────────────────────────────────────────────
print("\nAltitude over time (target = 0 m):\n")
for i in range(0, N, max(1, N // 15)):
    z = z_arr[i]
    bar = int(min(abs(z) / 0.5 * 20, 20))
    print(f"  t={i*dt:5.2f}s |{'#'*bar:<20}| z={z:+.3f} m")

print("\nQuadrotor simulation complete.")

"""
Example 4: Fleet -- Two Robots in Parallel
==========================================
Demonstrates running two robots (balancing_bot + quadrotor) simultaneously
using FleetManager. Shows per-robot health and residual tracking.

Run:
    python examples/fleet_two_robots.py
    physicore run fleet_two_robots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from physicore import PhysiCore, PLATFORM_DYNAMICS
from physicore.core.fleet import FleetManager, FleetRobotSpec
from physicore.core.robot_config import RobotConfig

print("=" * 60)
print("  PhysiCore -- Fleet: Two Robots in Parallel")
print("=" * 60)

# ── 1. Build engines directly ──────────────────────────────────────────────────
bot_params  = {"mass": 1.0, "friction": 0.15, "inertia": 0.01}
quad_params = {"mass": 1.5, "arm_length": 0.25, "k_thrust": 3.16e-6,
               "k_torque": 7.94e-9, "Ixx": 0.02, "Iyy": 0.02, "Izz": 0.04}

bot_engine  = PhysiCore.for_platform("balancing_bot", initial_params=bot_params,  control_hz=50)
quad_engine = PhysiCore.for_platform("quadrotor",     initial_params=quad_params, control_hz=50)

bot_dyn  = PLATFORM_DYNAMICS["balancing_bot"][0]
quad_dyn = PLATFORM_DYNAMICS["quadrotor"][0]

bot_dt  = 1 / bot_engine.cfg.control_hz
quad_dt = 1 / quad_engine.cfg.control_hz

print(f"\nEngines created:")
print(f"  bot-001  : balancing_bot  state_dim={bot_engine.cfg.state_dim}")
print(f"  quad-001 : quadrotor      state_dim={quad_engine.cfg.state_dim}")

# ── 2. Initial states ─────────────────────────────────────────────────────────
states = {
    "bot-001":  np.array([0.1, 0.0, 0.0, 0.0]),
    "quad-001": np.zeros(quad_engine.cfg.state_dim),
}
states["quad-001"][2] = -1.0  # 1 m below origin

refs = {
    "bot-001":  np.zeros(bot_engine.cfg.state_dim),
    "quad-001": np.zeros(quad_engine.cfg.state_dim),
}

engines = {"bot-001": bot_engine, "quad-001": quad_engine}
dyn_fns = {"bot-001": bot_dyn,   "quad-001": quad_dyn}
params  = {"bot-001": bot_params, "quad-001": quad_params}
dts     = {"bot-001": bot_dt,     "quad-001": quad_dt}

# ── 3. Run parallel steps ─────────────────────────────────────────────────────
N = 100
print(f"\nRunning {N} synchronised steps...\n")
print(f"  {'Step':>4}  {'bot-001 angle':>14}  {'quad-001 alt':>13}  {'bot resid':>10}  {'quad resid':>11}")
print(f"  {'-'*4}  {'-'*14}  {'-'*13}  {'-'*10}  {'-'*11}")

residuals = {"bot-001": [], "quad-001": []}

for t in range(N):
    for rid, engine in engines.items():
        x = states[rid]
        step = engine.step(x, refs[rid])
        residuals[rid].append(float(step.residual_norm))
        xdot = dyn_fns[rid](x, step.action, params[rid])
        states[rid] = x + dts[rid] * xdot

    if t % 20 == 0:
        angle = states["bot-001"][0]
        alt   = states["quad-001"][2]
        br    = residuals["bot-001"][-1]
        qr    = residuals["quad-001"][-1]
        print(f"  {t:>4}  {angle:>+14.4f}  {alt:>+13.4f}  {br:>10.6f}  {qr:>11.6f}")

# ── 4. Summary ────────────────────────────────────────────────────────────────
print(f"\n{'-'*60}")
print("  Final states:")
print(f"    bot-001  angle = {states['bot-001'][0]:+.4f} rad")
print(f"    quad-001 alt   = {states['quad-001'][2]:+.4f} m")
print()
print("  Residual summary:")
for rid in ["bot-001", "quad-001"]:
    r = np.array(residuals[rid])
    print(f"    {rid:<12}  mean={r.mean():.6f}  max={r.max():.6f}")
print(f"{'-'*60}")
print("\nFleet simulation complete.")

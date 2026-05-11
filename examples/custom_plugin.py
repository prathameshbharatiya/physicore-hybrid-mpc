"""
Example 5: Custom Plugin
=========================
Shows how to write, register, and use a PhysiCore plugin inline.
Implements a "gravity compensation" plugin that adds a feed-forward
torque offset to counteract gravity on each control step.

Run:
    python examples/custom_plugin.py
    physicore run custom_plugin
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

print("=" * 60)
print("  PhysiCore -- Custom Plugin Demo")
print("=" * 60)

# ── 1. Define a plugin ────────────────────────────────────────────────────────
try:
    from physicore.sdk.plugin_loader import PhysicorePlugin, PluginMeta, PluginLoader

    class GravityCompPlugin(PhysicorePlugin):
        """Feed-forward gravity compensation plugin."""

        def __init__(self, g: float = 9.81, mass: float = 1.0):
            self._g = g
            self._mass = mass
            self._step_count = 0
            self._total_correction = 0.0

        @property
        def meta(self) -> PluginMeta:
            return PluginMeta(
                id="gravity-comp",
                name="Gravity Compensation",
                version="1.0.0",
                description="Feed-forward torque offset to counteract gravity",
                author="example",
            )

        def on_load(self):
            print(f"  [GravityComp] loaded: g={self._g}, mass={self._mass}")

        def on_step(self, state, action, dt: float):
            theta = float(state[0]) if len(state) > 0 else 0.0
            l = 0.3
            torque = self._mass * self._g * l * np.sin(theta)
            self._step_count += 1
            self._total_correction += abs(torque)
            return {"gravity_torque": torque, "step": self._step_count}

        def on_unload(self):
            avg = self._total_correction / max(self._step_count, 1)
            print(f"  [GravityComp] unloaded after {self._step_count} steps, avg torque = {avg:.4f} Nm")

    HAS_SDK = True

except ImportError as e:
    HAS_SDK = False
    print(f"\n  [INFO] Plugin SDK not available ({e}). Running simplified demo.\n")


# ── 2. Simplified inline plugin (no SDK dependency) ───────────────────────────
class SimpleGravityComp:
    """Minimal plugin demo without SDK."""

    def __init__(self, g=9.81, mass=1.0):
        self._g, self._mass = g, mass
        self._steps = 0

    def step(self, state, dt):
        theta = float(state[0])
        torque = self._mass * self._g * 0.3 * np.sin(theta)
        self._steps += 1
        return {"gravity_torque": torque}


# ── 3. Demo with or without SDK ───────────────────────────────────────────────
from physicore import PhysiCore, PLATFORM_DYNAMICS

params = {"mass": 1.2, "friction": 0.1, "inertia": 0.015}
engine = PhysiCore.for_platform("balancing_bot", initial_params=params)
dyn_fn = PLATFORM_DYNAMICS["balancing_bot"][0]
_dt = 1 / engine.cfg.control_hz

if HAS_SDK:
    plugin = GravityCompPlugin(g=9.81, mass=params["mass"])
    loader = PluginLoader()
    loader.load_inline(plugin)
    print(f"\n  Plugin loaded: {plugin.meta.name} v{plugin.meta.version}")
else:
    plugin = SimpleGravityComp(g=9.81, mass=params["mass"])

# ── 4. Run with plugin ────────────────────────────────────────────────────────
x = np.array([0.2, 0.0, 0.0, 0.0])
x_ref = np.zeros(4)
N = 100

print(f"\n  Running {N} steps with gravity compensation plugin...")
print(f"\n  {'Step':>4}  {'Angle (rad)':>12}  {'Grav torque':>12}  {'Residual':>10}")
print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*10}")

for t in range(N):
    step_result = engine.step(x, x_ref)

    if HAS_SDK:
        plugin_out = plugin.on_step(x, step_result.action, _dt)
    else:
        plugin_out = plugin.step(x, _dt)

    torque = plugin_out["gravity_torque"]
    action_with_comp = step_result.action + np.array([torque * 0.1])

    xdot = dyn_fn(x, action_with_comp, params)
    x = x + _dt * xdot

    if t % 20 == 0:
        print(f"  {t:>4}  {x[0]:>+12.4f}  {torque:>+12.4f}  {step_result.residual_norm:>10.6f}")

print(f"\n  Final angle: {x[0]:.4f} rad ({np.degrees(x[0]):.2f} deg)")

if HAS_SDK:
    plugin.on_unload()
    loader.unload("gravity-comp")

# ── 5. Plugin scaffold ────────────────────────────────────────────────────────
print(f"\n{'-'*60}")
print("  Plugin scaffold (save as my_plugin.py):")
print(f"{'-'*60}")
scaffold = '''
from physicore.sdk.plugin_loader import PhysicorePlugin, PluginMeta

class MyPlugin(PhysicorePlugin):
    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(id="my-plugin", name="My Plugin",
                          version="1.0.0", description="...", author="you")

    def on_load(self): pass
    def on_step(self, state, action, dt): return {}
    def on_unload(self): pass
'''
print(scaffold)
print("Custom plugin demo complete.")

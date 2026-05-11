<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

<div align="center">

[![CI](https://github.com/prathameshbharatiya/physicore-hybrid-mpc/actions/workflows/ci.yml/badge.svg)](https://github.com/prathameshbharatiya/physicore-hybrid-mpc/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/physicore.svg)](https://pypi.org/project/physicore/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com)

**Hybrid Uncertainty-Aware Sim-to-Real Synchronization Engine**

[Quick Start](#quick-start) · [Documentation](docs/index.html) · [Examples](examples/) · [API Reference](docs/api-reference.html) · [Plugin SDK](docs/plugin-sdk.html)

</div>

---

PhysiCore is a production-grade **Model Predictive Control (MPC)** engine for real robots. It combines physics-based dynamics, residual neural networks, online system identification, and CEM optimization into a single cohesive runtime — from simulation to deployment in one command.

## Features

| Feature | Description |
|---|---|
| **Hybrid MPC** | CEM optimizer + physics layer + residual ensembles in one loop |
| **Online System ID** | Real-time mass/drag/friction estimation from observations |
| **12 Built-in Platforms** | Quadrotor, balancing bot, legged robot, satellite, rocket, AUV… |
| **URDF Loading** | Parse any robot description, build numerical FK + Jacobian |
| **Fleet Manager** | Multi-robot coordination with per-robot health monitoring |
| **Perception Fusion** | EKF-based multi-sensor fusion (pose, depth, IMU, encoders) |
| **Trajectory Planning** | Joint/task space, waypoints, circular arcs, collision avoidance |
| **REST API + WebSocket** | FastAPI server with live telemetry streaming |
| **React Dashboard** | Real-time visualisation, fleet control, telemetry replay |
| **Plugin Marketplace** | Sandboxed plugins with AST safety scanning |
| **Multi-tenant** | Orgs, roles, quotas, audit logs, usage metering |
| **CLI** | `physicore serve`, `robot new`, `plugins install`, `fleet status`… |

---

## Quick Start

```bash
# Option A — one-line installer (Linux / macOS)
curl -fsSL https://raw.githubusercontent.com/prathameshbharatiya/physicore-hybrid-mpc/main/install.sh | bash

# Option B — from source
git clone https://github.com/prathameshbharatiya/physicore-hybrid-mpc.git
cd physicore-hybrid-mpc
python install.py
npm install

# Option C — pip
pip install physicore[all]
```

**Start everything:**

```bash
physicore serve          # API on :8000
npm run dev              # Dashboard on :5173
```

**Docker:**

```bash
docker compose up        # API + built frontend on :8000
```

---

## 30-Second Example

```python
import numpy as np
from physicore import PhysiCore, PLATFORM_DYNAMICS

# Build MPC engine for a balancing bot
engine = PhysiCore.for_platform(
    "balancing_bot",
    params={"mass": 1.0, "friction": 0.15, "inertia": 0.01}
)

# Simulate 200 steps from a tilted initial condition
x = np.array([0.15, 0.0, 0.0, 0.0])   # [angle, angular_vel, pos, vel]

for t in range(200):
    step = engine.step(x, np.zeros(1))
    xdot = PLATFORM_DYNAMICS["balancing_bot"](x, step.action, engine.platform_params)
    x += engine.cfg.dt * xdot

print(f"Final angle: {x[0]:.4f} rad  ({step.residual_norm:.5f} residual)")
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Browser / Client                         │
│   React 18 + TypeScript + Tailwind  │  WebSocket telemetry  │
└──────────────────┬───────────────────────────────┬───────────┘
                   │ HTTP REST                      │ WS
┌──────────────────▼───────────────────────────────▼───────────┐
│                   FastAPI  (port 8000)                        │
│  /api/fleet/*   /api/plan/*   /api/marketplace/*             │
│  /api/orgs/*    /api/telemetry/*   /ws/telemetry             │
└──────────────────┬────────────────────────────────────────────┘
                   │
┌──────────────────▼────────────────────────────────────────────┐
│                    PhysiCore Engine                           │
│  CEMOptimizer · OnlineSystemID · ResidualEnsemble            │
│  PhysicsLayer · StateEstimator (EKF) · SentinelOS            │
└──────────────────┬────────────────────────────────────────────┘
                   │
┌──────────────────▼────────────────────────────────────────────┐
│                    Platform Layer                             │
│  12 dynamics functions · URDF loader · Custom dynamics fn    │
└───────────────────────────────────────────────────────────────┘
```

---

## Supported Platforms

| Platform | State | Action | Notes |
|---|---|---|---|
| `balancing_bot` | 4 | 1 | Inverted pendulum |
| `quadrotor` | 12 | 4 | 3D flight |
| `fixed_wing` | 12 | 4 | Aerodynamic model |
| `evtol` | 12 | 6 | Tilt-rotor VTOL |
| `manipulator_arm` | 2n | n | n-DOF serial arm |
| `legged_robot` | 18 | 12 | Quadruped |
| `ground_rover` | 6 | 2 | Differential drive |
| `rocket` | 12 | 4 | TVC launch vehicle |
| `auv` | 12 | 6 | Underwater vehicle |
| `satellite` | 7 | 3 | Attitude control |
| `rover` | 8 | 4 | Planetary rover |
| `surgical_robot` | 14 | 7 | Cable-driven arm |

Plus 4 composite platforms: `mobile_manipulator`, `dual_arm`, `cable_driven`, `exoskeleton`.

---

## CLI

```bash
physicore serve                          # Start API server
physicore serve --host 0.0.0.0 --reload  # Hot-reload dev mode
physicore status                         # System health
physicore run balancing_bot_sim          # Run an example
physicore robot new my_arm --platform manipulator_arm
physicore robot load my_arm.yaml
physicore plugins list
physicore plugins install gravity-comp
physicore plugins new my-plugin --author alice
physicore fleet status
physicore data sessions
physicore data export <session-id> --format csv
physicore docs quickstart
physicore version
```

---

## Examples

| Script | Description |
|---|---|
| [`balancing_bot_sim.py`](examples/balancing_bot_sim.py) | 200-step balancing bot with ASCII trajectory |
| [`quadrotor_sim.py`](examples/quadrotor_sim.py) | 3D quadrotor hover from below |
| [`load_any_urdf.py`](examples/load_any_urdf.py) | URDF loading, FK, Jacobian inspection |
| [`fleet_two_robots.py`](examples/fleet_two_robots.py) | Two robots running in parallel |
| [`custom_plugin.py`](examples/custom_plugin.py) | Inline gravity compensation plugin |
| [`full_pipeline.py`](examples/full_pipeline.py) | End-to-end: perception → planning → MPC → telemetry |

```bash
physicore run balancing_bot_sim
python examples/full_pipeline.py
```

---

## Plugin SDK

```python
from physicore.sdk.plugin_loader import PhysicorePlugin, PluginMeta

class GravityCompPlugin(PhysicorePlugin):
    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(id="gravity-comp", name="Gravity Comp",
                          version="1.0.0", description="...", author="alice")

    def on_load(self): pass

    def on_step(self, state, action, dt):
        torque = 9.81 * 0.3 * state[0]   # gravity torque
        return {"gravity_torque": torque}

    def on_unload(self): pass
```

```bash
physicore plugins new my-plugin --author alice
```

Every plugin is AST-scanned before installation — `socket`, `subprocess`, `eval`, `exec`, and network libs are blocked.

---

## Documentation

Open the docs locally:

```bash
physicore docs            # opens docs/index.html
physicore docs quickstart
physicore docs api-reference
```

Or browse the HTML files in [`docs/`](docs/):

- [Introduction](docs/index.html)
- [Quick Start](docs/quickstart.html)
- [Architecture](docs/architecture.html)
- [Platforms](docs/platforms.html)
- [Robot Loading](docs/robot-loading.html)
- [API Reference](docs/api-reference.html)
- [Plugin SDK](docs/plugin-sdk.html)
- [Safety](docs/safety.html)
- [Deployment](docs/deployment.html)

---

## Development

```bash
# Run tests
pytest tests/ -v

# Type check
npm run lint    # or: npx tsc --noEmit

# Run specific test modules
pytest tests/test_phase6.py tests/test_phase7.py tests/test_launch_readiness.py -v
```

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feat/my-feature`
3. Run tests: `pytest tests/ -v && npm run lint`
4. Submit a PR

Please follow the existing code style and add tests for new features.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built by <a href="https://github.com/prathameshbharatiya">Prathamesh Shirbhate</a> · <a href="https://physicore.ai">physicore.ai</a>
</div>

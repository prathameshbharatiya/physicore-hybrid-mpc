<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# PhysiCore — Hybrid MPC Engine

A production-grade Model Predictive Control engine with online system identification, residual learning, and multi-robot fleet management. Runs a FastAPI backend and React dashboard in one command.

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/prathameshbharatiya/physicore-hybrid-mpc.git && cd physicore-hybrid-mpc

# 2. Launch backend + frontend (installs deps automatically)
./scripts/start.sh

# 3. Open the dashboard
open http://localhost:5173
```

> **Docker alternative:** `docker compose up` — serves everything on port 8000.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser / Client                         │
│   React 18 + TypeScript + Tailwind  │  WebSocket (port 8765)   │
└──────────────────┬──────────────────┴────────────┬─────────────┘
                   │ HTTP REST                      │ Telemetry
┌──────────────────▼────────────────────────────────▼─────────────┐
│                    FastAPI  (port 8000)                          │
│  /api/platforms  /fleet/*  /health  /ws/telemetry               │
└──────────────────┬───────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────────┐
│                     PhysiCore Engine                             │
│                                                                   │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────────┐  │
│  │ CEMOptimizer │  │ OnlineSystemID│  │  ResidualEnsemble     │  │
│  │  (MPC loop)  │  │  (mass, drag) │  │  (neural residuals)   │  │
│  └──────────────┘  └───────────────┘  └───────────────────────┘  │
│                                                                   │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────────┐  │
│  │ ModelRegistry│  │  SentinelOS   │  │   FleetManager        │  │
│  │ (persistence)│  │  (safety)     │  │   (multi-robot)       │  │
│  └──────────────┘  └───────────────┘  └───────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────────┐
│             Platform Dynamics Library                            │
│  quadrotor · humanoid · car · rocket · auv · rover              │
│  mobile_manipulator · dual_arm · cable_driven · exoskeleton      │
│  + any URDF / MJCF robot via load_robot()                        │
└───────────────────────────────────────────────────────────────────┘
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check — returns `{"status":"ok"}` |
| `GET` | `/api/platforms` | List all registered platform names |
| `POST` | `/api/optimize` | Run one MPC step; body: `{platform, state, goal, config?}` |
| `GET` | `/fleet/health` | Health snapshot for all fleet robots |
| `GET` | `/fleet/robots` | List robot IDs currently in the fleet |
| `POST` | `/fleet/add` | Add a robot; body: `{robot_id, platform}` or `{robot_id, urdf_path}` |
| `DELETE` | `/fleet/robot/{id}` | Remove a robot from the fleet |
| `WS` | `/ws/telemetry` | Real-time telemetry stream (JSON frames per robot) |

### `/api/optimize` request body

```json
{
  "platform": "quadrotor",
  "state": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "goal":  [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "config": {
    "horizon": 20,
    "n_samples": 512,
    "dt": 0.05
  }
}
```

---

## Loading Your Own Robot

Point `load_robot()` at any URDF or MJCF file and get a fully configured `PhysiCore` engine back in one call:

```python
from physicore import load_robot

engine = load_robot("path/to/my_robot.urdf")
action = engine.step(state, goal)
```

The loader auto-detects DOF, extracts joint limits, builds the kinematic chain, and wires up the CEM optimizer. No config required.

**With custom MPC settings:**

```python
from physicore import load_robot, PhysiCoreConfig

config = PhysiCoreConfig(horizon=30, n_samples=1024, dt=0.02)
engine = load_robot("my_robot.urdf", config=config)
```

**Register it for REST API access:**

```python
from physicore import PhysiCore
PhysiCore.register_platform("my_robot", "path/to/my_robot.urdf")
```

After registration, `POST /api/optimize` with `"platform": "my_robot"` works immediately.

---

## Supported Platforms

| Key | DOF | State dim | Description |
|-----|-----|-----------|-------------|
| `quadrotor` | 4 | 12 | Position + velocity + attitude |
| `humanoid` | 12 | 24 | Full-body with contact model |
| `car` | 2 | 6 | Bicycle model |
| `rocket` | 3 | 9 | Thrust-vector control |
| `auv` | 5 | 10 | Underwater vehicle |
| `rover` | 2 | 6 | Differential drive |
| `mobile_manipulator` | 6 | 14 | Mobile base + 6-DOF arm |
| `dual_arm` | 14 | 20 | Bimanual manipulation |
| `cable_driven` | 6 | 12 | Tendon-driven robot |
| `exoskeleton` | 10 | 16 | Human-assist wearable |

---

## Environment Variables

Copy `.env.local.example` to `.env.local` and fill in the values:

```bash
GEMINI_API_KEY=your_key_here
VITE_FIREBASE_API_KEY=your_key_here
VITE_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your_project_id
```

For Docker deployment, pass these via `docker compose` environment or a `.env` file — never hardcode them.

---

## Development

```bash
# Run Python tests
pytest tests/ -v

# TypeScript type check
npm run lint

# Build for production
npm run build

# Health check against a running instance
./scripts/healthcheck.sh localhost 8000
```

## License

MIT

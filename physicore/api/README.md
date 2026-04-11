# PhysiCore API Server

Connects the PhysiCore engine to the dashboard and any external integration.

## Start

```bash
pip install fastapi uvicorn
uvicorn physicore.api.server:app --host 0.0.0.0 --port 8000
```

## Quick integration example

```python
import requests
import numpy as np

# Configure engine for your platform
requests.post("http://localhost:8000/api/engine/configure", json={
    "platform": "quadrotor",
    "initial_params": {"mass": 1.5, "friction": 0.1, "inertia": 0.02},
    "control_hz": 60.0
})

# Control loop
state = np.zeros(12).tolist()
x_ref = [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0]

while True:
    # Get optimal action
    resp = requests.post("http://localhost:8000/api/engine/step", json={
        "state": state,
        "x_ref": x_ref
    }).json()
    
    action = resp["action"]
    # Apply to hardware...
    
    # Feed back real observation
    next_state = state  # replace with real sensor reading
    requests.post("http://localhost:8000/api/engine/observe", json={
        "state": state,
        "action": action,
        "next_state": next_state
    })
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Health check |
| GET | /api/status | Full engine diagnostics |
| POST | /api/engine/configure | Configure for a platform |
| POST | /api/engine/step | Get optimal action |
| POST | /api/engine/observe | Feed real transition |
| GET | /api/engine/params | Current estimated params |
| GET | /api/engine/residual | Current residual norm |
| GET | /api/engine/uncertainty | Current uncertainty |
| POST | /api/engine/reset | Reset engine |
| GET | /api/platforms | List all platforms |
| WS | /ws/telemetry | Live diagnostics stream |

## Supported platforms

- `quadrotor` — PX4/ArduPilot drones
- `fixed_wing` — Fixed wing aircraft
- `manipulator_arm` — 6-DOF robot arms
- `balancing_bot` — Self-balancing robots
- `rocket` — Sounding rockets
- `ground_rover` — Differential drive robots

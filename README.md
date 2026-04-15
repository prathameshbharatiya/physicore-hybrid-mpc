# PhysiCore

**Physics Intelligence Engine — real-time sim-to-real synchronization for autonomous systems.**

Robots trained in simulation fail on real hardware. PhysiCore closes that gap automatically at 60Hz. Connects to any hardware in 30 minutes. Learns your real physics from sensor data. No retraining.

## Supported platforms

`quadrotor` · `fixed_wing` · `evtol` · `manipulator_arm` · `surgical_robot` · `legged_robot` · `balancing_bot` · `rocket` · `ground_rover` · `auv` · `satellite`

## Quick start

```bash
pip install pymavlink websockets aiohttp pyserial

# Balancing bot (Arduino + MPU6050)
python physicore/bridge/physicore_bridge.py --platform balancing_bot_arduino --connection COM3

# PX4 drone
python physicore/bridge/physicore_bridge.py --platform px4_quadrotor --connection udp:14550

# Custom rocket flight computer
python physicore/bridge/physicore_bridge.py --platform custom_rocket_fc --connection /dev/ttyUSB0 --baud 115200
```

## How it works

1. Connect your hardware to the bridge in one command
2. Open the dashboard, click MAVLINK, connect to `ws://localhost:8765`
3. Click **ACTIVE CONTROL ON**
4. PhysiCore learns your real hardware mass, friction, and dynamics from sensor data in real time

## Architecture

- **RK4 Physics Core** — 4th-order integration, accurate to machine precision
- **Residual Ensemble** — 3 neural networks learn what your simulator got wrong
- **CEM-MPC Optimizer** — optimal control action every 16.7ms
- **Online SystemID** — learns your real hardware parameters from sensor data, converges in seconds
- **Sentinel OS** — mathematical safety layer, Lyapunov enforcement, SHA-256 forensic log

## Built by

Prathamesh Shirbhate — Founders Inc '26 · Momentum by DevLabs '26

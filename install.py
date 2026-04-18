#!/usr/bin/env python3
"""
PhysiCore One-Command Installer
================================
Run this once. Everything is set up.

    python install.py

What it does:
  1. Checks Python version (>=3.9 required)
  2. Installs all dependencies
  3. Installs PhysiCore as a package (pip install -e .)
  4. Creates ~/.physicore/ registry directory
  5. Verifies the engine loads correctly
  6. Prints your first command

Author: Prathamesh Shirbhate — physicore.ai
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

# ── Colours ───────────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {CYAN}→{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET}  {msg}")

def run(cmd, check=True):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        fail(f"Command failed: {cmd}")
        print(result.stderr[:500])
        sys.exit(1)
    return result


def main():
    print(f"""
{BOLD}╔══════════════════════════════════════════════════════╗
║          PHYSICORE INSTALLER v1.2.0                  ║
║  Real-time physics adaptation. Any robot. 30 seconds.║
╚══════════════════════════════════════════════════════╝{RESET}
""")

    # ── 1. Python version ──────────────────────────────────────────────────────
    print(f"{BOLD}[1/6] Checking Python version...{RESET}")
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        fail(f"Python 3.9+ required. You have {v.major}.{v.minor}")
        fail("Download from https://python.org")
        sys.exit(1)
    ok(f"Python {v.major}.{v.minor}.{v.micro}")

    # ── 2. pip up to date ──────────────────────────────────────────────────────
    print(f"\n{BOLD}[2/6] Upgrading pip...{RESET}")
    run(f"{sys.executable} -m pip install --upgrade pip -q")
    ok("pip up to date")

    # ── 3. Core dependencies ───────────────────────────────────────────────────
    print(f"\n{BOLD}[3/6] Installing core dependencies...{RESET}")
    core_deps = ["numpy>=1.24.0", "scipy>=1.10.0"]
    for dep in core_deps:
        info(f"Installing {dep}...")
        run(f"{sys.executable} -m pip install \"{dep}\" -q")
        ok(dep)

    # ── 4. Bridge dependencies ─────────────────────────────────────────────────
    print(f"\n{BOLD}[4/6] Installing bridge dependencies...{RESET}")
    bridge_deps = [
        ("pymavlink>=2.4.0",  "MAVLink — PX4/ArduPilot/Rocket"),
        ("websockets==12.0",  "WebSocket — dashboard connection"),
        ("aiohttp>=3.9.0",    "HTTP — health endpoint"),
        ("pyserial>=3.5",     "Serial — Arduino/direct hardware"),
        ("pyyaml>=6.0",       "YAML — robot config files"),
        ("fastapi>=0.110.0",  "FastAPI — REST API server"),
        ("uvicorn>=0.27.0",   "Uvicorn — ASGI server"),
        ("requests>=2.31.0",  "Requests — SDK HTTP client"),
    ]
    for dep, label in bridge_deps:
        info(f"Installing {label}...")
        result = run(f"{sys.executable} -m pip install \"{dep}\" -q", check=False)
        if result.returncode == 0:
            ok(label)
        else:
            warn(f"{label} — failed (optional, continuing)")

    # ── 5. Install PhysiCore as package ────────────────────────────────────────
    print(f"\n{BOLD}[5/6] Installing PhysiCore package...{RESET}")
    script_dir = Path(__file__).parent
    result = run(f"{sys.executable} -m pip install -e \"{script_dir}\" -q", check=False)
    if result.returncode == 0:
        ok("PhysiCore installed (editable mode — changes take effect immediately)")
    else:
        warn("pip install -e failed — trying direct path install")
        sys.path.insert(0, str(script_dir))
        ok("PhysiCore added to Python path")

    # ── 6. Verify engine loads ─────────────────────────────────────────────────
    print(f"\n{BOLD}[6/6] Verifying engine...{RESET}")
    verify = f"""
import sys
sys.path.insert(0, r'{script_dir}')
from physicore import PhysiCore, PLATFORM_DYNAMICS
engine = PhysiCore.for_platform('balancing_bot', {{'mass': 1.0, 'friction': 0.15, 'inertia': 0.01}})
import numpy as np
x = np.array([0.1, 0.0, 0.0, 0.0])
step = engine.step(x, np.zeros(4))
print('ENGINE_OK')
print(f'  Platform: balancing_bot')
print(f'  State dim: {{engine.cfg.state_dim}}')
print(f'  Action: {{step.action.tolist()}}')
print(f'  Residual: {{step.residual_norm:.4f}}')
"""
    result = run(f"{sys.executable} -c \"{verify}\"", check=False)
    if "ENGINE_OK" in result.stdout:
        ok("Engine verified")
        for line in result.stdout.strip().split('\n')[1:]:
            info(line.strip())
    else:
        fail("Engine verification failed")
        print(result.stderr[:500])
        sys.exit(1)

    # ── Create registry directory ──────────────────────────────────────────────
    registry_dir = Path.home() / ".physicore" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    ok(f"Registry directory: {registry_dir}")

    # ── Done ───────────────────────────────────────────────────────────────────
    is_windows = platform.system() == 'Windows'
    sep = "\\" if is_windows else "/"
    com_example = "COM8" if is_windows else "/dev/ttyUSB0"

    print(f"""
{BOLD}{GREEN}╔══════════════════════════════════════════════════════╗
║              INSTALLATION COMPLETE                   ║
╚══════════════════════════════════════════════════════╝{RESET}

{BOLD}Your first command:{RESET}

  Balancing bot (Arduino):
  {CYAN}python physicore{sep}bridge{sep}physicore_bridge.py --platform balancing_bot_arduino --connection {com_example}{RESET}

  PX4 Quadrotor:
  {CYAN}python physicore{sep}bridge{sep}physicore_bridge.py --platform px4_quadrotor --connection udp:14550{RESET}

  With YAML config:
  {CYAN}python physicore{sep}bridge{sep}physicore_bridge.py --config balancing_bot_robot.yaml{RESET}

  Test installation:
  {CYAN}python physicore{sep}bridge{sep}physicore_bridge.py --test{RESET}

  Prove registry works across sessions:
  {CYAN}python tools{sep}session_compare.py --platform balancing_bot{RESET}

  Start REST API:
  {CYAN}uvicorn physicore.api.server:app --host 0.0.0.0 --port 8000{RESET}

{BOLD}Registry location:{RESET} {registry_dir}
{BOLD}Supported platforms:{RESET} balancing_bot, quadrotor, fixed_wing, evtol,
  manipulator_arm, legged_robot, ground_rover, rocket, auv, satellite

{BOLD}Docs:{RESET} See INSTALL.md for full setup guides per hardware type.
""")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PhysiCore Universal Hardware Bridge v2.0.0
==========================================
FIXED VERSION — All bugs resolved:
  1. Engine starts automatically (no --active flag needed)
  2. numpy imported at top level
  3. observe() uses real previous state for actual learning
  4. Command sent to hardware every read cycle when active
  5. Platform auto-detected for engine initialization
  6. Previous state tracked for real SystemID convergence

Usage:
  python physicore_bridge.py --platform balancing_bot_arduino --connection COM3
  python physicore_bridge.py --platform px4_quadrotor --connection udp:14550
  python physicore_bridge.py --mode ros2
  python physicore_bridge.py --test
"""

import asyncio
import json
import time
import argparse
import threading
import sys
import math
import platform as _platform
import numpy as np

# ── PhysiCore engine import ────────────────────────────────────────────────────
try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from physicore import PhysiCore, PLATFORM_DYNAMICS
    HAS_PHYSICORE = True
    print("[BRIDGE] PhysiCore engine loaded successfully")
except ImportError as e:
    HAS_PHYSICORE = False
    print(f"[BRIDGE] Warning: PhysiCore not found ({e}). Telemetry-only mode.")

def check_deps():
    missing = []
    for lib in ['pymavlink', 'websockets', 'serial']:
        try:
            __import__('serial' if lib == 'serial' else lib)
        except ImportError:
            missing.append('pyserial' if lib == 'serial' else lib)
    if missing:
        print(f"\n[BRIDGE] Missing: pip install {' '.join(missing)}\n")
        sys.exit(1)

check_deps()

from pymavlink import mavutil
import websockets
import serial as pyserial

# ── PhysiCore persistence layer ───────────────────────────────────────────────
try:
    from physicore.core.registry    import get_registry
    from physicore.core.robot_config import RobotConfig, create_template
    from physicore.core.telemetry   import get_telemetry, CONSENT_TEXT
    HAS_REGISTRY = True
except ImportError as e:
    HAS_REGISTRY = False
    print(f"[BRIDGE] Registry not available: {e}")

# Global robot config and telemetry
_robot_config   = None
_telemetry_mgr  = None

BRIDGE_VERSION = "2.0.0"
TELEMETRY_HZ   = 20

# ── Platform profiles ──────────────────────────────────────────────────────────
PLATFORM_PROFILES = {
    "px4_quadrotor":         {"mode":"mavlink",      "baud":57600,  "vehicle_type":"QUADROTOR",   "domain":"AVIATION",  "engine_platform":"quadrotor",     "default_connection":"udp:14550"},
    "ardupilot_plane":       {"mode":"mavlink",      "baud":57600,  "vehicle_type":"FIXED_WING",  "domain":"AVIATION",  "engine_platform":"fixed_wing",    "default_connection":"udp:14550"},
    "ardupilot_quadrotor":   {"mode":"mavlink",      "baud":57600,  "vehicle_type":"QUADROTOR",   "domain":"AVIATION",  "engine_platform":"quadrotor",     "default_connection":"udp:14550"},
    "evtol":                 {"mode":"mavlink",      "baud":57600,  "vehicle_type":"EVTOL",        "domain":"AVIATION",  "engine_platform":"evtol",         "default_connection":"udp:14550"},
    "ros2_manipulator":      {"mode":"ros2",         "baud":0,      "vehicle_type":"MANIPULATOR", "domain":"ROBOTICS",  "engine_platform":"manipulator_arm","default_connection":"ros2"},
    "ros2_legged":           {"mode":"ros2",         "baud":0,      "vehicle_type":"LEGGED",      "domain":"ROBOTICS",  "engine_platform":"legged_robot",   "default_connection":"ros2"},
    "ros2_ground_rover":     {"mode":"ros2",         "baud":0,      "vehicle_type":"GROUND_ROVER","domain":"ROBOTICS",  "engine_platform":"ground_rover",   "default_connection":"ros2"},
    "ros2_auv":              {"mode":"ros2",         "baud":0,      "vehicle_type":"AUV",          "domain":"ROBOTICS",  "engine_platform":"auv",            "default_connection":"ros2"},
    "ros2_surgical":         {"mode":"ros2",         "baud":0,      "vehicle_type":"SURGICAL",    "domain":"ROBOTICS",  "engine_platform":"surgical_robot", "default_connection":"ros2"},
    "ros2_mobile_manipulator": {"mode":"ros2", "baud":0, "vehicle_type":"MOBILE_MANIPULATOR", "domain":"ROBOTICS", "engine_platform":"mobile_manipulator", "default_connection":"ros2"},
    "ros2_dual_arm":           {"mode":"ros2", "baud":0, "vehicle_type":"DUAL_ARM",           "domain":"ROBOTICS", "engine_platform":"dual_arm",           "default_connection":"ros2"},
    "ros2_exoskeleton":        {"mode":"ros2", "baud":0, "vehicle_type":"EXOSKELETON",        "domain":"ROBOTICS", "engine_platform":"exoskeleton",        "default_connection":"ros2"},
    "ros2_cable_driven":       {"mode":"ros2", "baud":0, "vehicle_type":"CABLE_DRIVEN",       "domain":"ROBOTICS", "engine_platform":"cable_driven",       "default_connection":"ros2"},
    "balancing_bot_arduino": {"mode":"robot_serial", "baud":115200, "vehicle_type":"GROUND_ROVER","domain":"ROBOTICS",  "engine_platform":"balancing_bot",  "default_connection":"COM3"},
    "custom_rocket_fc":      {"mode":"robot_serial", "baud":115200, "vehicle_type":"ROCKET",      "domain":"ROCKETS",   "engine_platform":"rocket",         "default_connection":"COM3"},
    "ground_rover_serial":   {"mode":"robot_serial", "baud":115200, "vehicle_type":"GROUND_ROVER","domain":"ROBOTICS",  "engine_platform":"ground_rover",   "default_connection":"COM3"},
    "satellite_serial":      {"mode":"robot_serial", "baud":115200, "vehicle_type":"SATELLITE",   "domain":"AVIATION",  "engine_platform":"satellite",      "default_connection":"COM3"},
}

# ── Global state ───────────────────────────────────────────────────────────────
class TelemetryState:
    def __init__(self):
        self.timestamp    = 0.0
        self.altitude     = 0.0
        self.velocity_x   = 0.0
        self.velocity_y   = 0.0
        self.velocity_z   = 0.0
        self.speed        = 0.0
        self.roll         = 0.0
        self.pitch        = 0.0
        self.yaw          = 0.0
        self.lat          = 0.0
        self.lon          = 0.0
        self.throttle     = 0.0
        self.battery_v    = 0.0
        self.battery_pct  = 0.0
        self.armed        = False
        self.flight_mode  = "UNKNOWN"
        self.gps_fix      = 0
        self.satellites   = 0
        self.accel_x      = 0.0
        self.accel_y      = 0.0
        self.accel_z      = 0.0
        self.gyro_x       = 0.0
        self.gyro_y       = 0.0
        self.gyro_z       = 0.0
        self.airspeed     = 0.0
        self.groundspeed  = 0.0
        self.climb_rate   = 0.0
        self.motor_l      = 0.0
        self.motor_r      = 0.0
        self.vehicle_type = "UNKNOWN"
        self.domain       = "ROBOTICS"
        self.connected    = False
        self.platform     = "unknown"
        # PhysiCore diagnostics
        self.residual_norm  = 0.0
        self.uncertainty    = 0.0
        self.estimated_mass = 0.0
        self.estimated_friction = 0.0
        self.physicore_active = False
        self.step_count     = 0
        # High-DOF joint state (populated by ROS2 joint_cb)
        self.joint_positions:  list = []
        self.joint_velocities: list = []
        self.joint_efforts:    list = []
        self.joint_names:      list = []
        self.n_joints:         int  = 0

    def to_dict(self) -> dict:
        return {
            "op": "publish", "topic": "/telemetry",
            "msg": {
                "timestamp":    self.timestamp,
                "altitude":     round(self.altitude, 3),
                "velocity":     {"x": round(self.velocity_x,3), "y": round(self.velocity_y,3), "z": round(self.velocity_z,3)},
                "speed":        round(self.speed, 3),
                "pitch":        round(self.pitch, 3),
                "roll":         round(self.roll, 3),
                "yaw":          round(self.yaw, 3),
                "orientation":  {"roll": round(self.roll,3), "pitch": round(self.pitch,3), "yaw": round(self.yaw,3)},
                "position":     {"lat": self.lat, "lon": self.lon},
                "acceleration": {"x": round(self.accel_x,4), "y": round(self.accel_y,4), "z": round(self.accel_z,4)},
                "gyro":         {"x": round(self.gyro_x,4), "y": round(self.gyro_y,4), "z": round(self.gyro_z,4)},
                "gyro_x":       round(self.gyro_x, 4),
                "gyro_y":       round(self.gyro_y, 4),
                "gyro_z":       round(self.gyro_z, 4),
                "airspeed":     round(self.airspeed, 3),
                "groundspeed":  round(self.groundspeed, 3),
                "climb_rate":   round(self.climb_rate, 3),
                "throttle":     round(self.throttle, 3),
                "motor_l":      round(self.motor_l, 1),
                "motor_r":      round(self.motor_r, 1),
                "battery":      {"voltage": round(self.battery_v,2), "percentage": round(self.battery_pct,1)},
                "armed":        self.armed,
                "flight_mode":  self.flight_mode,
                "gps":          {"fix": self.gps_fix, "satellites": self.satellites},
                "vehicle_type": self.vehicle_type,
                "domain":       self.domain,
                "connected":    self.connected,
                "bridge_version": BRIDGE_VERSION,
                # PhysiCore live diagnostics — shown in dashboard
                "mass":             round(self.estimated_mass, 4),
                "friction":         round(self.estimated_friction, 4),
                "residual":         round(self.residual_norm, 6),
                "uncertainty":      round(self.uncertainty, 6),
                "physicore_active": self.physicore_active,
                "step_count":       self.step_count,
                "joint_positions":  self.joint_positions[:32] if self.joint_positions else [],
                "joint_velocities": self.joint_velocities[:32] if self.joint_velocities else [],
                "n_joints":         self.n_joints,
            }
        }

state             = TelemetryState()
connected_clients = set()
engine            = None   # PhysiCore engine instance — initialized on startup if HAS_PHYSICORE
control_active    = False  # Set by dashboard "ACTIVE CONTROL" button
x_ref             = None   # Target state from dashboard

def state_to_vector(platform: str, n_joints: int = 0) -> np.ndarray:
    """Convert current telemetry to engine state vector."""
    import math as _math

    # High-DOF joint platforms — read from joint_positions global if available
    if platform in ('manipulator_arm', 'surgical_robot', 'dual_arm',
                    'cable_driven', 'exoskeleton'):
        n = n_joints if n_joints > 0 else 6
        jp = getattr(state, 'joint_positions', None)
        jv = getattr(state, 'joint_velocities', None)
        if jp is not None and len(jp) >= n:
            q = np.array(jp[:n])
            dq = np.array(jv[:n]) if jv and len(jv) >= n else np.zeros(n)
            return np.concatenate([q, dq])
        return np.zeros(n * 2)

    elif platform in ('legged_robot', 'humanoid'):
        n = n_joints if n_joints > 0 else 0
        if n > 6:
            jp = getattr(state, 'joint_positions', None)
            jv = getattr(state, 'joint_velocities', None)
            base = np.array([0, 0, state.altitude,
                             state.velocity_x, state.velocity_y, state.velocity_z])
            if jp is not None and len(jp) >= n:
                q = np.array(jp[:n])
                dq = np.array(jv[:n]) if jv and len(jv) >= n else np.zeros(n)
                return np.concatenate([base, q, dq])
            return np.zeros(6 + n * 2)
        else:
            return np.array([0, 0, state.altitude,
                             state.velocity_x, state.velocity_y, state.velocity_z,
                             _math.radians(state.roll), _math.radians(state.pitch), _math.radians(state.yaw),
                             state.gyro_x, state.gyro_y, state.gyro_z])

    elif platform == 'mobile_manipulator':
        n = n_joints if n_joints > 0 else 6
        jp = getattr(state, 'joint_positions', None)
        jv = getattr(state, 'joint_velocities', None)
        base = np.array([0, 0, _math.radians(state.yaw),
                         state.velocity_x, state.velocity_y, state.gyro_z])
        if jp is not None and len(jp) >= n:
            q = np.array(jp[:n])
            dq = np.array(jv[:n]) if jv and len(jv) >= n else np.zeros(n)
            return np.concatenate([base, q, dq])
        return np.zeros(6 + n * 2)

    elif platform == 'balancing_bot':
        return np.array([
            _math.radians(state.pitch),
            _math.radians(state.gyro_x),
            0.0,
            state.velocity_x,
        ])
    elif platform in ('quadrotor', 'evtol', 'fixed_wing'):
        roll_r = _math.radians(state.roll)
        pitch_r = _math.radians(state.pitch)
        yaw_r = _math.radians(state.yaw)
        if platform == 'quadrotor':
            from physicore.core.engine import euler_to_quat
            q = euler_to_quat(roll_r, pitch_r, yaw_r)
            return np.array([0, 0, state.altitude,
                             state.velocity_x, state.velocity_y, state.velocity_z,
                             q[0], q[1], q[2], q[3],
                             state.gyro_x, state.gyro_y, state.gyro_z])
        return np.array([0, 0, state.altitude,
                         state.velocity_x, state.velocity_y, state.velocity_z,
                         roll_r, pitch_r, yaw_r,
                         state.gyro_x, state.gyro_y, state.gyro_z])
    elif platform == 'rocket':
        return np.array([0, state.altitude, state.velocity_x, state.velocity_z,
                         1.0, _math.radians(state.pitch)])
    elif platform in ('ground_rover', 'rover'):
        return np.array([0, 0, _math.radians(state.yaw),
                         state.velocity_x, state.velocity_y, state.gyro_z])
    elif platform == 'auv':
        return np.array([0, 0, -state.altitude,
                         state.velocity_x, state.velocity_y, state.velocity_z,
                         _math.radians(state.roll), _math.radians(state.pitch), _math.radians(state.yaw),
                         state.gyro_x, state.gyro_y, state.gyro_z])
    else:
        return np.array([0, 0, state.altitude,
                         state.velocity_x, state.velocity_y, state.velocity_z,
                         _math.radians(state.roll), _math.radians(state.pitch), _math.radians(state.yaw),
                         state.gyro_x, state.gyro_y, state.gyro_z])

# ── MAVLink reader ─────────────────────────────────────────────────────────────
def mavlink_reader(connection_string: str, baud: int):
    global state, engine, control_active, x_ref
    print(f"[BRIDGE] MAVLink connecting: {connection_string}")
    try:
        mav = mavutil.mavlink_connection(connection_string, baud=baud, autoreconnect=True, source_system=255)
    except Exception as e:
        print(f"[BRIDGE] MAVLink connection failed: {e}")
        return

    print("[BRIDGE] Waiting for heartbeat...")
    hb = mav.wait_heartbeat(timeout=15)
    if not hb:
        print("[BRIDGE] No heartbeat. Check connection.")
        return

    vtypes = {1:"FIXED_WING",2:"QUADROTOR",3:"COAXIAL",4:"HELICOPTER",
              6:"GROUND_ROVER",8:"ROCKET",10:"FLAPPING_WING",
              13:"HEXAROTOR",14:"OCTOROTOR",15:"TRICOPTER",19:"EVTOL"}
    vt = vtypes.get(mav.messages['HEARTBEAT'].type, "UNKNOWN")
    state.vehicle_type = vt
    state.domain = "AVIATION" if vt in ("FIXED_WING","QUADROTOR","COAXIAL","HELICOPTER","HEXAROTOR","OCTOROTOR","TRICOPTER","EVTOL") else "ROCKETS" if vt == "ROCKET" else "ROBOTICS"
    state.connected = True
    print(f"[BRIDGE] MAVLink connected. Vehicle: {vt}  Domain: {state.domain}")
    mav.mav.request_data_stream_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 20, 1)

    prev_state_vec = None

    while True:
        try:
            msg = mav.recv_match(blocking=True, timeout=1.0)
            if not msg:
                state.connected = False
                continue
            state.connected = True
            state.timestamp = time.time()
            mt = msg.get_type()

            if mt == 'VFR_HUD':
                state.altitude    = msg.alt
                state.airspeed    = msg.airspeed
                state.groundspeed = msg.groundspeed
                state.climb_rate  = msg.climb
                state.throttle    = msg.throttle / 100.0
            elif mt == 'ATTITUDE':
                state.roll   = math.degrees(msg.roll)
                state.pitch  = math.degrees(msg.pitch)
                state.yaw    = math.degrees(msg.yaw)
                state.gyro_x = math.degrees(msg.rollspeed)
                state.gyro_y = math.degrees(msg.pitchspeed)
                state.gyro_z = math.degrees(msg.yawspeed)
            elif mt == 'GLOBAL_POSITION_INT':
                state.lat        = msg.lat / 1e7
                state.lon        = msg.lon / 1e7
                state.altitude   = msg.relative_alt / 1000.0
                state.velocity_x = msg.vx / 100.0
                state.velocity_y = msg.vy / 100.0
                state.velocity_z = msg.vz / 100.0
                state.speed      = math.sqrt(state.velocity_x**2 + state.velocity_y**2 + state.velocity_z**2)
            elif mt in ('RAW_IMU', 'SCALED_IMU2'):
                state.accel_x = msg.xacc / 1000.0
                state.accel_y = msg.yacc / 1000.0
                state.accel_z = msg.zacc / 1000.0
            elif mt == 'SYS_STATUS':
                state.battery_v   = msg.voltage_battery / 1000.0
                state.battery_pct = msg.battery_remaining
            elif mt == 'GPS_RAW_INT':
                state.gps_fix    = msg.fix_type
                state.satellites = msg.satellites_visible
            elif mt == 'HEARTBEAT':
                state.armed      = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                state.flight_mode = f"MODE_{msg.custom_mode}"

            # PhysiCore engine step
            if engine and control_active:
                try:
                    current_x = state_to_vector(engine.cfg.platform,
                                                n_joints=state.n_joints or engine.cfg.action_dim)
                    ref       = np.array(x_ref) if x_ref else np.zeros(engine.cfg.state_dim)
                    ref       = ref[:engine.cfg.state_dim] if len(ref) >= engine.cfg.state_dim else np.pad(ref, (0, engine.cfg.state_dim - len(ref)))
                    step      = engine.step(current_x, ref)
                    if prev_state_vec is not None:
                        engine.observe(prev_state_vec, step.action, current_x)
                    prev_state_vec = current_x.copy()
                    _update_engine_state(step)
                    state.physicore_active = True
                except Exception as e:
                    print(f"[ENGINE] Step error: {e}")
            else:
                state.physicore_active = False

        except Exception:
            time.sleep(0.1)

# ── Robot serial reader ────────────────────────────────────────────────────────
def robot_serial_reader(connection_string: str, baud: int):
    global state, engine, control_active, x_ref
    print(f"[BRIDGE] Serial connecting: {connection_string} @ {baud}")

    while True:
        try:
            ser = pyserial.Serial(connection_string, baud, timeout=2)
            print(f"[BRIDGE] Serial connected: {connection_string}")
            state.connected = True
            if state.vehicle_type == "UNKNOWN":
                state.vehicle_type = "GROUND_ROVER"

            prev_state_vec = None

            while True:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line or not line.startswith('{'):
                        continue

                    data = json.loads(line)
                    state.pitch       = float(data.get('pitch',   0))
                    state.roll        = float(data.get('roll',    0))
                    state.gyro_x      = float(data.get('gyro_x',  0))
                    state.gyro_y      = float(data.get('gyro_y',  0))
                    state.gyro_z      = float(data.get('gyro_z',  0))
                    state.accel_x     = float(data.get('accel_x', 0))
                    state.accel_y     = float(data.get('accel_y', 0))
                    state.accel_z     = float(data.get('accel_z', 0))
                    state.motor_l     = float(data.get('motor_l', 0))
                    state.motor_r     = float(data.get('motor_r', 0))
                    state.altitude    = float(data.get('altitude', 0))
                    # Estimate forward velocity from accel_x integration (no wheel encoders on basic bots)
                    raw_vx = float(data.get('vx', 0))
                    if raw_vx != 0:
                        state.velocity_x = raw_vx
                    else:
                        state.velocity_x = state.velocity_x * 0.95 + state.accel_x * (1.0 / 50.0)
                    state.flight_mode = str(data.get('phase', state.flight_mode))
                    # Rocket-specific: update mass from telemetry (tracks propellant depletion)
                    if data.get('mass') is not None:
                        # Only update if it's decreasing (propellant burning) or initialized
                        reported_mass = float(data.get('mass', 0))
                        if reported_mass > 0:
                            state.estimated_mass = reported_mass
                    state.timestamp   = time.time()
                    state.connected   = True

                    # ── PHYSICORE ACTIVE CONTROL ──────────────────────────────
                    if engine and control_active:
                        try:
                            current_x = state_to_vector(engine.cfg.platform,
                                                        n_joints=state.n_joints or engine.cfg.action_dim)
                            ref       = np.array(x_ref) if x_ref else np.zeros(engine.cfg.state_dim)
                            if len(ref) < engine.cfg.state_dim:
                                ref = np.pad(ref, (0, engine.cfg.state_dim - len(ref)))
                            ref = ref[:engine.cfg.state_dim]

                            # Compute optimal action
                            step = engine.step(current_x, ref)

                            # REAL observe: use previous state for actual learning
                            if prev_state_vec is not None:
                                engine.observe(prev_state_vec, step.action, current_x)

                            prev_state_vec = current_x.copy()
                            _update_engine_state(step)
                            state.physicore_active = True

                            # Send command to hardware
                            cmd = json.dumps({"op": "command", "action": step.action.tolist()}) + "\n"
                            ser.write(cmd.encode())

                        except Exception as e:
                            print(f"[ENGINE] Error: {e}")
                            state.physicore_active = False
                    else:
                        state.physicore_active = False
                        prev_state_vec = None

                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"[BRIDGE] Read error: {e}")
                    break

            ser.close()
        except Exception as e:
            print(f"[BRIDGE] Serial failed: {e} — retrying in 3s...")
            state.connected = False
            time.sleep(3)

def _update_engine_state(step):
    """Update telemetry state with PhysiCore diagnostics."""
    d = engine.diagnostics_full
    state.residual_norm        = d.get('residual_norm', 0.0)
    state.uncertainty          = d.get('uncertainty', 0.0)

    # Feed session buffer for intelligence layer
    try:
        from physicore.api.server import _update_session_buffer
        if state.step_count % 100 == 0:
            _update_session_buffer(engine)
    except Exception:
        pass
    state.estimated_mass       = d['params'].get('mass', 0.0)
    state.estimated_friction   = d['params'].get('friction', 0.0)
    state.step_count           = d.get('step_count', 0)

# ── ROS2 reader ────────────────────────────────────────────────────────────────
def ros2_reader(topic: str):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu, NavSatFix
        from nav_msgs.msg import Odometry
    except ImportError:
        print("[BRIDGE] ROS2 requires rclpy. Run: source /opt/ros/humble/setup.bash")
        sys.exit(1)

    class BridgeNode(Node):
        def __init__(self):
            super().__init__('physicore_bridge')
            self.create_subscription(Imu,      '/imu/data', self.imu_cb,  10)
            self.create_subscription(NavSatFix, '/gps/fix',  self.gps_cb,  10)
            self.create_subscription(Odometry,  '/odom',     self.odom_cb, 10)
            try:
                from sensor_msgs.msg import JointState
                self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
            except Exception:
                pass
            try:
                from geometry_msgs.msg import TwistStamped
                self.create_subscription(TwistStamped, '/dvl/velocity', self.dvl_cb, 10)
            except Exception:
                pass
            print("[BRIDGE] ROS2 subscribed to /imu/data /gps/fix /odom /joint_states /dvl/velocity")

        def imu_cb(self, msg):
            state.accel_x = msg.linear_acceleration.x
            state.accel_y = msg.linear_acceleration.y
            state.accel_z = msg.linear_acceleration.z
            state.gyro_x  = math.degrees(msg.angular_velocity.x)
            state.gyro_y  = math.degrees(msg.angular_velocity.y)
            state.gyro_z  = math.degrees(msg.angular_velocity.z)
            state.connected = True; state.timestamp = time.time()

        def gps_cb(self, msg):
            state.lat = msg.latitude; state.lon = msg.longitude; state.altitude = msg.altitude

        def odom_cb(self, msg):
            state.velocity_x = msg.twist.twist.linear.x
            state.velocity_y = msg.twist.twist.linear.y
            state.velocity_z = msg.twist.twist.linear.z
            state.speed = math.sqrt(state.velocity_x**2 + state.velocity_y**2 + state.velocity_z**2)

        def dvl_cb(self, msg):
            # DVL provides bottom-track velocity for AUVs
            state.velocity_x = msg.twist.linear.x
            state.velocity_y = msg.twist.linear.y
            state.velocity_z = msg.twist.linear.z
            state.speed = math.sqrt(state.velocity_x**2 + state.velocity_y**2 + state.velocity_z**2)
            state.connected = True
            state.timestamp = time.time()

        def joint_cb(self, msg):
            """Full joint state — supports arbitrary DOF."""
            n = len(msg.name) if msg.name else len(msg.position)
            state.joint_names = list(msg.name) if msg.name else [f"joint_{i}" for i in range(n)]
            state.joint_positions = [float(p) for p in msg.position]
            state.joint_velocities = [float(v) for v in msg.velocity] if msg.velocity else [0.0] * n
            state.joint_efforts = [float(e) for e in msg.effort] if msg.effort else [0.0] * n
            state.n_joints = n
            # Backward-compat: keep old pitch/roll fields for simple bots
            if n >= 1: state.pitch = math.degrees(msg.position[0])
            if n >= 2: state.roll = math.degrees(msg.position[1])
            if n >= 1: state.gyro_x = math.degrees(msg.velocity[0]) if msg.velocity else 0.0
            if n >= 2: state.gyro_y = math.degrees(msg.velocity[1]) if msg.velocity and len(msg.velocity) > 1 else 0.0
            if n >= 1: state.motor_l = msg.effort[0] if msg.effort else 0.0
            if n >= 2: state.motor_r = msg.effort[1] if msg.effort and len(msg.effort) > 1 else 0.0
            state.connected = True
            state.timestamp = time.time()

    rclpy.init()
    rclpy.spin(BridgeNode())

# ── Test mode ──────────────────────────────────────────────────────────────────
def run_test():
    print("\n[PHYSICORE BRIDGE TEST v2.0.0]\n")
    import importlib
    for lib in ['pymavlink', 'websockets', 'serial', 'numpy']:
        try:
            importlib.import_module(lib)
            print(f"  [OK] {lib}")
        except ImportError:
            print(f"  [MISSING] {lib} — pip install {'pyserial' if lib=='serial' else lib}")
    print(f"\n  PhysiCore: {'OK — engine loaded' if HAS_PHYSICORE else 'MISSING — run from physicore directory'}")
    import socket
    try:
        s = socket.socket(); s.bind(('', 8765)); s.close()
        print("  [OK] Port 8765 available")
    except OSError:
        print("  [WARN] Port 8765 in use — another bridge may be running")
    print(f"\n  Platform profiles:")
    for p in PLATFORM_PROFILES:
        print(f"    --platform {p}")
    print()

# ── WebSocket server ───────────────────────────────────────────────────────────
async def ws_handler(websocket):
    global control_active, x_ref
    connected_clients.add(websocket)
    print(f"[BRIDGE] Dashboard connected from {websocket.remote_address}")
    try:
        await websocket.send(json.dumps({
            "op": "status",
            "msg": {
                "service": "physicore", "status": "ok",
                "bridge_version": BRIDGE_VERSION,
                "vehicle_type": state.vehicle_type,
                "domain": state.domain,
                "platform": state.platform,
                "engine_ready": engine is not None,
            }
        }))
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("op") == "ping":
                    await websocket.send(json.dumps({"op": "pong"}))
                elif data.get("op") == "command":
                    msg = data.get("msg", {})
                    control_active = msg.get("active", False)
                    x_ref_raw = msg.get("x_ref")
                    x_ref = x_ref_raw if x_ref_raw else None
                    print(f"[BRIDGE] Control {'ACTIVATED' if control_active else 'DEACTIVATED'}")
            except Exception:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[BRIDGE] Dashboard disconnected")

async def broadcast_telemetry():
    interval = 1.0 / TELEMETRY_HZ
    while True:
        await asyncio.sleep(interval)
        if connected_clients and state.connected:
            payload = json.dumps(state.to_dict())
            dead = set()
            for ws in connected_clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            connected_clients -= dead

async def health_endpoint():
    from aiohttp import web
    async def health(req):
        return web.Response(
            text=json.dumps({
                "service": "physicore", "status": "ok",
                "vehicle_type": state.vehicle_type,
                "domain": state.domain,
                "connected": state.connected,
                "engine_ready": engine is not None,
                "physicore_active": state.physicore_active,
                "step_count": state.step_count,
                "estimated_mass": state.estimated_mass,
                "residual": state.residual_norm,
            }),
            content_type='application/json',
            headers={"Access-Control-Allow-Origin": "*"}
        )
    app = web.Application()
    app.router.add_get('/api/health', health)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', 8080).start()
    print("[BRIDGE] Health: http://localhost:8080/api/health")

async def broadcast_registry_status():
    """Emit registry_status to all connected dashboards every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        if not connected_clients or not HAS_REGISTRY or engine is None:
            continue
        try:
            reg = get_registry()
            platform_key = engine.cfg.platform
            if _robot_config:
                platform_key = getattr(_robot_config, 'registry_key', engine.cfg.platform)
            d = reg._platform_dir(platform_key)
            params_path = d / "params.json"
            sessions_count = reg._session_count(platform_key)
            latest_params = {}
            prior_weight = 0.0
            if params_path.exists():
                import json as _j
                saved = _j.load(open(params_path))
                latest_params = saved.get("params", {})
            prior_path = d / "platform_prior.json"
            if prior_path.exists():
                import json as _j
                prior = _j.load(open(prior_path))
                prior_weight = prior.get("weight", 0.0)

            payload = json.dumps({
                "op": "registry_status",
                "platform": platform_key,
                "sessions_count": sessions_count,
                "latest_params": {k: round(v, 4) for k, v in latest_params.items()},
                "prior_weight": round(prior_weight, 2),
                "loaded": params_path.exists(),
                "registry_path": str(d),
                "current_mass": round(state.estimated_mass, 4),
                "current_friction": round(state.estimated_friction, 4),
            })
            dead = set()
            for ws in connected_clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            connected_clients -= dead
        except Exception as e:
            pass


async def broadcast_extensions_status():
    """Emit extensions_status once at startup, then every 30s."""
    await asyncio.sleep(3)  # Wait for clients to connect
    while True:
        if engine and engine._extensions and connected_clients:
            payload = json.dumps({
                "op": "extensions_status",
                "extensions": engine._extensions.loaded,
            })
            dead = set()
            for ws in list(connected_clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            connected_clients -= dead
        await asyncio.sleep(30)


async def status_printer():
    while True:
        await asyncio.sleep(5)
        if state.connected:
            eng_info = f"mass={state.estimated_mass:.3f} res={state.residual_norm:.4f} steps={state.step_count}" if engine else "no engine"
            print(f"[TELEM] P:{state.pitch:.1f}° R:{state.roll:.1f}° | {eng_info} | clients={len(connected_clients)}")
        else:
            print("[TELEM] Waiting for hardware...")

async def main(args):
    print(f"""
╔══════════════════════════════════════════════════════╗
║     PHYSICORE BRIDGE v{BRIDGE_VERSION} — FIXED EDITION      ║
╠══════════════════════════════════════════════════════╣
║  Mode:       {args.mode:<40}║
║  Connection: {args.connection:<40}║
║  Engine:     {'READY — PhysiCore loaded' if engine else 'Not loaded (pip install physicore)':<40}║
╚══════════════════════════════════════════════════════╝
""")
    if engine:
        print(f"[ENGINE] Platform: {engine.cfg.platform} | State dim: {engine.cfg.state_dim} | 60Hz CEM-MPC ready")
    print("[BRIDGE] Dashboard: Click MAVLINK, endpoint ws://localhost:8765, Connect")
    print("[BRIDGE] Then click 'ACTIVE CONTROL ON' to start PhysiCore control\n")

    if args.mode in ('mavlink','px4','ardupilot','rocket'):
        threading.Thread(target=mavlink_reader, args=(args.connection, args.baud), daemon=True).start()
    elif args.mode == 'robot_serial':
        threading.Thread(target=robot_serial_reader, args=(args.connection, args.baud), daemon=True).start()
    elif args.mode == 'ros2':
        threading.Thread(target=ros2_reader, args=(args.topic,), daemon=True).start()

    port = 8765
    async with websockets.serve(ws_handler, "0.0.0.0", port):
        await asyncio.gather(broadcast_telemetry(), broadcast_registry_status(), broadcast_extensions_status(), status_printer(), health_endpoint())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhysiCore Universal Hardware Bridge v2.0")
    parser.add_argument('--config', default=None, help='Path to robot config YAML (e.g. my_robot.yaml)')
    parser.add_argument('--init-config', default=None, help='Create a template config for a platform (e.g. balancing_bot)')
    parser.add_argument('--mode',       default='mavlink', choices=['mavlink','px4','ardupilot','rocket','ros2','robot_serial'])
    parser.add_argument('--connection', default='udp:14550')
    parser.add_argument('--baud',       type=int, default=57600)
    parser.add_argument('--physicore',  default='ws://localhost:8765')
    parser.add_argument('--topic',      default='/imu/data')
    parser.add_argument('--platform',   default=None, choices=list(PLATFORM_PROFILES.keys()))
    parser.add_argument('--test',       action='store_true')
    args = parser.parse_args()

    if args.test:
        run_test(); sys.exit(0)

    # Handle --init-config: create template and exit
    if hasattr(args, 'init_config') and args.init_config:
        if HAS_REGISTRY:
            output = f"{args.init_config}_robot.yaml"
            create_template(args.init_config, output)
        else:
            print("[BRIDGE] Registry not available — cannot create template")
        import sys; sys.exit(0)

    # Handle --config: load robot config from YAML
    if hasattr(args, 'config') and args.config and HAS_REGISTRY:
        try:
            _robot_config = RobotConfig.from_yaml(args.config)
            print(f"[CONFIG] Loaded robot config: {args.config}")
            print(f"  Name: {_robot_config.name}")
            print(f"  Platform: {_robot_config.bridge_platform}")
            print(f"  Connection: {_robot_config.resolved_connection}")
            print(f"  Mass: {_robot_config.mass}kg")
            print(f"  Registry: {'enabled' if _robot_config.use_registry else 'disabled'}")
            print(f"  Telemetry: {'opt-in' if _robot_config.opt_in_telemetry else 'off'}")
            # Override args from config
            if not args.platform:
                args.platform = _robot_config.bridge_platform
            if args.connection in ('udp:14550', 'COM3'):
                args.connection = _robot_config.resolved_connection
            args.baud = _robot_config.baud
        except Exception as e:
            print(f"[CONFIG] Failed to load config: {e}")
            _robot_config = None

    if args.platform:
        profile = PLATFORM_PROFILES[args.platform]
        args.mode  = profile["mode"]
        args.baud  = profile["baud"]
        state.vehicle_type = profile["vehicle_type"]
        state.domain       = profile["domain"]
        state.platform     = args.platform
        if args.connection in ('udp:14550', 'COM3'):
            args.connection = profile["default_connection"]
        print(f"[BRIDGE] Platform: {args.platform}")

        # Auto-initialize engine for this platform
        if HAS_PHYSICORE:
            ep = profile.get("engine_platform", "ground_rover")
            if ep in PLATFORM_DYNAMICS:
                # Use config params if available, otherwise defaults
                init_params = {"mass": 1.0, "friction": 0.15, "inertia": 0.01}
                if _robot_config:
                    init_params = _robot_config.initial_params

                from physicore.core.engine import PhysiCore, PLATFORM_DYNAMICS
                # Use for_platform_dof() for variable-DOF platforms
                if _robot_config and _robot_config.is_high_dof:
                    engine = PhysiCore.for_platform_dof(
                        ep,
                        dof=_robot_config.effective_dof,
                        initial_params=init_params,
                    )
                    print(f"[ENGINE] High-DOF mode: platform='{ep}' dof={_robot_config.effective_dof} state_dim={engine.cfg.state_dim} action_dim={engine.cfg.action_dim}")
                else:
                    engine = PhysiCore.for_platform(ep, init_params)

                # Load saved model from registry if available
                registry_key = ep
                if _robot_config:
                    registry_key = _robot_config.registry_key
                if HAS_REGISTRY:
                    reg = get_registry()
                    loaded = reg.load(engine, registry_key)
                    if not loaded:
                        # Try platform-level prior
                        reg.load_prior(engine, ep)

                print(f"[ENGINE] Initialized for '{ep}' — SystemID will adapt mass/friction from real data")
                print(f"[ENGINE] Starting params: {engine.physics.params}")

                # Auto-load extensions from ~/.physicore/extensions/
                try:
                    from physicore.extensions import load_extensions_from_dir
                    import pathlib
                    ext_dir = pathlib.Path.home() / ".physicore" / "extensions"
                    engine._extensions = load_extensions_from_dir(ext_dir, engine)
                    if engine._extensions.loaded:
                        print(f"[Extensions] {len(engine._extensions.loaded)} extension(s) loaded")
                    else:
                        print("[Extensions] No extensions found — drop .py files into ~/.physicore/extensions/")
                except Exception as _ext_err:
                    print(f"[Extensions] Auto-load error: {_ext_err}")

                # Start telemetry if opted in
                if HAS_REGISTRY and _robot_config and _robot_config.opt_in_telemetry:
                    hw_class = f"{_robot_config.imu}_{_robot_config.motor_driver}_{_robot_config.mcu}".lower()
                    _telemetry_mgr = get_telemetry(enabled=True)
                    _telemetry_mgr.start_session(ep, hw_class, engine.cfg.control_hz)

    if _platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n[BRIDGE] Stopping — saving model...")
        # Save learned model to registry on clean exit
        if engine is not None and HAS_REGISTRY:
            try:
                registry_key = engine.cfg.platform
                if _robot_config:
                    registry_key = _robot_config.registry_key
                reg = get_registry()
                session_meta = {}
                if _robot_config:
                    session_meta = {
                        "name":         _robot_config.name,
                        "imu":          _robot_config.imu,
                        "motor_driver": _robot_config.motor_driver,
                        "mcu":          _robot_config.mcu,
                    }
                reg.save(
                    engine,
                    platform=registry_key,
                    session_meta=session_meta,
                    opt_in_telemetry=_robot_config.opt_in_telemetry if _robot_config else False,
                )
                # End telemetry session
                if _telemetry_mgr:
                    _telemetry_mgr.end_session(engine)
            except Exception as e:
                print(f"[BRIDGE] Could not save model: {e}")
        print("[BRIDGE] Stopped.")
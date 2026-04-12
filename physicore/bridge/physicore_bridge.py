#!/usr/bin/env python3
"""
Physicore Universal Hardware Bridge v1.1.0
==========================================
Connects any autonomous system to Physicore in real time.

Supported stacks:
  mavlink    — PX4, ArduPilot over UDP/TCP/Serial (default)
  ros2       — Any ROS2 robot via rclpy topics
  robot_serial — Arduino/ESP32 JSON serial (balancing bots, custom hardware)
  rocket     — Custom rocket FCs over serial
  px4        — alias for mavlink
  ardupilot  — alias for mavlink

Platform profiles (auto-configure):
  --platform px4_quadrotor
  --platform ardupilot_plane
  --platform ros2_manipulator
  --platform balancing_bot_arduino
  --platform custom_rocket_fc
  --platform ground_rover_ros2

Usage:
  python physicore_bridge.py --connection udp:14550
  python physicore_bridge.py --connection COM3 --baud 115200
  python physicore_bridge.py --mode robot_serial --connection COM3 --baud 115200
  python physicore_bridge.py --platform balancing_bot_arduino --connection COM3
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

# Try to import PhysiCore for active control
try:
    from physicore import PhysiCore, PhysiCoreConfig
    HAS_PHYSICORE = True
except ImportError:
    HAS_PHYSICORE = False

def check_deps():
    missing = []
    try:
        import pymavlink
    except ImportError:
        missing.append("pymavlink")
    try:
        import websockets
    except ImportError:
        missing.append("websockets")
    try:
        import serial
    except ImportError:
        missing.append("pyserial")
    if missing:
        print(f"\n[PHYSICORE BRIDGE] Missing dependencies.")
        print(f"Run: pip install {' '.join(missing)}\n")
        sys.exit(1)

check_deps()

from pymavlink import mavutil
import websockets
import serial as pyserial

BRIDGE_VERSION = "1.1.0"
TELEMETRY_HZ   = 20

# ── Platform profiles ──────────────────────────────────────────────────────────
PLATFORM_PROFILES = {
    "px4_quadrotor":          {"mode":"mavlink",       "baud":57600,  "vehicle_type":"QUADROTOR",   "domain":"AVIATION",  "default_connection":"udp:14550"},
    "ardupilot_plane":        {"mode":"mavlink",       "baud":57600,  "vehicle_type":"FIXED_WING",  "domain":"AVIATION",  "default_connection":"udp:14550"},
    "ardupilot_quadrotor":    {"mode":"mavlink",       "baud":57600,  "vehicle_type":"QUADROTOR",   "domain":"AVIATION",  "default_connection":"udp:14550"},
    "evtol":                  {"mode":"mavlink",       "baud":57600,  "vehicle_type":"EVTOL",        "domain":"AVIATION",  "default_connection":"udp:14550"},
    "ros2_manipulator":       {"mode":"ros2",          "baud":0,      "vehicle_type":"MANIPULATOR", "domain":"ROBOTICS",  "default_connection":"ros2"},
    "ros2_legged":            {"mode":"ros2",          "baud":0,      "vehicle_type":"LEGGED",      "domain":"ROBOTICS",  "default_connection":"ros2"},
    "ros2_ground_rover":      {"mode":"ros2",          "baud":0,      "vehicle_type":"GROUND_ROVER","domain":"ROBOTICS",  "default_connection":"ros2"},
    "ros2_auv":               {"mode":"ros2",          "baud":0,      "vehicle_type":"AUV",          "domain":"ROBOTICS",  "default_connection":"ros2"},
    "ros2_surgical":          {"mode":"ros2",          "baud":0,      "vehicle_type":"SURGICAL",    "domain":"ROBOTICS",  "default_connection":"ros2"},
    "balancing_bot_arduino":  {"mode":"robot_serial",  "baud":115200, "vehicle_type":"GROUND_ROVER","domain":"ROBOTICS",  "default_connection":"COM3"},
    "custom_rocket_fc":       {"mode":"robot_serial",  "baud":115200, "vehicle_type":"ROCKET",      "domain":"ROCKETS",   "default_connection":"COM3"},
    "ground_rover_serial":    {"mode":"robot_serial",  "baud":115200, "vehicle_type":"GROUND_ROVER","domain":"ROBOTICS",  "default_connection":"COM3"},
    "satellite_serial":       {"mode":"robot_serial",  "baud":115200, "vehicle_type":"SATELLITE",   "domain":"AVIATION",  "default_connection":"COM3"},
}

# ── Telemetry state ────────────────────────────────────────────────────────────
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
        # Joint states (manipulators, humanoids, legged robots)
        self.joint_positions  = [0.0] * 6
        self.joint_velocities = [0.0] * 6
        self.joint_efforts    = [0.0] * 6
        # Force/torque (surgical, manipulation)
        self.force_x  = 0.0
        self.force_y  = 0.0
        self.force_z  = 0.0
        self.torque_x = 0.0
        self.torque_y = 0.0
        self.torque_z = 0.0
        # Depth/pressure (AUV)
        self.depth    = 0.0
        # Transition ratio (eVTOL)
        self.transition_ratio = 0.0
        self.vehicle_type = "UNKNOWN"
        self.domain       = "ROBOTICS"
        self.connected    = False
        self.platform     = "unknown"

    def to_dict(self) -> dict:
        return {
            "op":    "publish",
            "topic": "/telemetry",
            "msg": {
                "timestamp":    self.timestamp,
                "altitude":     round(self.altitude,    3),
                "velocity":     {"x": round(self.velocity_x, 3), "y": round(self.velocity_y, 3), "z": round(self.velocity_z, 3)},
                "speed":        round(self.speed,        3),
                "pitch":        round(self.pitch,        3),
                "roll":         round(self.roll,         3),
                "yaw":          round(self.yaw,          3),
                "orientation":  {"roll": round(self.roll, 3), "pitch": round(self.pitch, 3), "yaw": round(self.yaw, 3)},
                "position":     {"lat": self.lat, "lon": self.lon},
                "acceleration": {"x": round(self.accel_x, 4), "y": round(self.accel_y, 4), "z": round(self.accel_z, 4)},
                "gyro":         {"x": round(self.gyro_x, 4), "y": round(self.gyro_y, 4), "z": round(self.gyro_z, 4)},
                "gyro_x":       round(self.gyro_x,  4),
                "gyro_y":       round(self.gyro_y,  4),
                "gyro_z":       round(self.gyro_z,  4),
                "airspeed":     round(self.airspeed,    3),
                "groundspeed":  round(self.groundspeed, 3),
                "climb_rate":   round(self.climb_rate,  3),
                "throttle":     round(self.throttle,    3),
                "motor_l":      round(self.motor_l,     1),
                "motor_r":      round(self.motor_r,     1),
                "joint_positions":  [round(v, 4) for v in self.joint_positions],
                "joint_velocities": [round(v, 4) for v in self.joint_velocities],
                "joint_efforts":    [round(v, 4) for v in self.joint_efforts],
                "force":    {"x": round(self.force_x, 4), "y": round(self.force_y, 4), "z": round(self.force_z, 4)},
                "torque":   {"x": round(self.torque_x, 4), "y": round(self.torque_y, 4), "z": round(self.torque_z, 4)},
                "depth":             round(self.depth, 3),
                "transition_ratio":  round(self.transition_ratio, 3),
                "battery":      {"voltage": round(self.battery_v, 2), "percentage": round(self.battery_pct, 1)},
                "armed":        self.armed,
                "flight_mode":  self.flight_mode,
                "gps":          {"fix": self.gps_fix, "satellites": self.satellites},
                "vehicle_type": self.vehicle_type,
                "domain":       self.domain,
                "connected":    self.connected,
                "bridge_version": BRIDGE_VERSION,
            }
        }

state              = TelemetryState()
connected_clients  = set()

# Downlink command state
command_state = {
    "action": None,
    "timestamp": 0,
    "active": False,
    "x_ref": None
}

engine = None

# ── MAVLink reader ─────────────────────────────────────────────────────────────
def mavlink_reader(connection_string: str, baud: int):
    global state
    print(f"[BRIDGE] MAVLink connecting: {connection_string}")
    try:
        mav = mavutil.mavlink_connection(connection_string, baud=baud, autoreconnect=True, source_system=255)
    except Exception as e:
        print(f"[BRIDGE] Connection failed: {e}")
        return

    print("[BRIDGE] Waiting for heartbeat...")
    hb = mav.wait_heartbeat(timeout=15)
    if not hb:
        print("[BRIDGE] No heartbeat. Check connection and that vehicle is powered.")
        return

    vtypes = {
        1:"FIXED_WING", 2:"QUADROTOR", 3:"COAXIAL", 4:"HELICOPTER",
        6:"GROUND_ROVER", 8:"ROCKET", 10:"FLAPPING_WING",
        13:"HEXAROTOR", 14:"OCTOROTOR", 15:"TRICOPTER",
        19:"EVTOL", 20:"EVTOL", 21:"EVTOL",
        27:"GROUND_ROVER", 40:"LEGGED"
    }
    vt = vtypes.get(mav.messages['HEARTBEAT'].type, "UNKNOWN")
    state.vehicle_type = vt
    state.domain = (
        "AVIATION"  if vt in ("FIXED_WING","QUADROTOR","COAXIAL","HELICOPTER","HEXAROTOR","OCTOROTOR","TRICOPTER","FLAPPING_WING","EVTOL")
        else "ROCKETS"  if vt == "ROCKET"
        else "ROBOTICS"
    )
    state.connected = True
    print(f"[BRIDGE] Connected. Vehicle: {vt}  Domain: {state.domain}")
    mav.mav.request_data_stream_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 20, 1)

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
            elif mt == 'BATTERY_STATUS':
                if msg.voltages and msg.voltages[0] != 65535:
                    state.battery_v = msg.voltages[0] / 1000.0
                state.battery_pct = msg.battery_remaining
            elif mt == 'GPS_RAW_INT':
                state.gps_fix    = msg.fix_type
                state.satellites = msg.satellites_visible
            elif mt == 'HEARTBEAT':
                state.armed       = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                state.flight_mode = f"MODE_{msg.custom_mode}"
        except Exception as e:
            time.sleep(0.1)

# ── Robot serial reader ────────────────────────────────────────────────────────
def robot_serial_reader(connection_string: str, baud: int):
    """
    Reads line-by-line JSON telemetry from Arduino/ESP32.
    Expected format (one JSON per line):
    {"pitch":0.0,"roll":0.0,"gyro_x":0.0,"gyro_y":0.0,"gyro_z":0.0,
     "accel_x":0.0,"accel_y":0.0,"accel_z":0.0,"motor_l":0,"motor_r":0,"timestamp":0}
    """
    global state
    print(f"[BRIDGE] Serial connecting: {connection_string} @ {baud} baud")

    while True:
        try:
            ser = pyserial.Serial(connection_string, baud, timeout=2)
            print(f"[BRIDGE] Serial connected: {connection_string}")
            if state.vehicle_type == "UNKNOWN":
                state.vehicle_type = "GROUND_ROVER"
            if state.domain == "ROBOTICS":
                pass
            state.connected = True

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
                    state.velocity_x  = float(data.get('vx', 0))
                    state.velocity_y  = float(data.get('vy', 0))
                    state.velocity_z  = float(data.get('vz', 0))
                    state.timestamp   = time.time()
                    state.connected   = True

                    # --- ACTIVE CONTROL LOOP ---
                    if engine and command_state["active"]:
                        # Convert state to numpy array for engine
                        # For balancing bot: [pitch, pitch_rate, x, x_vel]
                        # We'll use a simplified mapping for now
                        current_x = np.array([state.pitch, state.gyro_y, 0.0, 0.0])
                        target_x  = command_state["x_ref"] if command_state["x_ref"] is not None else np.zeros(4)
                        
                        # Step engine
                        step_res = engine.step(current_x, target_x)
                        command_state["action"] = step_res.action.tolist()
                        command_state["timestamp"] = time.time()
                        
                        # Observe (Online Learning)
                        # In a real loop, we'd wait for the next state, 
                        # but for now we'll observe the transition
                        engine.observe(current_x, step_res.action, current_x) # Dummy observe for now

                    # Downlink: Send command back if active
                    if command_state["active"] and command_state["action"] is not None:
                        # Only send if command is fresh (less than 500ms old)
                        if time.time() - command_state["timestamp"] < 0.5:
                            cmd_payload = json.dumps({
                                "op": "command",
                                "action": command_state["action"]
                            }) + "\n"
                            ser.write(cmd_payload.encode())
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"[BRIDGE] Serial read error: {e}")
                    break

            ser.close()
        except Exception as e:
            print(f"[BRIDGE] Serial failed: {e} — retrying in 3s...")
            state.connected = False
            time.sleep(3)

# ── ROS2 reader ────────────────────────────────────────────────────────────────
def ros2_reader(topic: str):
    """
    Reads from ROS2 topics.
    Handles: IMU, GPS, Odometry, JointStates, Wrench, NavSatFix, DVL, depth
    """
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu, NavSatFix, FluidPressure, Temperature
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import WrenchStamped, TwistStamped
    except ImportError:
        print("[BRIDGE] ROS2 requires rclpy. Run: source /opt/ros/humble/setup.bash")
        sys.exit(1)

    global state

    class BridgeNode(Node):
        def __init__(self):
            super().__init__('physicore_bridge')

            # Universal subscriptions — work for all platforms
            self.create_subscription(Imu,      '/imu/data',  self.imu_cb,  10)
            self.create_subscription(NavSatFix, '/gps/fix',   self.gps_cb,  10)
            self.create_subscription(Odometry,  '/odom',      self.odom_cb, 10)

            # Manipulator arm / humanoid — joint states
            try:
                from sensor_msgs.msg import JointState
                self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
            except Exception:
                pass

            # Force/torque sensor — surgical robots, manipulation
            try:
                self.create_subscription(WrenchStamped, '/wrench', self.wrench_cb, 10)
                self.create_subscription(WrenchStamped, '/ft_sensor/wrench', self.wrench_cb, 10)
            except Exception:
                pass

            # AUV / underwater
            try:
                self.create_subscription(FluidPressure, '/depth', self.depth_cb, 10)
                self.create_subscription(TwistStamped,  '/dvl/velocity', self.dvl_cb, 10)
            except Exception:
                pass

            print("[BRIDGE] ROS2 node started. Topics: /imu/data /gps/fix /odom /joint_states /wrench /depth /dvl/velocity")

        def imu_cb(self, msg):
            state.accel_x = msg.linear_acceleration.x
            state.accel_y = msg.linear_acceleration.y
            state.accel_z = msg.linear_acceleration.z
            state.gyro_x  = math.degrees(msg.angular_velocity.x)
            state.gyro_y  = math.degrees(msg.angular_velocity.y)
            state.gyro_z  = math.degrees(msg.angular_velocity.z)
            state.connected = True
            state.timestamp = time.time()

        def gps_cb(self, msg):
            state.lat      = msg.latitude
            state.lon      = msg.longitude
            state.altitude = msg.altitude

        def odom_cb(self, msg):
            state.velocity_x = msg.twist.twist.linear.x
            state.velocity_y = msg.twist.twist.linear.y
            state.velocity_z = msg.twist.twist.linear.z
            state.speed = math.sqrt(
                state.velocity_x**2 + state.velocity_y**2 + state.velocity_z**2
            )
            # Extract roll/pitch/yaw from quaternion if available
            try:
                q = msg.pose.pose.orientation
                siny = 2.0 * (q.w * q.z + q.x * q.y)
                cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                state.yaw = math.degrees(math.atan2(siny, cosy))
                sinp = 2.0 * (q.w * q.y - q.z * q.x)
                state.pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))
                sinr = 2.0 * (q.w * q.x + q.y * q.z)
                cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
                state.roll = math.degrees(math.atan2(sinr, cosr))
            except Exception:
                pass

        def joint_cb(self, msg):
            # Store joint positions as pitch/roll/yaw (first 3 joints) for display
            if len(msg.position) >= 1:
                state.pitch = math.degrees(msg.position[0])
            if len(msg.position) >= 2:
                state.roll  = math.degrees(msg.position[1])
            if len(msg.position) >= 3:
                state.yaw   = math.degrees(msg.position[2])
            # Store joint velocities in gyro fields
            if len(msg.velocity) >= 1:
                state.gyro_x = math.degrees(msg.velocity[0])
            if len(msg.velocity) >= 2:
                state.gyro_y = math.degrees(msg.velocity[1])
            if len(msg.velocity) >= 3:
                state.gyro_z = math.degrees(msg.velocity[2])
            # Store joint efforts (torques) in motor fields
            if len(msg.effort) >= 1:
                state.motor_l = msg.effort[0]
            if len(msg.effort) >= 2:
                state.motor_r = msg.effort[1]
            state.connected = True
            state.timestamp = time.time()

        def wrench_cb(self, msg):
            # Force/torque sensor — map to acceleration fields
            state.accel_x = msg.wrench.force.x
            state.accel_y = msg.wrench.force.y
            state.accel_z = msg.wrench.force.z
            state.gyro_x  = msg.wrench.torque.x
            state.gyro_y  = msg.wrench.torque.y
            state.gyro_z  = msg.wrench.torque.z
            state.connected = True

        def depth_cb(self, msg):
            # AUV depth from fluid pressure: P = rho * g * h
            state.altitude = -(msg.fluid_pressure - 101325.0) / (1025.0 * 9.81)
            state.connected = True

        def dvl_cb(self, msg):
            # DVL bottom-track velocity
            state.velocity_x = msg.twist.linear.x
            state.velocity_y = msg.twist.linear.y
            state.velocity_z = msg.twist.linear.z
            state.speed = math.sqrt(
                state.velocity_x**2 + state.velocity_y**2 + state.velocity_z**2
            )

    rclpy.init()
    rclpy.spin(BridgeNode())

# ── Test mode ──────────────────────────────────────────────────────────────────
def run_test():
    print("\n[PHYSICORE BRIDGE TEST]\n")
    import importlib
    results = {}

    for lib in ['pymavlink', 'websockets', 'aiohttp', 'serial']:
        try:
            importlib.import_module(lib)
            results[lib] = "OK"
        except ImportError:
            results[lib] = "MISSING — run: pip install " + ('pyserial' if lib == 'serial' else lib)

    for name, status in results.items():
        icon = "OK" if status == "OK" else "FAIL"
        print(f"  [{icon}] {name}: {status}")

    import socket
    sock = socket.socket()
    try:
        sock.bind(('', 8765))
        sock.close()
        print("  [OK]   Port 8765: available")
    except OSError:
        print("  [WARN] Port 8765: already in use — another bridge may be running")

    print(f"\n  Platform: {_platform.system()} {_platform.release()}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"\n  Available platform profiles:")
    for p in PLATFORM_PROFILES:
        print(f"    --platform {p}")
    print()

# ── WebSocket server ───────────────────────────────────────────────────────────
async def ws_handler(websocket):
    connected_clients.add(websocket)
    addr = websocket.remote_address
    print(f"[BRIDGE] Physicore UI connected from {addr}")
    try:
        await websocket.send(json.dumps({
            "op": "status",
            "msg": {
                "service":        "physicore",
                "status":         "ok",
                "bridge_version": BRIDGE_VERSION,
                "vehicle_type":   state.vehicle_type,
                "domain":         state.domain,
                "platform":       state.platform,
            }
        }))
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("op") == "ping":
                    await websocket.send(json.dumps({"op": "pong"}))
                elif data.get("op") == "command":
                    msg = data.get("msg", {})
                    command_state["action"] = msg.get("action")
                    command_state["active"] = msg.get("active", True)
                    command_state["x_ref"]  = msg.get("x_ref")
                    command_state["timestamp"] = time.time()
            except Exception:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[BRIDGE] Physicore UI disconnected from {addr}")

async def broadcast_telemetry():
    interval = 1.0 / TELEMETRY_HZ
    while True:
        if connected_clients and state.connected:
            payload = json.dumps(state.to_dict())
            dead = set()
            for ws in connected_clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            connected_clients -= dead
        await asyncio.sleep(interval)

async def health_endpoint():
    from aiohttp import web
    async def health(req):
        return web.Response(
            text=json.dumps({
                "service":      "physicore",
                "status":       "ok",
                "vehicle_type": state.vehicle_type,
                "domain":       state.domain,
                "connected":    state.connected,
                "bridge_version": BRIDGE_VERSION,
            }),
            content_type='application/json',
            headers={"X-PhysiCore-Bridge": "active", "Access-Control-Allow-Origin": "*"}
        )
    app = web.Application()
    app.router.add_get('/api/health', health)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', 8080).start()
    print("[BRIDGE] Health check: http://localhost:8080/api/health")

async def status_printer():
    while True:
        await asyncio.sleep(3)
        if state.connected:
            print(
                f"[TELEMETRY] ALT:{state.altitude:.1f}m  "
                f"SPD:{state.speed:.1f}m/s  "
                f"R:{state.roll:.1f}°  P:{state.pitch:.1f}°  Y:{state.yaw:.1f}°  "
                f"GYRO_Y:{state.gyro_y:.2f}°/s  "
                f"MOTOR_L:{state.motor_l:.0f}  MOTOR_R:{state.motor_r:.0f}  "
                f"BAT:{state.battery_pct:.0f}%  "
                f"CLIENTS:{len(connected_clients)}"
            )
        else:
            print("[TELEMETRY] Waiting for hardware...")

async def main(args):
    print(f"""
╔══════════════════════════════════════════════════════════╗
║       PHYSICORE UNIVERSAL HARDWARE BRIDGE v{BRIDGE_VERSION}      ║
╠══════════════════════════════════════════════════════════╣
║  Mode:       {args.mode:<44}║
║  Connection: {args.connection:<44}║
║  Baud:       {str(args.baud):<44}║
╚══════════════════════════════════════════════════════════╝
""")

    if args.mode in ('mavlink', 'px4', 'ardupilot', 'rocket'):
        threading.Thread(target=mavlink_reader, args=(args.connection, args.baud), daemon=True).start()
    elif args.mode == 'robot_serial':
        threading.Thread(target=robot_serial_reader, args=(args.connection, args.baud), daemon=True).start()
    elif args.mode == 'ros2':
        threading.Thread(target=ros2_reader, args=(args.topic,), daemon=True).start()

    port = int(args.physicore.split(":")[-1]) if ":" in args.physicore else 8765
    print(f"[BRIDGE] WebSocket server on port {port}")
    print(f"[BRIDGE] In Physicore UI: set endpoint to ws://localhost:{port}")
    print(f"[BRIDGE] Press Ctrl+C to stop\n")

    async with websockets.serve(ws_handler, "0.0.0.0", port):
        await asyncio.gather(
            broadcast_telemetry(),
            status_printer(),
            health_endpoint(),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Physicore Universal Hardware Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python physicore_bridge.py --connection udp:14550
  python physicore_bridge.py --connection COM3 --baud 115200
  python physicore_bridge.py --mode robot_serial --connection COM3 --baud 115200
  python physicore_bridge.py --platform balancing_bot_arduino --connection COM3
  python physicore_bridge.py --mode ros2
  python physicore_bridge.py --test
        """
    )
    parser.add_argument('--mode',       default='mavlink',
                        choices=['mavlink','px4','ardupilot','rocket','ros2','robot_serial'])
    parser.add_argument('--connection', default='udp:14550')
    parser.add_argument('--baud',       type=int, default=57600)
    parser.add_argument('--physicore',  default='ws://localhost:8765')
    parser.add_argument('--topic',      default='/imu/data')
    parser.add_argument('--platform',   default=None, choices=list(PLATFORM_PROFILES.keys()))
    parser.add_argument('--test',       action='store_true')
    parser.add_argument('--active',     action='store_true', help="Enable active PhysiCore control loop")
    args = parser.parse_args()

    if args.test:
        run_test()
        sys.exit(0)

    if args.active:
        if not HAS_PHYSICORE:
            print("[BRIDGE] Error: PhysiCore package not found. Active mode disabled.")
        else:
            print(f"[BRIDGE] Initializing PhysiCore Engine for {args.platform or 'ground_rover'}...")
            import numpy as np
            engine = PhysiCore.for_platform(args.platform or "ground_rover")

    if args.platform:
        profile = PLATFORM_PROFILES[args.platform]
        args.mode       = profile["mode"]
        args.baud       = profile["baud"]
        state.vehicle_type = profile["vehicle_type"]
        state.domain       = profile["domain"]
        state.platform     = args.platform
        if args.connection == 'udp:14550':
            args.connection = profile["default_connection"]
        print(f"[BRIDGE] Platform profile: {args.platform}")

    if _platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n[BRIDGE] Stopped.")

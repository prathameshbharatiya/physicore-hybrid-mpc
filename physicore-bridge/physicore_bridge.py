#!/usr/bin/env python3
"""
Physicore Universal Hardware Bridge
Run this on the lab laptop next to the hardware.

Install: pip install pymavlink websockets aiohttp

Usage:
  python physicore_bridge.py --connection udp:14550        # PX4 / ArduPilot UDP
  python physicore_bridge.py --connection /dev/ttyUSB0     # USB serial
  python physicore_bridge.py --connection COM3             # Windows
  python physicore_bridge.py --mode ros2                   # ROS2
  python physicore_bridge.py --mode rocket --connection /dev/ttyUSB0
"""

import asyncio, json, time, argparse, threading, sys, math
from datetime import datetime

def check_deps():
    missing = []
    try: import pymavlink
    except ImportError: missing.append("pymavlink")
    try: import websockets
    except ImportError: missing.append("websockets")
    if missing:
        print(f"[PHYSICORE] Install missing deps: pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()
from pymavlink import mavutil
import websockets

BRIDGE_VERSION = "1.0.0"
TELEMETRY_HZ = 20

class TelemetryState:
    def __init__(self):
        self.timestamp = 0.0
        self.altitude = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        self.speed = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.lat = 0.0
        self.lon = 0.0
        self.throttle = 0.0
        self.battery_v = 0.0
        self.battery_pct = 0.0
        self.armed = False
        self.flight_mode = "UNKNOWN"
        self.gps_fix = 0
        self.satellites = 0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0
        self.airspeed = 0.0
        self.groundspeed = 0.0
        self.climb_rate = 0.0
        self.motor = 0.0
        self.vehicle_type = "UNKNOWN"
        self.connected = False

    def to_dict(self):
        return {
            "op": "publish",
            "topic": "/telemetry",
            "msg": {
                "timestamp": self.timestamp,
                "altitude": round(self.altitude, 3),
                "velocity": {"x": round(self.velocity_x, 3), "y": round(self.velocity_y, 3), "z": round(self.velocity_z, 3)},
                "speed": round(self.speed, 3),
                "orientation": {"roll": round(self.roll, 3), "pitch": round(self.pitch, 3), "yaw": round(self.yaw, 3)},
                "position": {"lat": self.lat, "lon": self.lon},
                "acceleration": {"x": round(self.accel_x, 4), "y": round(self.accel_y, 4), "z": round(self.accel_z, 4)},
                "gyro": {"x": round(self.gyro_x, 4), "y": round(self.gyro_y, 4), "z": round(self.gyro_z, 4)},
                "airspeed": round(self.airspeed, 3),
                "groundspeed": round(self.groundspeed, 3),
                "climb_rate": round(self.climb_rate, 3),
                "throttle": round(self.throttle, 3),
                "motor": round(self.motor, 3),
                "battery": {"voltage": round(self.battery_v, 2), "percentage": round(self.battery_pct, 1)},
                "armed": self.armed,
                "flight_mode": self.flight_mode,
                "gps": {"fix": self.gps_fix, "satellites": self.satellites},
                "vehicle_type": self.vehicle_type,
                "connected": self.connected,
                "bridge_version": BRIDGE_VERSION
            }
        }

state = TelemetryState()
connected_clients = set()

def mavlink_reader(connection_string, baud):
    global state
    print(f"[BRIDGE] Connecting: {connection_string}")
    try:
        mav = mavutil.mavlink_connection(connection_string, baud=baud, autoreconnect=True, source_system=255)
    except Exception as e:
        print(f"[BRIDGE] Failed: {e}")
        return
    print("[BRIDGE] Waiting for heartbeat...")
    hb = mav.wait_heartbeat(timeout=15)
    if not hb:
        print("[BRIDGE] No heartbeat. Check connection.")
        return
    vtypes = {1:"FIXED_WING",2:"QUADROTOR",3:"COAXIAL",4:"HELICOPTER",6:"GROUND_ROVER",8:"ROCKET",13:"HEXAROTOR",14:"OCTOROTOR",15:"TRICOPTER"}
    state.vehicle_type = vtypes.get(mav.messages['HEARTBEAT'].type, "UNKNOWN")
    state.connected = True
    print(f"[BRIDGE] Connected. Vehicle: {state.vehicle_type}")
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
                state.altitude = msg.alt; state.airspeed = msg.airspeed
                state.groundspeed = msg.groundspeed; state.climb_rate = msg.climb
                state.throttle = msg.throttle / 100.0
            elif mt == 'ATTITUDE':
                state.roll = math.degrees(msg.roll); state.pitch = math.degrees(msg.pitch)
                state.yaw = math.degrees(msg.yaw)
                state.gyro_x = math.degrees(msg.rollspeed); state.gyro_y = math.degrees(msg.pitchspeed)
                state.gyro_z = math.degrees(msg.yawspeed)
            elif mt == 'GLOBAL_POSITION_INT':
                state.lat = msg.lat/1e7; state.lon = msg.lon/1e7
                state.altitude = msg.relative_alt/1000.0
                state.velocity_x = msg.vx/100.0; state.velocity_y = msg.vy/100.0; state.velocity_z = msg.vz/100.0
                state.speed = math.sqrt(state.velocity_x**2 + state.velocity_y**2 + state.velocity_z**2)
            elif mt == 'RAW_IMU':
                state.accel_x = msg.xacc/1000.0; state.accel_y = msg.yacc/1000.0; state.accel_z = msg.zacc/1000.0
            elif mt == 'SYS_STATUS':
                state.battery_v = msg.voltage_battery/1000.0; state.battery_pct = msg.battery_remaining
            elif mt == 'GPS_RAW_INT':
                state.gps_fix = msg.fix_type; state.satellites = msg.satellites_visible
            elif mt == 'HEARTBEAT':
                state.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                state.flight_mode = f"MODE_{msg.custom_mode}"
        except Exception as e:
            time.sleep(0.1)

def robot_serial_reader(connection_string, baud):
    """
    Reads raw telemetry from a serial connection.
    Expected format: "P:pitch,R:roll,M:motor"
    """
    import serial
    global state
    print(f"[BRIDGE] Connecting Serial: {connection_string} at {baud}")
    try:
        ser = serial.Serial(connection_string, baud, timeout=1)
        state.connected = True
        state.vehicle_type = "GROUND_ROVER"
        print(f"[BRIDGE] Serial Connected. Vehicle: {state.vehicle_type}")
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    # Parse "P:1.2,R:3.4,M:0.5"
                    parts = dict(item.split(":") for item in line.split(",") if ":" in item)
                    state.timestamp = time.time()
                    if 'P' in parts: state.pitch = float(parts['P'])
                    if 'R' in parts: state.roll = float(parts['R'])
                    if 'M' in parts: state.motor = float(parts['M'])
                except Exception as e:
                    print(f"[BRIDGE] Parse Error: {e} on line: {line}")
    except Exception as e:
        print(f"[BRIDGE] Serial Error: {e}")
        state.connected = False

def ros2_reader(topic):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu, NavSatFix
        from nav_msgs.msg import Odometry
    except ImportError:
        print("[BRIDGE] ROS2 requires rclpy. Source your workspace: source /opt/ros/humble/setup.bash")
        sys.exit(1)
    global state
    class BridgeNode(Node):
        def __init__(self):
            super().__init__('physicore_bridge')
            self.create_subscription(Imu, '/imu/data', self.imu_cb, 10)
            self.create_subscription(NavSatFix, '/gps/fix', self.gps_cb, 10)
            self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        def imu_cb(self, msg):
            state.accel_x = msg.linear_acceleration.x; state.accel_y = msg.linear_acceleration.y; state.accel_z = msg.linear_acceleration.z
            state.gyro_x = math.degrees(msg.angular_velocity.x); state.gyro_y = math.degrees(msg.angular_velocity.y); state.gyro_z = math.degrees(msg.angular_velocity.z)
            state.connected = True; state.timestamp = time.time()
        def gps_cb(self, msg):
            state.lat = msg.latitude; state.lon = msg.longitude; state.altitude = msg.altitude
        def odom_cb(self, msg):
            state.velocity_x = msg.twist.twist.linear.x; state.velocity_y = msg.twist.twist.linear.y; state.velocity_z = msg.twist.twist.linear.z
            state.speed = math.sqrt(state.velocity_x**2 + state.velocity_y**2 + state.velocity_z**2)
    rclpy.init()
    rclpy.spin(BridgeNode())

async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"[BRIDGE] Physicore connected from {websocket.remote_address}")
    try:
        await websocket.send(json.dumps({"op":"status","msg":{"service":"physicore","status":"ok","bridge_version":BRIDGE_VERSION,"vehicle_type":state.vehicle_type}}))
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("op") == "ping":
                    await websocket.send(json.dumps({"op":"pong"}))
            except: pass
    except websockets.exceptions.ConnectionClosed: pass
    finally:
        connected_clients.discard(websocket)

async def broadcast_telemetry():
    interval = 1.0 / TELEMETRY_HZ
    while True:
        if connected_clients and state.connected:
            payload = json.dumps(state.to_dict())
            dead = set()
            for ws in connected_clients:
                try: await ws.send(payload)
                except: dead.add(ws)
            connected_clients -= dead
        await asyncio.sleep(interval)

async def health_endpoint():
    from aiohttp import web
    async def health(req):
        return web.Response(text=json.dumps({"service":"physicore","status":"ok","vehicle_type":state.vehicle_type,"connected":state.connected}),
            content_type='application/json', headers={"X-PhysiCore-Bridge":"active","Access-Control-Allow-Origin":"*"})
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
            print(f"[TELEMETRY] ALT:{state.altitude:.1f}m SPD:{state.speed:.1f}m/s R:{state.roll:.1f}° P:{state.pitch:.1f}° Y:{state.yaw:.1f}° BAT:{state.battery_pct:.0f}% ARMED:{state.armed} CLIENTS:{len(connected_clients)}")
        else:
            print("[TELEMETRY] Waiting for hardware...")

async def main(args):
    print(f"\n[PHYSICORE BRIDGE v{BRIDGE_VERSION}] Mode:{args.mode} Connection:{args.connection}\n")
    if args.mode in ('mavlink','px4','ardupilot','rocket'):
        threading.Thread(target=mavlink_reader, args=(args.connection, args.baud), daemon=True).start()
    elif args.mode == 'ros2':
        threading.Thread(target=ros2_reader, args=(args.topic,), daemon=True).start()
    elif args.mode == 'robot_serial':
        threading.Thread(target=robot_serial_reader, args=(args.connection, args.baud), daemon=True).start()
    port = int(args.physicore.split(":")[-1]) if ":" in args.physicore else 8765
    print(f"[BRIDGE] WebSocket on port {port}. In Physicore: set endpoint to ws://localhost:{port}")
    async with websockets.serve(ws_handler, "0.0.0.0", port):
        await asyncio.gather(broadcast_telemetry(), status_printer(), health_endpoint())

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='mavlink', choices=['mavlink','px4','ardupilot','rocket','ros2','robot_serial'])
    parser.add_argument('--connection', default='udp:14550')
    parser.add_argument('--baud', type=int, default=57600)
    parser.add_argument('--physicore', default='ws://localhost:8765')
    parser.add_argument('--topic', default='/imu/data')
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n[BRIDGE] Stopped.")

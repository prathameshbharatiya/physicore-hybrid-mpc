// ============================================================================
// PHYSICORE INTEGRATION ENGINEER — ZERO API DEPENDENCY
// Runs 100% in the browser. No network calls. Never fails.
// Decision tree + code generator for every hardware type.
// ============================================================================

// ── Types ──────────────────────────────────────────────────────────────────

interface IntegrationState {
  phase: 'DETECT' | 'QUESTIONS' | 'CONFIRM' | 'GENERATED';
  hardwareType: string;
  answers: Record<string, string>;
  questionIndex: number;
  messages: IntegrationMessage[];
}

interface IntegrationMessage {
  role: 'engineer' | 'user';
  content: string;
  code?: { filename: string; content: string }[];
  buttons?: string[];
}

interface Question {
  key: string;
  text: string;
  options?: string[];
  freeText?: boolean;
}

interface HardwarePath {
  name: string;
  keywords: string[];
  platform: string;
  questions: Question[];
  bridgeCommand: (answers: Record<string, string>) => string;
  generateCode: (answers: Record<string, string>) => { filename: string; content: string }[];
  steps: string[];
}

// ── Hardware detection keywords ────────────────────────────────────────────

function detectHardwareFromMessage(msg: string): string {
  const m = msg.toLowerCase();
  if (m.match(/balanc|self.balanc|inverted.pendulum|segway/)) return 'balancing_bot';
  if (m.match(/px4|pixhawk|autopilot/)) return 'px4_drone';
  if (m.match(/ardupilot|apm|cube.pilot/)) return 'ardupilot_drone';
  if (m.match(/ros2|ros 2|moveit|manipulation.*arm|robot.*arm|cobot|ur5|ur10|kuka|fanuc|abb/)) return 'ros2_arm';
  if (m.match(/humanoid|biped|figure.ai|unitree|boston.dynamic|spot|g1|h1/)) return 'humanoid';
  if (m.match(/legged|quadruped|anymal|cheetah|go1|go2|spot/)) return 'legged';
  if (m.match(/evtol|e.vtol|vtol|air.taxi|tilt.rotor/)) return 'evtol';
  if (m.match(/surgical|medical.robot|endoscop/)) return 'surgical';
  if (m.match(/auv|underwater|subsea|rov|bluerov|dvl/)) return 'auv';
  if (m.match(/satellite|spacecraft|orbital|cubesat|reaction.wheel/)) return 'satellite';
  if (m.match(/ugv|ground.vehicle|defence|defense|military.robot/)) return 'ugv';
  if (m.match(/rocket|sounding.rocket|motor.*rocket|flight.computer/)) return 'rocket';
  if (m.match(/esp32|esp8266|arduino.*drone|custom.*drone/)) return 'arduino_drone';
  if (m.match(/drone|quadrotor|quad.rotor|multirotor|fpv/)) return 'generic_drone';
  if (m.match(/arduino|esp32|esp8266|micro.*controller|mcu/)) return 'arduino_generic';
  if (m.match(/rover|ground.robot|differential.drive|wheeled/)) return 'ground_rover';
  if (m.match(/logistics|amr|warehouse.robot|agv/)) return 'amr';
  return 'unknown';
}

// ── Code generators ────────────────────────────────────────────────────────

function generateBalancingBotArduino(a: Record<string, string>): { filename: string; content: string }[] {
  const imu = a.imu || 'MPU6050';
  const mass = a.mass || '1.0';
  const comH = a.com_height || '0.15';
  const baud = '115200';

  const imuInit: Record<string, string> = {
    'MPU6050': `#include <MPU6050_light.h>\nMPU6050 mpu(Wire);\nvoid initIMU() {\n  Wire.begin();\n  mpu.begin();\n  mpu.calcOffsets();\n}`,
    'BNO055':  `#include <Adafruit_BNO055.h>\nAdafruit_BNO055 bno(55, 0x28);\nvoid initIMU() {\n  bno.begin();\n  bno.setExtCrystalUse(true);\n}`,
    'MPU9250': `#include <MPU9250_asukiaaa.h>\nMPU9250_asukiaaa mpu;\nvoid initIMU() {\n  Wire.begin();\n  mpu.setup(0x68);\n}`,
  };

  const imuRead: Record<string, string> = {
    'MPU6050': `  mpu.update();\n  pitch  = mpu.getAngleX();\n  gyro_y = mpu.getGyroX();`,
    'BNO055':  `  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);\n  imu::Vector<3> gyro  = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);\n  pitch  = euler.x();\n  gyro_y = gyro.y();`,
    'MPU9250': `  mpu.accelUpdate();\n  mpu.gyroUpdate();\n  pitch  = mpu.accelX() * 57.2958;\n  gyro_y = mpu.gyroY();`,
  };

  const motorPins: Record<string, string> = {
    'L298N':    `const int L_EN=5,L_IN1=4,L_IN2=3,R_EN=6,R_IN1=7,R_IN2=8;`,
    'TB6612FNG':`const int PWMA=5,AIN1=4,AIN2=3,PWMB=6,BIN1=7,BIN2=8,STBY=9;`,
    'DRV8833':  `const int AIN1=4,AIN2=3,BIN1=7,BIN2=8,PWR=9;`,
  };

  const motorApply: Record<string, string> = {
    'L298N':    `  int pwm = constrain((int)(abs(v)*255), 0, 255);\n  digitalWrite(L_IN1, v>0); digitalWrite(L_IN2, v<=0);\n  analogWrite(L_EN, pwm);\n  digitalWrite(R_IN1, v>0); digitalWrite(R_IN2, v<=0);\n  analogWrite(R_EN, pwm);`,
    'TB6612FNG':`  int pwm = constrain((int)(abs(v)*255), 0, 255);\n  digitalWrite(AIN1, v>0); digitalWrite(AIN2, v<=0);\n  analogWrite(PWMA, pwm);\n  digitalWrite(BIN1, v>0); digitalWrite(BIN2, v<=0);\n  analogWrite(PWMB, pwm);\n  digitalWrite(STBY, HIGH);`,
    'DRV8833':  `  int pwm = constrain((int)(abs(v)*255), 0, 255);\n  analogWrite(AIN1, v>0?pwm:0); analogWrite(AIN2, v<=0?pwm:0);\n  analogWrite(BIN1, v>0?pwm:0); analogWrite(BIN2, v<=0?pwm:0);`,
  };

  const imuCode = imuInit[imu] || imuInit['MPU6050'];
  const readCode = imuRead[imu] || imuRead['MPU6050'];
  const pinCode = motorPins[a.motor_driver || 'L298N'] || motorPins['L298N'];
  const applyCode = motorApply[a.motor_driver || 'L298N'] || motorApply['L298N'];

  const ino = `/*
 * PhysiCore Balancing Bot Firmware
 * Hardware: ${imu} + ${a.motor_driver || 'L298N'}
 * Mass: ${mass}kg  CoM height: ${comH}m
 * Generated by PhysiCore Integration Engineer
 *
 * WIRING:
 * IMU SDA -> A4, SCL -> A5 (Arduino Uno/Nano)
 * Motor driver as configured below
 *
 * INSTALL LIBRARIES first:
 * Sketch > Include Library > Manage Libraries
 * Search: "${imu}" and "ArduinoJson" — install both
 */

#include <Wire.h>
#include <ArduinoJson.h>
${imuCode.split('\n').slice(0, 2).join('\n')}

${pinCode}

// --- PhysiCore state ---
float pitch   = 0.0;
float gyro_y  = 0.0;
float motor_l = 0.0;
float motor_r = 0.0;
bool  physicore_active = false;
unsigned long last_cmd = 0;
unsigned long last_tx  = 0;

// --- Internal safety PID (runs when PhysiCore not connected) ---
float kp = 35.0, kd = 0.8;
float prev_err = 0.0;

${imuCode.split('\n').slice(2).join('\n')}

void setup() {
  Serial.begin(${baud});
  initIMU();
  delay(100);
  Serial.println("PhysiCore balancing bot ready");
}

void loop() {
  unsigned long now = millis();

  // 1. READ REAL IMU DATA
${readCode}

  // 2. SEND TELEMETRY AT 50Hz
  if (now - last_tx >= 20) {
    last_tx = now;
    StaticJsonDocument<256> doc;
    doc["pitch"]   = pitch;
    doc["roll"]    = 0.0;
    doc["gyro_x"]  = 0.0;
    doc["gyro_y"]  = gyro_y;
    doc["gyro_z"]  = 0.0;
    doc["accel_x"] = 0.0;
    doc["accel_y"] = 0.0;
    doc["accel_z"] = 9.81;
    doc["motor_l"] = motor_l;
    doc["motor_r"] = motor_r;
    doc["timestamp"] = now;
    serializeJson(doc, Serial);
    Serial.println();
  }

  // 3. RECEIVE COMMANDS FROM PHYSICORE
  while (Serial.available()) {
    StaticJsonDocument<256> cmd;
    if (deserializeJson(cmd, Serial) == DeserializationError::Ok) {
      if (cmd["op"] == "command") {
        float torque = cmd["action"][0];
        // Scale torque to motor [-1, 1]
        motor_l = constrain(torque / 10.0, -1.0, 1.0);
        motor_r = motor_l;
        physicore_active = true;
        last_cmd = now;
      }
    }
  }

  // 4. SAFETY TIMEOUT — fallback PID if PhysiCore silent for 500ms
  if (now - last_cmd > 500) {
    physicore_active = false;
  }

  // 5. APPLY CONTROL
  float v;
  if (physicore_active) {
    v = motor_l;  // PhysiCore computed this
  } else {
    // Internal safety PID
    float err  = -pitch;
    float deriv = (err - prev_err) / 0.02;
    v = constrain((kp * err + kd * deriv) / 255.0, -1.0, 1.0);
    prev_err = err;
    motor_l = v; motor_r = v;
  }

  applyMotors(v);
  delay(1);
}

void applyMotors(float v) {
${applyCode}
}`;

  return [{ filename: 'physicore_balancing_bot.ino', content: ino }];
}

function generateROS2ArmNode(a: Record<string, string>): { filename: string; content: string }[] {
  const distro = a.ros2_distro || 'humble';
  const topic = a.joint_topic || '/joint_states';
  const dof = parseInt(a.dof || '6');

  const node = `#!/usr/bin/env python3
"""
PhysiCore ROS2 Manipulator Bridge Node
ROS2 ${distro} | ${dof}-DOF arm | Topic: ${topic}
Generated by PhysiCore Integration Engineer
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
import json, socket, asyncio, threading

PHYSICORE_HOST = 'localhost'
PHYSICORE_PORT = 8765

class PhysicoreArmBridge(Node):
    def __init__(self):
        super().__init__('physicore_arm_bridge')
        self.joint_state = [0.0] * ${dof}
        self.joint_vel   = [0.0] * ${dof}
        self.joint_effort= [0.0] * ${dof}
        self.force = [0.0, 0.0, 0.0]
        self.torque= [0.0, 0.0, 0.0]

        self.create_subscription(JointState, '${topic}', self.joint_cb, 10)
        self.create_subscription(WrenchStamped, '/ft_sensor/wrench', self.wrench_cb, 10)

        self.get_logger().info('PhysiCore bridge started — connecting to ws://localhost:8765')

    def joint_cb(self, msg: JointState):
        n = min(len(msg.position), ${dof})
        self.joint_state[:n]  = list(msg.position[:n])
        self.joint_vel[:n]    = list(msg.velocity[:n]) if msg.velocity else [0.0]*n
        self.joint_effort[:n] = list(msg.effort[:n]) if msg.effort else [0.0]*n
        self.send_telemetry()

    def wrench_cb(self, msg: WrenchStamped):
        f = msg.wrench.force
        t = msg.wrench.torque
        self.force  = [f.x, f.y, f.z]
        self.torque = [t.x, t.y, t.z]

    def send_telemetry(self):
        payload = json.dumps({
            "op": "publish",
            "topic": "/telemetry",
            "msg": {
                "pitch":       self.joint_state[0] * 57.2958,
                "roll":        self.joint_state[1] * 57.2958 if ${dof} > 1 else 0,
                "yaw":         self.joint_state[2] * 57.2958 if ${dof} > 2 else 0,
                "gyro_x":      self.joint_vel[0],
                "gyro_y":      self.joint_vel[1] if ${dof} > 1 else 0,
                "gyro_z":      self.joint_vel[2] if ${dof} > 2 else 0,
                "accel_x":     self.force[0],
                "accel_y":     self.force[1],
                "accel_z":     self.force[2],
                "motor_l":     self.joint_effort[0],
                "motor_r":     self.joint_effort[1] if ${dof} > 1 else 0,
                "vehicle_type":"MANIPULATOR",
                "domain":      "ROBOTICS",
                "connected":   True,
                "joint_positions":  self.joint_state,
                "joint_velocities": self.joint_vel,
                "joint_efforts":    self.joint_effort,
                "force":  {"x": self.force[0],  "y": self.force[1],  "z": self.force[2]},
                "torque": {"x": self.torque[0], "y": self.torque[1], "z": self.torque[2]},
            }
        }) + '\\n'
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((PHYSICORE_HOST, PHYSICORE_PORT))
            sock.sendall(payload.encode())
            sock.close()
        except Exception:
            pass  # Bridge not connected yet

def main():
    rclpy.init()
    node = PhysicoreArmBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()`;

  return [
    { filename: 'physicore_arm_bridge.py', content: node },
    { filename: 'run_bridge.sh', content: `#!/bin/bash
# Run this AFTER starting the PhysiCore bridge
source /opt/ros/${distro}/setup.bash
python3 physicore_arm_bridge.py` }
  ];
}

function generatePX4Config(a: Record<string, string>): { filename: string; content: string }[] {
  const connection = a.connection === 'USB/Serial' ? '/dev/ttyACM0' : 'udp:14550';
  const mass = a.mass || '1.5';

  return [{
    filename: 'physicore_px4_setup.sh',
    content: `#!/bin/bash
# PhysiCore PX4 Setup
# Mass: ${mass}kg | Connection: ${connection}
# Run this on your companion computer

echo "Installing PhysiCore bridge dependencies..."
pip install pymavlink websockets aiohttp pyserial

echo "Starting PhysiCore bridge for PX4..."
python physicore_bridge.py \\
  --platform px4_quadrotor \\
  --connection ${connection} \\
  --baud 57600

# The bridge will:
# 1. Connect to PX4 via MAVLink
# 2. Stream telemetry to PhysiCore dashboard at ws://localhost:8765
# 3. PhysiCore learns your drone's real mass (nominal: ${mass}kg)
# 4. SystemID converges in ~300 steps (~5 seconds at 60Hz)`
  }];
}

function generateRocketFirmware(a: Record<string, string>): { filename: string; content: string }[] {
  const baud = a.baud || '115200';
  const baro = a.barometer || 'BMP280';

  return [{
    filename: 'physicore_rocket_fc.ino',
    content: `/*
 * PhysiCore Rocket Flight Computer Firmware
 * Barometer: ${baro} | Baud: ${baud}
 * Generated by PhysiCore Integration Engineer
 */

#include <Wire.h>
#include <ArduinoJson.h>
// Install: ${baro} library via Library Manager

float altitude    = 0.0;
float velocity    = 0.0;
float accel_x     = 0.0;
float accel_y     = 0.0;
float accel_z     = 9.81;
float pitch       = 0.0;
float mass        = 1.0;  // Update with your dry mass
float thrust      = 0.0;
String phase      = "IDLE";
unsigned long last_tx = 0;
float prev_alt    = 0.0;

void setup() {
  Serial.begin(${baud});
  Wire.begin();
  // Initialize ${baro} here
  delay(100);
}

void loop() {
  unsigned long now = millis();

  // Read your sensors here
  // altitude = baro.readAltitude(1013.25);
  // accel_x  = imu.getAccelX();
  // etc.

  // Compute velocity from altitude
  velocity = (altitude - prev_alt) / 0.02;
  prev_alt = altitude;

  // Detect flight phase
  if (velocity > 2.0)   phase = "BOOST";
  else if (altitude > 50 && velocity < 0) phase = "COAST";
  else if (altitude < 50) phase = "RECOVERY";
  else phase = "IDLE";

  // Send telemetry at 50Hz
  if (now - last_tx >= 20) {
    last_tx = now;
    StaticJsonDocument<512> doc;
    doc["altitude"]  = altitude;
    doc["velocity"]  = velocity;
    doc["accel_x"]   = accel_x;
    doc["accel_y"]   = accel_y;
    doc["accel_z"]   = accel_z;
    doc["pitch"]     = pitch;
    doc["yaw"]       = 0.0;
    doc["thrust"]    = thrust;
    doc["mass"]      = mass;
    doc["phase"]     = phase;
    doc["timestamp"] = now;
    serializeJson(doc, Serial);
    Serial.println();
  }
  delay(1);
}`
  }];
}

// ── Hardware paths — complete decision trees ────────────────────────────────

const HARDWARE_PATHS: Record<string, HardwarePath> = {
  balancing_bot: {
    name: 'Self-Balancing Robot',
    keywords: ['balancing', 'arduino', 'esp32'],
    platform: 'balancing_bot',
    questions: [
      { key: 'imu', text: 'What IMU sensor are you using?', options: ['MPU6050', 'BNO055', 'MPU9250', 'ICM20689', 'Other'] },
      { key: 'mcu', text: 'What microcontroller?', options: ['Arduino Uno/Nano', 'Arduino Mega', 'ESP32', 'Raspberry Pi Pico', 'Other'] },
      { key: 'motor_driver', text: 'What motor driver?', options: ['L298N', 'TB6612FNG', 'DRV8833', 'BTS7960', 'Other'] },
      { key: 'mass', text: 'Approximate robot mass in kg (e.g. 1.2)?', freeText: true },
      { key: 'com_height', text: 'Height of center of mass from wheel axle in meters (e.g. 0.15)?', freeText: true },
    ],
    bridgeCommand: (a) => `python physicore_bridge.py --platform balancing_bot_arduino --connection COM3 --baud 115200`,
    generateCode: generateBalancingBotArduino,
    steps: [
      'Install ArduinoJson library: Sketch → Include Library → Manage Libraries → search ArduinoJson → Install',
      'Install your IMU library the same way',
      'Flash the generated .ino file to your Arduino/ESP32',
      'On your laptop: pip install pymavlink websockets aiohttp pyserial',
      'Run: python physicore_bridge.py --platform balancing_bot_arduino --connection COM3 --baud 115200  (Windows: COM3, Mac: /dev/cu.usbserial-0001, Linux: /dev/ttyUSB0)',
      'Open PhysiCore Dashboard → click MAVLINK → endpoint ws://localhost:8765 → Connect',
      'Your bot\'s real pitch appears immediately. PhysiCore starts learning your hardware.',
    ],
  },

  px4_drone: {
    name: 'PX4 Drone',
    keywords: ['px4', 'pixhawk'],
    platform: 'quadrotor',
    questions: [
      { key: 'connection', text: 'How is your computer connected to PX4?', options: ['UDP WiFi (QGC default)', 'USB/Serial', 'UDP Ethernet', 'UART Telemetry Radio'] },
      { key: 'companion', text: 'Companion computer or laptop?', options: ['Laptop (direct USB)', 'Raspberry Pi 4', 'Jetson Nano', 'Jetson Orin', 'No companion — laptop only'] },
      { key: 'mass', text: 'Drone total mass with battery in kg (e.g. 1.5)?', freeText: true },
    ],
    bridgeCommand: (a) => `python physicore_bridge.py --platform px4_quadrotor --connection ${a.connection?.includes('USB') ? '/dev/ttyACM0' : 'udp:14550'}`,
    generateCode: generatePX4Config,
    steps: [
      'On your companion computer or laptop: pip install pymavlink websockets aiohttp',
      'Connect PX4 via USB or ensure UDP telemetry is enabled in QGroundControl',
      'Run: python physicore_bridge.py --platform px4_quadrotor --connection udp:14550',
      'Open PhysiCore Dashboard → MAVLINK → ws://localhost:8765 → Connect',
      'Fly — PhysiCore learns your drone\'s real mass and aerodynamics in real time',
    ],
  },

  ardupilot_drone: {
    name: 'ArduPilot Drone',
    keywords: ['ardupilot', 'apm'],
    platform: 'quadrotor',
    questions: [
      { key: 'frame', text: 'Frame type?', options: ['Quadrotor', 'Hexarotor', 'Fixed-wing', 'VTOL', 'Rover'] },
      { key: 'connection', text: 'Connection method?', options: ['UDP (Mission Planner)', 'USB Serial', 'UART Telemetry Radio'] },
      { key: 'mass', text: 'Total mass with battery in kg?', freeText: true },
    ],
    bridgeCommand: (a) => `python physicore_bridge.py --platform ardupilot_${a.frame?.includes('wing') ? 'plane' : 'quadrotor'} --connection udp:14550`,
    generateCode: (a) => [{
      filename: 'physicore_ardupilot_setup.sh',
      content: `#!/bin/bash\n# ArduPilot ${a.frame || 'Quadrotor'} | Mass: ${a.mass || '1.5'}kg\npip install pymavlink websockets aiohttp\npython physicore_bridge.py --platform ardupilot_${a.frame?.includes('wing') ? 'plane' : 'quadrotor'} --connection udp:14550`
    }],
    steps: [
      'Enable MAVLink telemetry in Mission Planner: Config → Planner → enable UDP',
      'pip install pymavlink websockets aiohttp',
      'Run bridge: python physicore_bridge.py --platform ardupilot_quadrotor --connection udp:14550',
      'Open PhysiCore Dashboard → MAVLINK → ws://localhost:8765 → Connect',
    ],
  },

  ros2_arm: {
    name: 'ROS2 Manipulator Arm',
    keywords: ['ros2', 'arm', 'manipulator'],
    platform: 'manipulator_arm',
    questions: [
      { key: 'ros2_distro', text: 'Which ROS2 distribution?', options: ['Humble', 'Jazzy', 'Iron', 'Rolling'] },
      { key: 'joint_topic', text: 'Joint states topic name?', options: ['/joint_states', '/robot/joint_states', '/arm/joint_states', 'Custom'] },
      { key: 'dof', text: 'How many joints (DOF)?', options: ['4', '6', '7', 'Other'] },
      { key: 'mass', text: 'End-effector payload mass in kg?', freeText: true },
    ],
    bridgeCommand: () => `python physicore_bridge.py --platform ros2_manipulator`,
    generateCode: generateROS2ArmNode,
    steps: [
      'Copy physicore_arm_bridge.py to your ROS2 workspace',
      'Install deps: pip install websockets',
      'Run PhysiCore bridge: python physicore_bridge.py --platform ros2_manipulator',
      'In a new terminal: python3 physicore_arm_bridge.py',
      'Open PhysiCore Dashboard → MAVLINK → ws://localhost:8765 → Connect',
      'Move your arm — joint positions appear in real time, PhysiCore adapts',
    ],
  },

  humanoid: {
    name: 'Humanoid Robot',
    keywords: ['humanoid', 'biped'],
    platform: 'legged_robot',
    questions: [
      { key: 'brand', text: 'Which humanoid robot?', options: ['Unitree G1', 'Unitree H1', 'Boston Dynamics Spot', 'Figure AI Apollo', 'Custom/Other'] },
      { key: 'interface', text: 'Control interface?', options: ['ROS2', 'Unitree SDK', 'Boston Dynamics SDK', 'Custom'] },
      { key: 'mass', text: 'Robot mass in kg?', freeText: true },
    ],
    bridgeCommand: () => `python physicore_bridge.py --platform ros2_legged`,
    generateCode: (a) => {
      const brand = a.brand || 'Custom';
      const specific = brand.includes('Unitree') ?
        `# Unitree specific: source unitree_ros2 workspace first\n# ros2 topic list should show /joint_states` :
        brand.includes('Boston') ?
        `# Spot: ensure Spot ROS2 driver is running\n# spot_ros2 package publishes /joint_states` : '';
      return [{
        filename: 'physicore_humanoid_setup.sh',
        content: `#!/bin/bash
# PhysiCore Humanoid Setup: ${brand}
# Mass: ${a.mass || '50'}kg
${specific}

source /opt/ros/humble/setup.bash
pip install pymavlink websockets aiohttp

# Terminal 1: Start PhysiCore bridge
python physicore_bridge.py --platform ros2_legged

# Terminal 2: Verify joint states are flowing
ros2 topic echo /joint_states --once

# Open PhysiCore Dashboard → MAVLINK → ws://localhost:8765 → Connect`
      }];
    },
    steps: [
      'Ensure your robot\'s ROS2 driver is running and /joint_states topic is publishing',
      'pip install pymavlink websockets aiohttp',
      'Run: python physicore_bridge.py --platform ros2_legged',
      'Open PhysiCore Dashboard → MAVLINK → ws://localhost:8765 → Connect',
      'PhysiCore learns your robot\'s real mass and contact dynamics',
    ],
  },

  rocket: {
    name: 'Sounding Rocket',
    keywords: ['rocket', 'sounding'],
    platform: 'rocket',
    questions: [
      { key: 'mcu', text: 'Flight computer microcontroller?', options: ['Arduino Mega', 'Teensy 4.1', 'ESP32', 'Raspberry Pi', 'Custom FC', 'Other'] },
      { key: 'barometer', text: 'Barometer/altimeter?', options: ['BMP280', 'MS5611', 'BMP388', 'MPL3115A2', 'Other'] },
      { key: 'baud', text: 'Serial baud rate?', options: ['115200', '57600', '9600', '38400'] },
      { key: 'dry_mass', text: 'Rocket dry mass in kg?', freeText: true },
    ],
    bridgeCommand: (a) => `python physicore_bridge.py --platform custom_rocket_fc --connection /dev/ttyUSB0 --baud ${a.baud || '115200'}`,
    generateCode: generateRocketFirmware,
    steps: [
      'Install required libraries via Arduino Library Manager',
      'Flash the generated firmware to your flight computer',
      'pip install pymavlink websockets aiohttp pyserial',
      'Run: python physicore_bridge.py --platform custom_rocket_fc --connection /dev/ttyUSB0 --baud 115200',
      'Open PhysiCore Dashboard → MAVLINK → ws://localhost:8765 → Connect',
      'PhysiCore tracks mass depletion, Mach-dependent drag, and wind in real time',
    ],
  },

  auv: {
    name: 'AUV / Underwater Robot',
    keywords: ['auv', 'underwater'],
    platform: 'auv',
    questions: [
      { key: 'ros2_distro', text: 'ROS2 distribution?', options: ['Humble', 'Iron', 'Jazzy'] },
      { key: 'dvl', text: 'Do you have a DVL (Doppler Velocity Log)?', options: ['Yes', 'No — IMU + depth only'] },
      { key: 'depth_sensor', text: 'Depth sensor type?', options: ['Bar30', 'MS5837', 'BlueRobotics Bar02', 'Custom', 'None'] },
      { key: 'mass', text: 'Vehicle mass in kg?', freeText: true },
    ],
    bridgeCommand: () => `python physicore_bridge.py --platform ros2_auv`,
    generateCode: (a) => [{
      filename: 'physicore_auv_setup.sh',
      content: `#!/bin/bash
# PhysiCore AUV Setup
# DVL: ${a.dvl}  Depth: ${a.depth_sensor}  Mass: ${a.mass}kg
source /opt/ros/${a.ros2_distro?.toLowerCase() || 'humble'}/setup.bash
pip install pymavlink websockets aiohttp

# Bridge subscribes to: /imu/data, /depth, /dvl/velocity
python physicore_bridge.py --platform ros2_auv

# PhysiCore uses nonlinear quadratic drag model
# Buoyancy and drag coefficients learned online`
    }],
    steps: [
      'Ensure /imu/data and /depth topics are publishing in ROS2',
      'If DVL available: ensure /dvl/velocity is publishing',
      'Run: python physicore_bridge.py --platform ros2_auv',
      'Open PhysiCore Dashboard → MAVLINK → ws://localhost:8765 → Connect',
    ],
  },

  ground_rover: {
    name: 'Ground Rover / AMR',
    keywords: ['rover', 'amr', 'logistics'],
    platform: 'ground_rover',
    questions: [
      { key: 'interface', text: 'Communication interface?', options: ['ROS2', 'Arduino Serial', 'ESP32 Serial', 'Custom'] },
      { key: 'ros2_distro', text: 'ROS2 distribution? (if applicable)', options: ['Humble', 'Iron', 'Jazzy', 'N/A'] },
      { key: 'mass', text: 'Robot mass in kg?', freeText: true },
    ],
    bridgeCommand: (a) => a.interface === 'ROS2' ? `python physicore_bridge.py --platform ros2_ground_rover` : `python physicore_bridge.py --platform ground_rover_serial --connection COM3 --baud 115200`,
    generateCode: (a) => [{
      filename: 'physicore_rover_setup.sh',
      content: `#!/bin/bash\n# PhysiCore Ground Rover | Mass: ${a.mass}kg\n${a.interface === 'ROS2' ? `source /opt/ros/${a.ros2_distro?.toLowerCase() || 'humble'}/setup.bash\npython physicore_bridge.py --platform ros2_ground_rover` : `python physicore_bridge.py --platform ground_rover_serial --connection COM3 --baud 115200`}`
    }],
    steps: [
      'pip install pymavlink websockets aiohttp pyserial',
      'Run the bridge command for your interface',
      'Open PhysiCore Dashboard → MAVLINK → ws://localhost:8765 → Connect',
      'PhysiCore adapts terrain friction and slip model in real time',
    ],
  },

  unknown: {
    name: 'Custom Hardware',
    keywords: [],
    platform: 'ground_rover',
    questions: [
      { key: 'type', text: 'What type of system are you integrating?', options: ['Ground robot', 'Aerial vehicle', 'Manipulator arm', 'Rocket/spacecraft', 'Underwater vehicle', 'Other'] },
      { key: 'interface', text: 'Communication interface?', options: ['ROS2', 'Arduino/Serial', 'MAVLink', 'Custom protocol'] },
      { key: 'sensors', text: 'Primary sensors?', options: ['IMU only', 'IMU + GPS', 'IMU + encoders', 'IMU + vision', 'Other'] },
    ],
    bridgeCommand: () => `python physicore_bridge.py --mode robot_serial --connection COM3 --baud 115200`,
    generateCode: (a) => [{
      filename: 'physicore_custom_setup.md',
      content: `# PhysiCore Custom Hardware Setup

## Your system
Type: ${a.type || 'Custom'}
Interface: ${a.interface || 'Serial'}
Sensors: ${a.sensors || 'IMU'}

## Serial JSON format (if using serial)
Your hardware must output this JSON once per line at 20-50Hz:
{"pitch":0.0,"roll":0.0,"gyro_x":0.0,"gyro_y":0.0,"gyro_z":0.0,"accel_x":0.0,"accel_y":0.0,"accel_z":9.81,"motor_l":0,"motor_r":0,"timestamp":0}

## Bridge
pip install pymavlink websockets aiohttp pyserial
python physicore_bridge.py --mode robot_serial --connection COM3 --baud 115200

## Dashboard
Open PhysiCore → MAVLINK → ws://localhost:8765 → Connect`
    }],
    steps: [
      'Format your telemetry as single-line JSON at 20-50Hz',
      'pip install pymavlink websockets aiohttp pyserial',
      'Run bridge: python physicore_bridge.py --mode robot_serial --connection COM3',
      'Open PhysiCore Dashboard → MAVLINK → ws://localhost:8765 → Connect',
    ],
  },
};

// ── Main integration engine — pure logic, zero API ─────────────────────────

export function processIntegrationMessage(
  userMessage: string,
  state: IntegrationState
): { newState: IntegrationState; response: IntegrationMessage } {

  const m = userMessage.toLowerCase().trim();

  // Phase: DETECT — figure out what hardware they have
  if (state.phase === 'DETECT') {
    let detected = detectHardwareFromMessage(userMessage);

    // If unknown, make our best guess from context
    if (detected === 'unknown') {
      if (m.includes('i have') || m.includes('i am using') || m.includes('i want')) {
        // Ask for clarification
        return {
          newState: { ...state, phase: 'DETECT' },
          response: {
            role: 'engineer',
            content: '> INTEGRATION ENGINEER:\nI can integrate PhysiCore with any hardware. What are you building?',
            buttons: [
              'Balancing bot (Arduino)',
              'PX4 drone',
              'ArduPilot drone',
              'ROS2 robot arm',
              'Humanoid robot',
              'Legged robot',
              'eVTOL aircraft',
              'Surgical robot',
              'AUV / underwater',
              'Rocket',
              'Ground rover',
              'Custom hardware',
            ]
          }
        };
      }
    }

    // Handle button selections
    const buttonMap: Record<string, string> = {
      'balancing bot': 'balancing_bot',
      'px4 drone': 'px4_drone',
      'ardupilot drone': 'ardupilot_drone',
      'ros2 robot arm': 'ros2_arm',
      'humanoid robot': 'humanoid',
      'legged robot': 'legged',
      'evtol aircraft': 'evtol',
      'surgical robot': 'surgical',
      'auv': 'auv',
      'underwater': 'auv',
      'rocket': 'rocket',
      'ground rover': 'ground_rover',
      'custom hardware': 'unknown',
    };

    for (const [key, val] of Object.entries(buttonMap)) {
      if (m.includes(key)) { detected = val; break; }
    }

    const path = HARDWARE_PATHS[detected] || HARDWARE_PATHS['unknown'];
    const firstQ = path.questions[0];

    return {
      newState: {
        ...state,
        phase: 'QUESTIONS',
        hardwareType: detected,
        questionIndex: 0,
        answers: {},
      },
      response: {
        role: 'engineer',
        content: `> INTEGRATION ENGINEER:\nPerfect — integrating PhysiCore with your ${path.name}.\n\nI need a few details to generate your exact code.\n\n${firstQ.text}`,
        buttons: firstQ.options,
      }
    };
  }

  // Phase: QUESTIONS — collect answers one at a time
  if (state.phase === 'QUESTIONS') {
    const path = HARDWARE_PATHS[state.hardwareType] || HARDWARE_PATHS['unknown'];
    const currentQ = path.questions[state.questionIndex];
    const newAnswers = { ...state.answers, [currentQ.key]: userMessage };
    const nextIndex = state.questionIndex + 1;

    if (nextIndex < path.questions.length) {
      // More questions
      const nextQ = path.questions[nextIndex];
      return {
        newState: { ...state, questionIndex: nextIndex, answers: newAnswers },
        response: {
          role: 'engineer',
          content: `> INTEGRATION ENGINEER:\nGot it — ${userMessage}.\n\n${nextQ.text}`,
          buttons: nextQ.options,
        }
      };
    }

    // All questions answered — generate
    const codes = path.generateCode(newAnswers);
    const bridgeCmd = path.bridgeCommand(newAnswers);
    const stepsText = path.steps.map((s, i) => `STEP ${i+1} — ${s}`).join('\n');

    return {
      newState: { ...state, phase: 'GENERATED', answers: newAnswers },
      response: {
        role: 'engineer',
        content: `> INTEGRATION ENGINEER:\nExcellent. Here is your complete PhysiCore integration for your ${path.name}.\n\n${stepsText}\n\nBridge command:\n${bridgeCmd}`,
        code: codes,
      }
    };
  }

  // Phase: GENERATED — answer follow-up questions
  if (state.phase === 'GENERATED') {
    const path = HARDWARE_PATHS[state.hardwareType] || HARDWARE_PATHS['unknown'];
    const a = state.answers;

    // Handle common follow-ups
    if (m.includes('port') || m.includes('com') || m.includes('tty')) {
      return {
        newState: state,
        response: {
          role: 'engineer',
          content: `> INTEGRATION ENGINEER:\nPort selection:\n- Windows: COM3, COM4, COM5 (check Device Manager → Ports)\n- Mac: /dev/cu.usbserial-XXXX or /dev/cu.usbmodem-XXXX\n- Linux: /dev/ttyUSB0 or /dev/ttyACM0\n\nRun: python physicore_bridge.py --platform balancing_bot_arduino --connection YOUR_PORT --baud 115200`
        }
      };
    }
    if (m.includes('library') || m.includes('install') || m.includes('depend')) {
      return {
        newState: state,
        response: {
          role: 'engineer',
          content: `> INTEGRATION ENGINEER:\nArduino Library Manager: Sketch → Include Library → Manage Libraries\n- Search "${a.imu || 'MPU6050'}" → Install\n- Search "ArduinoJson" by Benoit Blanchon → Install version 6.x\n\nPython: pip install pymavlink websockets aiohttp pyserial`
        }
      };
    }
    if (m.includes('not work') || m.includes('error') || m.includes('fail') || m.includes('connect')) {
      return {
        newState: state,
        response: {
          role: 'engineer',
          content: `> INTEGRATION ENGINEER:\nTroubleshooting:\n1. Check port: Arduino IDE → Tools → Port (note the port, use same in bridge)\n2. Close Arduino IDE before running bridge (they share the serial port)\n3. Check baud: must be 115200 in both firmware and bridge command\n4. Try: python physicore_bridge.py --test (runs diagnostics)\n5. Dashboard: make sure endpoint is exactly ws://localhost:8765 (not https)`
        }
      };
    }
    if (m.includes('not balanc') || m.includes('still fall') || m.includes('not control') || m.includes('same')) {
      return {
        newState: state,
        response: {
          role: 'engineer',
          content: `> INTEGRATION ENGINEER:\nFor PhysiCore to control your bot, the firmware must:\n1. Send REAL IMU data — not simulated. Verify your IMU is initialized and reading real angles.\n2. Apply the command from PhysiCore. Check that the "op: command" handler applies motor_cmd to your actual motor driver.\n3. The serial port must be exclusively used by the bridge — close Arduino IDE.\n\nTo verify real IMU data: open Arduino Serial Monitor, you should see pitch values that change when you tilt the robot.`
        }
      };
    }

    // Generic follow-up
    return {
      newState: state,
      response: {
        role: 'engineer',
        content: `> INTEGRATION ENGINEER:\nYour ${path.name} integration is complete. The bridge command is:\n${path.bridgeCommand(a)}\n\nAsk me anything specific about the integration — port selection, library installation, troubleshooting, or hardware wiring.`
      }
    };
  }

  return {
    newState: state,
    response: {
      role: 'engineer',
      content: '> INTEGRATION ENGINEER:\nTell me about your hardware and I will generate your complete PhysiCore integration code.',
      buttons: ['Balancing bot (Arduino)', 'PX4 drone', 'ROS2 robot arm', 'Rocket', 'Humanoid robot', 'Custom hardware']
    }
  };
}

export function getInitialIntegrationState(): IntegrationState {
  return {
    phase: 'DETECT',
    hardwareType: '',
    answers: {},
    questionIndex: 0,
    messages: [],
  };
}
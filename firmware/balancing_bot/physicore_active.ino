/*
 * PhysiCore Balancing Bot Firmware v2.0
 * Reads REAL MPU6050 data. Applies PhysiCore commands to real motors.
 *
 * INSTALL LIBRARIES FIRST (Sketch → Include Library → Manage Libraries):
 *   1. "MPU6050_light" by rfetick
 *   2. "ArduinoJson" by Benoit Blanchon (version 6.x)
 *
 * WIRING:
 *   MPU6050: SDA→A4, SCL→A5, VCC→3.3V, GND→GND
 *   L298N:   ENA→5, IN1→4, IN2→3, ENB→6, IN3→7, IN4→8
 */

#include <Wire.h>
#include <MPU6050_light.h>
#include <ArduinoJson.h>

// ── CONFIGURE FOR YOUR HARDWARE ───────────────────────────────────────────
const int L_EN=5, L_IN1=4, L_IN2=3;   // Left motor (L298N)
const int R_EN=6, R_IN1=7, R_IN2=8;   // Right motor (L298N)
const float BALANCE_POINT = 0.0;        // Tune: degrees when robot stands upright
const float MAX_TORQUE    = 100.0;      // Scale factor for PhysiCore torque

// Internal PID (safety fallback when PhysiCore not connected)
const float KP=35.0, KI=0.5, KD=1.2;
// ─────────────────────────────────────────────────────────────────────────

const int BAUD_RATE=115200, LOOP_MS=20;
MPU6050 mpu(Wire);

float pitch=0, pitch_rate=0, motor_l=0, motor_r=0;
bool  physicore_active=false;
unsigned long last_cmd=0, last_tx=0;
float pid_integral=0, prev_error=0;

void setup() {
  Serial.begin(BAUD_RATE);
  Wire.begin();
  pinMode(L_EN,OUTPUT); pinMode(L_IN1,OUTPUT); pinMode(L_IN2,OUTPUT);
  pinMode(R_EN,OUTPUT); pinMode(R_IN1,OUTPUT); pinMode(R_IN2,OUTPUT);
  applyMotors(0);

  byte status = mpu.begin();
  while (status != 0) {
    Serial.println("{\"error\":\"MPU6050 not found — check wiring SDA->A4 SCL->A5\"}");
    delay(500);
    status = mpu.begin();
  }
  Serial.println("{\"status\":\"calibrating\",\"message\":\"Keep robot STILL for 3 seconds\"}");
  mpu.calcOffsets(true, true);
  Serial.println("{\"status\":\"ready\",\"message\":\"PhysiCore firmware ready\"}");
}

void loop() {
  unsigned long now = millis();

  // 1. READ REAL IMU
  mpu.update();
  pitch      = mpu.getAngleX() - BALANCE_POINT;
  pitch_rate = mpu.getGyroX();

  // 2. RECEIVE PHYSICORE COMMANDS
  if (Serial.available() > 0) {
    StaticJsonDocument<256> cmd;
    if (deserializeJson(cmd, Serial) == DeserializationError::Ok) {
      if (strcmp(cmd["op"], "command") == 0) {
        JsonArray action = cmd["action"].as<JsonArray>();
        if (action.size() > 0) {
          float torque = action[0].as<float>();
          motor_l = constrain(torque / MAX_TORQUE, -1.0f, 1.0f);
          motor_r = motor_l;
          physicore_active = true;
          last_cmd = now;
        }
      }
    }
  }

  // 3. SAFETY TIMEOUT
  if (now - last_cmd > 500) physicore_active = false;

  // 4. APPLY CONTROL
  if (physicore_active) {
    applyMotors(motor_l);
  } else {
    float err = -pitch;
    pid_integral = constrain(pid_integral + err*(LOOP_MS/1000.0f), -50.0f, 50.0f);
    float deriv  = (err - prev_error) / (LOOP_MS/1000.0f);
    float v      = constrain((KP*err + KI*pid_integral + KD*deriv)/255.0f, -1.0f, 1.0f);
    prev_error   = err;
    motor_l = motor_r = v;
    applyMotors(v);
  }

  // 5. SEND TELEMETRY AT 50Hz
  if (now - last_tx >= LOOP_MS) {
    last_tx = now;
    StaticJsonDocument<256> doc;
    doc["pitch"]   = pitch;
    doc["roll"]    = mpu.getAngleY();
    doc["gyro_x"]  = pitch_rate;
    doc["gyro_y"]  = mpu.getGyroY();
    doc["gyro_z"]  = mpu.getGyroZ();
    doc["accel_x"] = mpu.getAccX();
    doc["accel_y"] = mpu.getAccY();
    doc["accel_z"] = mpu.getAccZ();
    doc["motor_l"] = motor_l * MAX_TORQUE;
    doc["motor_r"] = motor_r * MAX_TORQUE;
    doc["active"]  = physicore_active;
    doc["timestamp"] = now;
    serializeJson(doc, Serial);
    Serial.println();
  }

  while (millis() - now < LOOP_MS);
}

void applyMotors(float v) {
  int pwm     = constrain((int)(abs(v)*255), 0, 255);
  bool fwd    = (v >= 0);
  digitalWrite(L_IN1, fwd); digitalWrite(L_IN2, !fwd); analogWrite(L_EN, pwm);
  digitalWrite(R_IN1, fwd); digitalWrite(R_IN2, !fwd); analogWrite(R_EN, pwm);
}

/*
 * PhysiCore Active Control Firmware: Balancing Bot
 * ===============================================
 * This firmware allows a self-balancing robot to be controlled 
 * by the PhysiCore Engine via the Universal Bridge.
 * 
 * Features:
 *  - Real-time JSON telemetry (50Hz)
 *  - Command listener for "op: command"
 *  - Seamless handover from internal PID to PhysiCore
 * 
 * Dependencies:
 *  - ArduinoJson (Install via Library Manager)
 *  - MPU6050 (or your preferred IMU library)
 */

#include <ArduinoJson.h>
#include <Wire.h>

// --- CONFIGURATION ---
const int BAUD_RATE = 115200;
const int LOOP_MS   = 20; // 50Hz

// Motor Pins (Example for L298N or similar)
const int L_PWM = 5;
const int L_IN1 = 4;
const int L_IN2 = 3;
const int R_PWM = 6;
const int R_IN1 = 7;
const int R_IN2 = 8;

// --- STATE ---
float pitch = 0.0;
float gyro_y = 0.0;
float motor_cmd = 0.0;
bool  physicore_active = false;
unsigned long last_cmd_time = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  Wire.begin();
  
  pinMode(L_PWM, OUTPUT);
  pinMode(L_IN1, OUTPUT);
  pinMode(L_IN2, OUTPUT);
  pinMode(R_PWM, OUTPUT);
  pinMode(R_IN1, OUTPUT);
  pinMode(R_IN2, OUTPUT);

  // Initialize IMU here...
}

void loop() {
  unsigned long start_time = millis();

  // 1. READ SENSORS
  // Replace with real IMU code (e.g. mpu.getRotation(&gx, &gy, &gz))
  pitch  += (random(-100, 100) / 1000.0); // Simulated drift
  gyro_y  = (random(-50, 50) / 10.0);

  // 2. SEND TELEMETRY TO PHYSICORE
  StaticJsonDocument<256> telemetry;
  telemetry["pitch"]   = pitch;
  telemetry["gyro_y"]  = gyro_y;
  telemetry["motor_l"] = motor_cmd;
  telemetry["motor_r"] = motor_cmd;
  telemetry["active"]  = physicore_active;
  
  serializeJson(telemetry, Serial);
  Serial.println();

  // 3. LISTEN FOR COMMANDS
  if (Serial.available()) {
    StaticJsonDocument<256> cmd;
    DeserializationError err = deserializeJson(cmd, Serial);
    
    if (err == DeserializationError::Ok) {
      if (cmd["op"] == "command") {
        JsonArray action = cmd["action"];
        motor_cmd = action[0]; // PhysiCore sends torque/PWM
        physicore_active = true;
        last_cmd_time = millis();
      }
    }
  }

  // 4. SAFETY TIMEOUT
  // If no command from PhysiCore for 500ms, revert to internal safety
  if (millis() - last_cmd_time > 500) {
    physicore_active = false;
  }

  // 5. APPLY CONTROL
  if (physicore_active) {
    apply_motors(motor_cmd);
  } else {
    // INTERNAL SAFETY PID
    float error = 0.0 - pitch;
    float safety_torque = error * 20.0; // Simple P-gain
    apply_motors(safety_torque);
  }

  // Maintain loop rate
  while (millis() - start_time < LOOP_MS);
}

void apply_motors(float val) {
  // Map -1.0...1.0 to PWM 0...255
  int pwm = constrain(abs(val) * 255, 0, 255);
  bool dir = val > 0;

  digitalWrite(L_IN1, dir);
  digitalWrite(L_IN2, !dir);
  analogWrite(L_PWM, pwm);

  digitalWrite(R_IN1, dir);
  digitalWrite(R_IN2, !dir);
  analogWrite(R_PWM, pwm);
}

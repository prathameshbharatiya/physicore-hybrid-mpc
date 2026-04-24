/*
 * PhysiCore — ATmega328PB Serial Bridge Firmware
 * ================================================
 * Compatible with: ATmega328PB (Insight 2.0, custom flight computers)
 *
 * NOTE: ATmega328PB vs ATmega328P differences handled here:
 *   - ATmega328PB has 2x UART (UART0 + UART1), 2x SPI, 2x I2C (TWI0 + TWI1)
 *   - Uses Wire (TWI0) for I2C — same as standard Arduino Wire library
 *   - UART0 = USB Serial (pins 0/1), UART1 = Serial1 (pins on PD2/PD3 on 328PB)
 *   - Timer/counter registers identical to 328P — Arduino core works directly
 *
 * Sensors supported (configure below):
 *   - MPU-6050 (I2C, ±16g accel, ±2000°/s gyro)
 *   - BMP-388 / BMP-390 (I2C, barometric pressure + temperature)
 *   - BMP-280 (I2C, pressure only — no temperature compensation)
 *
 * Required libraries (install via Arduino Library Manager):
 *   - MPU6050 by Electronic Cats (or Adafruit MPU6050)
 *   - Adafruit BMP3XX (for BMP-388/390) OR Adafruit BMP280
 *   - ArduinoJson by Benoit Blanchon (v6.x)
 *
 * Wiring (ATmega328PB):
 *   MPU-6050: SDA → A4 (PC4/SDA0), SCL → A5 (PC5/SCL0), VCC → 3.3V, GND → GND
 *   BMP-388:  SDA → A4,            SCL → A5,             VCC → 3.3V, GND → GND
 *   (Both share the same I2C bus — I2C addresses are different: 0x68 vs 0x77)
 *
 * PhysiCore receives this JSON at SEND_HZ and sends back:
 *   {"op":"command","action":[TORQUE_VALUE]}
 *
 * Author: PhysiCore — physicore.ai
 */

#include <Wire.h>
#include <ArduinoJson.h>

// ── Sensor selection — uncomment what you have ────────────────────────────────
#define HAS_MPU6050        // Comment out if not using MPU-6050
#define HAS_BMP388         // Comment out if using BMP280 instead
// #define HAS_BMP280      // Uncomment if using BMP-280 instead of BMP-388

#ifdef HAS_MPU6050
  #include <MPU6050.h>
  MPU6050 mpu;
#endif

#ifdef HAS_BMP388
  #include <Adafruit_BMP3XX.h>
  Adafruit_BMP3XX bmp;
#endif

#ifdef HAS_BMP280
  #include <Adafruit_BMP280.h>
  Adafruit_BMP280 bmp;
#endif

// ── Configuration ─────────────────────────────────────────────────────────────
#define SEND_HZ         50      // Telemetry rate to PhysiCore (Hz)
#define BAUD_RATE       115200  // Must match --baud in bridge command
#define GROUND_PRESSURE 1013.25 // hPa — set this at your launch site

// ── Calibration offsets (run calibration sketch first) ────────────────────────
// These correct for IMU mounting offset and sensor bias.
// Defaults are zero — run MPU6050 calibration sketch to get your values.
float ACCEL_OFFSET_X = 0.0;  // m/s²
float ACCEL_OFFSET_Y = 0.0;
float ACCEL_OFFSET_Z = 0.0;
float GYRO_OFFSET_X  = 0.0;  // deg/s
float GYRO_OFFSET_Y  = 0.0;
float GYRO_OFFSET_Z  = 0.0;

// ── IMU Frame — change if your IMU is mounted non-standard ───────────────────
// Options: 1 (no flip) or -1 (flip axis)
// e.g. if Z is pointing down: Z_SIGN = -1
#define X_SIGN  1
#define Y_SIGN  1
#define Z_SIGN  1

// ── State ─────────────────────────────────────────────────────────────────────
float accel_x = 0, accel_y = 0, accel_z = 0;  // m/s²
float gyro_x  = 0, gyro_y  = 0, gyro_z  = 0;  // deg/s
float altitude = 0;   // m AGL (from barometer)
float pressure = 0;   // hPa
float temperature = 0; // °C
float motor_l = 0, motor_r = 0;  // last commanded values
float ground_pressure = GROUND_PRESSURE;

// PhysiCore command received
float last_command = 0.0;
bool physicore_active = false;

// Timing
unsigned long last_send_ms = 0;
const unsigned long SEND_INTERVAL_MS = 1000 / SEND_HZ;

// ── Altitude from pressure ────────────────────────────────────────────────────
float pressure_to_altitude(float pres_hpa, float temp_c) {
  // Hypsometric formula with temperature correction
  float T = temp_c + 273.15;
  return (8.31432 * T / (0.0289644 * 9.80665)) * log(ground_pressure / pres_hpa);
}

void setup() {
  Serial.begin(BAUD_RATE);
  Wire.begin();
  Wire.setClock(400000);  // Fast mode I2C — ATmega328PB supports 400kHz

  // ── MPU-6050 init ──────────────────────────────────────────────────────────
  #ifdef HAS_MPU6050
    mpu.initialize();
    if (!mpu.testConnection()) {
      Serial.println("{\"error\":\"MPU6050 not found. Check wiring: SDA->A4, SCL->A5\"}");
    } else {
      // Set ±16g range and ±2000°/s gyro range
      mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_16);
      mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_2000);
      mpu.setDLPFMode(MPU6050_DLPF_BW_42);  // 42Hz low-pass filter
    }
  #endif

  // ── BMP-388 init ───────────────────────────────────────────────────────────
  #ifdef HAS_BMP388
    if (!bmp.begin_I2C(0x77)) {  // BMP-388 default address 0x77
      if (!bmp.begin_I2C(0x76)) {
        Serial.println("{\"error\":\"BMP388 not found. Check wiring or I2C address (0x76 or 0x77)\"}");
      }
    } else {
      // Configure for high accuracy + reasonable speed
      bmp.setTemperatureOversampling(BMP3_OVERSAMPLING_8X);
      bmp.setPressureOversampling(BMP3_OVERSAMPLING_4X);
      bmp.setIIRFilterCoeff(BMP3_IIR_FILTER_COEFF_3);
      bmp.setOutputDataRate(BMP3_ODR_50_HZ);
      // Set ground pressure reference at startup (average first 5 readings)
      float sum = 0;
      for (int i = 0; i < 5; i++) {
        bmp.performReading();
        sum += bmp.pressure / 100.0;
        delay(100);
      }
      ground_pressure = sum / 5.0;
    }
  #endif

  #ifdef HAS_BMP280
    if (!bmp.begin(0x77)) {
      Serial.println("{\"error\":\"BMP280 not found. Check wiring.\"}");
    } else {
      bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,
                      Adafruit_BMP280::SAMPLING_X2,
                      Adafruit_BMP280::SAMPLING_X16,
                      Adafruit_BMP280::FILTER_X4,
                      Adafruit_BMP280::STANDBY_MS_1);
      float sum = 0;
      for (int i = 0; i < 5; i++) { sum += bmp.readPressure() / 100.0; delay(100); }
      ground_pressure = sum / 5.0;
    }
  #endif

  delay(500);
}

void read_imu() {
  #ifdef HAS_MPU6050
    int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
    mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);

    // Convert raw to SI units
    // MPU-6050 at ±16g: 1g = 2048 LSB
    // MPU-6050 at ±2000°/s: 1°/s = 16.4 LSB
    accel_x = (float(ax_raw) / 2048.0) * 9.81 * X_SIGN + ACCEL_OFFSET_X;
    accel_y = (float(ay_raw) / 2048.0) * 9.81 * Y_SIGN + ACCEL_OFFSET_Y;
    accel_z = (float(az_raw) / 2048.0) * 9.81 * Z_SIGN + ACCEL_OFFSET_Z;
    gyro_x  = float(gx_raw)  / 16.4  * X_SIGN + GYRO_OFFSET_X;
    gyro_y  = float(gy_raw)  / 16.4  * Y_SIGN + GYRO_OFFSET_Y;
    gyro_z  = float(gz_raw)  / 16.4  * Z_SIGN + GYRO_OFFSET_Z;
  #endif
}

void read_baro() {
  #ifdef HAS_BMP388
    if (bmp.performReading()) {
      pressure    = bmp.pressure / 100.0;   // Pa → hPa
      temperature = bmp.temperature;         // °C
      altitude    = pressure_to_altitude(pressure, temperature);
    }
  #endif
  #ifdef HAS_BMP280
    pressure    = bmp.readPressure() / 100.0;
    temperature = bmp.readTemperature();
    altitude    = pressure_to_altitude(pressure, temperature);
  #endif
}

void send_telemetry() {
  StaticJsonDocument<256> doc;

  doc["pitch"]       = gyro_x > 100 ? accel_x / 9.81 * 57.3 : accel_x / 9.81 * 57.3;  // rough pitch from accel
  doc["roll"]        = accel_y / 9.81 * 57.3;
  doc["accel_x"]     = accel_x;
  doc["accel_y"]     = accel_y;
  doc["accel_z"]     = accel_z;
  doc["gyro_x"]      = gyro_x;   // deg/s — PhysiCore expects deg/s
  doc["gyro_y"]      = gyro_y;
  doc["gyro_z"]      = gyro_z;
  doc["altitude"]    = altitude;  // m AGL
  doc["pressure"]    = pressure;  // hPa — for PhysiCore temperature compensation
  doc["temperature"] = temperature; // °C
  doc["motor_l"]     = motor_l;
  doc["motor_r"]     = motor_r;
  doc["timestamp"]   = millis();  // ms — PhysiCore uses for dt calculation

  serializeJson(doc, Serial);
  Serial.println();  // PhysiCore reads line-by-line
}

void read_command() {
  // Read PhysiCore command if available
  // Format: {"op":"command","action":[TORQUE]}
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    StaticJsonDocument<128> cmd;
    if (deserializeJson(cmd, line) == DeserializationError::Ok) {
      if (cmd["op"] == "command") {
        JsonArray action = cmd["action"];
        if (action.size() > 0) {
          last_command = action[0];
          physicore_active = true;
          // ── APPLY COMMAND TO YOUR ACTUATORS HERE ──────────────────────────
          // For a balancing bot: apply torque to motors
          // motor_l = constrain(last_command, -255, 255);
          // motor_r = constrain(-last_command, -255, 255);  // opposite sign for balance
          // analogWrite(MOTOR_L_PIN, abs(motor_l));
          // analogWrite(MOTOR_R_PIN, abs(motor_r));
          // ─────────────────────────────────────────────────────────────────
        }
      }
    }
  }
}

void loop() {
  read_imu();

  // BMP-388 at 32Hz — only read every ~31ms
  static unsigned long last_baro = 0;
  if (millis() - last_baro > 31) {
    read_baro();
    last_baro = millis();
  }

  read_command();

  // Send telemetry at SEND_HZ
  if (millis() - last_send_ms >= SEND_INTERVAL_MS) {
    send_telemetry();
    last_send_ms = millis();
  }
}

/*
 * ── Bridge command to connect ─────────────────────────────────────────────────
 *
 * Windows:
 *   python physicore/bridge/physicore_bridge.py --platform custom_rocket_fc --connection COM3 --baud 115200
 *
 * Mac:
 *   python physicore/bridge/physicore_bridge.py --platform custom_rocket_fc --connection /dev/cu.usbserial-0001 --baud 115200
 *   (Find your port: ls /dev/cu.* in terminal)
 *
 * Linux:
 *   python physicore/bridge/physicore_bridge.py --platform custom_rocket_fc --connection /dev/ttyUSB0 --baud 115200
 *   (Find your port: ls /dev/ttyUSB* or ls /dev/ttyACM*)
 *
 * With IMU frame correction (if mounted sideways):
 *   python physicore/bridge/physicore_bridge.py --platform custom_rocket_fc --connection /dev/ttyUSB0 --imu-frame z_down
 *
 * ── Troubleshooting ───────────────────────────────────────────────────────────
 *
 * "MPU6050 not found":
 *   → Check SDA→A4, SCL→A5. Both need 4.7kΩ pull-ups to 3.3V.
 *   → ATmega328PB has two I2C buses. Wire.h uses TWI0 (A4/A5). If you wired to TWI1, use Wire1.begin().
 *
 * "BMP388 not found":
 *   → Check I2C address. SDO pin LOW = 0x76, SDO pin HIGH (or floating) = 0x77.
 *   → Try both addresses in begin_I2C() call.
 *
 * Altitude jumping wildly:
 *   → Normal during ejection charge firing — PhysiCore's pressure spike filter handles this.
 *   → If jumping on ground: electromagnetic interference from motors. Add decoupling caps near BMP388.
 *
 * PhysiCore residual not decreasing:
 *   → Check that accel_z = 9.81 when stationary (correct sign and units).
 *   → Check that gyro = 0 when stationary (bias correction working).
 *   → Make sure BAUD_RATE matches --baud in bridge command.
 */

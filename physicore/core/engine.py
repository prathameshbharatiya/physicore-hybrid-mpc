
import numpy as np
import math

# ──────────────────────────────────────────────────────────────────────────────
#  Pre-built Platform Dynamics
# ──────────────────────────────────────────────────────────────────────────────

def quadrotor_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    6-DOF quadrotor dynamics in body frame.
    State:  [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]  (12-dim)
    Action: [thrust, roll_cmd, pitch_cmd, yaw_cmd]               (4-dim)
    """
    m   = params.get("mass", 1.5)
    b   = params.get("friction", 0.1)
    g   = 9.81
    
    # Simple 6-DOF model for demonstration
    x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r = state
    thrust, roll_cmd, pitch_cmd, yaw_cmd = action
    
    # Acceleration in body frame
    ax = (thrust / m) * (math.sin(roll) * math.sin(yaw) + math.cos(roll) * math.cos(yaw) * math.sin(pitch)) - b * vx
    ay = (thrust / m) * (math.cos(roll) * math.sin(pitch) * math.sin(yaw) - math.cos(yaw) * math.sin(roll)) - b * vy
    az = (thrust / m) * (math.cos(roll) * math.cos(pitch)) - g - b * vz
    
    return np.array([vx, vy, vz, ax, ay, az, p, q, r, roll_cmd, pitch_cmd, yaw_cmd])

def rover_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    2D Rover dynamics.
    State: [x, y, theta, v, omega]
    Action: [throttle, steer]
    """
    m = params.get("mass", 5.0)
    b = params.get("friction", 0.5)
    
    x, y, theta, v, omega = state
    throttle, steer = action
    
    dv = (throttle / m) - b * v
    dtheta = omega
    dx = v * math.cos(theta)
    dy = v * math.sin(theta)
    domega = steer # Simplified
    
def rk4_step(state: np.ndarray, action: np.ndarray, params: dict, dynamics_fn, dt: float = 0.01) -> np.ndarray:
    """
    Standard RK4 integration step.
    """
    k1 = dynamics_fn(state, action, params)
    k2 = dynamics_fn(state + k1 * dt / 2, action, params)
    k3 = dynamics_fn(state + k2 * dt / 2, action, params)
    k4 = dynamics_fn(state + k3 * dt, action, params)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class PhysicoreSimulator:
    def __init__(self, platform: str = "quadrotor", params: dict = None):
        self.platform = platform
        self.params = params or {}
        
        if platform == "quadrotor":
            self.dynamics_fn = quadrotor_dynamics
            self.state = np.zeros(12)
        elif platform == "rover":
            self.dynamics_fn = rover_dynamics
            self.state = np.zeros(5)
        else:
            raise ValueError(f"Unknown platform: {platform}")
            
    def step(self, action: np.ndarray, dt: float = 0.01):
        self.state = rk4_step(self.state, action, self.params, self.dynamics_fn, dt)
        return self.state

    def reset(self):
        self.state = np.zeros_like(self.state)
        return self.state

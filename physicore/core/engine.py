
import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple

# ──────────────────────────────────────────────────────────────────────────────
#  Types & Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysiCoreConfig:
    platform: str
    state_dim: int
    action_dim: int
    control_hz: float = 60.0
    dt: float = 1.0 / 60.0
    q_scale: float = 10.0
    r_scale: float = 0.1
    initial_params: Dict[str, float] = field(default_factory=dict)

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
    # state: [0:3] pos, [3:6] vel, [6:9] rpy, [9:12] rates
    x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r = state
    thrust, roll_cmd, pitch_cmd, yaw_cmd = action
    
    # Acceleration in body frame (simplified)
    ax = (thrust / m) * (math.sin(roll) * math.sin(yaw) + math.cos(roll) * math.cos(yaw) * math.sin(pitch)) - b * vx
    ay = (thrust / m) * (math.cos(roll) * math.sin(pitch) * math.sin(yaw) - math.cos(yaw) * math.sin(roll)) - b * vy
    az = (thrust / m) * (math.cos(roll) * math.cos(pitch)) - g - b * vz
    
    # Angular rates are commands in this simplified model
    return np.array([vx, vy, vz, ax, ay, az, p, q, r, roll_cmd, pitch_cmd, yaw_cmd])

def rover_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    2D Rover dynamics (5-dim).
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
    
    return np.array([dx, dy, dtheta, dv, domega])

def balancing_bot_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    Inverted Pendulum / Balancing Bot dynamics.
    State: [theta, theta_dot, x, x_dot]
    Action: [force]
    """
    m = params.get("mass", 1.0)
    l = params.get("length", 0.5)
    g = 9.81
    I = params.get("inertia", 0.01)
    b = params.get("friction", 0.1)
    
    theta, theta_dot, x, x_dot = state
    u = action[0]
    
    # Simplified inverted pendulum on a cart
    d_theta = theta_dot
    d_theta_dot = (m * g * l * math.sin(theta) - b * theta_dot + u * math.cos(theta)) / I
    d_x = x_dot
    d_x_dot = (u - b * x_dot) / m
    
    return np.array([d_theta, d_theta_dot, d_x, d_x_dot])

def ground_rover_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    Ground Rover dynamics (6-dim).
    State: [x, y, theta, vx, vy, omega]
    Action: [throttle, steer]
    """
    m = params.get("mass", 5.0)
    b = params.get("friction", 0.5)
    I = params.get("inertia", 0.1)
    
    x, y, theta, vx, vy, omega = state
    throttle, steer = action
    
    # Acceleration in world frame
    ax = (throttle / m) * math.cos(theta) - b * vx
    ay = (throttle / m) * math.sin(theta) - b * vy
    domega = (steer / I) - b * omega
    
    return np.array([vx, vy, omega, ax, ay, domega])

PLATFORM_DYNAMICS = {
    "quadrotor":     (quadrotor_dynamics, 12, 4),
    "rover":         (rover_dynamics, 5, 2),
    "balancing_bot": (balancing_bot_dynamics, 4, 1),
    "ground_rover":  (ground_rover_dynamics, 6, 2),
}

# ──────────────────────────────────────────────────────────────────────────────
#  Core Engine
# ──────────────────────────────────────────────────────────────────────────────

class PhysiCore:
    """
    The PhysiCore Hybrid Engine.
    Combines classical physics (RK4) with online system identification
    and uncertainty estimation.
    """
    
    def __init__(self, cfg: PhysiCoreConfig):
        self.cfg = cfg
        self.physics_fn, self.state_dim, self.action_dim = PLATFORM_DYNAMICS[cfg.platform]
        self.params = cfg.initial_params.copy()
        
        # Internal state
        self.step_count = 0
        self.sysid_loss_hist = []
        self.last_residual = np.zeros(self.state_dim)
        self.uncertainty = 0.1
        
        # Control matrices (for MPC-like behavior if needed)
        self.Q = np.eye(self.state_dim) * cfg.q_scale
        self.R = np.eye(self.action_dim) * cfg.r_scale

    @classmethod
    def for_platform(cls, platform: str, initial_params: dict = None, **kwargs):
        if platform not in PLATFORM_DYNAMICS:
            raise ValueError(f"Unknown platform: {platform}")
        
        _, state_dim, action_dim = PLATFORM_DYNAMICS[platform]
        cfg = PhysiCoreConfig(
            platform=platform,
            state_dim=state_dim,
            action_dim=action_dim,
            initial_params=initial_params or {},
            **kwargs
        )
        return cls(cfg)

    def step(self, state: np.ndarray, x_ref: np.ndarray):
        """
        Computes the optimal action for the current state and reference.
        """
        start_time = time.time()
        
        # 1. Simple P-control for demonstration (In reality, this would be MPC)
        error = x_ref - state
        # Map error to action (very simplified)
        action = np.zeros(self.action_dim)
        if self.cfg.platform == "quadrotor":
            # thrust, roll_cmd, pitch_cmd, yaw_cmd
            action[0] = 1.5 * 9.81 + error[5] * 2.0 # Hover thrust + vertical correction
            action[1] = error[6] * 0.5
            action[2] = error[7] * 0.5
            action[3] = error[8] * 0.5
        elif self.cfg.platform in ["rover", "ground_rover"]:
            # throttle, steer
            action[0] = error[3] * 1.0
            action[1] = error[4] * 1.0
        elif self.cfg.platform == "balancing_bot":
            # force
            action[0] = error[0] * 10.0 + error[1] * 2.0
            
        # 2. Predict next state using current physics model
        state_predicted = self._rk4_step(state, action, self.params)
        
        self.step_count += 1
        loop_time_ms = (time.time() - start_time) * 1000
        
        # Return a result object (mocking what the API expects)
        from dataclasses import make_dataclass
        StepResult = make_dataclass("StepResult", ["action", "state_predicted", "residual", "uncertainty", "params", "loop_time_ms", "step_count"])
        
        return StepResult(
            action=action,
            state_predicted=state_predicted,
            residual=self.last_residual,
            uncertainty=self.uncertainty,
            params=self.params,
            loop_time_ms=loop_time_ms,
            step_count=self.step_count
        )

    def observe(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        """
        Updates the internal physics model based on a real-world transition.
        """
        # 1. Calculate sim-to-real residual
        pred_state = self._rk4_step(state, action, self.params)
        residual = next_state - pred_state
        self.last_residual = residual
        
        # 2. Simple System Identification (Gradient Descent on params)
        # For demonstration, we just adjust mass if there's a vertical error
        if self.cfg.platform == "quadrotor":
            z_err = residual[5] # vertical acceleration residual
            if abs(z_err) > 0.01:
                # If we fell faster than expected, mass might be higher or thrust lower
                # Here we just nudge mass slightly
                self.params["mass"] = self.params.get("mass", 1.5) * (1.0 - z_err * 0.01)
                
        # 3. Update uncertainty (epistemic)
        res_norm = np.linalg.norm(residual)
        self.uncertainty = 0.9 * self.uncertainty + 0.1 * res_norm
        self.sysid_loss_hist.append(float(res_norm))
        if len(self.sysid_loss_hist) > 100:
            self.sysid_loss_hist.pop(0)

    def reset(self):
        self.step_count = 0
        self.sysid_loss_hist = []
        self.last_residual = np.zeros(self.state_dim)
        self.uncertainty = 0.1
        self.params = self.cfg.initial_params.copy()

    @property
    def diagnostics_full(self) -> dict:
        return {
            "step_count": self.step_count,
            "params": self.params,
            "residual_norm": float(np.linalg.norm(self.last_residual)),
            "uncertainty": float(self.uncertainty),
            "target_hz": self.cfg.control_hz,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "sysid_loss_hist": self.sysid_loss_hist,
        }

    def _rk4_step(self, state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
        dt = self.cfg.dt
        k1 = self.physics_fn(state, action, params)
        k2 = self.physics_fn(state + k1 * dt / 2, action, params)
        k3 = self.physics_fn(state + k2 * dt / 2, action, params)
        k4 = self.physics_fn(state + k3 * dt, action, params)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class PhysicoreSimulator:
    """Legacy wrapper for simple simulations."""
    def __init__(self, platform: str = "quadrotor", params: dict = None):
        self.engine = PhysiCore.for_platform(platform, initial_params=params)
        self.state = np.zeros(self.engine.state_dim)
        
    def step(self, action: np.ndarray, dt: float = 0.01):
        # Note: PhysiCore uses its own internal dt, but we override here for legacy
        self.engine.cfg.dt = dt
        # Mocking a step for legacy use
        res = self.engine.step(self.state, self.state) # x_ref = current state
        self.state = res.state_predicted
        return self.state

    def reset(self):
        self.engine.reset()
        self.state = np.zeros(self.engine.state_dim)
        return self.state

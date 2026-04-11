
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

def manipulator_arm_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    6-DOF manipulator arm (joint space).
    State:  [q1,q2,q3,q4,q5,q6, dq1,dq2,dq3,dq4,dq5,dq6]  12-dim
    Action: [tau1,tau2,tau3,tau4,tau5,tau6]                   6-dim
    """
    n     = 6
    q     = state[:n]
    dq    = state[n:]
    tau   = action
    m     = params.get("mass", 2.0)
    fric  = params.get("friction", 0.3)
    M     = np.ones(n) * m * 0.1 + np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    ddq   = (tau - fric * dq) / M
    return np.concatenate([dq, ddq])


def fixed_wing_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    Fixed-wing aircraft 6-DOF.
    State:  [x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r]  12-dim
    Action: [throttle,aileron,elevator,rudder]        4-dim
    """
    m      = params.get("mass", 12.5)
    cd0    = params.get("friction", 0.025)
    cla    = params.get("inertia", 5.7)
    g      = 9.81
    x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r = state
    throttle,aileron,elevator,rudder = action
    v      = max(np.sqrt(vx**2+vy**2+vz**2), 1.0)
    lift   = 0.5*1.225*v**2*cla*0.85*np.deg2rad(pitch)
    drag   = 0.5*1.225*v**2*cd0*0.85
    ax     = throttle*10.0/m - drag*vx/(m*v)
    ay     = -drag*vy/(m*v)
    az     = lift/m - g
    dp,dq_,dr = aileron*2.0, elevator*2.0, rudder*1.5
    return np.array([vx,vy,vz,ax,ay,az,p,q,r,dp,dq_,dr])


def evtol_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    eVTOL transition dynamics (VTOL + cruise blend).
    State:  [x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r]  12-dim
    Action: [thrust,roll_cmd,pitch_cmd,fwd_thrust]    4-dim
    """
    m      = params.get("mass", 500.0)
    b      = params.get("friction", 0.05)
    g      = 9.81
    x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r = state
    thrust,roll_cmd,pitch_cmd,fwd_thrust = action
    v      = max(np.sqrt(vx**2+vy**2+vz**2), 0.1)
    transition_ratio = min(v / 30.0, 1.0)
    lift_vtol  = (thrust / m) * np.cos(pitch) * np.cos(roll)
    lift_wing  = 0.5*1.225*v**2*5.0*0.85*np.deg2rad(pitch)/m
    ax  = fwd_thrust/m - b*vx/m
    ay  = -b*vy/m
    az  = (1-transition_ratio)*lift_vtol + transition_ratio*lift_wing - g
    dp  = (roll_cmd - p) / 0.08
    dq_ = (pitch_cmd - q) / 0.08
    dr  = -r * 0.5
    return np.array([vx,vy,vz,ax,ay,az,p,q,r,dp,dq_,dr])


def legged_robot_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    Legged robot (biped/quadruped) simplified whole-body dynamics.
    State:  [x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r]  12-dim
    Action: [fx,fy,fz,tau_roll,tau_pitch,tau_yaw]    6-dim
    """
    m    = params.get("mass", 30.0)
    fric = params.get("friction", 0.7)
    g    = 9.81
    x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r = state
    fx,fy,fz,tr,tp,ty_ = action
    ax   = fx/m - fric*vx/m
    ay   = fy/m - fric*vy/m
    az   = fz/m - g - fric*vz/m
    Ixx  = params.get("inertia", 0.5)
    dp   = tr/Ixx - fric*p
    dq_  = tp/Ixx - fric*q
    dr   = ty_/Ixx - fric*r
    return np.array([vx,vy,vz,ax,ay,az,p,q,r,dp,dq_,dr])


def auv_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    Autonomous Underwater Vehicle dynamics.
    State:  [x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r]  12-dim
    Action: [surge,sway,heave,yaw_rate]               4-dim
    """
    m    = params.get("mass", 50.0)
    drag = params.get("friction", 2.0)
    g    = 9.81
    buoy = params.get("inertia", 0.02)
    x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r = state
    surge,sway,heave,yaw_cmd = action
    ax   = surge/m - drag*vx/m
    ay   = sway/m  - drag*vy/m
    az   = heave/m - drag*vz/m + buoy
    dp   = -drag*p*0.1
    dq_  = -drag*q*0.1
    dr   = (yaw_cmd - r) / 0.1
    return np.array([vx,vy,vz,ax,ay,az,p,q,r,dp,dq_,dr])


def satellite_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    Spacecraft/satellite attitude control dynamics.
    State:  [x,y,z,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz]  13-dim — approximated as 12
    Action: [Tx,Ty,Tz,thrust]                          4-dim
    """
    m    = params.get("mass", 100.0)
    Ixx  = params.get("inertia", 10.0)
    drag = params.get("friction", 1e-5)
    x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r = state
    Tx,Ty,Tz,thrust = action
    ax   = thrust*np.cos(pitch)*np.cos(yaw)/m - drag*vx
    ay   = thrust*np.cos(pitch)*np.sin(yaw)/m - drag*vy
    az   = thrust*np.sin(pitch)/m            - drag*vz
    dp   = Tx/Ixx - (Ixx*0.1)*q*r/Ixx
    dq_  = Ty/Ixx - (Ixx*0.1)*p*r/Ixx
    dr   = Tz/Ixx - (Ixx*0.1)*p*q/Ixx
    return np.array([vx,vy,vz,ax,ay,az,p,q,r,dp,dq_,dr])


def surgical_robot_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    Surgical robot / micro-manipulator dynamics.
    State:  [q1..q6, dq1..dq6]  12-dim (same as manipulator but sub-mm scale)
    Action: [tau1..tau6]          6-dim
    """
    n     = 6
    dq    = state[n:]
    tau   = action
    m     = params.get("mass", 0.05)
    fric  = params.get("friction", 0.8)
    tissue_k = params.get("inertia", 0.1)
    M     = np.ones(n) * m * 0.001 + np.array([0.005,0.004,0.003,0.002,0.001,0.0005])
    ddq   = (tau - fric*dq - tissue_k*state[:n]) / M
    return np.concatenate([dq, ddq])

def rocket_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    Simplified 2D Rocket dynamics.
    State: [x, y, theta, vx, vy, omega]
    Action: [thrust, gimbal]
    """
    m = params.get("mass", 500.0)
    g = 9.81
    x, y, theta, vx, vy, omega = state
    thrust, gimbal = action
    ax = (thrust / m) * math.sin(theta + gimbal)
    ay = (thrust / m) * math.cos(theta + gimbal) - g
    domega = gimbal * 10.0 # Simplified
    return np.array([vx, vy, omega, ax, ay, domega])

PLATFORM_DYNAMICS = {
    "quadrotor":         (quadrotor_dynamics,       12, 4),
    "fixed_wing":        (fixed_wing_dynamics,       12, 4),
    "evtol":             (evtol_dynamics,            12, 4),
    "manipulator_arm":   (manipulator_arm_dynamics,  12, 6),
    "surgical_robot":    (surgical_robot_dynamics,   12, 6),
    "legged_robot":      (legged_robot_dynamics,     12, 6),
    "balancing_bot":     (balancing_bot_dynamics,     4, 1),
    "rocket":            (rocket_dynamics,            6, 2),
    "ground_rover":      (ground_rover_dynamics,      6, 2),
    "auv":               (auv_dynamics,              12, 4),
    "satellite":         (satellite_dynamics,         12, 4),
    "rover":             (ground_rover_dynamics,      6, 2),
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

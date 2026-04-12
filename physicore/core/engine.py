"""
PhysiCore Core Engine v1.2.0
============================
Hybrid Uncertainty-Aware Sim-to-Real Synchronization Engine.

Architecture:
  Physics Layer (RK4) → Residual Ensemble → CEM-MPC → Online System ID

Paper: Prathamesh Shirbhate — Independent Research
"""

from __future__ import annotations
import numpy as np
import time
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, List

# ── Platform Dynamics ─────────────────────────────────────────────────────────

def quadrotor_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    m   = params.get("mass", 1.5)
    b   = params.get("friction", 0.1)
    g   = 9.81
    x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r = state
    thrust,roll_cmd,pitch_cmd,yaw_cmd = action
    ax = (thrust/m)*(np.cos(yaw)*np.sin(pitch)+np.sin(yaw)*np.sin(roll)) - b*vx/m
    ay = (thrust/m)*(np.sin(yaw)*np.sin(pitch)-np.cos(yaw)*np.sin(roll)) - b*vy/m
    az = (thrust/m)*np.cos(pitch)*np.cos(roll) - g - b*vz/m
    return np.array([vx,vy,vz,ax,ay,az,p,q,r,(roll_cmd-p)/0.05,(pitch_cmd-q)/0.05,(yaw_cmd-r)/0.05])

def fixed_wing_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    m=params.get("mass",12.5); cd0=params.get("friction",0.025); cla=params.get("inertia",5.7); g=9.81
    x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r = state
    throttle,aileron,elevator,rudder = action
    v=max(np.sqrt(vx**2+vy**2+vz**2),1.0)
    lift=0.5*1.225*v**2*cla*0.85*np.deg2rad(pitch); drag=0.5*1.225*v**2*cd0*0.85
    ax=throttle*10.0/m-drag*vx/(m*v); ay=-drag*vy/(m*v); az=lift/m-g
    return np.array([vx,vy,vz,ax,ay,az,p,q,r,aileron*2.0,elevator*2.0,rudder*1.5])

def evtol_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    m=params.get("mass",500.0); b=params.get("friction",0.05); g=9.81
    x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r = state
    thrust,roll_cmd,pitch_cmd,fwd_thrust = action
    v=max(np.sqrt(vx**2+vy**2+vz**2),0.1); tr=min(v/30.0,1.0)
    az=(1-tr)*(thrust/m)*np.cos(pitch)*np.cos(roll)+tr*0.5*1.225*v**2*5.0*0.85*np.deg2rad(pitch)/m-g
    return np.array([vx,vy,vz,fwd_thrust/m-b*vx/m,-b*vy/m,az,p,q,r,(roll_cmd-p)/0.08,(pitch_cmd-q)/0.08,-r*0.5])

def manipulator_arm_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    n=6; dq=state[n:]; m=params.get("mass",2.0); fric=params.get("friction",0.3)
    M=np.ones(n)*m*0.1+np.array([0.5,0.4,0.3,0.2,0.1,0.05])
    return np.concatenate([dq,(action-fric*dq)/M])

def surgical_robot_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    n=6; dq=state[n:]; m=params.get("mass",0.05); fric=params.get("friction",0.8); tk=params.get("inertia",0.1)
    M=np.ones(n)*m*0.001+np.array([0.005,0.004,0.003,0.002,0.001,0.0005])
    return np.concatenate([dq,(action-fric*dq-tk*state[:n])/M])

def legged_robot_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    m=params.get("mass",30.0); fric=params.get("friction",0.7); Ixx=params.get("inertia",0.5); g=9.81
    x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r = state
    fx,fy,fz,tr,tp,ty_ = action
    return np.array([vx,vy,vz,fx/m-fric*vx/m,fy/m-fric*vy/m,fz/m-g-fric*vz/m,p,q,r,tr/Ixx-fric*p,tp/Ixx-fric*q,ty_/Ixx-fric*r])

def balancing_bot_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    pitch,pitch_rate,x_pos,x_vel = state; torque=action[0]
    m=params.get("mass",1.0); l=params.get("friction",0.15); I=params.get("inertia",0.01); g=9.81
    ddpitch=(m*g*l*np.sin(pitch)-torque*np.cos(pitch))/(I+m*l**2)
    ddx=(torque-m*l*ddpitch*np.cos(pitch))/m
    return np.array([pitch_rate,ddpitch,x_vel,ddx])

def rocket_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    x,y,vx,vy,mass,angle = state; thrust_mag,gimbal = action
    cd=params.get("friction",0.45); isp=params.get("inertia",220.0); dia=params.get("mass",0.15); g=9.80665
    alt=max(y,0); T=288.15-0.0065*alt
    rho=101325*(1-0.0065*alt/288.15)**(9.80665/(287.05*0.0065))/(287.05*max(T,1))
    area=math.pi*(dia/2)**2; v=math.sqrt(vx**2+vy**2); drag=0.5*rho*v**2*cd*area
    ta=angle+gimbal
    Ftx=thrust_mag*math.sin(ta); Fty=thrust_mag*math.cos(ta)
    Fdx=-drag*(vx/v) if v>0.1 else 0.0; Fdy=-drag*(vy/v) if v>0.1 else 0.0
    dm=-thrust_mag/(g*isp) if thrust_mag>0 else 0.0
    return np.array([vx,vy,(Ftx+Fdx)/mass,(Fty+Fdy-mass*g)/mass,dm,0.0])

def ground_rover_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    x,y,theta,vx,vy,omega = state; v_left,v_right = action
    m=params.get("mass",5.0); fric=params.get("friction",0.5); wb=0.3
    v=(v_left+v_right)/2.0; w=(v_right-v_left)/wb
    return np.array([vx,vy,omega,(v*np.cos(theta)-fric*vx)/m,(v*np.sin(theta)-fric*vy)/m,(w-fric*omega)/(m*wb**2/12)])

def rover_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    return ground_rover_dynamics(state, action, params)

def auv_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    m=params.get("mass",50.0); drag=params.get("friction",2.0); buoy=params.get("inertia",0.02)
    x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r = state; surge,sway,heave,yaw_cmd = action
    return np.array([vx,vy,vz,surge/m-drag*vx/m,sway/m-drag*vy/m,heave/m-drag*vz/m+buoy,p,q,r,-drag*p*0.1,-drag*q*0.1,(yaw_cmd-r)/0.1])

def satellite_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    m=params.get("mass",100.0); Ixx=params.get("inertia",10.0); drag=params.get("friction",1e-5)
    x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r = state; Tx,Ty,Tz,thrust = action
    return np.array([vx,vy,vz,
        thrust*np.cos(pitch)*np.cos(yaw)/m-drag*vx,
        thrust*np.cos(pitch)*np.sin(yaw)/m-drag*vy,
        thrust*np.sin(pitch)/m-drag*vz,
        p,q,r,Tx/Ixx,Ty/Ixx,Tz/Ixx])

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
    "rover":             (ground_rover_dynamics,      6, 2),
    "auv":               (auv_dynamics,              12, 4),
    "satellite":         (satellite_dynamics,         12, 4),
}

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PhysiCoreConfig:
    state_dim:   int   = 12
    action_dim:  int   = 4
    control_hz:  float = 60.0
    dt:          float = field(init=False)
    horizon:     int   = 10
    cem_samples: int   = 12
    cem_elites:  int   = 4
    cem_iters:   int   = 3
    cem_min_std: float = 1e-3
    lam_unc:     float = 0.1
    ensemble_size: int = 3
    hidden_dim:  int   = 64
    residual_lr: float = 1e-3
    sysid_lr:    float = 0.01
    sysid_every: int   = 50
    param_bounds: dict = field(default_factory=lambda: {
        "mass":     (0.1, 1000.0),
        "friction": (0.0, 10.0),
        "inertia":  (1e-4, 1000.0),
    })

    def __post_init__(self):
        self.dt = 1.0 / self.control_hz

# ── Physics Layer ─────────────────────────────────────────────────────────────

class PhysicsLayer:
    def __init__(self, dynamics_fn: Callable, params: dict):
        self.dynamics_fn = dynamics_fn
        self.params = params.copy()

    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.dynamics_fn(state, action, self.params)
        k2 = self.dynamics_fn(state + dt*k1/2, action, self.params)
        k3 = self.dynamics_fn(state + dt*k2/2, action, self.params)
        k4 = self.dynamics_fn(state + dt*k3,   action, self.params)
        return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def rollout(self, state: np.ndarray, actions: np.ndarray, dt: float) -> np.ndarray:
        traj = [state]
        x = state.copy()
        for u in actions:
            x = self.step(x, u, dt)
            traj.append(x)
        return np.array(traj)

    def update_params(self, new_params: dict):
        self.params.update(new_params)

# ── Residual MLP ──────────────────────────────────────────────────────────────

class ResidualMLP:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, lr: float):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        inp = state_dim + action_dim
        self.W1 = np.random.randn(inp, hidden_dim)  * np.sqrt(2.0/inp)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0/hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, state_dim)  * np.sqrt(2.0/hidden_dim)
        self.b3 = np.zeros(state_dim)
        self.lr = lr
        self._replay: list = []

    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x  = np.concatenate([state, action])
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def add_experience(self, state: np.ndarray, action: np.ndarray, residual: np.ndarray):
        self._replay.append((np.concatenate([state, action]), residual))
        if len(self._replay) > 10000:
            self._replay.pop(0)

    def update(self, batch_size: int = 32):
        if len(self._replay) < batch_size:
            return
        idxs = np.random.choice(len(self._replay), batch_size, replace=False)
        total_loss = 0.0
        for i in idxs:
            inp, target = self._replay[i]
            state  = inp[:self.state_dim]
            action = inp[self.state_dim:]
            pred   = self.forward(state, action)
            err    = pred - target
            total_loss += float(np.sum(err**2))
            grad = 2 * err / batch_size
            self.W3 -= self.lr * np.outer(np.maximum(0, inp @ self.W1 + self.b1) @ self.W2, grad)
            self.b3 -= self.lr * grad

# ── Residual Ensemble ─────────────────────────────────────────────────────────

class ResidualEnsemble:
    def __init__(self, cfg: PhysiCoreConfig):
        self.members = [
            ResidualMLP(cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.residual_lr)
            for _ in range(cfg.ensemble_size)
        ]
        self.batch_size = 32

    def predict(self, state: np.ndarray, action: np.ndarray):
        preds = np.array([m.forward(state, action) for m in self.members])
        residual    = preds.mean(axis=0)
        uncertainty = float(np.mean(np.var(preds, axis=0)))
        return residual, uncertainty

    def add_experience(self, state, action, sim_pred, real_next):
        target = real_next - sim_pred
        for m in self.members:
            m.add_experience(state, action, target)

    def update_all(self):
        for m in self.members:
            m.update(self.batch_size)

# ── CEM Optimizer ─────────────────────────────────────────────────────────────

class CEMOptimizer:
    def __init__(self, cfg: PhysiCoreConfig, action_bounds=None):
        self.H   = cfg.horizon
        self.M   = cfg.cem_samples
        self.I   = cfg.cem_iters
        self.K   = max(1, cfg.cem_elites)
        self.lam = cfg.lam_unc
        self.min_std    = cfg.cem_min_std
        self.action_dim = cfg.action_dim
        self.bounds     = action_bounds
        self.mu  = np.zeros((self.H, cfg.action_dim))
        self.std = np.ones((self.H,  cfg.action_dim))

    def optimize(self, state, physics, ensemble, Q, R, x_ref, dt):
        for _ in range(self.I):
            seqs = np.random.normal(self.mu[np.newaxis], self.std[np.newaxis], (self.M, self.H, self.action_dim))
            if self.bounds is not None:
                seqs = np.clip(seqs, self.bounds[0], self.bounds[1])
            costs = np.array([self._cost(state, seqs[j], physics, ensemble, Q, R, x_ref, dt) for j in range(self.M)])
            elite = seqs[np.argsort(costs)[:self.K]]
            self.mu  = elite.mean(axis=0)
            self.std = np.maximum(elite.std(axis=0), self.min_std)
        u = self.mu[0].copy()
        self.mu  = np.roll(self.mu,  -1, axis=0); self.mu[-1]  = 0
        self.std = np.roll(self.std, -1, axis=0); self.std[-1] = 1
        return u

    def _cost(self, state, actions, physics, ensemble, Q, R, x_ref, dt):
        x = state.copy(); total = 0.0
        for u in actions:
            x_sim  = physics.step(x, u, dt)
            res, s2 = ensemble.predict(x, u)
            x      = x_sim + res
            dx     = x - x_ref
            total += float(dx@Q@dx + u@R@u) + self.lam*s2
        return total

# ── Online System ID ──────────────────────────────────────────────────────────

class OnlineSystemID:
    def __init__(self, cfg: PhysiCoreConfig, initial_params: dict):
        self.lr      = cfg.sysid_lr
        self.bounds  = cfg.param_bounds
        self.params  = initial_params.copy()
        self.every   = cfg.sysid_every
        self._step   = 0
        self._history: list = []

    def update(self, state, action, next_state_real, physics):
        self._step += 1
        if self._step % self.every != 0:
            return self.params
        eps = 1e-4
        for name in list(self.params.keys()):
            if name not in self.bounds:
                continue
            p_plus  = {**self.params, name: self.params[name]+eps}
            p_minus = {**self.params, name: self.params[name]-eps}
            physics.update_params(p_plus)
            xp = physics.step(state, action, 1.0/60.0)
            physics.update_params(p_minus)
            xm = physics.step(state, action, 1.0/60.0)
            physics.update_params(self.params)
            grad = (np.sum((xp-next_state_real)**2) - np.sum((xm-next_state_real)**2)) / (2*eps)
            grad = np.clip(grad, -1.0, 1.0)
            lo, hi = self.bounds[name]
            self.params[name] = float(np.clip(self.params[name] - self.lr*grad, lo, hi))
        physics.update_params(self.params)
        xpred = physics.step(state, action, 1.0/60.0)
        loss  = float(np.sum((xpred-next_state_real)**2))
        self._history.append(loss)
        if len(self._history) > 1000:
            self._history.pop(0)
        return self.params

    @property
    def convergence_history(self):
        return list(self._history)

# ── Control Step result ───────────────────────────────────────────────────────

@dataclass
class ControlStep:
    action:          np.ndarray
    state_predicted: np.ndarray
    residual:        np.ndarray
    uncertainty:     float
    params:          dict
    loop_time_ms:    float
    step_count:      int

# ── PhysiCore Engine ──────────────────────────────────────────────────────────

class PhysiCore:
    """
    PhysiCore Hybrid Uncertainty-Aware Sim-to-Real Engine.
    Runs at 60 Hz. Plug into any rigid-body robot.

    Usage:
        engine = PhysiCore.for_platform("quadrotor", {"mass":1.5})
        step   = engine.step(state, x_ref)
        engine.observe(state, step.action, next_state)
    """

    def __init__(self, cfg: PhysiCoreConfig, dynamics_fn: Callable,
                 initial_params: dict, Q: np.ndarray, R: np.ndarray,
                 action_bounds=None):
        self.cfg      = cfg
        self.Q        = Q
        self.R        = R
        self.physics  = PhysicsLayer(dynamics_fn, initial_params)
        self.ensemble = ResidualEnsemble(cfg)
        self.cem      = CEMOptimizer(cfg, action_bounds)
        self.sysid    = OnlineSystemID(cfg, initial_params)
        self._step_count    = 0
        self._last_action   = None
        self._last_state    = None
        self._last_sim_pred = None

    @classmethod
    def for_platform(cls, platform: str, initial_params: dict = None,
                     Q=None, R=None, action_bounds=None,
                     control_hz: float = 60.0, **kwargs) -> "PhysiCore":
        if platform not in PLATFORM_DYNAMICS:
            raise ValueError(f"Unknown platform '{platform}'. Available: {list(PLATFORM_DYNAMICS.keys())}")
        dynamics_fn, state_dim, action_dim = PLATFORM_DYNAMICS[platform]
        cfg = PhysiCoreConfig(state_dim=state_dim, action_dim=action_dim, control_hz=control_hz)
        if initial_params is None:
            initial_params = {"mass":1.0,"friction":0.3,"inertia":0.1}
        if Q is None:
            Q = np.eye(state_dim)  * 10.0
        if R is None:
            R = np.eye(action_dim) * 0.1
        return cls(cfg, dynamics_fn, initial_params, Q, R, action_bounds)

    def step(self, state: np.ndarray, x_ref: np.ndarray) -> ControlStep:
        t0 = time.perf_counter()
        action = self.cem.optimize(state, self.physics, self.ensemble, self.Q, self.R, x_ref, self.cfg.dt)
        x_sim  = self.physics.step(state, action, self.cfg.dt)
        res, unc = self.ensemble.predict(state, action)
        self._last_action   = action.copy()
        self._last_state    = state.copy()
        self._last_sim_pred = x_sim.copy()
        self._step_count   += 1
        return ControlStep(
            action=action, state_predicted=x_sim+res, residual=res,
            uncertainty=unc, params=self.physics.params.copy(),
            loop_time_ms=(time.perf_counter()-t0)*1000, step_count=self._step_count
        )

    def observe(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        if self._last_sim_pred is None:
            return
        self.ensemble.add_experience(state, action, self._last_sim_pred, next_state)
        if self._step_count % 10 == 0:
            self.ensemble.update_all()
        new_params = self.sysid.update(state, action, next_state, self.physics)
        self.physics.update_params(new_params)

    @property
    def diagnostics(self) -> dict:
        return {"step_count": self._step_count, "params": self.physics.params, "target_hz": self.cfg.control_hz}

    @property
    def diagnostics_full(self) -> dict:
        res_norm = 0.0; unc = 0.0
        if self._last_state is not None and self._last_action is not None:
            r, unc = self.ensemble.predict(self._last_state, self._last_action)
            res_norm = float(np.linalg.norm(r))
        return {
            "step_count":    self._step_count,
            "params":        self.physics.params.copy(),
            "residual_norm": res_norm,
            "uncertainty":   unc,
            "sysid_loss_hist": self.sysid.convergence_history[-20:],
            "target_hz":     self.cfg.control_hz,
            "state_dim":     self.cfg.state_dim,
            "action_dim":    self.cfg.action_dim,
        }

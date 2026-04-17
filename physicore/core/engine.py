"""
PhysiCore Core Engine v2.1.0
============================
Hybrid Uncertainty-Aware Sim-to-Real Synchronization Engine.

New in v2.1.0:
  - FailureLog: structured failure detection and logging per step
  - Real-time residual telemetry: step-level L2 + per-axis breakdown
  - Jitter reduction: exponential smoothing on CEM output (alpha=0.35)
  - Innovation-driven adaptive SysID learning rate (RLS-style forgetting)
  - SHA-256 hash chain on every ControlStep for forensic traceability
  - Richer diagnostics_full: failure_log, residual_axis, innovation_ema

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations
import numpy as np
import math
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
#  QUATERNION MATH
# ═══════════════════════════════════════════════════════════════════════════════

def quat_multiply(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def quat_to_rotmat(q):
    w,x,y,z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def euler_to_quat(roll, pitch, yaw):
    cr,cp,cy = math.cos(roll/2),math.cos(pitch/2),math.cos(yaw/2)
    sr,sp,sy = math.sin(roll/2),math.sin(pitch/2),math.sin(yaw/2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])

def quat_to_euler(q):
    w,x,y,z = q
    roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    sinp  = 2*(w*y-z*x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return roll, pitch, yaw

# ═══════════════════════════════════════════════════════════════════════════════
#  FAILURE LOG
# ═══════════════════════════════════════════════════════════════════════════════

FAILURE_TYPES = {
    "RESIDUAL_HIGH":    "Sim-to-real residual exceeded threshold",
    "UNCERTAINTY_HIGH": "Ensemble epistemic uncertainty too high",
    "LOOP_SLOW":        "Control loop exceeded timing budget",
    "SYSID_DIVERGE":    "System ID loss increasing for 5+ consecutive steps",
    "ACTION_CLIPPED":   "Optimal action was clipped to bounds",
    "STATE_EXPLODED":   "State norm exceeded safety ceiling",
}

@dataclass
class FailureEvent:
    step:         int
    timestamp:    float
    failure_type: str
    description:  str
    severity:     str   # "WARNING" | "ERROR" | "CRITICAL"
    value:        float
    threshold:    float
    params_snapshot: dict

    def to_dict(self):
        return {"step": self.step, "timestamp": round(self.timestamp,4),
                "type": self.failure_type, "description": self.description,
                "severity": self.severity, "value": round(self.value,6),
                "threshold": round(self.threshold,6), "params": self.params_snapshot}


class FailureLog:
    """Real-time failure detection. Checks residual, uncertainty, loop time,
    state explosion, action clipping, and SysID divergence every step."""

    THRESHOLDS = {
        "residual_warn": 0.30, "residual_error": 0.80,
        "uncertainty_warn": 0.05, "uncertainty_err": 0.15,
        "loop_warn_ms": 20.0, "loop_error_ms": 50.0,
        "state_ceiling": 1e4,
    }

    def __init__(self, max_events=2000):
        self._events: List[FailureEvent] = []
        self._max = max_events
        self._sysid_loss_prev: Optional[float] = None
        self._sysid_bad_streak = 0

    def check(self, step, residual, uncertainty, loop_ms,
              state, params, action_clipped, sysid_loss=None):
        new: List[FailureEvent] = []
        t = time.time()
        snap = {k: round(v,4) for k,v in params.items()}

        if residual > self.THRESHOLDS["residual_error"]:
            new.append(FailureEvent(step,t,"RESIDUAL_HIGH",FAILURE_TYPES["RESIDUAL_HIGH"],"ERROR",residual,self.THRESHOLDS["residual_error"],snap))
        elif residual > self.THRESHOLDS["residual_warn"]:
            new.append(FailureEvent(step,t,"RESIDUAL_HIGH",FAILURE_TYPES["RESIDUAL_HIGH"],"WARNING",residual,self.THRESHOLDS["residual_warn"],snap))

        if uncertainty > self.THRESHOLDS["uncertainty_err"]:
            new.append(FailureEvent(step,t,"UNCERTAINTY_HIGH",FAILURE_TYPES["UNCERTAINTY_HIGH"],"ERROR",uncertainty,self.THRESHOLDS["uncertainty_err"],snap))
        elif uncertainty > self.THRESHOLDS["uncertainty_warn"]:
            new.append(FailureEvent(step,t,"UNCERTAINTY_HIGH",FAILURE_TYPES["UNCERTAINTY_HIGH"],"WARNING",uncertainty,self.THRESHOLDS["uncertainty_warn"],snap))

        if loop_ms > self.THRESHOLDS["loop_error_ms"]:
            new.append(FailureEvent(step,t,"LOOP_SLOW",FAILURE_TYPES["LOOP_SLOW"],"ERROR",loop_ms,self.THRESHOLDS["loop_error_ms"],snap))
        elif loop_ms > self.THRESHOLDS["loop_warn_ms"]:
            new.append(FailureEvent(step,t,"LOOP_SLOW",FAILURE_TYPES["LOOP_SLOW"],"WARNING",loop_ms,self.THRESHOLDS["loop_warn_ms"],snap))

        state_norm = float(np.linalg.norm(state))
        if state_norm > self.THRESHOLDS["state_ceiling"]:
            new.append(FailureEvent(step,t,"STATE_EXPLODED",FAILURE_TYPES["STATE_EXPLODED"],"CRITICAL",state_norm,self.THRESHOLDS["state_ceiling"],snap))

        if action_clipped:
            new.append(FailureEvent(step,t,"ACTION_CLIPPED",FAILURE_TYPES["ACTION_CLIPPED"],"WARNING",1.0,1.0,snap))

        if sysid_loss is not None and self._sysid_loss_prev is not None:
            if sysid_loss > self._sysid_loss_prev * 1.05:
                self._sysid_bad_streak += 1
            else:
                self._sysid_bad_streak = 0
            if self._sysid_bad_streak >= 5:
                new.append(FailureEvent(step,t,"SYSID_DIVERGE",FAILURE_TYPES["SYSID_DIVERGE"],"WARNING",sysid_loss,self._sysid_loss_prev,snap))
        self._sysid_loss_prev = sysid_loss

        self._events.extend(new)
        if len(self._events) > self._max:
            self._events = self._events[-self._max:]
        return new

    @property
    def recent(self): return list(self._events[-50:])

    @property
    def counts(self):
        c = {}
        for ev in self._events:
            c[ev.failure_type] = c.get(ev.failure_type,0)+1
        return c

    @property
    def last_critical(self):
        for ev in reversed(self._events):
            if ev.severity == "CRITICAL": return ev
        return None

    def summary(self):
        return {"total_events": len(self._events), "counts": self.counts,
                "last_critical": self.last_critical.to_dict() if self.last_critical else None,
                "recent_10": [e.to_dict() for e in self._events[-10:]]}


# ═══════════════════════════════════════════════════════════════════════════════
#  WIND FIELD — Dryden turbulence (MIL-SPEC-F-8785C)
# ═══════════════════════════════════════════════════════════════════════════════

class WindField:
    def __init__(self, intensity=0.0, seed=None):
        self.intensity = intensity
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(3)

    def sample(self, altitude, dt):
        if self.intensity < 1e-6: return np.zeros(3)
        alt = max(altitude, 10.0)
        Lu  = alt / (0.177+0.000823*alt)**1.2
        sigma_w = self.intensity*0.1*(0.177+0.000823*alt)**0.4
        sigma_u = sigma_w/(0.177+0.000823*alt)**0.4
        sigmas = np.array([sigma_u,sigma_u,sigma_w])
        taus   = np.array([Lu,Lu,alt]) / max(1.0, abs(self._state[0])+5.0)
        alpha  = np.exp(-dt/(taus+1e-6))
        noise  = self._rng.standard_normal(3)*sigmas*np.sqrt(1-alpha**2)
        self._state = alpha*self._state + noise
        return self._state.copy()

    @staticmethod
    def calm():    return WindField(0.0)
    @staticmethod
    def moderate():return WindField(0.5)
    @staticmethod
    def severe():  return WindField(1.0)

_DEFAULT_WIND = WindField(0.0)

# ═══════════════════════════════════════════════════════════════════════════════
#  ATMOSPHERE + DRAG
# ═══════════════════════════════════════════════════════════════════════════════

def isa_atmosphere(altitude):
    alt = max(0.0, min(altitude, 86000.0))
    if alt <= 11000:   T=288.15-0.0065*alt; P=101325.0*(T/288.15)**5.2561
    elif alt <= 20000: T=216.65;             P=22632.1*math.exp(-0.0001577*(alt-11000))
    elif alt <= 32000: T=216.65+0.001*(alt-20000); P=5474.89*(T/216.65)**(-34.1632)
    else:              T=228.65+0.0028*(alt-32000); P=868.019*(T/228.65)**(-17.0816)
    return T, P, P/(287.05*T)

def mach_drag_factor(mach, cd0):
    if mach < 0.8:   return cd0
    elif mach < 1.0: return cd0*(1.0+0.3*(mach-0.8)/0.2)/math.sqrt(max(1e-6,1-mach**2))
    elif mach < 1.2: return cd0*(1.8-0.3*(mach-1.0))/math.sqrt(mach**2-1+0.01)
    else:            return cd0*(1.5/math.sqrt(mach**2-1+0.01))

# J2
_MU=3.986004418e14; _RE=6378137.0; _J2=1.08262668e-3

def j2_acceleration(pos):
    r=np.linalg.norm(pos)
    if r<1e3: return np.zeros(3)
    x,y,z=pos; j2c=1.5*_J2*_MU*_RE**2/r**5; zr2=(z/r)**2
    return -(_MU/r**3)*pos + j2c*np.array([x*(5*zr2-1),y*(5*zr2-1),z*(5*zr2-3)])

# ═══════════════════════════════════════════════════════════════════════════════
#  PLATFORM DYNAMICS — all 12 platforms
# ═══════════════════════════════════════════════════════════════════════════════

def quadrotor_dynamics(state, action, params):
    m=max(params.get("mass",1.5),0.01); b=params.get("friction",0.1)
    x,y,z,vx,vy,vz,qw,qx,qy,qz,p,q,r=state; thrust,rc,pc,yc=action
    qa=np.array([qw,qx,qy,qz]); R=quat_to_rotmat(qa)
    tw=R@np.array([0,0,thrust/m])
    ax=tw[0]-b*vx/m; ay=tw[1]-b*vy/m; az=tw[2]-9.81
    dq=0.5*quat_multiply(qa,np.array([0,p,q,r])); tau=0.05
    return np.array([vx,vy,vz,ax,ay,az,dq[0],dq[1],dq[2],dq[3],(rc-p)/tau,(pc-q)/tau,(yc-r)/tau])

def fixed_wing_dynamics(state, action, params):
    m=max(params.get("mass",12.5),0.1); cd0=params.get("friction",0.025); cla=params.get("inertia",5.7)
    _,_,_,vx,vy,vz,roll,pitch,yaw,p,q,r=state; throttle,ail,elev,rud=action
    alt=max(-state[2],0); _,_,rho=isa_atmosphere(alt); wind=_DEFAULT_WIND.sample(alt,1/60)
    vxe,vye,vze=vx-wind[0],vy-wind[1],vz-wind[2]; v=max(math.sqrt(vxe**2+vye**2+vze**2),0.5)
    cd_m=mach_drag_factor(v/340.3,cd0); qdyn=0.5*rho*v**2; S=params.get("wing_area",0.85)
    lift=qdyn*cla*S*math.sin(pitch); drag=qdyn*cd_m*S
    return np.array([vx,vy,vz,throttle*80/m-drag*vxe/(m*v),-drag*vye/(m*v),lift/m-9.81,p,q,r,ail*2.5,elev*2.5,rud*2])

def evtol_dynamics(state, action, params):
    m=max(params.get("mass",500),1); b=params.get("friction",0.05)
    _,_,_,vx,vy,vz,roll,pitch,yaw,p,q,r=state; thrust,rc,pc,fwd=action
    v=max(math.sqrt(vx**2+vy**2+vz**2),0.1); tr=min(v/30,1)
    _,_,rho=isa_atmosphere(max(-state[2],0)); S=params.get("wing_area",12)
    lv=(thrust/m)*math.cos(pitch)*math.cos(roll); lw=0.5*rho*v**2*5*S*math.sin(pitch)/m if v>5 else 0
    tau=0.08
    return np.array([vx,vy,vz,fwd/m-b*vx/m,-b*vy/m,(1-tr)*lv+tr*lw-9.81,p,q,r,(rc-p)/tau,(pc-q)/tau,-r*0.5])

def manipulator_arm_dynamics(state, action, params):
    n=6; q_=state[:n]; dq=state[n:]; m=max(params.get("mass",2),0.01); fric=params.get("friction",0.3)
    M=np.ones(n)*m*0.1+np.array([0.5,0.4,0.3,0.2,0.1,0.05])
    gc=np.array([m*9.81*0.3*math.cos(q_[0]),m*9.81*0.2*math.cos(q_[1]),m*9.81*0.1*math.cos(q_[2]),0,0,0])
    return np.concatenate([dq,(action+gc-fric*dq)/M])

def surgical_robot_dynamics(state, action, params):
    n=6; dq=state[n:]; m=max(params.get("mass",0.05),1e-5); fric=params.get("friction",0.8); tk=params.get("inertia",0.1)
    M=np.ones(n)*m*0.001+np.array([5e-3,4e-3,3e-3,2e-3,1e-3,5e-4])
    return np.concatenate([dq,(action-fric*dq-tk*state[:n])/M])

def legged_robot_dynamics(state, action, params):
    m=max(params.get("mass",30),0.1); fric=params.get("friction",0.7); Ixx=max(params.get("inertia",0.5),0.001)
    _,_,z,vx,vy,vz,_,_,_,p,q,r=state; fx,fy,fz,tr,tp,ty=action
    fzc=max(fz,0) if z<=0.01 else 0
    return np.array([vx,vy,vz,fx/m-fric*vx/m,fy/m-fric*vy/m,fzc/m-9.81*(1-min(1,max(0,-z*100))),
                     p,q,r,tr/Ixx-fric*p,tp/Ixx-fric*q,ty/Ixx-fric*r])

def balancing_bot_dynamics(state, action, params):
    """Nonlinear inverted pendulum. State=[pitch,pitch_rate,x,v]. Action=[torque]."""
    pitch,pitch_rate,x_pos,x_vel=state; torque=float(action[0])
    m=max(params.get("mass",1.0),0.01)
    l=max(params.get("friction",0.15),0.01)   # CoM height reused as l
    I=max(params.get("inertia",0.01),1e-5); g=9.81
    denom=I+m*l**2
    ddpitch=(m*g*l*math.sin(pitch)-torque*math.cos(pitch))/denom
    ddx=(torque-m*l*ddpitch*math.cos(pitch))/m
    return np.array([pitch_rate,ddpitch,x_vel,ddx])

def rocket_dynamics(state, action, params):
    x,y,vx,vy,mass,angle=state; thrust_mag,gimbal=action
    cd=params.get("friction",0.45); isp=max(params.get("inertia",220),1); dia=max(params.get("mass",0.15),0.01)
    alt=max(y,0); _,_,rho=isa_atmosphere(alt); wind=_DEFAULT_WIND.sample(alt,1/60)
    vxe=vx-wind[0]; vye=vy-wind[2]; v=math.sqrt(vxe**2+vye**2)
    T,_,_=isa_atmosphere(alt); a_s=math.sqrt(1.4*287.05*max(T,1)); mach=v/a_s if a_s>0 else 0
    area=math.pi*(dia/2)**2; drag=0.5*rho*v**2*mach_drag_factor(mach,cd)*area
    ta=angle+gimbal; Ftx=thrust_mag*math.sin(ta); Fty=thrust_mag*math.cos(ta)
    Fdx=-drag*(vxe/v) if v>0.1 else 0; Fdy=-drag*(vye/v) if v>0.1 else 0
    m_=max(mass,0.001)
    return np.array([vx,vy,(Ftx+Fdx)/m_,(Fty+Fdy)/m_-9.80665,-thrust_mag/(9.80665*isp) if thrust_mag>0 else 0,0])

def ground_rover_dynamics(state, action, params):
    x,y,theta,vx,vy,omega=state; vl,vr=action
    m=max(params.get("mass",5),0.01); fric=params.get("friction",0.5); wb=params.get("inertia",0.3)
    v=(vl+vr)/2; w=(vr-vl)/wb; Iz=m*wb**2/12
    return np.array([vx,vy,omega,(v*math.cos(theta)-fric*vx)/m,(v*math.sin(theta)-fric*vy)/m,(w-fric*omega)/Iz])

def auv_dynamics(state, action, params):
    _,_,depth,vx,vy,vz,_,_,_,p,q,r=state; surge,sway,heave,yc=action
    m=max(params.get("mass",50),0.1); drag=params.get("friction",2); buoy=params.get("inertia",0.5)
    v=math.sqrt(vx**2+vy**2+vz**2); dc=drag*(1+0.1*v)
    return np.array([vx,vy,vz,surge/m-dc*vx/m,sway/m-dc*vy/m,heave/m-dc*vz/m+buoy/m,
                     p,q,r,-drag*p*0.1,-drag*q*0.1,(yc-r)/0.1])

def satellite_dynamics(state, action, params):
    pos=state[:3]; vel=state[3:6]; _,_,_,_,_,_,roll,pitch,yaw,p,q,r=state
    Tx,Ty,Tz,thrust_mag=action; m=max(params.get("mass",100),0.1); Ixx=max(params.get("inertia",10),0.001)
    acc=j2_acceleration(pos)+quat_to_rotmat(euler_to_quat(roll,pitch,yaw))@np.array([0,0,thrust_mag/m])
    return np.array([vel[0],vel[1],vel[2],acc[0],acc[1],acc[2],p,q,r,Tx/Ixx,Ty/Ixx,Tz/Ixx])

def rover_dynamics(state, action, params): return ground_rover_dynamics(state, action, params)

PLATFORM_DYNAMICS: Dict[str, Tuple[Callable, int, int]] = {
    "quadrotor":       (quadrotor_dynamics,       13, 4),
    "fixed_wing":      (fixed_wing_dynamics,       12, 4),
    "evtol":           (evtol_dynamics,            12, 4),
    "manipulator_arm": (manipulator_arm_dynamics,  12, 6),
    "surgical_robot":  (surgical_robot_dynamics,   12, 6),
    "legged_robot":    (legged_robot_dynamics,     12, 6),
    "balancing_bot":   (balancing_bot_dynamics,     4, 1),
    "rocket":          (rocket_dynamics,            6, 2),
    "ground_rover":    (ground_rover_dynamics,      6, 2),
    "rover":           (ground_rover_dynamics,      6, 2),
    "auv":             (auv_dynamics,              12, 4),
    "satellite":       (satellite_dynamics,         12, 4),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysiCoreConfig:
    platform:            str   = "quadrotor"
    state_dim:           int   = 12
    action_dim:          int   = 4
    control_hz:          float = 60.0
    dt:                  float = field(init=False)
    q_scale:             float = 10.0
    r_scale:             float = 0.1
    initial_params:      Dict[str,float] = field(default_factory=dict)
    horizon:             int   = 6
    cem_samples:         int   = 8
    cem_elites:          int   = 3
    cem_iters:           int   = 2
    cem_min_std:         float = 1e-3
    lam_unc:             float = 0.1
    action_smooth_alpha: float = 0.35   # jitter reduction: 0=none, higher=smoother
    ensemble_size:       int   = 3
    hidden_dim:          int   = 64
    residual_lr:         float = 1e-3
    residual_batch:      int   = 32
    sysid_lr:            float = 0.05
    sysid_clip:          float = 2.0
    sysid_buffer:        int   = 30
    param_bounds:        dict  = field(default_factory=lambda: {
        "mass":     (0.001, 5000.0),
        "friction": (0.0,   100.0),
        "inertia":  (1e-6,  10000.0),
        "wing_area":(0.01,  200.0),
        "gravity":  (0.0,   20.0),
    })
    def __post_init__(self): self.dt = 1.0/self.control_hz

# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSICS LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsLayer:
    def __init__(self, dynamics_fn, params):
        self.dynamics_fn = dynamics_fn
        self.params = params.copy()

    def step(self, state, action, dt):
        k1=self.dynamics_fn(state,action,self.params)
        k2=self.dynamics_fn(state+dt*k1/2,action,self.params)
        k3=self.dynamics_fn(state+dt*k2/2,action,self.params)
        k4=self.dynamics_fn(state+dt*k3,action,self.params)
        result=state+(dt/6)*(k1+2*k2+2*k3+k4)
        if len(result)==13:
            q=result[6:10]; n=np.linalg.norm(q)
            if n>1e-10: result[6:10]=q/n
        return result

    def update_params(self, new_params): self.params.update(new_params)

# ═══════════════════════════════════════════════════════════════════════════════
#  RESIDUAL ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualMLP:
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        inp=state_dim+action_dim
        self.sd=state_dim; self.ad=action_dim; self.lr=lr
        self.W1=np.random.randn(inp,hidden_dim)*math.sqrt(2/inp)
        self.b1=np.zeros(hidden_dim)
        self.W2=np.random.randn(hidden_dim,hidden_dim)*math.sqrt(2/hidden_dim)
        self.b2=np.zeros(hidden_dim)
        self.W3=np.random.randn(hidden_dim,state_dim)*math.sqrt(2/hidden_dim)
        self.b3=np.zeros(state_dim)
        self._replay=[]

    def forward(self, state, action):
        x=np.concatenate([state,action])
        h1=np.maximum(0,x@self.W1+self.b1)
        h2=np.maximum(0,h1@self.W2+self.b2)
        return h2@self.W3+self.b3

    def add_experience(self, state, action, residual):
        self._replay.append((np.concatenate([state,action]),residual))
        if len(self._replay)>10000: self._replay.pop(0)

    def update(self, batch_size=32):
        if len(self._replay)<batch_size: return None
        idxs=np.random.choice(len(self._replay),batch_size,replace=False)
        total=0.0
        for i in idxs:
            inp,target=self._replay[i]
            h1=np.maximum(0,inp@self.W1+self.b1)
            h2=np.maximum(0,h1@self.W2+self.b2)
            out=h2@self.W3+self.b3; err=out-target; total+=float(np.sum(err**2))
            g3=2*err/batch_size
            self.W3-=self.lr*np.clip(np.outer(h2,g3),-1,1)
            self.b3-=self.lr*np.clip(g3,-1,1)
        return total/batch_size


class ResidualEnsemble:
    def __init__(self, cfg):
        self.members=[ResidualMLP(cfg.state_dim,cfg.action_dim,cfg.hidden_dim,cfg.residual_lr)
                      for _ in range(cfg.ensemble_size)]
        self.batch_size=cfg.residual_batch

    def predict(self, state, action):
        """Returns (mean_residual, scalar_variance, per_axis_residual)."""
        preds=np.array([m.forward(state,action) for m in self.members])
        residual=preds.mean(axis=0)
        uncertainty=float(np.mean(np.var(preds,axis=0)))
        return residual, uncertainty, residual

    def add_experience(self, state, action, sim_pred, real_next):
        target=real_next-sim_pred
        for m in self.members: m.add_experience(state,action,target)

    def update_all(self):
        for m in self.members: m.update(self.batch_size)

# ═══════════════════════════════════════════════════════════════════════════════
#  CEM OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class CEMOptimizer:
    def __init__(self, cfg, action_bounds=None):
        self.H=cfg.horizon; self.M=cfg.cem_samples; self.I=cfg.cem_iters
        self.K=max(1,cfg.cem_elites); self.lam=cfg.lam_unc; self.min_std=cfg.cem_min_std
        self.action_dim=cfg.action_dim; self.bounds=action_bounds
        self.mu=np.zeros((self.H,cfg.action_dim)); self.std=np.ones((self.H,cfg.action_dim))

    def optimize(self, state, physics, ensemble, Q, R, x_ref, dt):
        """Returns (action, was_clipped)."""
        for _ in range(self.I):
            seqs=np.random.normal(self.mu[np.newaxis],self.std[np.newaxis],(self.M,self.H,self.action_dim))
            if self.bounds is not None: seqs=np.clip(seqs,self.bounds[0],self.bounds[1])
            costs=np.array([self._cost(state,seqs[j],physics,ensemble,Q,R,x_ref,dt) for j in range(self.M)])
            elite=seqs[np.argsort(costs)[:self.K]]
            self.mu=elite.mean(axis=0); self.std=np.maximum(elite.std(axis=0),self.min_std)
        u=self.mu[0].copy()
        self.mu=np.roll(self.mu,-1,axis=0); self.mu[-1]=0
        self.std=np.roll(self.std,-1,axis=0); self.std[-1]=1
        clipped=False
        if self.bounds is not None:
            uc=np.clip(u,self.bounds[0],self.bounds[1]); clipped=not np.allclose(u,uc); u=uc
        return u, clipped

    def _cost(self, state, actions, physics, ensemble, Q, R, x_ref, dt):
        x=state.copy(); total=0.0
        for u in actions:
            x_sim=physics.step(x,u,dt); res,s2,_=ensemble.predict(x,u); x=x_sim+res
            x_ref_p=np.pad(x_ref,(0,max(0,len(x)-len(x_ref))))[:len(x)]; dx=x-x_ref_p; n=min(len(dx),Q.shape[0])
            total+=float(dx[:n]@Q[:n,:n]@dx[:n]+u@R@u)+self.lam*s2
        return total

# ═══════════════════════════════════════════════════════════════════════════════
#  ONLINE SYSTEM ID — windowed gradient + momentum + innovation-driven LR
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineSystemID:
    """
    Innovation-driven adaptive learning rate:
    When recent prediction errors are large (new terrain, payload change),
    LR increases automatically to adapt faster. Settles back when converged.
    """
    def __init__(self, cfg, initial_params):
        self.lr=cfg.sysid_lr; self.clip=cfg.sysid_clip; self.bounds=cfg.param_bounds
        self.params=initial_params.copy(); self.buf_size=cfg.sysid_buffer
        self._buf=[]; self._vel={k:0.0 for k in initial_params}; self._beta=0.9
        self._history=[]; self._innovation_ema=0.1

    def update(self, state, action, next_state_real, physics):
        self._buf.append((state.copy(),action.copy(),next_state_real.copy()))
        if len(self._buf)>self.buf_size: self._buf.pop(0)
        if len(self._buf)<max(5,self.buf_size//3): return self.params

        # Innovation-driven LR
        x_pred=physics.step(self._buf[-1][0],self._buf[-1][1],1/60)
        innovation=float(np.linalg.norm(self._buf[-1][2]-x_pred))
        self._innovation_ema=0.9*self._innovation_ema+0.1*innovation
        adaptive_lr=self.lr*min(3.0,max(0.5,innovation/(self._innovation_ema+1e-8)))

        eps=5e-4
        for name in list(self.params.keys()):
            if name not in self.bounds: continue
            loss_p=loss_m=0.0
            for s,u,ns in self._buf:
                pp={**self.params,name:self.params[name]+eps}
                pm={**self.params,name:self.params[name]-eps}
                physics.update_params(pp); xp=physics.step(s,u,1/60)
                physics.update_params(pm); xm=physics.step(s,u,1/60)
                physics.update_params(self.params)
                loss_p+=float(np.sum((xp-ns)**2)); loss_m+=float(np.sum((xm-ns)**2))
            n=len(self._buf)
            grad=float(np.clip((loss_p-loss_m)/(2*eps*n),-self.clip,self.clip))
            self._vel[name]=self._beta*self._vel[name]+(1-self._beta)*grad
            lo,hi=self.bounds.get(name,(None,None))
            if lo is None: lo,hi=-1e9,1e9
            self.params[name]=float(np.clip(self.params[name]-adaptive_lr*self._vel[name],lo,hi))

        physics.update_params(self.params)
        loss=sum(np.sum((physics.step(s,u,1/60)-ns)**2) for s,u,ns in self._buf)/len(self._buf)
        self._history.append(float(loss))
        if len(self._history)>1000: self._history.pop(0)
        return self.params

    @property
    def convergence_history(self): return list(self._history)
    @property
    def innovation_ema(self): return self._innovation_ema

# ═══════════════════════════════════════════════════════════════════════════════
#  FORENSIC HASH CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

class HashChain:
    """SHA-256 hash chain — tamper-evident forensic ledger."""
    def __init__(self): self._prev="GENESIS"

    def sign(self, step_count, action, params, residual, uncertainty):
        payload=json.dumps({"step":step_count,"action":[round(float(a),6) for a in action],
                            "params":{k:round(v,6) for k,v in params.items()},
                            "residual":round(residual,8),"uncertainty":round(uncertainty,8),
                            "prev":self._prev},sort_keys=True)
        h=hashlib.sha256(payload.encode()).hexdigest()
        self._prev=h; return h

# ═══════════════════════════════════════════════════════════════════════════════
#  CONTROL STEP
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ControlStep:
    action:          np.ndarray
    state_predicted: np.ndarray
    residual:        np.ndarray
    residual_norm:   float
    residual_axis:   np.ndarray   # per-axis for dashboard breakdown
    uncertainty:     float
    params:          dict
    loop_time_ms:    float
    step_count:      int
    action_clipped:  bool
    certificate:     str          # SHA-256 hash chain
    failure_events:  List[FailureEvent]

# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSICORE ENGINE v2.1
# ═══════════════════════════════════════════════════════════════════════════════

class PhysiCore:
    def __init__(self, cfg, dynamics_fn, initial_params, Q, R, action_bounds=None):
        self.cfg=cfg; self.Q=Q; self.R=R
        self.physics=PhysicsLayer(dynamics_fn,initial_params)
        self.ensemble=ResidualEnsemble(cfg)
        self.cem=CEMOptimizer(cfg,action_bounds)
        self.sysid=OnlineSystemID(cfg,initial_params)
        self.failure_log=FailureLog()
        self.hash_chain=HashChain()
        self._step_count=0
        self._last_action=self._last_state=self._last_sim_pred=None
        self._smoothed_action=None

    @classmethod
    def for_platform(cls, platform, initial_params=None, Q=None, R=None,
                     action_bounds=None, control_hz=60.0, wind_intensity=0.0, **kw):
        if platform not in PLATFORM_DYNAMICS:
            raise ValueError(f"Unknown platform '{platform}'. Available: {sorted(PLATFORM_DYNAMICS.keys())}")
        global _DEFAULT_WIND
        if wind_intensity>0: _DEFAULT_WIND=WindField(intensity=wind_intensity)
        fn,state_dim,action_dim=PLATFORM_DYNAMICS[platform]
        cfg=PhysiCoreConfig(platform=platform,state_dim=state_dim,action_dim=action_dim,
                            control_hz=control_hz,initial_params=initial_params or {})
        if platform in ('quadrotor','satellite','fixed_wing','evtol','manipulator_arm',
                        'surgical_robot','legged_robot','rocket','ground_rover'):
            cfg.cem_samples=6; cfg.horizon=5; cfg.cem_iters=2
        if initial_params is None: initial_params={"mass":1.0,"friction":0.3,"inertia":0.1}
        if Q is None: Q=np.eye(state_dim)*cfg.q_scale
        if R is None: R=np.eye(action_dim)*cfg.r_scale
        return cls(cfg,fn,initial_params,Q,R,action_bounds)

    def step(self, state, x_ref):
        t0=time.perf_counter()
        raw_action,clipped=self.cem.optimize(state,self.physics,self.ensemble,self.Q,self.R,x_ref,self.cfg.dt)

        # Jitter reduction: exponential smoothing on action
        alpha=self.cfg.action_smooth_alpha
        if self._smoothed_action is None or len(self._smoothed_action)!=len(raw_action):
            self._smoothed_action=raw_action.copy()
        else:
            self._smoothed_action=(1-alpha)*raw_action+alpha*self._smoothed_action
        action=self._smoothed_action.copy()

        x_sim=self.physics.step(state,action,self.cfg.dt)
        residual,unc,r_axis=self.ensemble.predict(state,action)
        residual_norm=float(np.linalg.norm(residual))
        loop_ms=(time.perf_counter()-t0)*1000

        cert=self.hash_chain.sign(self._step_count,action,self.physics.params,residual_norm,unc)
        sysid_loss=self.sysid.convergence_history[-1] if self.sysid.convergence_history else None
        failures=self.failure_log.check(
            step=self._step_count,residual=residual_norm,uncertainty=unc,
            loop_ms=loop_ms,state=state,params=self.physics.params,
            action_clipped=clipped,sysid_loss=sysid_loss)

        self._last_action=action.copy(); self._last_state=state.copy()
        self._last_sim_pred=x_sim.copy(); self._step_count+=1

        return ControlStep(action=action,state_predicted=x_sim+residual,
                           residual=residual,residual_norm=residual_norm,residual_axis=r_axis,
                           uncertainty=unc,params=self.physics.params.copy(),
                           loop_time_ms=loop_ms,step_count=self._step_count,
                           action_clipped=clipped,certificate=cert,failure_events=failures)

    def observe(self, state, action, next_state):
        if self._last_sim_pred is None: return
        self.ensemble.add_experience(state,action,self._last_sim_pred,next_state)
        if self._step_count%10==0: self.ensemble.update_all()
        self.physics.update_params(self.sysid.update(state,action,next_state,self.physics))

    def set_wind(self, intensity):
        global _DEFAULT_WIND; _DEFAULT_WIND=WindField(intensity=intensity)

    @property
    def diagnostics(self):
        return {"step_count":self._step_count,"params":self.physics.params,"target_hz":self.cfg.control_hz}

    @property
    def diagnostics_full(self):
        res_norm=unc=0.0; r_axis=np.zeros(self.cfg.state_dim)
        if self._last_state is not None and self._last_action is not None:
            r,unc,r_axis=self.ensemble.predict(self._last_state,self._last_action)
            res_norm=float(np.linalg.norm(r))
        return {
            "step_count":      self._step_count,
            "params":          self.physics.params.copy(),
            "residual_norm":   res_norm,
            "residual_axis":   r_axis.tolist(),         # NEW: per-axis residual
            "uncertainty":     unc,
            "sysid_loss_hist": self.sysid.convergence_history[-20:],
            "innovation_ema":  self.sysid.innovation_ema,  # NEW: adaptive LR signal
            "target_hz":       self.cfg.control_hz,
            "state_dim":       self.cfg.state_dim,
            "action_dim":      self.cfg.action_dim,
            "failure_summary": self.failure_log.summary(),  # NEW: structured failure log
            "hash_chain_head": self.hash_chain._prev,        # NEW: forensic cert
        }


class PhysicoreSimulator:
    """Backward compatibility wrapper."""
    def __init__(self, platform="quadrotor", params=None):
        self.engine=PhysiCore.for_platform(platform,initial_params=params)
        self.state=np.zeros(self.engine.cfg.state_dim)
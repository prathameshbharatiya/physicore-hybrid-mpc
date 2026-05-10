"""
PhysiCore Extra Dynamics
========================
Physics models for the four platforms that were listed in RobotConfig /
PLATFORM_ALIASES but had no corresponding entry in PLATFORM_DYNAMICS:

  - mobile_manipulator  (wheeled base + serial arm, 14D state / 6D action)
  - dual_arm            (two 7-DOF arms, 20D state / 14D action)
  - cable_driven        (cable-driven parallel robot, 12D state / 6D action)
  - exoskeleton         (lower-limb exo with admittance control, 16D state / 10D action)

All functions follow the standard PhysiCore signature:
    f(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray
where the return value is dstate/dt (continuous-time derivative).

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations
import math
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  MOBILE MANIPULATOR
#  Base: differential-drive rover (6D)
#  Arm:  serial 6-DOF arm (8D: 4 joints q + 4 joints dq)
#  Total state (14): [base_x, base_y, base_theta, base_vx, base_vy, base_omega,
#                     q1, q2, q3, q4, dq1, dq2, dq3, dq4]
#  Action (6):       [vl, vr, tau1, tau2, tau3, tau4]
#                     vl/vr = wheel velocities, tau = arm joint torques
# ═══════════════════════════════════════════════════════════════════════════════

def mobile_manipulator_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    Holonomic mobile manipulator: differential-drive base + 4-DOF serial arm.
    """
    # Unpack state
    bx, by, btheta, bvx, bvy, bomega = state[0:6]
    q    = state[6:10]
    dq   = state[10:14]

    # Unpack action
    vl   = float(action[0]) if len(action) > 0 else 0.0
    vr   = float(action[1]) if len(action) > 1 else 0.0
    tau  = np.asarray(action[2:6], dtype=float) if len(action) >= 6 else np.zeros(4)

    # Physical params
    m_base  = max(params.get("mass",     15.0), 0.1)
    m_arm   = max(params.get("inertia",   5.0), 0.1)   # arm effective mass
    fric    = params.get("friction",  0.4)
    wb      = params.get("wheelbase", 0.4)              # metres between wheels
    Iz_base = m_base * wb ** 2 / 12.0

    # ── Base dynamics (differential drive) ────────────────────────────────────
    v_base  = (vl + vr) / 2.0
    w_base  = (vr - vl) / wb

    dbx     = bvx
    dby     = bvy
    dbtheta = bomega
    dbvx    = (v_base * math.cos(btheta) - fric * bvx) / m_base
    dbvy    = (v_base * math.sin(btheta) - fric * bvy) / m_base
    dbomega = (w_base - fric * bomega) / Iz_base

    # ── Arm dynamics (simplified planar 4-DOF) ────────────────────────────────
    # Diagonal mass matrix approximation (each link contributes equally)
    M_arm = np.array([m_arm * 0.25] * 4)

    # Gravity torques (planar arm, joints in vertical plane)
    L     = params.get("link_length", 0.25)   # uniform link length
    g     = 9.81
    cum_angle = btheta  # arm mounted on base, starts at base heading
    g_torque  = np.zeros(4)
    for i in range(4):
        cum_angle += q[i]
        # Gravity pulls COM of all distal links through joint i
        distal_mass = m_arm * (4 - i) / 4.0
        g_torque[i] = -distal_mass * g * L * math.sin(cum_angle) * 0.5

    # Friction damping
    fric_torque = -fric * 0.5 * dq

    # Contact: prevent arm going through floor (simplified)
    ddq = (tau + g_torque + fric_torque) / M_arm
    ddq = np.clip(ddq, -200.0, 200.0)

    return np.array([dbx, dby, dbtheta, dbvx, dbvy, dbomega,
                     dq[0], dq[1], dq[2], dq[3],
                     ddq[0], ddq[1], ddq[2], ddq[3]])


# ═══════════════════════════════════════════════════════════════════════════════
#  DUAL ARM
#  Two 7-DOF arms sharing a torso (or table mount).
#  State (20):  [q_L(7), q_R(7), dq_L(3), dq_R(3)]   (reduced to 20 for engine)
#               Practical: 7+7 joint angles + 3+3 velocities → 20D
#  Action (14): [tau_L(7), tau_R(7)]
# ═══════════════════════════════════════════════════════════════════════════════

def dual_arm_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    Dual 7-DOF arm manipulator (e.g. YuMi / Baxter style).
    State(20): [q_L(7), q_R(7), dq_L(3), dq_R(3)]
    Action(14): [tau_L(7), tau_R(7)]

    Note: velocities tracked for first 3 joints per arm (dominant inertia).
    """
    q_L  = state[0:7]
    q_R  = state[7:14]
    dq_L = state[14:17]   # velocities for joints 0-2 of left arm
    dq_R = state[17:20]   # velocities for joints 0-2 of right arm

    tau_L = np.asarray(action[0:7],  dtype=float) if len(action) >= 7  else np.zeros(7)
    tau_R = np.asarray(action[7:14], dtype=float) if len(action) >= 14 else np.zeros(7)

    m_arm  = max(params.get("mass",     3.0), 0.01)
    fric   = params.get("friction",  0.25)
    g      = 9.81
    L      = params.get("link_length", 0.3)

    def _arm_ddq3(q: np.ndarray, dq3: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Dynamics for the 3 dominant (shoulder/elbow) joints."""
        M    = np.array([m_arm * 0.5, m_arm * 0.3, m_arm * 0.2])
        g_q  = np.zeros(3)
        angle = 0.0
        for i in range(3):
            angle += q[i]
            distal = m_arm * (3 - i) / 3.0
            g_q[i] = -distal * g * L * math.sin(angle) * 0.5
        fric_q = -fric * dq3
        ddq    = np.clip((tau[:3] + g_q + fric_q) / M, -300.0, 300.0)
        return ddq

    # Position derivatives: full 7-joint, but velocity tracked for 3
    # Joints 3-6 are fast/light — approximate as critically damped
    dq_L4_7 = -fric * q_L[3:7] * 10.0   # fast decay for wrist joints
    dq_R4_7 = -fric * q_R[3:7] * 10.0

    ddq_L3 = _arm_ddq3(q_L, dq_L, tau_L)
    ddq_R3 = _arm_ddq3(q_R, dq_R, tau_R)

    # State derivatives
    dq_L_full = np.concatenate([dq_L, dq_L4_7])
    dq_R_full = np.concatenate([dq_R, dq_R4_7])

    return np.concatenate([
        dq_L_full,         # dq_L  (7)
        dq_R_full,         # dq_R  (7)
        ddq_L3,            # ddq_L (3)
        ddq_R3,            # ddq_R (3)
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  CABLE-DRIVEN PARALLEL ROBOT (CDPR)
#  State (12): [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
#  Action (6): [T1, T2, T3, T4, T5, T6]  — cable tensions (N)
#
#  Cable geometry is parameterised; default is a 6-cable spatial CDPR with
#  anchor points at a unit-cube workspace boundary.
# ═══════════════════════════════════════════════════════════════════════════════

# Default anchor points (6 cables, 3D workspace, unit cube 2×2×2 m)
_CDPR_ANCHORS = np.array([
    [ 1.0,  1.0,  2.0],
    [-1.0,  1.0,  2.0],
    [ 1.0, -1.0,  2.0],
    [-1.0, -1.0,  2.0],
    [ 1.0,  0.0,  0.0],
    [-1.0,  0.0,  0.0],
], dtype=float)

# Attachment points on the end-effector platform (relative to platform COM)
_CDPR_EE_ATTACH = np.array([
    [ 0.2,  0.2,  0.1],
    [-0.2,  0.2,  0.1],
    [ 0.2, -0.2,  0.1],
    [-0.2, -0.2,  0.1],
    [ 0.2,  0.0, -0.1],
    [-0.2,  0.0, -0.1],
], dtype=float)


def _rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,          cp*sr,            cp*cr  ],
    ])


def cable_driven_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    6-cable CDPR with rigid end-effector platform.
    """
    x, y, z, vx, vy, vz, roll, pitch, yaw, p, q_, r = state
    tensions = np.clip(np.asarray(action[:6], dtype=float), 0.0, None)  # cables can only pull

    m   = max(params.get("mass",    20.0), 0.1)
    I_s = max(params.get("inertia",  2.0), 0.01)
    g   = 9.81

    pos  = np.array([x, y, z])
    R_ee = _rpy_to_R(np.array([roll, pitch, yaw]))

    # Compute cable force vectors
    F_total  = np.zeros(3)
    T_torque = np.zeros(3)

    anchors    = params.get("anchors",   _CDPR_ANCHORS)
    ee_attach  = params.get("ee_attach", _CDPR_EE_ATTACH)

    for i in range(min(6, len(tensions))):
        if i >= len(anchors):
            break
        b_i  = np.asarray(anchors[i], dtype=float)       # anchor in world
        a_i  = R_ee @ np.asarray(ee_attach[i], dtype=float)   # EE attach in world
        v_cable = b_i - (pos + a_i)
        length  = np.linalg.norm(v_cable)
        if length < 1e-6:
            continue
        u_i      = v_cable / length
        f_i      = tensions[i] * u_i
        F_total  += f_i
        T_torque += np.cross(a_i, f_i)

    # Gravity
    F_total[2] -= m * g

    # Inertia tensor (uniform sphere approx)
    Ixx = I_s; Iyy = I_s; Izz = I_s

    ax = F_total[0] / m
    ay = F_total[1] / m
    az = F_total[2] / m

    # Angular dynamics (body frame, simplified diagonal inertia)
    dp = T_torque[0] / Ixx - (Izz - Iyy) * q_ * r / Ixx
    dq = T_torque[1] / Iyy - (Ixx - Izz) * p  * r / Iyy
    dr = T_torque[2] / Izz - (Iyy - Ixx) * p  * q_ / Izz

    return np.array([vx, vy, vz, ax, ay, az, p, q_, r, dp, dq, dr])


# ═══════════════════════════════════════════════════════════════════════════════
#  EXOSKELETON (lower-limb, bilateral)
#  State (16): [hip_L, knee_L, ankle_L, hip_R, knee_R, ankle_R,  ← angles (6)
#               dhip_L, dknee_L, dankle_L, dhip_R, dknee_R, dankle_R, ← velocities (6)
#               trunk_pitch, trunk_roll, dtrunk_pitch, dtrunk_roll]
#  Action (10): [tau_hip_L, tau_knee_L, tau_ankle_L,
#                tau_hip_R, tau_knee_R, tau_ankle_R,
#                F_assist_L, F_assist_R,   ← assistive force on shin (N)
#                trunk_Tp, trunk_Tr]       ← trunk stabilisation torques
#
#  Human-in-the-loop: human torque modelled as visco-elastic opposing motion.
#  Admittance: F_interaction = k_adm * (q_ref - q) + b_adm * (dq_ref - dq)
# ═══════════════════════════════════════════════════════════════════════════════

def exoskeleton_dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """
    Bilateral lower-limb exoskeleton with human admittance model.
    """
    q_L   = state[0:3]    # [hip, knee, ankle] left
    q_R   = state[3:6]    # [hip, knee, ankle] right
    dq_L  = state[6:9]
    dq_R  = state[9:12]
    trunk = state[12:14]  # [pitch, roll]
    dtrunk= state[14:16]

    tau_L  = np.asarray(action[0:3],  dtype=float) if len(action) >= 3  else np.zeros(3)
    tau_R  = np.asarray(action[3:6],  dtype=float) if len(action) >= 6  else np.zeros(3)
    F_L    = float(action[6]) if len(action) > 6 else 0.0
    F_R    = float(action[7]) if len(action) > 7 else 0.0
    T_tp   = float(action[8]) if len(action) > 8 else 0.0
    T_tr   = float(action[9]) if len(action) > 9 else 0.0

    # Parameters
    m_human  = max(params.get("mass",        80.0), 1.0)   # total human+exo
    m_exo    = max(params.get("inertia",     12.0), 0.5)   # exo structural mass
    fric     = params.get("friction",   0.6)
    k_adm    = params.get("admittance_k", 100.0)           # admittance stiffness
    b_adm    = params.get("admittance_b",  20.0)           # admittance damping
    g        = 9.81
    L_thigh  = params.get("link_length", 0.42)             # thigh length m
    L_shank  = params.get("link_length2", 0.40)            # shank length m

    def _leg_ddq(q3, dq3, tau3, F_assist):
        """Per-leg dynamics (3 joints: hip, knee, ankle)."""
        M = np.array([
            m_human * L_thigh**2 * 0.3,
            m_human * L_shank**2 * 0.2,
            m_human * 0.005,           # ankle (small)
        ])
        M = np.maximum(M, 0.001)

        # Gravity torques (2D sagittal plane)
        hip_a, knee_a, ankle_a = q3
        g_hip   = -m_human * g * L_thigh * 0.5 * math.sin(hip_a)
        g_knee  = -m_human * g * L_shank * 0.5 * math.sin(hip_a + knee_a)
        g_ankle = 0.0  # foot mass negligible
        g_q     = np.array([g_hip, g_knee, g_ankle])

        # Human interaction torque (visco-elastic admittance)
        tau_human = -k_adm * q3 - b_adm * dq3   # human resists deviation from neutral

        # Assistive force mapped to ankle (shin-level force arm)
        assist_torque = np.array([0.0, F_assist * L_shank * 0.5, 0.0])

        # Friction
        fric_q = -fric * dq3

        ddq = (tau3 + g_q + tau_human + assist_torque + fric_q) / M
        return np.clip(ddq, -500.0, 500.0)

    ddq_L  = _leg_ddq(q_L, dq_L, tau_L, F_L)
    ddq_R  = _leg_ddq(q_R, dq_R, tau_R, F_R)

    # Trunk dynamics (sagittal + coronal)
    I_trunk    = m_human * 0.15**2   # trunk radius of gyration ≈ 15 cm
    dtrunk_p   = trunk[0]
    dtrunk_r   = trunk[1]
    g_pitch    = -m_human * g * 0.1 * math.sin(trunk[0])   # gravity restoring
    g_roll     = -m_human * g * 0.05 * math.sin(trunk[1])
    ddtrunk_p  = (T_tp + g_pitch - fric * 5.0 * dtrunk[0]) / I_trunk
    ddtrunk_r  = (T_tr + g_roll  - fric * 5.0 * dtrunk[1]) / I_trunk

    return np.array([
        dq_L[0], dq_L[1], dq_L[2],
        dq_R[0], dq_R[1], dq_R[2],
        ddq_L[0], ddq_L[1], ddq_L[2],
        ddq_R[0], ddq_R[1], ddq_R[2],
        dtrunk_p, dtrunk_r,
        ddtrunk_p, ddtrunk_r,
    ])

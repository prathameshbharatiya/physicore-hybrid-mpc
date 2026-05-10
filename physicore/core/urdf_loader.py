"""
PhysiCore URDF/MJCF Loader
==========================
Auto-generates RobotConfig (and a matching PhysiCore engine) from any
standard robot description file — URDF (.urdf, .xml) or MuJoCo MJCF (.xml).

No more 13 hardcoded platforms.  Drop in any robot and go:

    from physicore.core.urdf_loader import load_robot

    engine, config = load_robot("my_robot.urdf")
    engine, config = load_robot("humanoid.xml")          # MJCF auto-detected
    engine, config = load_robot("arm.urdf", platform_hint="manipulator_arm")

The loader:
  1. Parses the file (URDF via xml.etree, MJCF via xml.etree — same parser, different schema)
  2. Extracts mass, inertia, joint names/limits/types, link geometry
  3. Builds a URDFRobotModel with forward-kinematics and Jacobian
  4. Picks or generates the best dynamics function for the DOF/topology
  5. Returns (PhysiCore instance, RobotConfig)

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations

import math
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

from physicore.core.robot_config import RobotConfig, PLATFORM_ALIASES, ENGINE_PLATFORM


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class JointInfo:
    name:   str
    jtype:  str        # "revolute" | "prismatic" | "fixed" | "continuous" | "floating"
    parent: str        # parent link name
    child:  str        # child link name
    axis:   np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    limit_lo: float = -math.pi
    limit_hi: float =  math.pi
    limit_effort: float = 100.0
    limit_velocity: float = 10.0

    @property
    def is_actuated(self) -> bool:
        return self.jtype in ("revolute", "prismatic", "continuous")


@dataclass
class LinkInfo:
    name: str
    mass: float = 0.0
    inertia_diagonal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    com_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    has_collision: bool = False
    # Geometry for contact model
    collision_type: str = "none"   # "box" | "cylinder" | "sphere" | "mesh" | "none"
    collision_size: np.ndarray = field(default_factory=lambda: np.zeros(3))


# ═══════════════════════════════════════════════════════════════════════════════
#  URDF / MJCF PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_vec3(text: Optional[str], default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.array(default, dtype=float)
    parts = text.strip().split()
    if len(parts) == 3:
        return np.array([float(p) for p in parts])
    return np.array(default, dtype=float)


def _parse_urdf(path: Path) -> Tuple[List[LinkInfo], List[JointInfo], str]:
    """Parse a URDF file. Returns (links, joints, robot_name)."""
    tree = ET.parse(str(path))
    root = tree.getroot()
    robot_name = root.get("name", path.stem)

    links: List[LinkInfo] = []
    for link_el in root.findall("link"):
        li = LinkInfo(name=link_el.get("name", "unnamed"))
        inertial = link_el.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            if mass_el is not None:
                li.mass = float(mass_el.get("value", 0))
            inertia_el = inertial.find("inertia")
            if inertia_el is not None:
                ixx = float(inertia_el.get("ixx", 0))
                iyy = float(inertia_el.get("iyy", 0))
                izz = float(inertia_el.get("izz", 0))
                li.inertia_diagonal = np.array([ixx, iyy, izz])
            com_el = inertial.find("origin")
            if com_el is not None:
                li.com_xyz = _parse_vec3(com_el.get("xyz"))
        collision = link_el.find("collision")
        if collision is not None:
            li.has_collision = True
            geom = collision.find("geometry")
            if geom is not None:
                for gtype in ("box", "cylinder", "sphere", "mesh"):
                    gel = geom.find(gtype)
                    if gel is not None:
                        li.collision_type = gtype
                        if gtype == "box":
                            li.collision_size = _parse_vec3(gel.get("size"))
                        elif gtype == "cylinder":
                            r = float(gel.get("radius", 0.05))
                            l = float(gel.get("length", 0.1))
                            li.collision_size = np.array([r, r, l])
                        elif gtype == "sphere":
                            r = float(gel.get("radius", 0.05))
                            li.collision_size = np.array([r, r, r])
                        break
        links.append(li)

    joints: List[JointInfo] = []
    for j_el in root.findall("joint"):
        ji = JointInfo(
            name=j_el.get("name", "unnamed"),
            jtype=j_el.get("type", "fixed"),
            parent=j_el.find("parent").get("link", "") if j_el.find("parent") is not None else "",
            child=j_el.find("child").get("link", "") if j_el.find("child") is not None else "",
        )
        axis_el = j_el.find("axis")
        if axis_el is not None:
            ji.axis = _parse_vec3(axis_el.get("xyz"), (0, 0, 1))
        origin_el = j_el.find("origin")
        if origin_el is not None:
            ji.origin_xyz = _parse_vec3(origin_el.get("xyz"))
            ji.origin_rpy = _parse_vec3(origin_el.get("rpy"))
        limit_el = j_el.find("limit")
        if limit_el is not None:
            ji.limit_lo = float(limit_el.get("lower", -math.pi))
            ji.limit_hi = float(limit_el.get("upper",  math.pi))
            ji.limit_effort = float(limit_el.get("effort", 100.0))
            ji.limit_velocity = float(limit_el.get("velocity", 10.0))
        joints.append(ji)

    return links, joints, robot_name


def _parse_mjcf(path: Path) -> Tuple[List[LinkInfo], List[JointInfo], str]:
    """Parse a MuJoCo MJCF file. Returns (links, joints, robot_name)."""
    tree = ET.parse(str(path))
    root = tree.getroot()
    robot_name = root.get("model", path.stem)

    links:  List[LinkInfo]  = []
    joints: List[JointInfo] = []
    link_map: Dict[str, LinkInfo] = {}

    def _walk_body(body_el: ET.Element, parent_name: str):
        bname = body_el.get("name", f"body_{len(links)}")
        li = LinkInfo(name=bname)
        # MJCF inertial
        inertial_el = body_el.find("inertial")
        if inertial_el is not None:
            li.mass = float(inertial_el.get("mass", 0))
            diaginertia = inertial_el.get("diaginertia", "")
            if diaginertia:
                parts = diaginertia.strip().split()
                if len(parts) == 3:
                    li.inertia_diagonal = np.array([float(p) for p in parts])
            pos_str = inertial_el.get("pos", "")
            if pos_str:
                li.com_xyz = _parse_vec3(pos_str)
        # Geometry from geom children
        for geom_el in body_el.findall("geom"):
            li.has_collision = True
            gtype = geom_el.get("type", "sphere")
            li.collision_type = gtype
            size_str = geom_el.get("size", "")
            if size_str:
                parts = size_str.strip().split()
                if len(parts) >= 3:
                    li.collision_size = np.array([float(p) for p in parts[:3]])
                elif len(parts) == 2:
                    r, h = float(parts[0]), float(parts[1])
                    li.collision_size = np.array([r, r, h])
                elif len(parts) == 1:
                    r = float(parts[0])
                    li.collision_size = np.array([r, r, r])
            break  # first geom only for simplicity
        links.append(li)
        link_map[bname] = li

        # Parse joints attached to this body
        for j_el in body_el.findall("joint"):
            jname = j_el.get("name", f"joint_{len(joints)}")
            jtype_raw = j_el.get("type", "hinge")
            # MJCF → URDF type mapping
            jtype = {"hinge": "revolute", "slide": "prismatic",
                     "free": "floating", "ball": "revolute"}.get(jtype_raw, jtype_raw)
            ji = JointInfo(
                name=jname, jtype=jtype,
                parent=parent_name, child=bname,
            )
            axis_str = j_el.get("axis", "0 0 1")
            ji.axis = _parse_vec3(axis_str, (0, 0, 1))
            pos_str = j_el.get("pos", "0 0 0")
            ji.origin_xyz = _parse_vec3(pos_str)
            range_str = j_el.get("range", "")
            if range_str:
                parts = range_str.strip().split()
                if len(parts) == 2:
                    ji.limit_lo = float(parts[0])
                    ji.limit_hi = float(parts[1])
            joints.append(ji)

        # Recurse into child bodies
        for child_body in body_el.findall("body"):
            _walk_body(child_body, bname)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body_el in worldbody.findall("body"):
            _walk_body(body_el, "world")

    return links, joints, robot_name


# ═══════════════════════════════════════════════════════════════════════════════
#  URDF ROBOT MODEL — FK + JACOBIAN
# ═══════════════════════════════════════════════════════════════════════════════

def _rpy_to_rotmat(rpy: np.ndarray) -> np.ndarray:
    """Roll-Pitch-Yaw → 3×3 rotation matrix (XYZ extrinsic)."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_angle_rotmat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = math.cos(angle), math.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


class URDFRobotModel:
    """
    Lightweight kinematic model built from parsed URDF/MJCF.

    Provides:
      - forward_kinematics(q)   → list of (R, p) for each link frame
      - jacobian(q, link_name)  → 6×n Jacobian at the given link
      - ee_position(q)           → end-effector position (last non-fixed link)
    """

    def __init__(self, links: List[LinkInfo], joints: List[JointInfo], robot_name: str):
        self.robot_name = robot_name
        self.all_links  = links
        self.all_joints = joints

        # Ordered actuated joints only
        self.actuated_joints: List[JointInfo] = [j for j in joints if j.is_actuated]
        self.dof = len(self.actuated_joints)

        # Build parent–child tree (link name → joint that moves it)
        self._link_joint: Dict[str, JointInfo] = {j.child: j for j in joints}
        self._link_map:   Dict[str, LinkInfo]  = {l.name: l for l in links}

        # Find base link (no parent joint)
        child_names = {j.child for j in joints}
        base_candidates = [l.name for l in links if l.name not in child_names]
        self.base_link = base_candidates[0] if base_candidates else (links[0].name if links else "base")

        # Compute total mass and principal inertia
        self.total_mass = sum(l.mass for l in links)
        all_diag = np.array([l.inertia_diagonal for l in links if l.mass > 0])
        self.principal_inertia = np.sum(all_diag, axis=0) if len(all_diag) > 0 else np.array([0.01, 0.01, 0.01])

        # End-effector link: last link in kinematic chain
        parent_names = {j.parent for j in joints}
        ee_candidates = [l.name for l in links if l.name not in parent_names and l.name != self.base_link]
        self.ee_link = ee_candidates[-1] if ee_candidates else (links[-1].name if links else "ee")

    # ── Kinematic chain from base to a target link ─────────────────────────────

    def _chain_to(self, target_link: str) -> List[JointInfo]:
        """Return ordered list of joints from base to target_link."""
        chain: List[JointInfo] = []
        current = target_link
        visited = set()
        while current in self._link_joint and current not in visited:
            visited.add(current)
            j = self._link_joint[current]
            chain.append(j)
            current = j.parent
        chain.reverse()
        return chain

    # ── Forward kinematics ─────────────────────────────────────────────────────

    def forward_kinematics(self, q: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute FK for all links.

        Parameters
        ----------
        q : (dof,) joint angles / displacements

        Returns
        -------
        Dict[link_name → (R 3×3, p 3)] in world frame.
        """
        q = np.asarray(q, dtype=float)
        # Map actuated joint name → current value
        q_map = {j.name: q[i] for i, j in enumerate(self.actuated_joints)}

        frames: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        # Base frame
        frames[self.base_link] = (np.eye(3), np.zeros(3))

        # BFS over joints
        from collections import deque
        queue = deque([self.base_link])
        visited = {self.base_link}

        while queue:
            parent_link = queue.popleft()
            R_parent, p_parent = frames[parent_link]

            for j in self.all_joints:
                if j.parent != parent_link or j.child in visited:
                    continue
                visited.add(j.child)

                # Fixed transform from parent frame
                R_fixed = _rpy_to_rotmat(j.origin_rpy)
                p_fixed = j.origin_xyz

                # Joint transform
                if j.jtype in ("revolute", "continuous"):
                    angle = q_map.get(j.name, 0.0)
                    R_joint = _axis_angle_rotmat(j.axis, angle)
                    p_joint = np.zeros(3)
                elif j.jtype == "prismatic":
                    d = q_map.get(j.name, 0.0)
                    R_joint = np.eye(3)
                    p_joint = j.axis * d
                else:  # fixed / floating
                    R_joint = np.eye(3)
                    p_joint = np.zeros(3)

                R_child = R_parent @ R_fixed @ R_joint
                p_child = p_parent + R_parent @ (p_fixed + R_fixed @ p_joint)
                frames[j.child] = (R_child, p_child)
                queue.append(j.child)

        return frames

    # ── Geometric Jacobian ─────────────────────────────────────────────────────

    def jacobian(self, q: np.ndarray, link_name: Optional[str] = None) -> np.ndarray:
        """
        Compute the 6×dof geometric Jacobian at the given link (default: ee_link).

        Row 0-2: linear velocity Jacobian
        Row 3-5: angular velocity Jacobian

        Parameters
        ----------
        q         : (dof,) joint configuration
        link_name : target link (default end-effector)

        Returns
        -------
        J : (6, dof) ndarray
        """
        if link_name is None:
            link_name = self.ee_link

        q = np.asarray(q, dtype=float)
        dof = len(self.actuated_joints)
        J = np.zeros((6, dof))

        frames = self.forward_kinematics(q)
        if link_name not in frames:
            return J

        p_ee = frames[link_name][1]

        chain = self._chain_to(link_name)
        # Map actuated joint index
        act_idx = {j.name: i for i, j in enumerate(self.actuated_joints)}

        for j in chain:
            if j.name not in act_idx:
                continue
            col = act_idx[j.name]
            if j.parent not in frames:
                continue
            R_j, p_j = frames[j.parent]
            # Joint pivot in world frame: parent origin + R_parent @ joint_xyz
            p_pivot = p_j + R_j @ j.origin_xyz

            if j.jtype in ("revolute", "continuous"):
                z = R_j @ _rpy_to_rotmat(j.origin_rpy) @ j.axis
                r = p_ee - p_pivot
                J[:3, col] = np.cross(z, r)
                J[3:, col] = z
            elif j.jtype == "prismatic":
                z = R_j @ _rpy_to_rotmat(j.origin_rpy) @ j.axis
                J[:3, col] = z
                # angular part stays zero

        return J

    def ee_position(self, q: np.ndarray) -> np.ndarray:
        """Return end-effector position in world frame."""
        frames = self.forward_kinematics(q)
        return frames.get(self.ee_link, (np.eye(3), np.zeros(3)))[1]

    def ee_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R 3×3, p 3) end-effector pose."""
        frames = self.forward_kinematics(q)
        return frames.get(self.ee_link, (np.eye(3), np.zeros(3)))


# ═══════════════════════════════════════════════════════════════════════════════
#  CONTACT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ProperContactModel:
    """
    Penalty-based rigid contact model replacing the toy ground-penetration heuristic.

    Physics
    -------
    Normal force:   F_n = k_n * max(−z, 0)^1.5   (Hertz-like)
    Damping:        F_d = −b_n * vz * (penetration > 0)
    Friction:       F_t = −μ * |F_n| * vt / (|vt| + ε)   (regularized Coulomb)

    The model is parameterised by the link geometry (radius from collision shape)
    so it scales automatically to different robot sizes.
    """

    def __init__(
        self,
        stiffness:    float = 5000.0,   # N/m^1.5  (Hertz contact stiffness)
        damping:      float =  200.0,   # N·s/m
        friction_mu:  float =    0.8,   # Coulomb coefficient
        restitution:  float =    0.0,   # coefficient of restitution (0=plastic)
        contact_eps:  float =   1e-4,   # velocity regularisation
    ):
        self.k_n  = stiffness
        self.b_n  = damping
        self.mu   = friction_mu
        self.e    = restitution
        self.eps  = contact_eps

    def contact_force(
        self,
        link_pos:   np.ndarray,   # (3,) centre of link in world frame
        link_vel:   np.ndarray,   # (3,) velocity of link COM
        radius:     float = 0.05, # effective contact radius from geometry
        ground_z:   float = 0.0,  # ground plane height
    ) -> np.ndarray:
        """
        Returns contact force (3,) acting on the link.

        Positive z is up.  Contact activates when the bottom of the link
        (link_pos[2] − radius) penetrates below ground_z.
        """
        penetration = ground_z - (link_pos[2] - radius)

        if penetration <= 0.0:
            return np.zeros(3)

        # Hertz-like normal force (avoids sharp kink at contact onset)
        F_n = self.k_n * penetration ** 1.5

        # Velocity along normal (z-axis)
        vz = link_vel[2]
        F_damp = -self.b_n * vz if vz < 0.0 else 0.0

        total_normal = max(F_n + F_damp, 0.0)

        # Tangential friction
        vt = np.array([link_vel[0], link_vel[1], 0.0])
        vt_mag = np.linalg.norm(vt) + self.eps
        F_friction = -self.mu * total_normal * vt / vt_mag

        return np.array([F_friction[0], F_friction[1], total_normal])

    def foot_contacts(
        self,
        foot_positions: np.ndarray,   # (n_feet, 3)
        foot_velocities: np.ndarray,  # (n_feet, 3)
        foot_radii: Optional[np.ndarray] = None,  # (n_feet,)
        ground_z: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (contact_forces (n_feet, 3), in_contact (n_feet,) bool).
        Convenience wrapper for legged robots.
        """
        n = len(foot_positions)
        if foot_radii is None:
            foot_radii = np.full(n, 0.03)
        forces    = np.zeros((n, 3))
        in_contact = np.zeros(n, dtype=bool)
        for i in range(n):
            f = self.contact_force(foot_positions[i], foot_velocities[i],
                                   radius=foot_radii[i], ground_z=ground_z)
            forces[i] = f
            in_contact[i] = (f[2] > 0.0)
        return forces, in_contact


# Singleton default contact model
_DEFAULT_CONTACT = ProperContactModel()


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERIC DYNAMICS GENERATED FROM URDF MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def _make_generic_dynamics(
    robot_model: URDFRobotModel,
    contact_model: ProperContactModel,
) -> Callable:
    """
    Build a dynamics function  f(state, action, params) → dstate/dt
    that works for any robot described by its URDF model.

    State layout (2*dof + 6):
        [q_0 … q_{dof-1}, dq_0 … dq_{dof-1}, base_x, base_y, base_z, roll, pitch, yaw]

    Action layout (dof):
        [tau_0 … tau_{dof-1}]  (joint torques / forces)

    Physics:
      - Mass matrix approximated from link inertia (diagonal for speed)
      - Gravity projected through Jacobian
      - Contact forces on last link (end-effector / foot)
      - Damping from joint friction
    """
    dof   = robot_model.dof
    g_vec = np.array([0.0, 0.0, -9.81])

    # Pre-compute per-joint effective inertia (diagonal mass matrix approx)
    # Σ m_i * ||J_i_col||^2  at zero configuration
    q_zero = np.zeros(dof)
    frames_zero = robot_model.forward_kinematics(q_zero)
    M_diag = np.ones(dof) * 0.1

    for i, j in enumerate(robot_model.actuated_joints):
        m_sum = 0.0
        for link in robot_model.all_links:
            if link.mass > 0 and link.name in frames_zero:
                R_l, p_l = frames_zero[link.name]
                # Contribution from this joint column
                J_col = robot_model.jacobian(q_zero, link.name)[:3, i]
                m_sum += link.mass * np.dot(J_col, J_col)
        M_diag[i] = max(m_sum, 1e-3)

    # Contact geometry from last (EE) link collision
    ee_info = robot_model._link_map.get(robot_model.ee_link)
    ee_radius = float(ee_info.collision_size[0]) if (ee_info and ee_info.has_collision) else 0.03

    def _dynamics(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
        # Unpack state
        q  = state[:dof]
        dq = state[dof:2*dof]
        base = state[2*dof:2*dof+6] if len(state) > 2*dof else np.zeros(6)

        tau = np.asarray(action[:dof], dtype=float)

        # Dynamic parameters (updated by SystemID)
        mass_scale  = params.get("mass",     1.0)
        fric_scale  = params.get("friction", 0.1)
        inertia_scl = params.get("inertia",  1.0)

        M_eff = M_diag * mass_scale * inertia_scl

        # Gravity vector in joint space: g_q = J^T * m * g  (column-wise)
        frames  = robot_model.forward_kinematics(q)
        J_ee    = robot_model.jacobian(q)[:3, :]
        # gravity torque (approximate — use EE Jacobian weighted by total mass)
        g_torque = J_ee.T @ (g_vec * robot_model.total_mass * mass_scale / max(dof, 1))

        # Joint friction damping
        fric_torque = -fric_scale * dq

        # Contact force at EE / foot
        ee_pos = robot_model.ee_position(q)
        if len(base) >= 3:
            ee_pos = ee_pos + base[:3]  # add base offset if floating base
        ee_vel = J_ee @ dq
        F_contact = contact_model.contact_force(ee_pos, ee_vel, radius=ee_radius)
        # Map contact force back to joint space
        contact_torque = J_ee.T @ F_contact

        # Newton-Euler: M * ddq = tau - g_torque - fric + contact
        ddq = (tau + g_torque + fric_torque + contact_torque) / M_eff

        # Clip to reasonable values
        ddq = np.clip(ddq, -500.0, 500.0)

        dstate = np.concatenate([dq, ddq, np.zeros(len(base))])
        return dstate

    return _dynamics


# ═══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGY CLASSIFIER → platform hint
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_platform(
    links: List[LinkInfo],
    joints: List[JointInfo],
    robot_name: str,
    dof: int,
    user_hint: Optional[str] = None,
) -> str:
    """
    Heuristically infer the best PhysiCore platform string from topology.
    Returns an engine_platform key (e.g. 'manipulator_arm', 'legged_robot').
    """
    if user_hint:
        # Resolve alias
        hint = user_hint.lower().replace("-", "_")
        return ENGINE_PLATFORM.get(PLATFORM_ALIASES.get(hint, hint), hint)

    name_lower = robot_name.lower()

    # Keyword matching
    keyword_map = [
        (["quadrotor", "drone", "multirotor", "uav", "copter"],   "quadrotor"),
        (["fixed_wing", "plane", "airplane", "wing"],              "fixed_wing"),
        (["rocket", "missile", "launch"],                          "rocket"),
        (["satellite", "spacecraft", "cubesat"],                   "satellite"),
        # rover must come before auv so "rover_bot" → ground_rover, not auv via "rov"
        (["rover", "car", "ugv", "wheeled"],                       "ground_rover"),
        (["auv", "underwater", "submarine", "rov"],                "auv"),
        (["humanoid", "biped", "atlas", "nao", "pepper"],          "humanoid"),
        (["legged", "quadruped", "spot", "anymal", "cheetah"],     "legged_robot"),
        (["surgical", "laparoscop", "davinci"],                    "surgical_robot"),
        (["exoskeleton", "exo", "orthosis"],                       "exoskeleton"),
        (["dual_arm", "bimanual", "yumi"],                         "dual_arm"),
        (["mobile_manipulator", "mobile_arm"],                     "mobile_manipulator"),
        (["cable", "cdpr"],                                        "cable_driven"),
        (["balancing", "segway", "inverted_pendulum"],             "balancing_bot"),
        (["arm", "manipulator", "ur", "kuka", "panda", "fanuc",
          "abb", "robot_arm", "robotic_arm"],                      "manipulator_arm"),
    ]
    for keywords, platform in keyword_map:
        if any(kw in name_lower for kw in keywords):
            return platform

    # DOF-based fallback
    if dof == 0:
        return "ground_rover"
    if dof <= 2:
        return "balancing_bot"
    if dof <= 4:
        return "ground_rover"
    if dof <= 7:
        return "manipulator_arm"
    if dof <= 12:
        return "legged_robot"
    return "humanoid"


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def parse_robot_file(
    path: str,
) -> Tuple[List[LinkInfo], List[JointInfo], str, str]:
    """
    Parse a URDF or MJCF file.

    Returns
    -------
    (links, joints, robot_name, file_format)
    file_format : "urdf" | "mjcf"
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Robot description not found: {path}")

    # Detect format
    # MJCF root tag is <mujoco>, URDF root tag is <robot>
    tree = ET.parse(str(p))
    root = tree.getroot()
    if root.tag == "mujoco":
        links, joints, name = _parse_mjcf(p)
        return links, joints, name, "mjcf"
    else:
        links, joints, name = _parse_urdf(p)
        return links, joints, name, "urdf"


def build_robot_model(
    path: str,
    platform_hint: Optional[str] = None,
    contact_stiffness: float = 5000.0,
    contact_damping:   float =  200.0,
    friction_mu:       float =    0.8,
) -> Tuple["URDFRobotModel", "ProperContactModel", "RobotConfig", str]:
    """
    Parse a robot file and build the kinematic model + contact model + RobotConfig.

    Returns
    -------
    (robot_model, contact_model, robot_config, inferred_platform)
    """
    links, joints, robot_name, fmt = parse_robot_file(path)
    robot_model = URDFRobotModel(links, joints, robot_name)

    contact_model = ProperContactModel(
        stiffness=contact_stiffness,
        damping=contact_damping,
        friction_mu=friction_mu,
    )

    platform = _infer_platform(links, joints, robot_name, robot_model.dof, platform_hint)

    # Aggregate physical parameters from URDF
    total_mass  = robot_model.total_mass if robot_model.total_mass > 0 else 1.0
    Ixx, Iyy, Izz = robot_model.principal_inertia
    inertia = float(np.mean([Ixx, Iyy, Izz]))
    if inertia == 0.0:
        inertia = total_mass * 0.01   # 1 cm² radius of gyration guess

    actuated = robot_model.actuated_joints
    joint_names    = [j.name  for j in actuated]
    joint_limits_lo = [j.limit_lo for j in actuated]
    joint_limits_hi = [j.limit_hi for j in actuated]
    joint_types    = [j.jtype for j in actuated]

    config = RobotConfig(
        name=robot_name,
        platform=platform,
        mass=total_mass,
        inertia=inertia,
        friction=friction_mu * 0.2,   # joint friction ≈ 20 % of contact mu
        dof=robot_model.dof,
        joint_names=joint_names,
        joint_limits_lo=joint_limits_lo,
        joint_limits_hi=joint_limits_hi,
        joint_types=joint_types,
        description=f"Auto-loaded from {Path(path).name} ({fmt.upper()})",
    )

    return robot_model, contact_model, config, platform


def load_robot(
    path: str,
    platform_hint: Optional[str] = None,
    control_hz:   float = 60.0,
    contact_stiffness: float = 5000.0,
    contact_damping:   float =  200.0,
    friction_mu:       float =    0.8,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
) -> Tuple["PhysiCore", RobotConfig]:
    """
    One-call loader: URDF/MJCF → (PhysiCore engine, RobotConfig).

    Parameters
    ----------
    path            : Path to .urdf or .xml file
    platform_hint   : Optional platform override (e.g. "manipulator_arm")
    control_hz      : Control loop frequency
    contact_stiffness, contact_damping, friction_mu : Contact model params
    Q, R            : Optional cost matrices (auto-sized if None)

    Returns
    -------
    (engine, config)

    Example
    -------
    >>> engine, cfg = load_robot("my_arm.urdf")
    >>> state  = np.zeros(cfg.dof * 2 + 6)
    >>> x_ref  = np.zeros_like(state)
    >>> result = engine.step(state, x_ref)
    """
    from physicore.core.engine import PhysiCore, PLATFORM_DYNAMICS, PhysiCoreConfig

    robot_model, contact_model, config, platform = build_robot_model(
        path, platform_hint=platform_hint,
        contact_stiffness=contact_stiffness,
        contact_damping=contact_damping,
        friction_mu=friction_mu,
    )

    print(f"[URDF Loader] Robot   : {config.name}")
    print(f"[URDF Loader] Format  : {Path(path).suffix.upper()}")
    print(f"[URDF Loader] DOF     : {robot_model.dof}")
    print(f"[URDF Loader] Mass    : {config.mass:.3f} kg")
    print(f"[URDF Loader] Platform: {platform}")

    # If platform is in the built-in catalogue use it directly.
    # Do NOT pass joint_action_bounds from the URDF — the URDF DOF may differ
    # from the built-in platform's action_dim, which causes CEM shape errors.
    if platform in PLATFORM_DYNAMICS:
        engine = PhysiCore.for_platform(
            platform,
            initial_params=config.initial_params,
            control_hz=control_hz,
            Q=Q,
            R=R,
        )
        # Patch the physics layer dynamics with the URDF-derived one
        # so the engine benefits from robot-specific kinematics
        _generic_fn = _make_generic_dynamics(robot_model, contact_model)
        # Keep the built-in dynamics but attach the URDF model for FK/Jacobian
        engine._urdf_model   = robot_model
        engine._contact_model = contact_model
        return engine, config

    # Unknown platform → use generic URDF dynamics
    dof = robot_model.dof
    state_dim  = dof * 2 + 6   # q, dq, base pose
    action_dim = dof

    dynamics_fn = _make_generic_dynamics(robot_model, contact_model)

    initial_params = {
        "mass":     config.mass,
        "friction": config.friction,
        "inertia":  config.inertia,
    }

    cfg_engine = PhysiCoreConfig(
        platform=platform,
        state_dim=state_dim,
        action_dim=action_dim,
        control_hz=control_hz,
        initial_params=initial_params,
    )
    cfg_engine.cem_samples = 4
    cfg_engine.horizon     = 4
    cfg_engine.cem_iters   = 2

    if Q is None:
        Q = np.eye(state_dim)
    if R is None:
        R = np.eye(action_dim) * 0.01

    action_bounds = config.joint_action_bounds

    engine = PhysiCore(cfg_engine, dynamics_fn, initial_params, Q, R, action_bounds)
    engine._urdf_model    = robot_model
    engine._contact_model = contact_model

    # Register the new platform so it can be used later
    PLATFORM_DYNAMICS[platform] = (dynamics_fn, state_dim, action_dim)
    print(f"[URDF Loader] Registered new platform '{platform}' ({state_dim}D state, {action_dim}D action)")

    return engine, config


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE CLI (python -m physicore.core.urdf_loader robot.urdf)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m physicore.core.urdf_loader <robot.urdf|robot.xml> [platform_hint]")
        sys.exit(1)
    hint = sys.argv[2] if len(sys.argv) > 2 else None
    engine, cfg = load_robot(sys.argv[1], platform_hint=hint)
    print("\n[Config YAML preview]")
    print(cfg.to_yaml())
    # Quick FK test
    if engine._urdf_model.dof > 0:
        q_test = np.zeros(engine._urdf_model.dof)
        ee = engine._urdf_model.ee_position(q_test)
        J  = engine._urdf_model.jacobian(q_test)
        print(f"[FK] EE position at q=0 : {ee}")
        print(f"[FK] Jacobian shape     : {J.shape}")

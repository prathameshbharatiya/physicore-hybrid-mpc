"""
physicore/core/contact_lcp.py

Rigid contact solver using Linear Complementarity Problem (LCP) formulation,
contact detection, and Newton-restitution impact resolution for the PhysiCore
robotics engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContactPoint:
    """
    Represents a single contact between a robot link and the environment
    (or another link).

    Attributes
    ----------
    link_name : str
        Name of the robot link involved in the contact.
    position : np.ndarray, shape (3,)
        World-frame position of the contact point.
    normal : np.ndarray, shape (3,)
        Unit outward normal of the contact surface (points *away* from surface,
        i.e. into the robot link).
    penetration_depth : float
        Signed interpenetration depth (>= 0 means penetrating).
    geometry_radius : float
        Radius of the collision sphere used to detect this contact.
    """

    link_name: str
    position: np.ndarray
    normal: np.ndarray
    penetration_depth: float
    geometry_radius: float


# ---------------------------------------------------------------------------
# Rigid Contact Solver
# ---------------------------------------------------------------------------

class RigidContactSolver:
    """
    Impulse-based rigid contact solver for multi-contact scenarios.

    Uses an LCP formulation (Dantzig complementary pivot) with Coulomb
    friction, falling back to Projected Gauss-Seidel if the pivot method
    fails to converge.

    Parameters
    ----------
    max_contacts : int
        Maximum number of simultaneous contacts to handle.
    """

    def __init__(self, max_contacts: int = 8) -> None:
        self.max_contacts: int = int(max_contacts)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        M: np.ndarray,
        J_contact: np.ndarray,
        mu: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute contact impulses and corrected generalised velocities.

        Parameters
        ----------
        q : np.ndarray, shape (n,)
            Generalised positions (unused directly; present for interface
            completeness and future position-correction extensions).
        dq : np.ndarray, shape (n,)
            Generalised velocities before contact resolution.
        M : np.ndarray, shape (n, n)
            Positive-definite mass matrix.
        J_contact : np.ndarray, shape (nc, n)
            Stacked contact Jacobian, one row per contact point.
        mu : float
            Coulomb friction coefficient.
        dt : float
            Simulation timestep (s).

        Returns
        -------
        lambda_n : np.ndarray, shape (nc,)
            Normal impulses (non-negative).
        lambda_t : np.ndarray, shape (nc, 2)
            Tangential (friction) impulses in 2-D per contact.
        dq_corrected : np.ndarray, shape (n,)
            Generalised velocities after contact resolution.
        """
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        M = np.asarray(M, dtype=float)
        J = np.asarray(J_contact, dtype=float)
        mu = float(mu)
        dt = float(dt)

        nc: int = J.shape[0]

        # 1. Delassus matrix: A = J M^{-1} J^T  (nc x nc)
        M_inv: np.ndarray = np.linalg.inv(M)
        A: np.ndarray = J @ M_inv @ J.T

        # 2. Relative normal velocity at contacts: vn = J @ dq  (nc,)
        vn: np.ndarray = J @ dq

        # 3-4. Solve the LCP for normal impulses
        lambda_n: np.ndarray = self._solve_lcp(A, vn)

        # 5. Friction impulses per contact
        lambda_t: np.ndarray = np.zeros((nc, 2), dtype=float)

        for i in range(nc):
            # Full 3-D velocity at contact i in generalised coordinates
            v3d: np.ndarray = J[i, :] @ dq  # scalar (normal component)

            # We approximate tangential velocity from the contact Jacobian row.
            # The Jacobian row already gives the scalar normal velocity.
            # For tangential directions we build two orthonormal tangent
            # vectors t1, t2 perpendicular to the normal (world z-axis here).
            # Because J_contact rows encode normal direction only, we compute
            # the full 3-D velocity of the contact body point from dq and
            # then project onto the tangent plane.
            n_dof = dq.shape[0]
            if n_dof >= 3:
                # velocity of body origin in 3-D
                v_body = dq[:3]
            else:
                v_body = np.zeros(3, dtype=float)
                v_body[:n_dof] = dq[:n_dof]

            # Simple tangent frame aligned with world x,y
            vt_x = v_body[0]
            vt_y = v_body[1]
            vt_vec = np.array([vt_x, vt_y], dtype=float)
            vt_norm = float(np.linalg.norm(vt_vec))

            if vt_norm < 1e-8:
                # No tangential motion: zero friction impulse
                lambda_t[i, :] = 0.0
            else:
                # Coulomb cone: limit tangential impulse magnitude
                friction_mag = mu * float(lambda_n[i])
                lambda_t[i, :] = -friction_mag * vt_vec / vt_norm

        # 6. Corrected generalised velocities
        #    dq_corr = dq + M^{-1} J^T lambda_n  (+friction contribution)
        dq_corrected: np.ndarray = dq + M_inv @ J.T @ lambda_n

        # Add friction contribution (map 2-D lambda_t back to generalised space)
        # We apply the friction impulse along the first two DOFs if available
        for i in range(nc):
            n_dof = dq.shape[0]
            friction_gen = np.zeros(n_dof, dtype=float)
            if n_dof >= 1:
                friction_gen[0] = lambda_t[i, 0]
            if n_dof >= 2:
                friction_gen[1] = lambda_t[i, 1]
            dq_corrected = dq_corrected + M_inv @ friction_gen

        return lambda_n, lambda_t, dq_corrected

    # ------------------------------------------------------------------
    # LCP solver
    # ------------------------------------------------------------------

    def _solve_lcp(
        self,
        A: np.ndarray,
        q_vec: np.ndarray,
        max_iter: int = 50,
    ) -> np.ndarray:
        """
        Solve the LCP:  w = A z + q >= 0,  z >= 0,  z · w = 0

        Uses Dantzig's complementary pivot algorithm.  Falls back to
        Projected Gauss-Seidel (PGS) if the pivot method does not converge
        within *max_iter* steps.

        Parameters
        ----------
        A : np.ndarray, shape (n, n)
            Positive semi-definite matrix (Delassus operator).
        q_vec : np.ndarray, shape (n,)
            Constant vector (relative velocity at contacts).
        max_iter : int
            Maximum pivot iterations before PGS fallback.

        Returns
        -------
        z : np.ndarray, shape (n,)
            Non-negative solution vector (normal impulses).
        """
        A = np.asarray(A, dtype=float)
        q = np.asarray(q_vec, dtype=float)
        n: int = q.shape[0]

        if n == 0:
            return np.zeros(0, dtype=float)

        # Trivial solution: if q >= 0 everywhere then z = 0 already satisfies
        if np.all(q >= 0.0):
            return np.zeros(n, dtype=float)

        # ------------------------------------------------------------------
        # Dantzig complementary pivot
        # ------------------------------------------------------------------
        # We maintain a tableau of size (n, 2n+1):
        #   columns 0..n-1   : coefficients for w variables  (initially I)
        #   columns n..2n-1  : coefficients for z variables  (initially A)
        #   column  2n       : right-hand side               (initially -q)
        #
        # Basic variables track which variable (w or z) occupies each row.
        # The sign convention:  w = A z + q  => w - A z = q
        # In tableau form: I w - A z = q  =>  RHS starts as q.
        # We want RHS >= 0 at all times for a feasible basis.
        # Following Murty (1988) notation.
        # ------------------------------------------------------------------

        # Build tableau: [I | A | q]
        # (We work with the feasibility tableau where RHS = q, not -q)
        tableau = np.zeros((n, 2 * n + 1), dtype=float)
        tableau[:, :n] = np.eye(n, dtype=float)        # w-columns
        tableau[:, n : 2 * n] = A.copy()               # z-columns
        tableau[:, 2 * n] = q.copy()                   # rhs

        # basic_vars[i] = index of variable in row i
        # 0..n-1  -> w_0..w_{n-1}
        # n..2n-1 -> z_0..z_{n-1}
        basic_vars: List[int] = list(range(n))

        solved = False

        for _iteration in range(max_iter):
            rhs = tableau[:, 2 * n]

            # Find most-negative RHS row (driving variable)
            neg_mask = rhs < -1e-12
            if not np.any(neg_mask):
                solved = True
                break

            # Entering column: the complement of the most negative basic var
            pivot_row = int(np.argmin(rhs))
            bvar = basic_vars[pivot_row]

            # Complement: w_i <-> z_i  (shift by n)
            if bvar < n:
                entering_col = bvar + n   # complement is z_i
            else:
                entering_col = bvar - n   # complement is w_i

            col = tableau[:, entering_col]

            # Minimum ratio test among rows with positive pivot column entry
            ratios = np.full(n, np.inf, dtype=float)
            for r in range(n):
                if col[r] > 1e-12:
                    ratios[r] = rhs[r] / col[r]

            if np.all(np.isinf(ratios)):
                # Degenerate / unbounded: fall through to PGS
                break

            leaving_row = int(np.argmin(ratios))

            # Pivot: row-reduce on tableau[leaving_row, entering_col]
            pivot_elem = tableau[leaving_row, entering_col]
            tableau[leaving_row, :] /= pivot_elem

            for r in range(n):
                if r != leaving_row:
                    tableau[r, :] -= tableau[r, entering_col] * tableau[leaving_row, :]

            basic_vars[leaving_row] = entering_col

        # Extract solution
        if solved:
            z = np.zeros(n, dtype=float)
            for r in range(n):
                var = basic_vars[r]
                if n <= var < 2 * n:        # it is a z-variable
                    z[var - n] = max(0.0, tableau[r, 2 * n])
            return z

        # ------------------------------------------------------------------
        # Fallback: Projected Gauss-Seidel (PGS)
        # ------------------------------------------------------------------
        return self._solve_lcp_pgs(A, q_vec)

    @staticmethod
    def _solve_lcp_pgs(
        A: np.ndarray,
        q_vec: np.ndarray,
        max_iter: int = 50,
    ) -> np.ndarray:
        """
        Projected Gauss-Seidel fallback LCP solver.

        Iteratively updates each component of *z* by the Gauss-Seidel step
        and projects to the non-negative orthant.

        Parameters
        ----------
        A : np.ndarray, shape (n, n)
        q_vec : np.ndarray, shape (n,)
        max_iter : int

        Returns
        -------
        z : np.ndarray, shape (n,)
        """
        A = np.asarray(A, dtype=float)
        q = np.asarray(q_vec, dtype=float)
        n = q.shape[0]
        z = np.zeros(n, dtype=float)

        for _ in range(max_iter):
            for i in range(n):
                # Gauss-Seidel residual
                aii = A[i, i]
                denom = max(aii, 1e-10)
                z_new = z[i] - (float(A[i, :] @ z) + float(q[i])) / denom
                z[i] = max(0.0, z_new)

        return z

    # ------------------------------------------------------------------
    # Impact resolution
    # ------------------------------------------------------------------

    def solve_impact(
        self,
        dq_pre: np.ndarray,
        J: np.ndarray,
        e: float,
    ) -> np.ndarray:
        """
        Newton's restitution-based impact resolution.

        For each contact that is *approaching* (negative normal velocity)
        an impulse is computed so that the post-impact normal velocity equals
        ``-e * vn_pre`` (coefficient of restitution model).

        Parameters
        ----------
        dq_pre : np.ndarray, shape (n,)
            Pre-impact generalised velocities.
        J : np.ndarray, shape (nc, n)
            Contact Jacobians (same layout as :meth:`solve`).
        e : float
            Coefficient of restitution in [0, 1].

        Returns
        -------
        dq_post : np.ndarray, shape (n,)
            Post-impact generalised velocities.
        """
        dq_pre = np.asarray(dq_pre, dtype=float)
        J = np.asarray(J, dtype=float)
        e = float(e)
        nc = J.shape[0]
        n_dof = dq_pre.shape[0]

        vn_pre: np.ndarray = J @ dq_pre  # (nc,)

        dq_post = dq_pre.copy()

        # Build a diagonal mass matrix approximation for impulse computation.
        # We use the effective mass at each contact independently (1-D problem).
        for i in range(nc):
            if vn_pre[i] >= 0.0:
                # Contact is separating or resting: no impulse needed
                continue

            ji = J[i, :]  # shape (n,)

            # Effective mass at this contact: m_eff = 1 / (j^T M^{-1} j)
            # Without an explicit mass matrix, approximate with unit mass
            # (the caller is responsible for passing a meaningful J that
            # already accounts for the mass distribution).
            # Here we use the Gram-Schmidt effective mass from J alone.
            jj = float(np.dot(ji, ji))
            if jj < 1e-14:
                continue

            m_eff_inv = jj  # = j^T I^{-1} j  (identity mass approximation)
            m_eff = 1.0 / m_eff_inv

            # Target post-impact normal velocity
            vn_post_target = -e * vn_pre[i]

            # Impulse magnitude
            delta_vn = vn_post_target - vn_pre[i]
            impulse = m_eff * delta_vn  # scalar

            # Apply impulse (assuming identity effective mass matrix)
            dq_post = dq_post + impulse * ji / jj

        return dq_post


# ---------------------------------------------------------------------------
# Contact Detector
# ---------------------------------------------------------------------------

class ContactDetector:
    """
    Geometry-based contact detector.

    Collision geometries are represented as spheres attached to named links.
    Ground contact is detected against the plane ``z = ground_z``.

    Attributes
    ----------
    _geometries : list of (link_name, local_position, radius) tuples
        Registered collision spheres in local (body) frame.
    """

    def __init__(self) -> None:
        # Each entry: (link_name: str, local_position: np.ndarray(3,), radius: float)
        self._geometries: List[Tuple[str, np.ndarray, float]] = []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_urdf_model(cls, urdf_model: Any) -> "ContactDetector":
        """
        Build a :class:`ContactDetector` from a URDF robot model.

        Parameters
        ----------
        urdf_model : URDFRobotModel (or duck-typed equivalent)
            Robot model with a ``.links`` attribute listing link descriptions.
            Attribute errors are handled gracefully.

        Returns
        -------
        ContactDetector
        """
        detector = cls()

        try:
            links = urdf_model.links
        except AttributeError:
            # Model has no link information: return empty detector
            return detector

        for idx, link in enumerate(links):
            try:
                name: str = str(link.name)
            except AttributeError:
                name = f"link_{idx}"

            # Try to get geometry radius from the model; fall back to 0.05 m
            try:
                radius = float(link.geometry_radius)
            except AttributeError:
                radius = 0.05

            # Local position of the collision sphere centre (default: origin)
            try:
                local_pos = np.asarray(link.collision_origin, dtype=float)
                if local_pos.shape != (3,):
                    local_pos = np.zeros(3, dtype=float)
            except AttributeError:
                local_pos = np.zeros(3, dtype=float)

            detector._geometries.append((name, local_pos, radius))

        return detector

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        q: np.ndarray,
        ground_z: float = 0.0,
    ) -> List[ContactPoint]:
        """
        Detect ground contacts for all registered geometries.

        The world position of each collision sphere is approximated from the
        first three components of *q* (translational DOFs).  Per-link offsets
        are ignored (suitable for floating-base robots where *q[0:3]* is the
        root translation).

        Parameters
        ----------
        q : np.ndarray, shape (n,)
            Generalised positions.
        ground_z : float
            Height of the ground plane (default 0.0).

        Returns
        -------
        contacts : list of ContactPoint
            One entry per geometry that penetrates the ground.
        """
        q = np.asarray(q, dtype=float)
        ground_z = float(ground_z)

        # Extract base translation
        base_pos = np.zeros(3, dtype=float)
        n_q = q.shape[0]
        base_pos[: min(3, n_q)] = q[: min(3, n_q)]

        contacts: List[ContactPoint] = []

        for link_name, local_pos, radius in self._geometries:
            world_pos = base_pos + local_pos

            # Check ground penetration
            if world_pos[2] < ground_z + radius:
                depth = float(max(0.0, ground_z + radius - world_pos[2]))
                cp = ContactPoint(
                    link_name=link_name,
                    position=world_pos.copy(),
                    normal=np.array([0.0, 0.0, 1.0], dtype=float),
                    penetration_depth=depth,
                    geometry_radius=radius,
                )
                contacts.append(cp)

        return contacts

    def detect_self_collision(
        self,
        q: np.ndarray,
    ) -> List[Tuple[str, str]]:
        """
        Detect self-collision between all pairs of registered geometries.

        World positions are approximated from *q* with a small per-link index
        offset so that links are spread out even when only a few generalised
        coordinates are available.

        Parameters
        ----------
        q : np.ndarray, shape (n,)
            Generalised positions.

        Returns
        -------
        colliding_pairs : list of (link_a, link_b) tuples
            Each pair of links whose collision spheres overlap.
        """
        q = np.asarray(q, dtype=float)
        n_q = q.shape[0]

        # Compute approximate world position for every geometry
        base_pos = np.zeros(3, dtype=float)
        base_pos[: min(3, n_q)] = q[: min(3, n_q)]

        positions: List[np.ndarray] = []
        for idx, (_, local_pos, _) in enumerate(self._geometries):
            # Add a small per-index offset to spread links spatially
            offset = np.array(
                [
                    0.05 * float(idx),       # x offset
                    0.0,
                    0.02 * float(idx),       # z offset (stacking)
                ],
                dtype=float,
            )
            positions.append(base_pos + local_pos + offset)

        # Sphere-sphere overlap check for every pair
        colliding_pairs: List[Tuple[str, str]] = []
        n_geom = len(self._geometries)

        for i in range(n_geom):
            for j in range(i + 1, n_geom):
                name_i, _, radius_i = self._geometries[i]
                name_j, _, radius_j = self._geometries[j]

                dist_vec = positions[j] - positions[i]
                dist = float(np.linalg.norm(dist_vec))

                if dist < radius_i + radius_j:
                    colliding_pairs.append((name_i, name_j))

        return colliding_pairs

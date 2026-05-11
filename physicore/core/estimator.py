"""
physicore/core/estimator.py

Extended Kalman Filter state estimator and IMU preintegrator for the
PhysiCore robotics engine.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Extended Kalman Filter
# ---------------------------------------------------------------------------

class StateEstimator:
    """
    Extended Kalman Filter (EKF) for state estimation with multi-sensor fusion.

    The estimator maintains a Gaussian belief over a state vector of dimension
    *state_dim*.  Observations of dimension *obs_dim* (default) can arrive via
    :meth:`update` or, for named sensors registered via
    :meth:`register_sensor`, via :meth:`fuse`.

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector.
    obs_dim : int
        Default observation dimension.
    Q_process : np.ndarray, shape (state_dim, state_dim)
        Process noise covariance.
    R_obs : np.ndarray, shape (obs_dim, obs_dim)
        Default observation noise covariance.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        Q_process: np.ndarray,
        R_obs: np.ndarray,
    ) -> None:
        self.state_dim: int = state_dim
        self.obs_dim: int = obs_dim

        self.Q: np.ndarray = np.asarray(Q_process, dtype=float).copy()
        self.R_obs: np.ndarray = np.asarray(R_obs, dtype=float).copy()

        # State estimate and covariance
        self.x: np.ndarray = np.zeros(state_dim, dtype=float)
        self.P: np.ndarray = np.eye(state_dim, dtype=float)

        # Named sensors: name -> (H_matrix, R_matrix)
        self._sensors: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        dynamics_fn: Callable[..., np.ndarray],
        params: Any,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF prediction step.

        Numerically linearises *dynamics_fn* around *state* to obtain the
        Jacobian **F** and then propagates the mean and covariance forward by
        *dt*.

        Parameters
        ----------
        state : np.ndarray, shape (state_dim,)
            Current state estimate used as linearisation point.
        action : np.ndarray
            Control action passed verbatim to *dynamics_fn*.
        dynamics_fn : callable(state, action, params) -> ndarray (state_dim,)
            Continuous-time dynamics: returns ``dx/dt``.
        params : Any
            Extra parameters forwarded to *dynamics_fn*.
        dt : float
            Integration timestep (seconds).

        Returns
        -------
        x_pred : np.ndarray, shape (state_dim,)
            Copy of the predicted state mean.
        P_pred : np.ndarray, shape (state_dim, state_dim)
            Copy of the predicted covariance.
        """
        state = np.asarray(state, dtype=float)
        eps = 1e-5
        n = self.state_dim

        # Evaluate nominal dynamics
        f0: np.ndarray = np.asarray(dynamics_fn(state, action, params), dtype=float)

        # Numerically compute Jacobian df/dx, column by column
        df_dx = np.zeros((n, n), dtype=float)
        for j in range(n):
            state_plus = state.copy()
            state_plus[j] += eps
            f_plus = np.asarray(dynamics_fn(state_plus, action, params), dtype=float)
            df_dx[:, j] = (f_plus - f0) / eps

        # Discrete-time linearisation: F = I + dt * df/dx
        F: np.ndarray = np.eye(n, dtype=float) + dt * df_dx

        # Propagate mean (Euler integration)
        x_pred: np.ndarray = state + dt * f0

        # Propagate covariance
        P_pred: np.ndarray = F @ self.P @ F.T + self.Q

        # Symmetrize for numerical stability
        P_pred = 0.5 * (P_pred + P_pred.T)

        self.x = x_pred.copy()
        self.P = P_pred.copy()

        return x_pred.copy(), P_pred.copy()

    def update(
        self,
        observation: np.ndarray,
        H: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF measurement update with the default (or supplied) observation model.

        Parameters
        ----------
        observation : np.ndarray, shape (obs_dim,)
            Measurement vector.
        H : np.ndarray, shape (obs_dim, state_dim), optional
            Observation matrix.  Defaults to a truncated identity matrix
            ``eye(min(obs_dim, state_dim), state_dim)``.

        Returns
        -------
        x_updated : np.ndarray
        P_updated : np.ndarray
        """
        z = np.asarray(observation, dtype=float)
        obs_size = z.shape[0]

        if H is None:
            H_use = np.eye(min(self.obs_dim, self.state_dim), self.state_dim, dtype=float)
        else:
            H_use = np.asarray(H, dtype=float)

        # Trim R to match actual observation size
        R_use = self.R_obs[:obs_size, :obs_size]

        self._update_with_sensor(z, H_use, R_use)
        return self.x.copy(), self.P.copy()

    def _update_with_sensor(
        self,
        obs: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> None:
        """
        Internal EKF measurement update step.

        Modifies ``self.x`` and ``self.P`` in place.

        Parameters
        ----------
        obs : np.ndarray
            Measurement vector.
        H : np.ndarray, shape (obs_size, state_dim)
            Observation matrix.
        R : np.ndarray, shape (obs_size, obs_size)
            Observation noise covariance.
        """
        z = np.asarray(obs, dtype=float)
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)

        # Innovation covariance: S = H P H^T + R
        S: np.ndarray = H @ self.P @ H.T + R

        # Kalman gain: K = P H^T S^{-1}
        K: np.ndarray = self.P @ H.T @ np.linalg.inv(S)

        # Innovation (residual)
        y: np.ndarray = z - H @ self.x

        # State update
        self.x = self.x + K @ y

        # Covariance update: P = (I - K H) P
        I = np.eye(self.state_dim, dtype=float)
        self.P = (I - K @ H) @ self.P

        # Numerical symmetrisation
        self.P = 0.5 * (self.P + self.P.T)

    def fuse(self, sources: Dict[str, np.ndarray]) -> None:
        """
        Sequential measurement fusion from multiple named sources.

        For each source whose name has been registered via
        :meth:`register_sensor` the stored ``(H, R)`` pair is used.
        Unknown sources fall back to :meth:`update`.

        Parameters
        ----------
        sources : dict
            Mapping ``{sensor_name: observation_array}``.
        """
        for name, obs in sources.items():
            obs_arr = np.asarray(obs, dtype=float)
            if name in self._sensors:
                H, R = self._sensors[name]
                self._update_with_sensor(obs_arr, H, R)
            else:
                self.update(obs_arr)

    def register_sensor(
        self,
        name: str,
        H_matrix: np.ndarray,
        R_matrix: np.ndarray,
    ) -> None:
        """
        Register a named sensor with its observation model.

        Parameters
        ----------
        name : str
            Unique sensor identifier used as the key in :meth:`fuse`.
        H_matrix : array-like, shape (obs_size, state_dim)
            Observation matrix.
        R_matrix : array-like, shape (obs_size, obs_size)
            Observation noise covariance.
        """
        self._sensors[name] = (
            np.asarray(H_matrix, dtype=float),
            np.asarray(R_matrix, dtype=float),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def estimate(self) -> np.ndarray:
        """Current state estimate (copy)."""
        return self.x.copy()

    @property
    def uncertainty(self) -> np.ndarray:
        """Diagonal of the current covariance matrix (copy)."""
        return np.diag(self.P).copy()


# ---------------------------------------------------------------------------
# IMU Pre-integrator
# ---------------------------------------------------------------------------

class IMUPreintegrator:
    """
    IMU pre-integration for combining high-frequency inertial measurements
    into a single relative motion factor between two keyframes.

    Follows the notation of Forster et al. (2017) but uses a simplified
    first-order rotation update (small-angle approximation per step) that
    is re-orthogonalised via SVD after every step.

    Parameters
    ----------
    gravity : float
        Gravitational acceleration magnitude (m/s²), default 9.81.
    """

    def __init__(self, gravity: float = 9.81) -> None:
        self.gravity: float = float(gravity)

        # Accumulated preintegrated quantities
        self._dp: np.ndarray = np.zeros(3, dtype=float)   # delta position
        self._dv: np.ndarray = np.zeros(3, dtype=float)   # delta velocity
        self._dR: np.ndarray = np.eye(3, dtype=float)     # delta rotation

        # Biases (can be updated externally before integration)
        self._accel_bias: np.ndarray = np.zeros(3, dtype=float)
        self._gyro_bias: np.ndarray = np.zeros(3, dtype=float)

        self._dt_total: float = 0.0

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _skew_sym(v: np.ndarray) -> np.ndarray:
        """Return the 3×3 skew-symmetric matrix of vector *v*."""
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        return np.array(
            [
                [0.0, -vz,  vy],
                [ vz, 0.0, -vx],
                [-vy,  vx, 0.0],
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def integrate(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: float,
    ) -> None:
        """
        Integrate one IMU sample into the accumulated preintegrated state.

        Parameters
        ----------
        accel : np.ndarray, shape (3,)
            Raw accelerometer measurement (body frame, m/s²).
        gyro : np.ndarray, shape (3,)
            Raw gyroscope measurement (body frame, rad/s).
        dt : float
            Sample duration (seconds).
        """
        accel = np.asarray(accel, dtype=float)
        gyro = np.asarray(gyro, dtype=float)
        dt = float(dt)

        # Bias correction
        a_corrected_body: np.ndarray = accel - self._accel_bias
        w_corrected: np.ndarray = gyro - self._gyro_bias

        # Rotate accelerometer measurement to world frame, then subtract gravity
        gravity_vec = np.array([0.0, 0.0, self.gravity], dtype=float)
        a_world: np.ndarray = self._dR @ a_corrected_body - gravity_vec

        # Position and velocity integration (trapezoidal / midpoint variant)
        self._dp += self._dv * dt + 0.5 * a_world * (dt ** 2)
        self._dv += a_world * dt

        # Rotation update: first-order Rodriguez step
        angle_axis: np.ndarray = w_corrected * dt          # small-angle step
        skew: np.ndarray = self._skew_sym(angle_axis)
        self._dR = self._dR @ (np.eye(3, dtype=float) + skew)

        # Re-orthogonalise via SVD to prevent drift
        U, _, Vt = np.linalg.svd(self._dR)
        self._dR = U @ Vt

        self._dt_total += dt

    def reset(self) -> Dict[str, Any]:
        """
        Return accumulated preintegrated quantities and reset internal state.

        Returns
        -------
        dict with keys:
            ``delta_position`` : np.ndarray (3,)
            ``delta_velocity`` : np.ndarray (3,)
            ``delta_rotation`` : np.ndarray (3, 3)
            ``dt_total``       : float
        """
        result: Dict[str, Any] = {
            "delta_position": self._dp.copy(),
            "delta_velocity": self._dv.copy(),
            "delta_rotation": self._dR.copy(),
            "dt_total": self._dt_total,
        }

        # Reset all accumulators
        self._dp = np.zeros(3, dtype=float)
        self._dv = np.zeros(3, dtype=float)
        self._dR = np.eye(3, dtype=float)
        self._dt_total = 0.0

        return result

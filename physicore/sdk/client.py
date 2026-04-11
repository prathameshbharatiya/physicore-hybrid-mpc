"""
PhysiCore SDK Client
====================
High-level client for the PhysiCore API server.
Use this when running PhysiCore as a service.
"""

from __future__ import annotations
import numpy as np
import requests
import json
import time
from typing import Optional, Dict, List


class PhysicoreClient:
    """
    Client for the PhysiCore API server.

    Usage:
        client = PhysicoreClient(host="localhost", port=8000)
        client.configure("quadrotor", {"mass": 1.5, "friction": 0.1})

        while True:
            state  = robot.get_state()
            action = client.step(state, x_ref)
            robot.apply(action)
            client.observe(state, action, robot.get_state())
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.base_url = f"http://{host}:{port}"
        self._check_connection()

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/", timeout=3)
            r.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to PhysiCore API at {self.base_url}. "
                f"Start it with: uvicorn physicore.api.server:app --port 8000\n"
                f"Error: {e}"
            )

    def configure(
        self,
        platform: str,
        initial_params: Dict[str, float],
        control_hz: float = 60.0,
        q_scale: float = 10.0,
        r_scale: float = 0.1,
    ) -> dict:
        """Configure the engine for a platform."""
        r = requests.post(f"{self.base_url}/api/engine/configure", json={
            "platform":       platform,
            "initial_params": initial_params,
            "control_hz":     control_hz,
            "q_scale":        q_scale,
            "r_scale":        r_scale,
        })
        r.raise_for_status()
        return r.json()

    def step(
        self,
        state: np.ndarray,
        x_ref: np.ndarray,
    ) -> np.ndarray:
        """
        One control step. Returns optimal action.

        Args:
            state: Current state vector
            x_ref: Reference target state

        Returns:
            action: Optimal control action
        """
        r = requests.post(f"{self.base_url}/api/engine/step", json={
            "state": state.tolist() if isinstance(state, np.ndarray) else state,
            "x_ref": x_ref.tolist() if isinstance(x_ref, np.ndarray) else x_ref,
        })
        r.raise_for_status()
        data = r.json()
        return np.array(data["action"])

    def step_full(
        self,
        state: np.ndarray,
        x_ref: np.ndarray,
    ) -> dict:
        """Step and return full response including uncertainty and params."""
        r = requests.post(f"{self.base_url}/api/engine/step", json={
            "state": state.tolist() if isinstance(state, np.ndarray) else state,
            "x_ref": x_ref.tolist() if isinstance(x_ref, np.ndarray) else x_ref,
        })
        r.raise_for_status()
        return r.json()

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        """Feed real transition back to engine for online learning."""
        requests.post(f"{self.base_url}/api/engine/observe", json={
            "state":      state.tolist()      if isinstance(state,      np.ndarray) else state,
            "action":     action.tolist()     if isinstance(action,     np.ndarray) else action,
            "next_state": next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
        })

    def get_params(self) -> Dict[str, float]:
        """Get current estimated physical parameters."""
        r = requests.get(f"{self.base_url}/api/engine/params")
        r.raise_for_status()
        return r.json()["params"]

    def get_residual(self) -> float:
        """Get current sim-to-real residual norm."""
        r = requests.get(f"{self.base_url}/api/engine/residual")
        r.raise_for_status()
        return r.json()["residual_norm"]

    def get_uncertainty(self) -> float:
        """Get current epistemic uncertainty."""
        r = requests.get(f"{self.base_url}/api/engine/uncertainty")
        r.raise_for_status()
        return r.json()["uncertainty"]

    def get_status(self) -> dict:
        """Get full engine diagnostics."""
        r = requests.get(f"{self.base_url}/api/status")
        r.raise_for_status()
        return r.json()

    def reset(self) -> None:
        """Reset engine state."""
        requests.post(f"{self.base_url}/api/engine/reset")

    def list_platforms(self) -> dict:
        """List all supported platforms."""
        r = requests.get(f"{self.base_url}/api/platforms")
        r.raise_for_status()
        return r.json()

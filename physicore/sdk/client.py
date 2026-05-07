"""
PhysiCore SDK Client
====================
Drop-in client for teams integrating PhysiCore into their robot stack.
Wraps the REST API so you never need to think about HTTP.

Install PhysiCore API:
    pip install physicore[api]
    uvicorn physicore.api.server:app --host 0.0.0.0 --port 8000

Then in your control loop:

    from physicore.sdk.client import PhysicoreClient

    client = PhysicoreClient()
    client.configure("balancing_bot", {"mass": 1.35, "friction": 0.18})

    while True:
        state  = robot.read_imu()          # your sensor read
        action = client.step(state, x_ref) # PhysiCore computes control
        robot.apply_motors(action)         # your actuator write
        client.observe(state, action, robot.read_imu())  # PhysiCore learns

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations

import time
import json
import numpy as np
import requests
from typing import Optional, Dict, List, Tuple, Any


class PhysicoreClient:
    """
    High-level PhysiCore SDK client.

    Connects to a running PhysiCore API server and provides a clean
    interface for real-time robot control with adaptive physics.

    All methods are synchronous and safe to call from a control loop.
    Thread-safe for single-client usage.
    """

    def __init__(
        self,
        host:    str   = "localhost",
        port:    int   = 8000,
        timeout: float = 2.0,
        retries: int   = 3,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout  = timeout
        self.retries  = retries
        self._platform: Optional[str]  = None
        self._step_count: int = 0
        self._last_action: Optional[np.ndarray] = None
        self._sentinel_active: bool = False
        self._verify_connection()

    # ── SETUP ──────────────────────────────────────────────────────────────────

    def configure(
        self,
        platform:       str,
        initial_params: Dict[str, float],
        control_hz:     float = 60.0,
        q_scale:        float = 10.0,
        r_scale:        float = 0.1,
    ) -> dict:
        """
        Configure the engine for a platform.
        Must be called before step() or observe().

        Args:
            platform:       One of the 12 supported platforms.
            initial_params: Starting physics params e.g. {"mass": 1.35, "friction": 0.18}
            control_hz:     Control loop frequency (default 60Hz)
            q_scale:        State cost weight for CEM-MPC
            r_scale:        Action cost weight for CEM-MPC

        Returns:
            dict with state_dim, action_dim, control_hz
        """
        r = self._post("/api/engine/configure", {
            "platform":       platform,
            "initial_params": initial_params,
            "control_hz":     control_hz,
            "q_scale":        q_scale,
            "r_scale":        r_scale,
        })
        self._platform = platform
        self._step_count = 0
        print(f"[PhysiCore] Configured: {platform} | "
              f"state_dim={r['state_dim']} action_dim={r['action_dim']} "
              f"hz={control_hz}")
        return r

    def attach_sentinel(self, platform: Optional[str] = None) -> dict:
        """
        Attach Sentinel OS safety layer to the engine.
        After this, step() uses sentinel.step() internally.

        Args:
            platform: Platform for sentinel config (defaults to configured platform)
        """
        plat = platform or self._platform or "balancing_bot"
        r = self._post("/api/sentinel/configure", {}, params={"platform": plat})
        self._sentinel_active = True
        print(f"[PhysiCore] Sentinel OS attached | platform={plat}")
        return r

    # ── REAL-TIME CONTROL ──────────────────────────────────────────────────────

    def step(
        self,
        state:    List[float],
        x_ref:    List[float],
        altitude: float = 0.0,
    ) -> np.ndarray:
        """
        One control step. Returns optimal action.

        This is the main method you call in your control loop.
        PhysiCore computes the action using CEM-MPC on the
        real-time adapted physics model.

        Args:
            state:    Current robot state vector
            x_ref:    Target state (zeros = balance/hover/zero-error)
            altitude: Altitude in metres (used for atmosphere model, default 0)

        Returns:
            action: numpy array of motor commands
        """
        if self._sentinel_active:
            r = self._post("/api/sentinel/step", {
                "state": list(state), "x_ref": list(x_ref), "altitude": altitude
            })
            action = np.array(r["action"])
        else:
            r = self._post("/api/engine/step", {
                "state": list(state), "x_ref": list(x_ref)
            })
            action = np.array(r["action"])

        self._step_count += 1
        self._last_action = action
        return action

    def observe(
        self,
        state:      List[float],
        action:     List[float],
        next_state: List[float],
    ) -> None:
        """
        Feed a real state transition back to PhysiCore.

        Call this after every step with what actually happened.
        This is how PhysiCore learns your real physics in real time.

        Args:
            state:      State before action was applied
            action:     Action that was applied
            next_state: State after action was applied (new sensor reading)
        """
        self._post("/api/engine/observe", {
            "state":      list(state),
            "action":     list(action),
            "next_state": list(next_state),
        })

    # ── DIAGNOSTICS ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Full engine diagnostics — params, residual, uncertainty, step count."""
        return self._get("/api/status")

    def params(self) -> Dict[str, float]:
        """Current estimated physics parameters (mass, friction, inertia...)."""
        return self._get("/api/engine/params")["params"]

    def residual(self) -> float:
        """Current sim-to-real residual norm. Lower = better model fit."""
        return self._get("/api/engine/residual")["residual_norm"]

    def uncertainty(self) -> float:
        """Current epistemic uncertainty. Lower = engine is more confident."""
        return self._get("/api/engine/uncertainty")["uncertainty"]

    def sentinel_status(self) -> dict:
        """Full Sentinel OS status (all 8 layers)."""
        return self._get("/api/sentinel/status")

    def sentinel_ledger(self, limit: int = 50) -> List[dict]:
        """Last N entries from the SHA-256 forensic ledger."""
        return self._get("/api/sentinel/ledger", params={"limit": limit})["entries"]

    def faults(self) -> List[dict]:
        """All fault events detected during this session."""
        return self._get("/api/sentinel/faults")["faults"]

    @property
    def is_safe(self) -> bool:
        """True if Sentinel mode is NOMINAL or CAUTIOUS (not FALLBACK)."""
        if not self._sentinel_active:
            return True
        try:
            return self._get("/api/sentinel/status").get("is_safe", True)
        except Exception:
            return True

    # ── REGISTRY (DATA FLYWHEEL) ───────────────────────────────────────────────

    def save_session(self, platform: Optional[str] = None) -> str:
        """
        Save this session's learned model to the registry.
        Call at the end of a hardware session (before shutting down).

        Returns: session_id (SHA-256 prefix)
        """
        plat = platform or self._platform
        if not plat:
            raise ValueError("No platform configured")
        r = self._post(f"/api/registry/{plat}/save", {})
        print(f"[PhysiCore] Session saved | id={r['session_id']}")
        return r["session_id"]

    def load_prior(self, platform: Optional[str] = None) -> bool:
        """
        Load saved model from registry into the engine.
        Call after configure() to warm-start from previous sessions.

        Returns: True if prior was loaded, False if no prior exists yet.
        """
        plat = platform or self._platform
        if not plat:
            raise ValueError("No platform configured")
        r = self._post(f"/api/registry/{plat}/load", {})
        if r["loaded"]:
            print(f"[PhysiCore] Prior loaded from registry | platform={plat}")
        else:
            print(f"[PhysiCore] No prior yet for '{plat}' — starting fresh")
        return r["loaded"]

    def registry_summary(self) -> dict:
        """Global registry summary — all platforms, session counts."""
        return self._get("/api/registry/summary")

    def platform_summary(self, platform: Optional[str] = None) -> dict:
        """Per-platform registry stats — sessions, params, prior weight."""
        plat = platform or self._platform or "balancing_bot"
        return self._get(f"/api/registry/{plat}")

    def convergence_proof(self, platform: Optional[str] = None) -> dict:
        """
        Show session-over-session improvement from the registry flywheel.
        Compares session 1 vs latest session convergence.
        """
        plat = platform or self._platform or "balancing_bot"
        return self._get(f"/api/registry/{plat}/convergence")

    def session_history(self, platform: Optional[str] = None,
                        limit: int = 20) -> List[dict]:
        """List recent hardware sessions from the registry."""
        plat = platform or self._platform or "balancing_bot"
        return self._get(f"/api/registry/{plat}/sessions",
                         params={"limit": limit})["sessions"]

    # ── PLATFORMS ──────────────────────────────────────────────────────────────

    def platforms(self) -> Dict[str, dict]:
        """List all 12 supported platforms with state/action dimensions."""
        return self._get("/api/platforms")

    def reset(self) -> None:
        """Reset the engine. Requires re-configure() before next use."""
        self._post("/api/engine/reset", {})
        self._platform    = None
        self._step_count  = 0
        self._last_action = None
        self._sentinel_active = False
        print("[PhysiCore] Engine reset")

    # ── CONTEXT MANAGER ────────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        """Auto-save session to registry on clean exit."""
        if self._platform and self._step_count > 10:
            try:
                self.save_session()
            except Exception as e:
                print(f"[PhysiCore] Auto-save failed: {e}")

    # ── HTTP HELPERS ───────────────────────────────────────────────────────────

    def _verify_connection(self) -> None:
        try:
            r = requests.get(f"{self.base_url}/", timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            print(f"[PhysiCore] Connected to API v{data.get('version','?')} "
                  f"at {self.base_url}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to PhysiCore API at {self.base_url}\n"
                f"Start it with:\n"
                f"  uvicorn physicore.api.server:app --host 0.0.0.0 --port 8000\n"
                f"Error: {e}"
            )

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        for attempt in range(self.retries):
            try:
                r = requests.get(f"{self.base_url}{path}",
                                 params=params, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"GET {path} failed: {e}")
                time.sleep(0.05)

    def _post(self, path: str, data: dict,
              params: Optional[dict] = None) -> Any:
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{self.base_url}{path}",
                                  json=data, params=params,
                                  timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"POST {path} failed: {e}")
                time.sleep(0.05)


# ── CONVENIENCE: INLINE CLIENT (no API server needed) ─────────────────────────

class PhysicoreInline:
    """
    Inline PhysiCore client — runs the engine directly in-process.
    No API server needed. Ideal for embedded deployment or testing.

    Usage:
        client = PhysicoreInline("balancing_bot", {"mass": 1.35, "friction": 0.18})

        while True:
            action = client.step(state, x_ref)
            robot.apply(action)
            client.observe(state, action, next_state)

        client.save()   # save to registry before shutdown
    """

    def __init__(
        self,
        platform:       str,
        initial_params: Dict[str, float],
        control_hz:     float = 60.0,
        use_sentinel:   bool  = True,
        load_registry:  bool  = True,
        verbose:        bool  = True,
    ):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

        from physicore import PhysiCore
        from physicore.core.registry import get_registry

        self._platform = platform
        self._verbose  = verbose

        self.engine = PhysiCore.for_platform(platform, initial_params,
                                              control_hz=control_hz)

        # Load registry prior if available
        if load_registry:
            reg = get_registry()
            loaded = reg.load(self.engine, platform)
            if verbose:
                if loaded:
                    print(f"[PhysiCore] Registry prior loaded | platform={platform}")
                else:
                    print(f"[PhysiCore] No prior for '{platform}' — fresh start")

        # Attach Sentinel
        self.sentinel = None
        if use_sentinel:
            from physicore.sentinel.core import SentinelOS
            self.sentinel = SentinelOS(self.engine, platform=platform,
                                       verbose=verbose)

        self._step_count = 0
        if verbose:
            print(f"[PhysiCore] Inline engine ready | platform={platform} | "
                  f"sentinel={'ON' if use_sentinel else 'OFF'}")

    def step(self, state: np.ndarray, x_ref: np.ndarray,
             altitude: float = 0.0) -> np.ndarray:
        """One control step. Returns safe action."""
        state = np.asarray(state, dtype=float)
        x_ref = np.asarray(x_ref, dtype=float)
        self._step_count += 1

        if self.sentinel:
            return self.sentinel.step(state, x_ref, altitude=altitude)
        else:
            return self.engine.step(state, x_ref).action

    def observe(self, state: np.ndarray, action: np.ndarray,
                next_state: np.ndarray) -> None:
        """Feed real transition back for learning."""
        if self.sentinel:
            self.sentinel.observe(state, action, next_state)
        else:
            self.engine.observe(state, action, next_state)

    def save(self) -> str:
        """Save learned model to registry."""
        from physicore.core.registry import get_registry
        reg = get_registry()
        session_id = reg.save(self.engine, platform=self._platform,
                              session_meta={"steps": self._step_count})
        if self._verbose:
            print(f"[PhysiCore] Session saved | id={session_id} | "
                  f"steps={self._step_count}")
        return session_id

    @property
    def status(self) -> dict:
        if self.sentinel:
            return self.sentinel.status
        return self.engine.diagnostics_full

    @property
    def is_safe(self) -> bool:
        if self.sentinel:
            return self.sentinel.is_safe
        return True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._step_count > 10:
            try:
                self.save()
            except Exception as e:
                print(f"[PhysiCore] Auto-save failed: {e}")

    def close(self):
        """Alias for __exit__ — call at end of session."""
        self.__exit__()

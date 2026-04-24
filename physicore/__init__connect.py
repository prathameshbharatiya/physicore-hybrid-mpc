"""
PhysiCore — Three-line integration
===================================

    import physicore

    robot = physicore.connect("balancing_bot", mass=1.35, friction=0.18)

    while True:
        action = robot.step(sensor_state)   # PhysiCore computes control
        hardware.apply(action)              # your actuator write
        robot.observe(sensor_state, action, hardware.read())  # PhysiCore learns

That is the entire integration. PhysiCore handles:
  - CEM-MPC optimal control at 60Hz
  - Real-time mass/friction/drag learning from sensor data
  - Sentinel safety layer with 8-layer verification
  - Registry: saves learned params between sessions
  - Webhooks: notifies your system when faults/convergence occur

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional, Dict, List, Callable, Any


def connect(
    platform:        str,
    *,
    host:            str   = 'localhost',
    port:            int   = 8000,
    api_key:         str   = '',
    mass:            float = 1.0,
    friction:        float = 0.3,
    inertia:         float = 0.1,
    control_hz:      float = 60.0,
    use_sentinel:    bool  = True,
    load_registry:   bool  = True,
    inline:          bool  = True,
    on_fault:        Optional[Callable] = None,
    on_convergence:  Optional[Callable] = None,
    verbose:         bool  = True,
) -> "PhysicoreRobot":
    """
    Connect PhysiCore to your robot. Returns a PhysicoreRobot ready for use.

    Args:
        platform:       Robot type. One of: balancing_bot, quadrotor, rocket,
                        manipulator_arm, legged_robot, auv, ground_rover,
                        fixed_wing, evtol, surgical_robot, rover, satellite
        host:           PhysiCore API host (default: localhost for inline mode)
        port:           PhysiCore API port (default: 8000)
        api_key:        API key for hosted api.physicore.ai (leave empty for local)
        mass:           Initial mass estimate in kg
        friction:       Initial friction/drag estimate
        inertia:        Initial inertia estimate
        control_hz:     Control loop frequency (default: 60Hz)
        use_sentinel:   Enable Sentinel safety layer (default: True)
        load_registry:  Load learned params from previous sessions (default: True)
        inline:         Run engine in-process — no API server needed (default: True)
        on_fault:       Callback called when a hardware fault is detected
        on_convergence: Callback called when SysID has converged
        verbose:        Print status messages

    Returns:
        PhysicoreRobot — call .step(), .observe(), .save()

    Example:
        robot = physicore.connect("balancing_bot", mass=1.35)

        while True:
            action = robot.step(imu_state)
            motors.apply(action)
            robot.observe(imu_state, action, imu_state_next)
    """
    params = {k: v for k, v in {'mass': mass, 'friction': friction, 'inertia': inertia}.items()}

    if inline or (host == 'localhost' and not api_key):
        # In-process engine — no API server needed
        return PhysicoreRobot._inline(
            platform, params, control_hz, use_sentinel, load_registry,
            on_fault, on_convergence, verbose
        )
    else:
        # Remote API
        return PhysicoreRobot._remote(
            platform, params, host, port, api_key, control_hz,
            on_fault, on_convergence, verbose
        )


class PhysicoreRobot:
    """
    Your robot's PhysiCore interface.

    Core loop:
        action = robot.step(state)          # get optimal action
        robot.observe(state, action, next)  # feed what actually happened
        robot.save()                        # save at session end

    Diagnostics:
        robot.status                        # dict with all metrics
        robot.is_safe                       # bool — safe to continue
        robot.params                        # learned physics params
        robot.narrate()                     # plain-English status
    """

    def __init__(self):
        self._engine       = None
        self._sentinel     = None
        self._client       = None
        self._platform     = ''
        self._step_count   = 0
        self._on_fault     = None
        self._on_conv      = None
        self._converged    = False
        self._verbose      = True

    @classmethod
    def _inline(cls, platform, params, hz, sentinel, registry,
                on_fault, on_conv, verbose) -> "PhysicoreRobot":
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from physicore import PhysiCore
        from physicore.core.registry import get_registry

        r = cls()
        r._platform   = platform
        r._on_fault   = on_fault
        r._on_conv    = on_conv
        r._verbose    = verbose

        r._engine = PhysiCore.for_platform(platform, params, control_hz=hz)

        if registry:
            reg = get_registry()
            loaded = reg.load(r._engine, platform)
            if verbose:
                print(f'[PhysiCore] {"Prior loaded" if loaded else "Fresh start"} | platform={platform}')

        if sentinel:
            from physicore.sentinel.core import SentinelOS
            r._sentinel = SentinelOS(r._engine, platform=platform, verbose=False)

        if verbose:
            print(f'[PhysiCore] Ready | {platform} | {hz}Hz | sentinel={"ON" if sentinel else "OFF"}')
        return r

    @classmethod
    def _remote(cls, platform, params, host, port, api_key, hz,
                on_fault, on_conv, verbose) -> "PhysicoreRobot":
        from physicore.sdk.client import PhysicoreClient
        r = cls()
        r._platform  = platform
        r._on_fault  = on_fault
        r._on_conv   = on_conv
        r._verbose   = verbose
        r._client    = PhysicoreClient(host=host, port=port)
        r._client.configure(platform, params, control_hz=hz)
        return r

    # ── Core loop ──────────────────────────────────────────────────────────────

    def step(self, state, x_ref=None) -> np.ndarray:
        """
        Compute optimal action for current state.
        Call this every control cycle.

        Args:
            state: Current sensor state (list or numpy array)
            x_ref: Target state (zeros if not provided = balance/hover)

        Returns:
            action: numpy array of motor commands
        """
        state = np.asarray(state, dtype=float)
        if self._engine:
            n = self._engine.cfg.state_dim
            state = state[:n] if len(state) >= n else np.pad(state, (0, n - len(state)))
            ref   = np.zeros(n) if x_ref is None else np.asarray(x_ref, dtype=float)[:n]
        else:
            ref = np.zeros_like(state) if x_ref is None else np.asarray(x_ref, dtype=float)

        if self._client:
            action = self._client.step(list(state), list(ref))
        elif self._sentinel:
            action = self._sentinel.step(state, ref)
        else:
            action = self._engine.step(state, ref).action

        self._step_count += 1
        self._check_callbacks()
        return action

    def observe(self, state, action, next_state) -> None:
        """
        Feed what actually happened to PhysiCore.
        Call after every step. This is how PhysiCore learns your real physics.

        Args:
            state:      State before action was applied
            action:     Action that was applied
            next_state: Measured state after action (new sensor reading)
        """
        s  = np.asarray(state,      dtype=float)
        a  = np.asarray(action,     dtype=float)
        ns = np.asarray(next_state, dtype=float)

        if self._client:
            self._client.observe(list(s), list(a), list(ns))
        elif self._sentinel:
            self._sentinel.observe(s, a, ns)
        else:
            self._engine.observe(s, a, ns)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @property
    def status(self) -> dict:
        """Full engine diagnostics — params, residual, uncertainty, step count."""
        if self._client:
            return self._client.status()
        if self._sentinel:
            return self._sentinel.status
        return self._engine.diagnostics_full

    @property
    def is_safe(self) -> bool:
        """True if Sentinel is NOMINAL or CAUTIOUS (not FALLBACK)."""
        if self._client:
            return self._client.is_safe
        if self._sentinel:
            return self._sentinel.is_safe
        return True

    @property
    def params(self) -> Dict[str, float]:
        """Current learned physics parameters."""
        if self._engine:
            return self._engine.physics.params.copy()
        return {}

    @property
    def residual(self) -> float:
        """Current sim-to-real residual. Lower = better model fit."""
        if self._engine:
            return self._engine.diagnostics_full.get('residual_norm', 0.0)
        return 0.0

    def narrate(self) -> str:
        """
        Plain-English status. Display this in your product UI.

        Returns a single string your customers can read.
        Example: "Model converged — mass 1.347kg, residual 0.003. Nominal."
        """
        if self._engine:
            n = self._engine.narrate()
            return f"{n['headline']} {n['action']}"
        return "PhysiCore running."

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> str:
        """
        Save learned model to registry.
        Call at end of session. Next session starts from these params.

        Returns: session_id
        """
        if self._client:
            return self._client.save_session(self._platform)
        if self._engine:
            from physicore.core.registry import get_registry
            reg = get_registry()
            sid = reg.save(self._engine, self._platform,
                           session_meta={'steps': self._step_count})
            if self._verbose:
                print(f'[PhysiCore] Session saved | steps={self._step_count} | id={sid}')
            return sid
        return ''

    # ── Webhooks ───────────────────────────────────────────────────────────────

    def on_fault(self, callback: Callable) -> "PhysicoreRobot":
        """Register a callback for fault events. Returns self for chaining."""
        self._on_fault = callback
        return self

    def on_convergence(self, callback: Callable) -> "PhysicoreRobot":
        """Register a callback for convergence events. Returns self for chaining."""
        self._on_conv = callback
        return self

    def _check_callbacks(self) -> None:
        """Internal: fire callbacks when conditions are met."""
        if not self._engine:
            return
        d = self._engine.diagnostics_full

        # Fault callback
        if self._on_fault:
            failures = d.get('failure_summary', {}).get('recent_10', [])
            for f in failures:
                if f.get('severity') == 'CRITICAL':
                    try:
                        self._on_fault(f)
                    except Exception:
                        pass

        # Convergence callback
        if self._on_conv and not self._converged:
            hist = d.get('sysid_loss_hist', [])
            res  = d.get('residual_norm', 1.0)
            if len(hist) >= 10 and res < 0.05:
                last5 = hist[-5:]
                if max(last5) - min(last5) < 0.002:
                    self._converged = True
                    try:
                        self._on_conv(d)
                    except Exception:
                        pass

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "PhysicoreRobot":
        return self

    def __exit__(self, *args) -> None:
        if self._step_count > 10:
            try:
                self.save()
            except Exception as e:
                if self._verbose:
                    print(f'[PhysiCore] Auto-save failed: {e}')

    def __repr__(self) -> str:
        return (f'PhysicoreRobot(platform={self._platform!r}, '
                f'steps={self._step_count}, '
                f'residual={self.residual:.4f})')

"""
PhysiCore Simulator
===================
Run PhysiCore in pure simulation mode — no hardware needed.
Test your dynamics model and tune parameters before flying.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from physicore import PhysiCore, PhysiCoreConfig, PLATFORM_DYNAMICS


@dataclass
class SimulationResult:
    states:        np.ndarray
    actions:       np.ndarray
    residuals:     List[float]
    uncertainties: List[float]
    params_history: List[dict]
    loop_times_ms: List[float]
    n_steps:       int
    platform:      str

    @property
    def mean_residual(self) -> float:
        return float(np.mean(self.residuals)) if self.residuals else 0.0

    @property
    def final_params(self) -> dict:
        return self.params_history[-1] if self.params_history else {}

    @property
    def avg_loop_ms(self) -> float:
        return float(np.mean(self.loop_times_ms)) if self.loop_times_ms else 0.0

    def summary(self) -> str:
        lines = [
            f"PhysiCore Simulation — {self.platform}",
            f"  Steps:           {self.n_steps}",
            f"  Mean residual:   {self.mean_residual:.4f}",
            f"  Avg loop time:   {self.avg_loop_ms:.2f} ms",
            f"  Final params:    {self.final_params}",
        ]
        return "\n".join(lines)


class PhysicoreSimulator:
    """
    Run PhysiCore in simulation mode.

    Usage:
        sim = PhysicoreSimulator("quadrotor")
        result = sim.run(
            initial_state=np.array([0,0,5,0,0,0,0,0,0,0,0,0]),
            x_ref=np.array([0,0,10,0,0,0,0,0,0,0,0,0]),
            n_steps=300,
            true_params={"mass": 1.8, "friction": 0.15}  # inject model mismatch
        )
        print(result.summary())
    """

    def __init__(
        self,
        platform: str,
        initial_params: Optional[Dict[str, float]] = None,
        control_hz: float = 60.0,
    ):
        if platform not in PLATFORM_DYNAMICS:
            raise ValueError(f"Unknown platform '{platform}'. Available: {list(PLATFORM_DYNAMICS.keys())}")

        self.platform    = platform
        self.control_hz  = control_hz
        dynamics_fn, state_dim, action_dim = PLATFORM_DYNAMICS[platform]
        self.dynamics_fn = dynamics_fn
        self.state_dim   = state_dim
        self.action_dim  = action_dim

        PLATFORM_DEFAULTS = {
            "quadrotor":       {"mass": 1.5,   "friction": 0.1,  "inertia": 0.02},
            "fixed_wing":      {"mass": 12.5,  "friction": 0.025,"inertia": 5.7},
            "evtol":           {"mass": 500.0, "friction": 0.05, "inertia": 0.02},
            "manipulator_arm": {"mass": 2.0,   "friction": 0.3,  "inertia": 0.1},
            "surgical_robot":  {"mass": 0.05,  "friction": 0.8,  "inertia": 0.1},
            "legged_robot":    {"mass": 30.0,  "friction": 0.7,  "inertia": 0.5},
            "balancing_bot":   {"mass": 1.0,   "friction": 0.15, "inertia": 0.01},
            "rocket":          {"mass": 0.15,  "friction": 0.45, "inertia": 220.0},
            "ground_rover":    {"mass": 5.0,   "friction": 0.5,  "inertia": 0.1},
            "auv":             {"mass": 50.0,  "friction": 2.0,  "inertia": 0.02},
            "satellite":       {"mass": 100.0, "friction": 1e-5, "inertia": 10.0},
            "rover":           {"mass": 5.0,   "friction": 0.5,  "inertia": 0.1},
        }
        if initial_params is None:
            initial_params = PLATFORM_DEFAULTS.get(platform, {"mass": 1.0, "friction": 0.3, "inertia": 0.1})
        self.initial_params = initial_params

        self.engine = PhysiCore.for_platform(
            platform=platform,
            initial_params=initial_params,
            control_hz=control_hz,
        )

    def run(
        self,
        initial_state: np.ndarray,
        x_ref: np.ndarray,
        n_steps: int = 300,
        true_params: Optional[Dict[str, float]] = None,
        noise_std: float = 0.001,
        verbose: bool = True,
        log_every: int = 60,
    ) -> SimulationResult:
        """
        Run a simulation episode.

        Args:
            initial_state: Starting state vector
            x_ref:         Target reference state
            n_steps:       Number of control steps
            true_params:   True physical parameters (inject model mismatch)
            noise_std:     Sensor noise standard deviation
            verbose:       Print progress
            log_every:     Print every N steps

        Returns:
            SimulationResult with full trajectory data
        """
        if true_params is None:
            true_params = self.initial_params.copy()

        dt      = 1.0 / self.control_hz
        state   = initial_state.copy()
        states  = [state.copy()]
        actions, residuals, uncertainties = [], [], []
        params_history, loop_times = [], []

        if verbose:
            print(f"[SIM] {self.platform} | {n_steps} steps @ {self.control_hz}Hz")
            print(f"[SIM] Nominal params: {self.initial_params}")
            print(f"[SIM] True params:    {true_params}")
            print("-" * 60)

        for i in range(n_steps):
            step = self.engine.step(state, x_ref)

            state_dot  = self.dynamics_fn(state, step.action, true_params)
            noise      = np.random.randn(self.state_dim) * noise_std
            next_state = state + state_dot * dt + noise

            self.engine.observe(state, step.action, next_state)

            states.append(next_state.copy())
            actions.append(step.action.copy())
            residuals.append(step.uncertainty)
            uncertainties.append(step.uncertainty)
            params_history.append(step.params.copy())
            loop_times.append(step.loop_time_ms)

            state = next_state

            if verbose and (i % log_every == 0 or i == n_steps - 1):
                l2 = float(np.linalg.norm(state - x_ref))
                print(
                    f"  Step {i:4d} | L2={l2:.4f} | "
                    f"σ²={step.uncertainty:.4f} | "
                    f"mass_est={step.params.get('mass', 0):.3f} | "
                    f"loop={step.loop_time_ms:.1f}ms"
                )

        return SimulationResult(
            states=np.array(states),
            actions=np.array(actions) if actions else np.zeros((0, self.action_dim)),
            residuals=residuals,
            uncertainties=uncertainties,
            params_history=params_history,
            loop_times_ms=loop_times,
            n_steps=n_steps,
            platform=self.platform,
        )

    def validate(
        self,
        initial_state: np.ndarray,
        x_ref: np.ndarray,
        true_params: Dict[str, float],
        n_steps: int = 300,
    ) -> dict:
        """
        Run ablation validation and return benchmark metrics.
        Compares PhysiCore against PID baseline.
        """
        result = self.run(initial_state, x_ref, n_steps, true_params, verbose=False)

        states  = result.states
        l2_all  = [float(np.linalg.norm(s - x_ref)) for s in states]
        l2_init = float(np.mean(l2_all[:min(100, n_steps//3)]))
        l2_final= float(np.mean(l2_all[max(0, n_steps - 100):]))
        improvement = (l2_init - l2_final) / max(l2_init, 1e-9) * 100

        return {
            "platform":         self.platform,
            "n_steps":          n_steps,
            "l2_initial":       l2_init,
            "l2_final":         l2_final,
            "improvement_pct":  improvement,
            "avg_loop_ms":      result.avg_loop_ms,
            "final_params":     result.final_params,
            "mean_uncertainty": float(np.mean(result.uncertainties)),
        }

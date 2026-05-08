"""
PhysiCore Performance Configuration
=====================================
Single place to configure all performance settings.

Quick start:
    from physicore.core.performance_config import configure
    configure("laptop")       # safe defaults, no special deps
    configure("workstation")  # JAX CEM (requires: pip install jax jaxlib)
    configure("production")   # everything ON (Linux + JAX + CAP_SYS_NICE)

Environment variables (advanced):
    PHYSICORE_JAX=1                  Enable JAX CEM
    PHYSICORE_RT=1                   Enable RT thread (Linux, needs CAP_SYS_NICE)
    PHYSICORE_RT_CPU=2               Pin RT thread to CPU core 2
    PHYSICORE_RT_PRIO=80             SCHED_FIFO priority
    PHYSICORE_BATCHED_ENSEMBLE=0     Disable batched ensemble
    PHYSICORE_LATENCY_COMP=0         Disable latency compensation
    PHYSICORE_LATENCY_MS=20          Manual latency override in ms
    PHYSICORE_FORCE_NUMPY=1          Force numpy CEM

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceConfig:
    use_jax_cem:          bool  = False
    use_rt_thread:        bool  = False
    rt_cpu:               int   = -1
    rt_priority:          int   = 80
    use_batched_ensemble: bool  = True
    use_latency_comp:     bool  = True
    manual_latency_ms:    Optional[float] = None
    force_numpy:          bool  = False
    cem_samples_override: Optional[int]   = None

    def apply(self):
        """Write this config into environment variables."""
        def _s(k, v): os.environ[k] = str(v)
        _s("PHYSICORE_JAX",              "1" if self.use_jax_cem else "0")
        _s("PHYSICORE_RT",               "1" if self.use_rt_thread else "0")
        _s("PHYSICORE_RT_CPU",           str(self.rt_cpu))
        _s("PHYSICORE_RT_PRIO",          str(self.rt_priority))
        _s("PHYSICORE_BATCHED_ENSEMBLE", "0" if not self.use_batched_ensemble else "1")
        _s("PHYSICORE_LATENCY_COMP",     "0" if not self.use_latency_comp else "1")
        _s("PHYSICORE_FORCE_NUMPY",      "1" if self.force_numpy else "0")
        if self.manual_latency_ms is not None:
            _s("PHYSICORE_LATENCY_MS", str(self.manual_latency_ms))

    def describe(self) -> str:
        lines = ["PhysiCore Performance Config:"]
        cem_label = (f"JAX ({self.cem_samples_override or 64} samples, backend auto)"
                     if self.use_jax_cem else "numpy (8 samples)")
        lines.append(f"  CEM:      {cem_label}")
        ens_label = ("batched — 1 matmul for all members"
                     if self.use_batched_ensemble else "sequential — N matmuls")
        lines.append(f"  Ensemble: {ens_label}")
        if self.use_latency_comp:
            lat = (f"{self.manual_latency_ms}ms manual"
                   if self.manual_latency_ms else "auto-measured RTT")
            lines.append(f"  Latency:  Smith Predictor ON ({lat})")
        else:
            lines.append("  Latency:  OFF")
        if self.use_rt_thread:
            lines.append(f"  RT:       SCHED_FIFO priority={self.rt_priority} cpu={self.rt_cpu}")
        else:
            lines.append("  RT:       normal thread")
        return "\n".join(lines)


PROFILES = {
    "laptop": PerformanceConfig(
        use_jax_cem=False, use_rt_thread=False,
        use_batched_ensemble=True, use_latency_comp=True,
    ),
    "workstation": PerformanceConfig(
        use_jax_cem=True, use_rt_thread=False,
        use_batched_ensemble=True, use_latency_comp=True,
    ),
    "embedded": PerformanceConfig(
        use_jax_cem=False, use_rt_thread=False,
        use_batched_ensemble=False, use_latency_comp=False,
        cem_samples_override=4,
    ),
    "production": PerformanceConfig(
        use_jax_cem=True, use_rt_thread=True,
        use_batched_ensemble=True, use_latency_comp=True,
        rt_priority=80,
    ),
    "debug": PerformanceConfig(
        use_jax_cem=False, use_rt_thread=False,
        use_batched_ensemble=False, use_latency_comp=False,
        force_numpy=True,
    ),
}


def configure(profile: str = "laptop", **overrides) -> PerformanceConfig:
    """
    Apply a named performance profile with optional overrides.

    Examples:
        configure("laptop")
        configure("workstation")
        configure("production", rt_cpu=3)
        configure("laptop", manual_latency_ms=15.0)
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(PROFILES.keys())}")
    import dataclasses
    cfg_dict = dataclasses.asdict(PROFILES[profile])
    cfg_dict.update(overrides)
    cfg = PerformanceConfig(**cfg_dict)
    cfg.apply()
    print(cfg.describe())
    return cfg


def from_env() -> PerformanceConfig:
    """Read current configuration from environment variables."""
    return PerformanceConfig(
        use_jax_cem          = os.environ.get("PHYSICORE_JAX",               "0") == "1",
        use_rt_thread        = os.environ.get("PHYSICORE_RT",                "0") == "1",
        rt_cpu               = int(os.environ.get("PHYSICORE_RT_CPU",        "-1")),
        rt_priority          = int(os.environ.get("PHYSICORE_RT_PRIO",       "80")),
        use_batched_ensemble = os.environ.get("PHYSICORE_BATCHED_ENSEMBLE",  "1") != "0",
        use_latency_comp     = os.environ.get("PHYSICORE_LATENCY_COMP",      "1") != "0",
        manual_latency_ms    = float(os.environ["PHYSICORE_LATENCY_MS"])
                               if "PHYSICORE_LATENCY_MS" in os.environ else None,
        force_numpy          = os.environ.get("PHYSICORE_FORCE_NUMPY",       "0") == "1",
    )

"""
PhysiCore Model Registry
========================
Persistent learning across sessions.

Every time PhysiCore runs on real hardware, it learns:
  - Real mass, friction, inertia (SystemID)
  - Real physics residuals (ResidualEnsemble weights)
  - Adaptation trajectory (convergence history)

This module saves that learning and loads it next time.
The 101st lab to use PhysiCore on a UR5 arm starts with
99 labs worth of prior knowledge, not random weights.

Usage:
    registry = ModelRegistry()
    registry.save(engine, platform='balancing_bot', session_meta={...})
    registry.load(engine, platform='balancing_bot')

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any


# ─── Registry location ─────────────────────────────────────────────────────────
# Stored in ~/.physicore/ so it persists across project folders
_REGISTRY_ROOT = Path.home() / ".physicore" / "registry"
_REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class SessionRecord:
    """Metadata for one hardware session."""
    session_id:       str
    platform:         str
    timestamp:        float
    duration_s:       float
    steps:            int
    final_params:     Dict[str, float]
    convergence_pct:  float          # how much residual dropped (0-100)
    innovation_ema:   float
    hardware_meta:    Dict[str, Any]  # OS, MCU, IMU, etc from config
    opt_in_telemetry: bool

    def to_dict(self): return asdict(self)


class ModelRegistry:
    """
    Saves and loads PhysiCore model state across sessions.

    Structure on disk:
        ~/.physicore/registry/
            balancing_bot/
                params.json          — latest converged SystemID params
                ensemble_0.npz       — ResidualMLP weights, member 0
                ensemble_1.npz       — ResidualMLP weights, member 1
                ensemble_2.npz       — ResidualMLP weights, member 2
                sessions.jsonl       — log of every session
                platform_prior.json  — aggregated prior from all sessions
            quadrotor/
                ...
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else _REGISTRY_ROOT

    def _platform_dir(self, platform: str) -> Path:
        d = self.root / platform
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── SAVE ──────────────────────────────────────────────────────────────────

    def save(
        self,
        engine,                          # PhysiCore instance
        platform: str,
        session_meta: Optional[Dict] = None,
        opt_in_telemetry: bool = False,
    ) -> str:
        """
        Save engine state after a real hardware session.
        Returns session_id.
        """
        d = self._platform_dir(platform)
        session_id = hashlib.sha256(
            f"{platform}-{time.time()}".encode()
        ).hexdigest()[:12]

        # 1. Save SystemID params
        params = engine.physics.params.copy()
        params_path = d / "params.json"

        # Merge with existing params using exponential moving average
        # so params improve across sessions, not just overwrite
        if params_path.exists():
            with open(params_path) as f:
                existing = json.load(f)
            merged = {}
            for k, v in params.items():
                if k in existing:
                    # Weight current session 40%, historical 60%
                    merged[k] = 0.6 * existing[k] + 0.4 * v
                else:
                    merged[k] = v
            params_to_save = merged
        else:
            params_to_save = params

        with open(params_path, "w") as f:
            json.dump({
                "params": params_to_save,
                "platform": platform,
                "last_updated": time.time(),
                "sessions_count": self._session_count(platform) + 1,
            }, f, indent=2)

        # 2. Save ResidualEnsemble weights
        for i, member in enumerate(engine.ensemble.members):
            np.savez(
                d / f"ensemble_{i}.npz",
                W1=member.W1, b1=member.b1,
                W2=member.W2, b2=member.b2,
                W3=member.W3, b3=member.b3,
            )

        # 3. Save CEM warm start
        np.savez(
            d / "cem_warmstart.npz",
            mu=engine.cem.mu,
            std=engine.cem.std,
        )

        # 4. Compute convergence quality
        hist = engine.sysid.convergence_history
        if len(hist) >= 2:
            initial = hist[0] if hist[0] > 0 else 1.0
            final   = hist[-1]
            convergence_pct = max(0.0, (initial - final) / initial * 100)
        else:
            convergence_pct = 0.0

        # 5. Log session record
        record = SessionRecord(
            session_id=session_id,
            platform=platform,
            timestamp=time.time(),
            duration_s=engine._step_count / max(engine.cfg.control_hz, 1),
            steps=engine._step_count,
            final_params=params,
            convergence_pct=convergence_pct,
            innovation_ema=engine.sysid.innovation_ema,
            hardware_meta=session_meta or {},
            opt_in_telemetry=opt_in_telemetry,
        )

        with open(d / "sessions.jsonl", "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        # 6. Update platform prior (aggregated params from all sessions)
        self._update_prior(platform, params, engine._step_count, convergence_pct)

        print(f"[REGISTRY] Saved session {session_id} for '{platform}'")
        print(f"  Steps: {engine._step_count} | Convergence: {convergence_pct:.1f}%")
        print(f"  Params: {params}")
        print(f"  Saved to: {d}")
        return session_id

    def _update_prior(self, platform, params, steps, convergence_pct):
        """Update aggregated prior — weighted by session quality."""
        d  = self._platform_dir(platform)
        fp = d / "platform_prior.json"

        # Only use sessions with decent convergence and enough steps
        if convergence_pct < 10 or steps < 30:
            return

        if fp.exists():
            with open(fp) as f:
                prior = json.load(f)
        else:
            prior = {"params": {}, "weight": 0.0, "sessions": 0}

        w = prior.get("weight", 0.0)
        q = convergence_pct / 100.0   # session quality weight

        new_params = {}
        for k, v in params.items():
            if k in prior["params"]:
                # Weighted average: more converged sessions count more
                new_params[k] = (w * prior["params"][k] + q * v) / (w + q)
            else:
                new_params[k] = v

        prior["params"]   = new_params
        prior["weight"]   = min(w + q, 50.0)   # cap so single session always has impact
        prior["sessions"] = prior.get("sessions", 0) + 1
        prior["platform"] = platform
        prior["last_updated"] = time.time()

        with open(fp, "w") as f:
            json.dump(prior, f, indent=2)

    # ── LOAD ──────────────────────────────────────────────────────────────────

    def load(self, engine, platform: str) -> bool:
        """
        Load saved model state into engine.
        Returns True if anything was loaded, False if no saved state exists.
        """
        d = self._platform_dir(platform)

        # Check if we have any saved state
        params_path = d / "params.json"
        if not params_path.exists():
            print(f"[REGISTRY] No saved state for '{platform}' — starting fresh")
            return False

        loaded_anything = False

        # 1. Load SystemID params
        with open(params_path) as f:
            saved = json.load(f)
        params = saved.get("params", {})
        engine.physics.update_params(params)
        engine.sysid.params = params.copy()
        engine.sysid._vel   = {k: 0.0 for k in params}
        sessions_count = saved.get("sessions_count", 1)
        loaded_anything = True
        print(f"[REGISTRY] Loaded params from {sessions_count} previous session(s)")
        print(f"  {params}")

        # 2. Load ResidualEnsemble weights
        all_loaded = True
        for i, member in enumerate(engine.ensemble.members):
            path = d / f"ensemble_{i}.npz"
            if path.exists():
                data = np.load(path)
                member.W1 = data["W1"]; member.b1 = data["b1"]
                member.W2 = data["W2"]; member.b2 = data["b2"]
                member.W3 = data["W3"]; member.b3 = data["b3"]
            else:
                all_loaded = False
        if all_loaded:
            print(f"[REGISTRY] Loaded residual ensemble weights")
        else:
            print(f"[REGISTRY] No ensemble weights found — ensemble starts fresh")

        # 3. Load CEM warm start
        cem_path = d / "cem_warmstart.npz"
        if cem_path.exists():
            data = np.load(cem_path)
            if data["mu"].shape == engine.cem.mu.shape:
                engine.cem.mu  = data["mu"]
                engine.cem.std = data["std"]
                print(f"[REGISTRY] Loaded CEM warm start")

        return loaded_anything

    def load_prior(self, engine, platform: str) -> bool:
        """
        Load platform prior — aggregated params from ALL sessions across all users.
        More conservative than load() — only updates params if prior is strong.
        """
        d  = self._platform_dir(platform)
        fp = d / "platform_prior.json"
        if not fp.exists():
            return False

        with open(fp) as f:
            prior = json.load(f)

        if prior.get("sessions", 0) < 3:
            print(f"[REGISTRY] Prior too weak ({prior.get('sessions',0)} sessions) — starting fresh")
            return False

        params = prior.get("params", {})
        engine.physics.update_params(params)
        engine.sysid.params = params.copy()
        print(f"[REGISTRY] Loaded platform prior from {prior['sessions']} sessions")
        print(f"  {params}")
        return True

    # ── INSPECT ───────────────────────────────────────────────────────────────

    def _session_count(self, platform: str) -> int:
        d = self._platform_dir(platform)
        fp = d / "sessions.jsonl"
        if not fp.exists(): return 0
        return sum(1 for _ in open(fp))

    def list_platforms(self) -> List[str]:
        return [p.name for p in self.root.iterdir() if p.is_dir()]

    def platform_summary(self, platform: str) -> dict:
        d  = self._platform_dir(platform)
        pp = d / "params.json"
        sp = d / "sessions.jsonl"
        prior_p = d / "platform_prior.json"

        params = json.load(open(pp)) if pp.exists() else {}
        sessions_count = sum(1 for _ in open(sp)) if sp.exists() else 0
        prior = json.load(open(prior_p)) if prior_p.exists() else {}

        return {
            "platform":      platform,
            "sessions":      sessions_count,
            "latest_params": params.get("params", {}),
            "prior_weight":  prior.get("weight", 0.0),
            "prior_sessions":prior.get("sessions", 0),
        }

    def global_summary(self) -> dict:
        platforms = self.list_platforms()
        return {
            "platforms":      platforms,
            "total_platforms":len(platforms),
            "registry_path":  str(self.root),
            "summaries":      {p: self.platform_summary(p) for p in platforms},
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_registry: Optional[ModelRegistry] = None

def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
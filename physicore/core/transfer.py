"""
physicore/core/transfer.py — Cross-robot transfer learning

PlatformPrior  — Bayesian prior over physics parameters, built from past sessions.
TransferEngine — Finds similar sessions and warm-starts new robots from prior knowledge.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── PlatformPrior ─────────────────────────────────────────────────────────────

class PlatformPrior:
    """
    Incremental Bayesian prior over physics parameters for a platform.

    Internally maintains a weighted mean (mu) and variance (var) per parameter
    using an online algorithm. Weight is proportional to:
        w = convergence_pct / 100  *  log10(max(n_steps, 10))

    This gives stronger sessions (more steps, better convergence) more influence.
    """

    def __init__(self, platform: str):
        self.platform  = platform
        self._mu:  Dict[str, float] = {}
        self._var: Dict[str, float] = {}
        self._total_weight: float   = 0.0
        self._n_sessions:   int     = 0

    def update(
        self,
        session_params:   Dict[str, float],
        n_steps:          int,
        convergence_pct:  float,
    ) -> None:
        """Update prior with one session's result."""
        if convergence_pct < 5.0 or n_steps < 10:
            return

        w = (convergence_pct / 100.0) * math.log10(max(n_steps, 10))

        for k, v in session_params.items():
            if k not in self._mu:
                self._mu[k]  = v
                self._var[k] = 1.0
                continue

            old_mu  = self._mu[k]
            old_var = self._var[k]
            W       = self._total_weight

            # Welford-style weighted update
            new_W       = W + w
            new_mu      = (W * old_mu + w * v) / new_W
            # Biased weighted variance
            new_var     = (W * (old_var + (old_mu - new_mu) ** 2)
                           + w * (v - new_mu) ** 2) / new_W
            self._mu[k]  = new_mu
            self._var[k] = max(new_var, 1e-8)

        self._total_weight += w
        self._n_sessions   += 1

    def map_estimate(self) -> Dict[str, float]:
        """Maximum a posteriori parameter estimate (= weighted mean)."""
        return dict(self._mu)

    def sample(self) -> Dict[str, float]:
        """Sample a parameter set from the prior (normal around MAP)."""
        result = {}
        for k in self._mu:
            std        = math.sqrt(self._var.get(k, 1.0))
            result[k]  = float(np.random.normal(self._mu[k], std))
        return result

    def uncertainty(self) -> Dict[str, float]:
        """Per-parameter standard deviation."""
        return {k: math.sqrt(max(v, 0.0)) for k, v in self._var.items()}

    @property
    def n_sessions(self) -> int:
        return self._n_sessions

    @property
    def total_weight(self) -> float:
        return self._total_weight

    def serialize(self) -> dict:
        return {
            "platform":      self.platform,
            "mu":            self._mu,
            "var":           self._var,
            "total_weight":  self._total_weight,
            "n_sessions":    self._n_sessions,
            "last_updated":  time.time(),
        }

    def deserialize(self, data: dict) -> "PlatformPrior":
        self.platform       = data.get("platform", self.platform)
        self._mu            = data.get("mu",  {})
        self._var           = data.get("var", {})
        self._total_weight  = data.get("total_weight", 0.0)
        self._n_sessions    = data.get("n_sessions",   0)
        return self

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.serialize(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, platform: str, path: Path) -> "PlatformPrior":
        prior = cls(platform)
        if Path(path).exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            prior.deserialize(data)
        return prior


# ── TransferEngine ────────────────────────────────────────────────────────────

class TransferEngine:
    """
    Finds similar past sessions and warm-starts new robots from prior knowledge.
    Integrates with ModelRegistry for session data.
    """

    def __init__(self, registry_root: Optional[Path] = None):
        from physicore.core.registry import _REGISTRY_ROOT
        self._root = Path(registry_root) if registry_root else _REGISTRY_ROOT
        self._priors: Dict[str, PlatformPrior] = {}

    def _load_prior(self, platform: str) -> PlatformPrior:
        if platform not in self._priors:
            prior_file = self._root / platform / "transfer_prior.json"
            self._priors[platform] = PlatformPrior.load(platform, prior_file)
        return self._priors[platform]

    def _save_prior(self, platform: str) -> None:
        prior      = self._priors.get(platform)
        if prior:
            prior_file = self._root / platform / "transfer_prior.json"
            prior.save(prior_file)

    def _load_sessions(self, platform: str) -> List[dict]:
        """Load all session records for a platform from the registry."""
        sessions_file = self._root / platform / "sessions.jsonl"
        if not sessions_file.exists():
            return []
        sessions = []
        for line in sessions_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    sessions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return sessions

    def build_prior_from_registry(self, platform: str) -> PlatformPrior:
        """(Re-)build a PlatformPrior from all registry sessions for a platform."""
        prior    = PlatformPrior(platform)
        sessions = self._load_sessions(platform)
        for s in sessions:
            params  = s.get("final_params", {})
            steps   = s.get("steps", 0)
            conv    = s.get("convergence_pct", 0.0)
            prior.update(params, steps, conv)
        self._priors[platform] = prior
        self._save_prior(platform)
        return prior

    def find_similar_sessions(
        self,
        platform:       str,
        current_params: Dict[str, float],
        n:              int = 5,
    ) -> List[dict]:
        """
        Find the n sessions whose final_params are most similar to current_params.
        Similarity = inverse L2 distance in normalised param space.
        """
        sessions = self._load_sessions(platform)
        if not sessions:
            return []

        keys     = list(current_params.keys())
        cur_vec  = np.array([current_params.get(k, 0.0) for k in keys])

        def _dist(s: dict) -> float:
            fp  = s.get("final_params", {})
            vec = np.array([fp.get(k, 0.0) for k in keys])
            return float(np.linalg.norm(cur_vec - vec))

        ranked = sorted(sessions, key=_dist)
        return ranked[:n]

    def warm_start(self, engine, platform: str) -> Dict[str, float]:
        """
        Load the MAP estimate from the platform prior and apply it to the engine.
        Returns the suggested initial params dict.

        If the prior has fewer than 2 sessions, falls back to registry load_prior().
        """
        prior = self._load_prior(platform)

        if prior.n_sessions >= 2:
            params = prior.map_estimate()
        else:
            # Rebuild from registry sessions
            prior  = self.build_prior_from_registry(platform)
            params = prior.map_estimate() if prior.n_sessions >= 1 else {}

        if params:
            engine.physics.update_params(params)
            engine.sysid.params = params.copy()
            print(
                f"[TransferEngine] Warm-started '{platform}' from prior "
                f"({prior.n_sessions} sessions): {params}"
            )
        return params

    def cross_platform_transfer(
        self,
        source_platform: str,
        target_platform: str,
        param_map:       Dict[str, str],
    ) -> Dict[str, float]:
        """
        Transfer params from source to target using an explicit key mapping.

        param_map example: {"mass": "link_mass", "friction": "joint_friction"}
        Keys not in param_map are not transferred.
        """
        source_prior = self._load_prior(source_platform)
        source_map   = source_prior.map_estimate()

        transferred: Dict[str, float] = {}
        for src_key, tgt_key in param_map.items():
            if src_key in source_map:
                transferred[tgt_key] = source_map[src_key]

        return transferred

    def update_prior(
        self,
        platform:       str,
        session_params: Dict[str, float],
        n_steps:        int,
        convergence_pct: float,
    ) -> None:
        """Update the prior after a new session completes."""
        prior = self._load_prior(platform)
        prior.update(session_params, n_steps, convergence_pct)
        self._save_prior(platform)

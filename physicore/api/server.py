"""
PhysiCore API Server
====================
FastAPI server that connects the PhysiCore engine to the dashboard
and exposes a REST API for external integration.

Run:
  uvicorn physicore.api.server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET  /                        — health check
  GET  /api/status              — engine status and diagnostics
  POST /api/engine/step         — send state, get optimal action
  POST /api/engine/observe      — feed real transition back
  GET  /api/engine/params       — current estimated parameters
  GET  /api/engine/residual     — current sim-to-real residual
  GET  /api/engine/uncertainty  — current epistemic uncertainty
  POST /api/engine/reset        — reset engine state
  POST /api/engine/configure    — configure engine for a platform
  WS   /ws/telemetry            — WebSocket stream of live engine output
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from physicore.api.auth import require_api_key
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from physicore import __version__
from physicore import PhysiCore, PhysiCoreConfig, PLATFORM_DYNAMICS

# Registry + Sentinel
try:
    from physicore.core.registry import get_registry, ModelRegistry
    from physicore.sentinel.core import SentinelOS, get_sentinel_config, SentinelMode
    HAS_REGISTRY = True
    HAS_SENTINEL = True
except ImportError:
    HAS_REGISTRY = False
    HAS_SENTINEL = False


# ── Global engine state ────────────────────────────────────────────────────────

class EngineState:
    def __init__(self):
        self.engine: Optional[PhysiCore] = None
        self.sentinel: Optional[object] = None    # SentinelOS instance
        self.platform: str = "none"
        self.running: bool = False
        self.last_step_time: float = 0.0
        self.telemetry_clients: List[WebSocket] = []
        self.last_diagnostics: dict = {}
        self.session_id: Optional[str] = None
        self.lock = threading.Lock()

engine_state = EngineState()


# ── Startup / shutdown ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[PHYSICORE API] Server starting...")
    asyncio.create_task(broadcast_diagnostics())
    yield
    print("[PHYSICORE API] Server stopping...")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PhysiCore API",
    description="Hybrid Uncertainty-Aware Sim-to-Real Engine — REST API",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS — allow physicore.ai domains + localhost for dev
_ALLOWED_ORIGINS = [
    "https://physicore.ai",
    "https://www.physicore.ai",
    "https://physicore-hybrid-mpc.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class ConfigureRequest(BaseModel):
    platform: str
    initial_params: Dict[str, float]
    control_hz: float = 60.0
    q_scale: float = 10.0
    r_scale: float = 0.1

class StepRequest(BaseModel):
    state: List[float]
    x_ref: List[float]

class ObserveRequest(BaseModel):
    state: List[float]
    action: List[float]
    next_state: List[float]

class StepResponse(BaseModel):
    action: List[float]
    state_predicted: List[float]
    residual: List[float]
    uncertainty: float
    params: Dict[str, float]
    loop_time_ms: float
    step_count: int

class DiagnosticsResponse(BaseModel):
    platform: str
    running: bool
    step_count: int
    params: Dict[str, float]
    residual_norm: float
    residual_axis: List[float]      # NEW: per-axis residual breakdown
    uncertainty: float
    target_hz: float
    state_dim: int
    action_dim: int
    sysid_loss_hist: List[float]
    innovation_ema: float           # NEW: adaptive LR signal
    failure_summary: Dict           # NEW: structured failure log
    hash_chain_head: str            # NEW: latest forensic certificate
    timestamp: float


# ── Helpers ────────────────────────────────────────────────────────────────────

def require_engine():
    if engine_state.engine is None:
        raise HTTPException(
            status_code=400,
            detail="Engine not configured. POST /api/engine/configure first."
        )
    return engine_state.engine


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "physicore",
        "version": "1.1.0",
        "status": "ok",
        "engine_ready": engine_state.engine is not None,
        "platform": engine_state.platform,
    }


@app.get("/api/status", response_model=DiagnosticsResponse)
async def get_status():
    if engine_state.engine is None:
        return DiagnosticsResponse(
            platform="none", running=False, step_count=0,
            params={}, residual_norm=0.0, residual_axis=[],
            uncertainty=0.0, target_hz=0.0, state_dim=0, action_dim=0,
            sysid_loss_hist=[], innovation_ema=0.0,
            failure_summary={}, hash_chain_head="", timestamp=time.time()
        )
    d = engine_state.engine.diagnostics_full
    return DiagnosticsResponse(
        platform=engine_state.platform,
        running=engine_state.running,
        step_count=d["step_count"],
        params=d["params"],
        residual_norm=d["residual_norm"],
        residual_axis=d.get("residual_axis", []),
        uncertainty=d["uncertainty"],
        target_hz=d["target_hz"],
        state_dim=d["state_dim"],
        action_dim=d["action_dim"],
        sysid_loss_hist=d["sysid_loss_hist"],
        innovation_ema=d.get("innovation_ema", 0.0),
        failure_summary=d.get("failure_summary", {}),
        hash_chain_head=d.get("hash_chain_head", ""),
        timestamp=time.time(),
    )


@app.post("/api/engine/configure")
async def configure_engine(req: ConfigureRequest):
    if req.platform not in PLATFORM_DYNAMICS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown platform '{req.platform}'. Available: {list(PLATFORM_DYNAMICS.keys())}"
        )

    _, state_dim, action_dim = PLATFORM_DYNAMICS[req.platform]
    Q = np.eye(state_dim)  * req.q_scale
    R = np.eye(action_dim) * req.r_scale

    with engine_state.lock:
        engine_state.engine = PhysiCore.for_platform(
            platform=req.platform,
            initial_params=req.initial_params,
            Q=Q,
            R=R,
            control_hz=req.control_hz,
        )
        engine_state.platform = req.platform
        engine_state.running  = True

    return {
        "status": "configured",
        "platform": req.platform,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "control_hz": req.control_hz,
    }


@app.post("/api/engine/step", response_model=StepResponse)
async def engine_step(req: StepRequest):
    engine = require_engine()
    state  = np.array(req.state,  dtype=float)
    x_ref  = np.array(req.x_ref, dtype=float)

    expected = engine.cfg.state_dim
    if len(state) != expected:
        raise HTTPException(400, f"State dim mismatch: expected {expected}, got {len(state)}")
    if len(x_ref) != expected:
        raise HTTPException(400, f"x_ref dim mismatch: expected {expected}, got {len(x_ref)}")

    with engine_state.lock:
        step = engine.step(state, x_ref)
    engine_state.last_step_time = time.time()

    return StepResponse(
        action=step.action.tolist(),
        state_predicted=step.state_predicted.tolist(),
        residual=step.residual.tolist(),
        uncertainty=step.uncertainty,
        params=step.params,
        loop_time_ms=step.loop_time_ms,
        step_count=step.step_count,
    )


@app.post("/api/engine/observe")
async def engine_observe(req: ObserveRequest):
    engine = require_engine()
    with engine_state.lock:
        engine.observe(
            np.array(req.state),
            np.array(req.action),
            np.array(req.next_state),
        )
    return {"status": "observed"}


@app.get("/api/engine/params")
async def get_params():
    engine = require_engine()
    return {"params": engine.physics.params, "timestamp": time.time()}


@app.get("/api/engine/residual")
async def get_residual():
    engine = require_engine()
    d = engine.diagnostics_full
    return {"residual_norm": d["residual_norm"], "timestamp": time.time()}


@app.get("/api/engine/uncertainty")
async def get_uncertainty():
    engine = require_engine()
    d = engine.diagnostics_full
    return {"uncertainty": d["uncertainty"], "timestamp": time.time()}


@app.get("/api/engine/failures")
async def get_failures():
    """Real-time failure log — structured events with severity, type, and params snapshot."""
    engine = require_engine()
    return {
        "failure_summary": engine.failure_log.summary(),
        "innovation_ema":  engine.sysid.innovation_ema,
        "hash_chain_head": engine.hash_chain._prev,
        "timestamp":       time.time(),
    }


@app.post("/api/engine/reset")
async def reset_engine():
    with engine_state.lock:
        engine_state.engine   = None
        engine_state.platform = "none"
        engine_state.running  = False
    return {"status": "reset"}


@app.get("/api/platforms")
async def list_platforms():
    return {
        name: {
            "state_dim":  sd,
            "action_dim": ad,
        }
        for name, (fn, sd, ad) in PLATFORM_DYNAMICS.items()
    }


# ── WebSocket telemetry stream ─────────────────────────────────────────────────

@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    engine_state.telemetry_clients.append(websocket)
    try:
        while True:
            await asyncio.sleep(1.0 / 20)  # 20 Hz stream to dashboard
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in engine_state.telemetry_clients:
            engine_state.telemetry_clients.remove(websocket)


async def broadcast_diagnostics():
    """Streams live engine diagnostics to all connected dashboard clients at 20Hz."""
    while True:
        await asyncio.sleep(1.0 / 20)
        if not engine_state.telemetry_clients or engine_state.engine is None:
            continue
        try:
            d = engine_state.engine.diagnostics_full
            payload = json.dumps({
                "type":         "diagnostics",
                "platform":     engine_state.platform,
                "step_count":   d["step_count"],
                "params":       d["params"],
                "residual_norm": d["residual_norm"],
                "uncertainty":  d["uncertainty"],
                "sysid_loss":   d["sysid_loss_hist"][-1] if d["sysid_loss_hist"] else 0.0,
                "timestamp":    time.time(),
            })
            dead = []
            for ws in engine_state.telemetry_clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                engine_state.telemetry_clients.remove(ws)
        except Exception:
            pass


# ── Registry endpoints (data flywheel) ────────────────────────────────────────

@app.get("/api/registry/summary")
async def registry_summary():
    """Global registry summary — all platforms, session counts, prior strength."""
    if not HAS_REGISTRY:
        raise HTTPException(503, "Registry not available")
    reg = get_registry()
    return reg.global_summary()


@app.get("/api/registry/{platform}")
async def registry_platform(platform: str):
    """Per-platform registry summary — sessions, latest params, prior weight."""
    if not HAS_REGISTRY:
        raise HTTPException(503, "Registry not available")
    reg = get_registry()
    return reg.platform_summary(platform)


@app.post("/api/registry/{platform}/save")
async def registry_save(platform: str):
    """Save current engine state to registry. Call at end of hardware session."""
    if not HAS_REGISTRY:
        raise HTTPException(503, "Registry not available")
    engine = require_engine()
    reg    = get_registry()
    session_id = reg.save(engine, platform=platform,
                          session_meta={"saved_via": "api", "ts": time.time()})
    engine_state.session_id = session_id
    return {"status": "saved", "session_id": session_id, "platform": platform}


@app.post("/api/registry/{platform}/load")
async def registry_load(platform: str):
    """Load saved model state from registry into current engine."""
    if not HAS_REGISTRY:
        raise HTTPException(503, "Registry not available")
    engine = require_engine()
    reg    = get_registry()
    loaded = reg.load(engine, platform)
    return {"status": "loaded" if loaded else "no_prior",
            "platform": platform, "loaded": loaded}


@app.get("/api/registry/{platform}/sessions")
async def registry_sessions(platform: str, limit: int = 20):
    """List recent sessions for a platform from the registry log."""
    if not HAS_REGISTRY:
        raise HTTPException(503, "Registry not available")
    from pathlib import Path
    import json as _json
    reg   = get_registry()
    d     = reg._platform_dir(platform)
    fp    = d / "sessions.jsonl"
    if not fp.exists():
        return {"sessions": [], "count": 0}
    sessions = []
    for line in open(fp):
        try:
            sessions.append(_json.loads(line.strip()))
        except Exception:
            pass
    sessions = sessions[-limit:]
    return {"sessions": sessions, "count": len(sessions), "platform": platform}


@app.get("/api/registry/{platform}/convergence")
async def registry_convergence(platform: str):
    """
    Convergence proof: compares session 1 vs latest session.
    Shows that the registry actually makes PhysiCore better over time.
    """
    if not HAS_REGISTRY:
        raise HTTPException(503, "Registry not available")
    from pathlib import Path
    import json as _json
    reg  = get_registry()
    d    = reg._platform_dir(platform)
    fp   = d / "sessions.jsonl"
    if not fp.exists():
        return {"message": "No sessions yet", "sessions_count": 0}
    sessions = [_json.loads(l.strip()) for l in open(fp) if l.strip()]
    if len(sessions) < 2:
        return {"message": f"Need >=2 sessions, have {len(sessions)}",
                "sessions_count": len(sessions)}
    first  = sessions[0]
    latest = sessions[-1]
    return {
        "sessions_count":    len(sessions),
        "platform":          platform,
        "session_1": {
            "id":              first["session_id"],
            "convergence_pct": first["convergence_pct"],
            "innovation_ema":  first["innovation_ema"],
            "steps":           first["steps"],
        },
        "session_latest": {
            "id":              latest["session_id"],
            "convergence_pct": latest["convergence_pct"],
            "innovation_ema":  latest["innovation_ema"],
            "steps":           latest["steps"],
        },
        "improvement": {
            "convergence_delta": round(
                latest["convergence_pct"] - first["convergence_pct"], 2),
            "innovation_delta":  round(
                first["innovation_ema"] - latest["innovation_ema"], 6),
            "interpretation":    (
                "Registry flywheel working — convergence improved across sessions"
                if latest["convergence_pct"] > first["convergence_pct"]
                else "Convergence stable — more sessions needed for improvement"
            ),
        },
    }



# ── Intelligence Layer ─────────────────────────────────────────────────────────
# All AI calls go through here. One key (yours), no user rate limits.
# Frontend calls these endpoints — never calls Gemini/Anthropic directly.

import os
import re as _re

_SESSION_BUFFER: list = []       # rolling 20-snapshot session context
_SESSION_BUFFER_MAX = 20

def _update_session_buffer(engine) -> None:
    """Call every ~100 steps from the control loop."""
    try:
        d   = engine.diagnostics_full
        nar = engine.narrate()
        snap = {
            "ts":          time.time(),
            "step":        d["step_count"],
            "residual":    round(d["residual_norm"], 4),
            "uncertainty": round(d["uncertainty"], 4),
            "params":      {k: round(v, 4) for k, v in d["params"].items()},
            "innovation":  round(d["innovation_ema"], 4),
            "status":      nar["status"],
            "headline":    nar["headline"],
            "failures":    d["failure_summary"].get("total_events", 0),
            "hash":        d["hash_chain_head"][:8],
        }
        _SESSION_BUFFER.append(snap)
        if len(_SESSION_BUFFER) > _SESSION_BUFFER_MAX:
            _SESSION_BUFFER.pop(0)
    except Exception:
        pass


def _call_gemini(system: str, user: str) -> str:
    """Call Gemini Flash. Falls back to deterministic narration if key missing."""
    try:
        import google.generativeai as genai
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("no_key")
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system,
        )
        resp = model.generate_content(user)
        return resp.text.strip()
    except Exception as e:
        # Graceful fallback — never crash
        if engine_state.engine:
            nar = engine_state.engine.narrate()
            return f"{nar['headline']}\n\n{nar['detail']}\n\nRecommendation: {nar['action']}"
        return "Engine not running — start the bridge first."


class IntelligenceAnalyzeRequest(BaseModel):
    context: Optional[str] = None   # extra context from frontend

class IntelligenceTroubleshootRequest(BaseModel):
    problem: str
    hw_type: str = ""
    hw_answers: dict = {}
    session_snapshot: Optional[dict] = None


@app.get("/api/intelligence/narrate")
async def get_narration():
    """
    Deterministic plain-English engine narration.
    No API key required. Always works.
    """
    engine = require_engine()
    return engine.narrate()


@app.get("/api/intelligence/session")
async def get_session_context():
    """Rolling 20-snapshot session buffer for IE live context."""
    return {
        "snapshots":  _SESSION_BUFFER,
        "count":      len(_SESSION_BUFFER),
        "has_data":   len(_SESSION_BUFFER) > 0,
    }


@app.post("/api/intelligence/analyze")
async def intelligence_analyze(req: IntelligenceAnalyzeRequest):
    """
    Meta-analyst: interprets live session data.
    Routes through backend — one API key, no user-facing rate limits.
    Falls back to narrate() if key missing.
    """
    engine = require_engine()
    d   = engine.diagnostics_full
    nar = engine.narrate()

    # Build rich context from real engine state
    buf_summary = ""
    if _SESSION_BUFFER:
        res_trend = [s["residual"] for s in _SESSION_BUFFER[-10:]]
        mass_vals = [s["params"].get("mass", 0) for s in _SESSION_BUFFER[-10:]]
        buf_summary = (
            f"Last {len(_SESSION_BUFFER)} snapshots: "
            f"residual {res_trend[0]:.3f}→{res_trend[-1]:.3f} "
            f"({'RISING' if res_trend[-1] > res_trend[0] else 'FALLING'}), "
            f"mass {mass_vals[0]:.3f}→{mass_vals[-1]:.3f}"
        )

    system_prompt = """You are the PhysiCore Meta-Analyst — an expert in real-time physics adaptation and sim-to-real robotics.
Interpret the telemetry data precisely. Use technical robotics language.
Return a JSON object with exactly these fields:
  insight: string (1-2 sentences, what is happening physically)
  diagnostics: array of strings (2-4 specific observations about the numbers)
  recommendation: string (one concrete action the operator should take now)
  q_weight: number (suggested MPC Q weight, 0.1-20.0)
  r_weight: number (suggested MPC R weight, 0.01-5.0)
No prose outside the JSON. No markdown. Just the JSON object."""

    user_prompt = f"""Current PhysiCore session:

NARRATION: {nar['status']} — {nar['headline']}
DETAIL: {nar['detail']}

LIVE METRICS:
  residual_norm:  {d['residual_norm']:.4f}
  uncertainty:    {d['uncertainty']:.4f}
  mass:           {d['params'].get('mass',0):.3f} kg
  friction:       {d['params'].get('friction',0):.4f}
  innovation_ema: {d['innovation_ema']:.4f}
  step_count:     {d['step_count']}
  sysid_trend:    {d['sysid_loss_hist'][-5:] if d['sysid_loss_hist'] else 'no data'}
  faults:         {d['failure_summary'].get('total_events',0)} total, recent: {d['failure_summary'].get('recent_10',[])}

SESSION TREND: {buf_summary if buf_summary else 'insufficient data (<100 steps)'}

{f"ADDITIONAL CONTEXT: {req.context}" if req.context else ""}

Analyze this and return JSON."""

    raw = _call_gemini(system_prompt, user_prompt)

    # Try to parse JSON — fallback to structured response
    try:
        # strip markdown fences if present
        clean = _re.sub(r'```[a-z]*\n?|```', '', raw).strip()
        parsed = json.loads(clean)
        return {
            "insight":        parsed.get("insight", nar["headline"]),
            "diagnostics":    parsed.get("diagnostics", [nar["detail"]]),
            "recommendation": parsed.get("recommendation", nar["action"]),
            "q_weight":       float(parsed.get("q_weight", 1.0)),
            "r_weight":       float(parsed.get("r_weight", 0.1)),
            "status":         nar["status"],
            "metrics":        nar["metrics"],
            "source":         "gemini",
        }
    except Exception:
        return {
            "insight":        nar["headline"],
            "diagnostics":    [nar["detail"]],
            "recommendation": nar["action"],
            "q_weight":       1.0,
            "r_weight":       0.1,
            "status":         nar["status"],
            "metrics":        nar["metrics"],
            "source":         "narrate_fallback",
        }


@app.post("/api/intelligence/troubleshoot")
async def intelligence_troubleshoot(req: IntelligenceTroubleshootRequest):
    """
    Troubleshooter: answers hardware questions with full session context.
    Routes through backend. Falls back to local logic if no key.
    """
    # Build session context if available
    session_ctx = ""
    if req.session_snapshot:
        snap = req.session_snapshot
        session_ctx = f"""
LIVE SESSION (step {snap.get('steps', 0)}):
  Sentinel: {snap.get('sentinelMode','UNKNOWN')}
  Mass: {snap.get('mass',{}).get('current','?')}kg (declared {snap.get('mass',{}).get('declared','?')}kg, {snap.get('mass',{}).get('driftPct','?')}% drift)
  Residual: {snap.get('residual',{}).get('current','?')} trend={snap.get('residual',{}).get('trend','?')}
  Residual history: {snap.get('residual',{}).get('history',[])}
  Stable: {snap.get('isStable','?')}, Faulted: {snap.get('isFaulted','?')}
  Pitch: {snap.get('pitch','?')}°, MotorL: {snap.get('motorL','?')}, MotorR: {snap.get('motorR','?')}"""

    engine_ctx = ""
    if engine_state.engine and _SESSION_BUFFER:
        last = _SESSION_BUFFER[-1]
        engine_ctx = f"""
ENGINE STATE (step {last['step']}):
  Status: {last['status']} — {last['headline']}
  Residual: {last['residual']}, Uncertainty: {last['uncertainty']}
  Params: {last['params']}
  Recent faults: {last['failures']} total"""

    system_prompt = """You are the PhysiCore Integration Engineer — expert troubleshooter.
You have real hardware session data. Give precise, numbered steps.
Each step must have a concrete command or action — no vague advice.
Maximum 6 steps. Be direct. The user is watching a live system."""

    user_prompt = f"""Hardware: {req.hw_type}
Setup answers: {json.dumps(req.hw_answers)}
{session_ctx}
{engine_ctx}

Problem: {req.problem}

Give exact numbered steps to fix this."""

    raw = _call_gemini(system_prompt, user_prompt)

    # Parse into steps
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    steps = []
    for line in lines[:10]:
        # Extract code/commands from backticks
        cmd_match = _re.search(r'`([^`]+)`', line)
        cmd = cmd_match.group(1) if cmd_match else ""
        label = _re.sub(r'^\d+\.\s*', '', line)[:120]
        if label:
            steps.append({"label": label, "cmd": cmd})

    if not steps:
        steps = [{"label": raw[:200], "cmd": ""}]

    return {
        "title": f"Diagnosis — {'live session data used' if req.session_snapshot else 'static context'}",
        "steps": steps,
        "raw":   raw,
        "source": "gemini" if os.environ.get("GEMINI_API_KEY") else "local",
    }




# ── Shared Registry endpoints (fleet learning / cross-team priors) ────────────
# These proxy to the hosted shared registry at api.physicore.ai
# Local registry is always the source of truth; shared is additive.

SHARED_REGISTRY_URL = os.environ.get(
    "PHYSICORE_SHARED_REGISTRY_URL",
    "https://api.physicore.ai/registry"
)

@app.get("/api/registry/shared/{platform}")
async def get_shared_prior(platform: str):
    """
    Fetch the globally aggregated platform prior from the hosted shared registry.
    Returns the best starting params learned from all opted-in teams worldwide.
    Falls back gracefully if shared registry is unreachable.
    """
    import urllib.request, urllib.error
    try:
        url = f"{SHARED_REGISTRY_URL}/prior/{platform}"
        req = urllib.request.Request(url, headers={"User-Agent": f"physicore/{__version__}"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {
            "platform": platform,
            "params": {},
            "sessions": 0,
            "source": "shared_registry_unavailable",
            "error": str(e),
        }


@app.post("/api/registry/{platform}/snapshot")
async def create_snapshot(platform: str):
    """
    Create a versioned snapshot of the current model state.
    Snapshots are immutable — you can always roll back to a known-good session.
    Stored in ~/.physicore/registry/{platform}/snapshots/{timestamp}/
    """
    if not HAS_REGISTRY:
        raise HTTPException(503, "Registry not available")
    engine = require_engine()
    reg = get_registry()

    import time, shutil
    snapshot_id = f"snap_{int(time.time())}"
    platform_dir = reg._platform_dir(platform)
    snap_dir = platform_dir / "snapshots" / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Copy current state into snapshot
    for fname in ["params.json", "ensemble_0.npz", "ensemble_1.npz",
                  "ensemble_2.npz", "cem_warmstart.npz"]:
        src = platform_dir / fname
        if src.exists():
            shutil.copy2(src, snap_dir / fname)

    # Save snapshot metadata
    meta = {
        "snapshot_id": snapshot_id,
        "platform": platform,
        "timestamp": time.time(),
        "step_count": engine._step_count,
        "params": engine.physics.params.copy(),
        "residual": engine.diagnostics_full.get("residual_norm", 0),
    }
    with open(snap_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"snapshot_id": snapshot_id, "path": str(snap_dir), "meta": meta}


@app.get("/api/registry/{platform}/snapshots")
async def list_snapshots(platform: str):
    """List all versioned snapshots for a platform."""
    if not HAS_REGISTRY:
        raise HTTPException(503, "Registry not available")
    reg = get_registry()
    snap_dir = reg._platform_dir(platform) / "snapshots"
    if not snap_dir.exists():
        return {"snapshots": []}

    snaps = []
    for d in sorted(snap_dir.iterdir()):
        meta_path = d / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                snaps.append(json.load(f))
    return {"snapshots": sorted(snaps, key=lambda x: x["timestamp"], reverse=True)}


@app.post("/api/registry/{platform}/rollback/{snapshot_id}")
async def rollback_snapshot(platform: str, snapshot_id: str):
    """
    Roll back engine to a previously saved snapshot.
    This is the safety net — if a session corrupted the model, roll back to last good.
    """
    if not HAS_REGISTRY:
        raise HTTPException(503, "Registry not available")
    engine = require_engine()
    reg = get_registry()

    snap_dir = reg._platform_dir(platform) / "snapshots" / snapshot_id
    if not snap_dir.exists():
        raise HTTPException(404, f"Snapshot {snapshot_id} not found")

    import shutil
    platform_dir = reg._platform_dir(platform)

    # Restore snapshot files
    restored = []
    for fname in ["params.json", "ensemble_0.npz", "ensemble_1.npz",
                  "ensemble_2.npz", "cem_warmstart.npz"]:
        src = snap_dir / fname
        if src.exists():
            shutil.copy2(src, platform_dir / fname)
            restored.append(fname)

    # Reload into running engine
    reg.load(engine, platform)

    with open(snap_dir / "meta.json") as f:
        meta = json.load(f)

    return {
        "rolled_back_to": snapshot_id,
        "restored_files": restored,
        "params": engine.physics.params,
        "original_step": meta.get("step_count"),
    }



# ── Sentinel endpoints ─────────────────────────────────────────────────────────

class SentinelStepRequest(BaseModel):
    state:    List[float]
    x_ref:    List[float]
    altitude: float = 0.0

@app.post("/api/sentinel/configure")
async def sentinel_configure(platform: str = "balancing_bot"):
    """Attach Sentinel OS to the current engine."""
    if not HAS_SENTINEL:
        raise HTTPException(503, "Sentinel not available")
    engine = require_engine()
    with engine_state.lock:
        engine_state.sentinel = SentinelOS(engine, platform=platform, verbose=False)
    return {"status": "sentinel_attached", "platform": platform}


@app.post("/api/sentinel/step")
async def sentinel_step(req: SentinelStepRequest):
    """One Sentinel-governed control step. Returns safe action."""
    if engine_state.sentinel is None:
        raise HTTPException(400, "Sentinel not configured. POST /api/sentinel/configure first.")
    import numpy as _np
    state  = _np.array(req.state, dtype=float)
    x_ref  = _np.array(req.x_ref, dtype=float)
    with engine_state.lock:
        action = engine_state.sentinel.step(state, x_ref, altitude=req.altitude)
    s = engine_state.sentinel.status
    return {
        "action":         action.tolist(),
        "sentinel_mode":  s["mode"],
        "is_safe":        s["is_safe"],
        "lyapunov_V":     s["lyapunov"]["V"],
        "ledger_hash":    s["ledger_hash"],
        "fault":          s["fault"],
        "rls_mass":       s["rls"]["total_mass"],
    }


@app.get("/api/sentinel/status")
async def sentinel_status():
    """Full Sentinel OS status — all layers."""
    if engine_state.sentinel is None:
        return {"status": "not_configured"}
    return engine_state.sentinel.status


@app.get("/api/sentinel/ledger")
async def sentinel_ledger(limit: int = 100):
    """Last N entries from the SHA-256 forensic ledger."""
    if engine_state.sentinel is None:
        raise HTTPException(400, "Sentinel not configured")
    entries = engine_state.sentinel.ledger[-limit:]
    return {"entries": entries, "chain_hash": engine_state.sentinel.chain_hash,
            "count": len(entries)}


@app.get("/api/sentinel/faults")
async def sentinel_faults():
    """All fault events detected during this session."""
    if engine_state.sentinel is None:
        raise HTTPException(400, "Sentinel not configured")
    return {"faults": engine_state.sentinel.fault_log,
            "count": len(engine_state.sentinel.fault_log)}



# ═════════════════════════════════════════════════════════════════════════════
#  ITEM 4 — WEBHOOK SYSTEM
#  Teams register endpoints. PhysiCore calls them when things happen.
#  Events: fault_detected, convergence_complete, session_saved, sentinel_mode_change
# =============================================================================

import hashlib as _hashlib
import hmac as _hmac
import threading as _threading
from typing import Optional as _Optional

_WEBHOOKS: list = []   # List of {url, secret, events}
_WEBHOOK_LOCK = _threading.Lock()

def _webhook_sign(payload: str, secret: str) -> str:
    """HMAC-SHA256 signature so teams can verify PhysiCore sent the event."""
    return _hmac.new(secret.encode(), payload.encode(), _hashlib.sha256).hexdigest()

def _fire_webhook(event_type: str, data: dict) -> None:
    """Fire all matching webhooks in background threads. Never blocks control loop."""
    import urllib.request, urllib.error
    with _WEBHOOK_LOCK:
        targets = [w for w in _WEBHOOKS if event_type in w.get('events', [])]

    for wh in targets:
        def _send(webhook=wh):
            try:
                payload = json.dumps({
                    'event':     event_type,
                    'timestamp': time.time(),
                    'data':      data,
                })
                sig = _webhook_sign(payload, webhook.get('secret', ''))
                req = urllib.request.Request(
                    webhook['url'],
                    data=payload.encode(),
                    headers={
                        'Content-Type':           'application/json',
                        'X-PhysiCore-Event':      event_type,
                        'X-PhysiCore-Signature':  f'sha256={sig}',
                        'X-PhysiCore-Timestamp':  str(int(time.time())),
                        'User-Agent':             'PhysiCore-Webhook/2.0',
                    },
                    method='POST',
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                print(f'[WEBHOOK] Failed to deliver {event_type} to {webhook["url"]}: {e}')

        _threading.Thread(target=_send, daemon=True).start()


class WebhookRegisterRequest(BaseModel):
    url:    str
    events: list  = ['fault_detected', 'convergence_complete', 'session_saved', 'sentinel_mode_change']
    secret: str   = ''   # Optional HMAC secret for signature verification


@app.post('/api/webhooks/register')
async def register_webhook(req: WebhookRegisterRequest):
    """
    Register a webhook endpoint.
    PhysiCore will POST to this URL when the specified events occur.

    Events:
      fault_detected          — Sentinel detected a hardware fault
      convergence_complete    — SysID has converged (residual stable < 0.05)
      session_saved           — Session ended and params saved to registry
      sentinel_mode_change    — Sentinel mode changed (NOMINAL → CAUTIOUS etc)

    Each POST includes:
      X-PhysiCore-Signature: sha256=<hmac> for verification
      X-PhysiCore-Event: the event type
      Body: JSON with event, timestamp, data

    Example payload (fault_detected):
      {
        "event": "fault_detected",
        "timestamp": 1714000000.0,
        "data": {
          "fault_type": "BEARING_WEAR",
          "confidence": 0.87,
          "params": {"friction": 0.52},
          "step": 1847
        }
      }
    """
    with _WEBHOOK_LOCK:
        # Deduplicate by URL
        _WEBHOOKS[:] = [w for w in _WEBHOOKS if w['url'] != req.url]
        _WEBHOOKS.append({'url': req.url, 'events': req.events, 'secret': req.secret})

    return {
        'registered': True,
        'url':        req.url,
        'events':     req.events,
        'total_webhooks': len(_WEBHOOKS),
    }


@app.delete('/api/webhooks/{url:path}')
async def deregister_webhook(url: str):
    """Remove a previously registered webhook."""
    with _WEBHOOK_LOCK:
        before = len(_WEBHOOKS)
        _WEBHOOKS[:] = [w for w in _WEBHOOKS if w['url'] != url]
    return {'removed': before - len(_WEBHOOKS)}


@app.get('/api/webhooks')
async def list_webhooks():
    """List all registered webhooks."""
    with _WEBHOOK_LOCK:
        return {'webhooks': [{'url': w['url'], 'events': w['events']} for w in _WEBHOOKS]}


@app.post('/api/webhooks/test')
async def test_webhook(req: WebhookRegisterRequest):
    """Send a test event to a webhook URL to verify delivery."""
    _fire_webhook('test', {
        'message': 'PhysiCore webhook delivery test',
        'physicore_version': '2.1.0',
    })
    return {'sent': True, 'url': req.url}


# ── Webhook triggers (called internally after key events) ─────────────────────

def _maybe_fire_fault(step_result) -> None:
    """Call after engine.step() — fires fault_detected if new critical fault."""
    try:
        failures = step_result.failure_events if hasattr(step_result, 'failure_events') else []
        for f in failures:
            if hasattr(f, 'severity') and f.severity == 'CRITICAL':
                _fire_webhook('fault_detected', {
                    'fault_type':  f.failure_type,
                    'description': f.description,
                    'severity':    f.severity,
                    'value':       f.value,
                    'step':        f.step,
                    'params':      f.params_snapshot,
                })
    except Exception:
        pass


def _maybe_fire_convergence(engine) -> None:
    """Call periodically — fires convergence_complete when residual first stabilises."""
    try:
        d = engine.diagnostics_full
        hist = d.get('sysid_loss_hist', [])
        res  = d.get('residual_norm', 1.0)
        if len(hist) >= 10 and res < 0.05:
            last5 = hist[-5:]
            if max(last5) - min(last5) < 0.002:   # stable
                _fire_webhook('convergence_complete', {
                    'residual':   round(res, 4),
                    'params':     d.get('params', {}),
                    'step_count': d.get('step_count', 0),
                    'message':    'PhysiCore has learned your hardware. Registry saving params.',
                })
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  ITEM 2 — REGISTRY AS A SERVICE
#  Versioned snapshots, rollback, and convergence proof.
#  The flywheel: teams can't leave without losing accumulated knowledge.
# =============================================================================

@app.get('/api/registry/{platform}/convergence')
async def registry_convergence(platform: str):
    """
    Convergence proof — show session-over-session improvement.
    This is the flywheel made visible: how much better each session starts
    than the previous one.

    Returns first vs latest session residual, param stability, sessions count.
    """
    if not HAS_REGISTRY:
        raise HTTPException(503, 'Registry not available')
    reg = get_registry()
    d   = reg._platform_dir(platform)
    fp  = d / 'sessions.jsonl'
    if not fp.exists():
        return {'platform': platform, 'sessions': 0, 'message': 'No sessions yet'}

    sessions = []
    with open(fp) as f:
        for line in f:
            try:
                sessions.append(_json.loads(line.strip()))
            except Exception:
                pass

    if not sessions:
        return {'platform': platform, 'sessions': 0}

    first  = sessions[0]
    latest = sessions[-1]

    # Compute improvement
    first_res  = first.get('convergence_pct',  0)
    latest_res = latest.get('convergence_pct', 0)
    improvement = latest_res - first_res

    return {
        'platform':       platform,
        'sessions':       len(sessions),
        'first_session':  first,
        'latest_session': latest,
        'improvement_pct': round(improvement, 1),
        'message': f'{len(sessions)} sessions. Convergence improved by {improvement:.1f}% since session 1.' if len(sessions) > 1 else 'First session — run more sessions to see flywheel effect.',
        'flywheel_active': len(sessions) >= 3 and improvement > 5,
    }


@app.post('/api/registry/{platform}/snapshot')
async def create_snapshot(platform: str):
    """
    Create a named snapshot of current model state.
    Snapshots are immutable — you can always roll back to a known-good model.
    """
    if not HAS_REGISTRY:
        raise HTTPException(503, 'Registry not available')
    engine = require_engine()
    reg    = get_registry()

    import shutil as _shutil
    snapshot_id = f'snap_{int(time.time())}'
    platform_dir = reg._platform_dir(platform)
    snap_dir     = platform_dir / 'snapshots' / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    restored = []
    for fname in ['params.json', 'ensemble_0.npz', 'ensemble_1.npz',
                  'ensemble_2.npz', 'cem_warmstart.npz']:
        src = platform_dir / fname
        if src.exists():
            _shutil.copy2(src, snap_dir / fname)
            restored.append(fname)

    d = engine.diagnostics_full
    meta = {
        'snapshot_id': snapshot_id,
        'platform':    platform,
        'timestamp':   time.time(),
        'step_count':  d.get('step_count', 0),
        'residual':    round(d.get('residual_norm', 0), 4),
        'params':      d.get('params', {}),
        'files':       restored,
    }
    with open(snap_dir / 'meta.json', 'w') as f:
        _json.dump(meta, f, indent=2)

    return {'snapshot_id': snapshot_id, 'files_saved': restored, 'meta': meta}


@app.get('/api/registry/{platform}/snapshots')
async def list_snapshots(platform: str):
    """List all versioned snapshots for a platform, newest first."""
    if not HAS_REGISTRY:
        raise HTTPException(503, 'Registry not available')
    reg      = get_registry()
    snap_dir = reg._platform_dir(platform) / 'snapshots'
    if not snap_dir.exists():
        return {'snapshots': [], 'count': 0}

    snaps = []
    for d in sorted(snap_dir.iterdir(), reverse=True):
        meta_p = d / 'meta.json'
        if meta_p.exists():
            with open(meta_p) as f:
                snaps.append(_json.load(f))

    return {'snapshots': snaps, 'count': len(snaps)}


@app.post('/api/registry/{platform}/rollback/{snapshot_id}')
async def rollback_to_snapshot(platform: str, snapshot_id: str):
    """
    Roll back engine to a previously saved snapshot.
    Safety net: if a session corrupted the model, restore last known-good.
    """
    if not HAS_REGISTRY:
        raise HTTPException(503, 'Registry not available')
    engine = require_engine()
    reg    = get_registry()

    import shutil as _shutil
    snap_dir     = reg._platform_dir(platform) / 'snapshots' / snapshot_id
    platform_dir = reg._platform_dir(platform)

    if not snap_dir.exists():
        raise HTTPException(404, f'Snapshot {snapshot_id} not found')

    restored = []
    for fname in ['params.json', 'ensemble_0.npz', 'ensemble_1.npz',
                  'ensemble_2.npz', 'cem_warmstart.npz']:
        src = snap_dir / fname
        if src.exists():
            _shutil.copy2(src, platform_dir / fname)
            restored.append(fname)

    reg.load(engine, platform)
    with open(snap_dir / 'meta.json') as f:
        meta = _json.load(f)

    _fire_webhook('session_saved', {
        'event':       'rollback',
        'snapshot_id': snapshot_id,
        'platform':    platform,
        'restored':    restored,
    })

    return {
        'rolled_back_to': snapshot_id,
        'restored_files': restored,
        'params':         engine.physics.params,
        'original_step':  meta.get('step_count'),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  ITEM 6 — NARRATION ENDPOINT (PRODUCT-GRADE)
#  Plain-English engine status. Teams display this in their product UI.
#  PhysiCore's words appear in front of their customers.
# =============================================================================

@app.get('/api/narrate')
async def narrate():
    """
    Plain-English interpretation of current PhysiCore state.
    No API key. No rate limits. Always works. Updates every step.

    Teams display this in their product UI so their customers understand
    what the robot is doing and how well it has adapted.

    Example response:
      {
        "status": "NOMINAL",
        "headline": "Model converged — residual 0.0031, mass 1.347kg",
        "detail": "PhysiCore has learned your hardware's real physics...",
        "action": "Nominal. Registry will save learned params on session end.",
        "badge": "🟢 NOMINAL",
        "badge_color": "#00FF88",
        "metrics": { "residual": 0.0031, "mass": 1.347, ... },
        "for_display": {
          "title": "PhysiCore Status",
          "subtitle": "Adapting to your hardware in real time",
          "status_line": "Mass learned: 1.347kg | Residual: 0.0031 | 847 steps"
        }
      }
    """
    engine = require_engine()
    nar    = engine.narrate()

    status = nar['status']
    badge_map = {
        'NOMINAL':    ('🟢 NOMINAL',    '#00FF88'),
        'CONVERGING': ('🔵 CONVERGING', '#0099FF'),
        'ELEVATED':   ('🟡 ELEVATED',   '#FFB800'),
        'FAULT':      ('🔴 FAULT',      '#FF2222'),
    }
    badge, color = badge_map.get(status, ('⚪ UNKNOWN', '#7A7A7A'))
    m = nar['metrics']

    return {
        **nar,
        'badge':       badge,
        'badge_color': color,
        'for_display': {
            'title':       'PhysiCore Adaptive Physics',
            'subtitle':    'Learning your hardware in real time',
            'status_line': f'Mass: {m["mass"]}kg | Residual: {m["residual"]} | {m["steps"]} steps',
            'detail':      nar['detail'],
            'action':      nar['action'],
        },
        'embed_html': f'''<div style="font-family:monospace;padding:12px;background:#0C0C0C;border:1px solid {color}40;border-radius:4px;">
  <span style="color:{color};font-weight:bold;">{badge}</span>
  <div style="color:#EFEFEF;margin-top:4px;font-size:13px;">{nar["headline"]}</div>
  <div style="color:#7A7A7A;font-size:11px;margin-top:4px;">Mass: {m["mass"]}kg | Residual: {m["residual"]} | {m["steps"]} steps</div>
</div>''',
    }


@app.get('/api/narrate/stream')
async def narrate_stream():
    """
    Server-sent events stream of narration updates.
    Teams connect once and receive live engine status as it changes.

    Usage (JavaScript):
        const es = new EventSource('http://localhost:8000/api/narrate/stream');
        es.onmessage = (e) => {
            const status = JSON.parse(e.data);
            document.getElementById('status').innerHTML = status.embed_html;
        };
    """
    from fastapi.responses import StreamingResponse

    async def event_stream():
        last_status = None
        while True:
            try:
                if engine_state.engine:
                    nar = engine_state.engine.narrate()
                    if nar['status'] != last_status:
                        last_status = nar['status']
                        m = nar['metrics']
                        badge_map = {
                            'NOMINAL':    ('🟢 NOMINAL', '#00FF88'),
                            'CONVERGING': ('🔵 CONVERGING', '#0099FF'),
                            'ELEVATED':   ('🟡 ELEVATED', '#FFB800'),
                            'FAULT':      ('🔴 FAULT', '#FF2222'),
                        }
                        badge, color = badge_map.get(nar['status'], ('⚪', '#7A7A7A'))
                        data = _json.dumps({
                            **nar,
                            'badge': badge,
                            'badge_color': color,
                            'status_line': f'Mass: {m["mass"]}kg | Residual: {m["residual"]} | {m["steps"]} steps',
                        })
                        yield f'data: {data}\n\n'
                await asyncio.sleep(1.0)
            except Exception:
                await asyncio.sleep(2.0)

    return StreamingResponse(event_stream(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache',
                                      'X-Accel-Buffering': 'no'})


def run():
    """Entry point for physicore-server console script."""
    import uvicorn
    host = os.environ.get("PHYSICORE_HOST", "0.0.0.0")
    port = int(os.environ.get("PHYSICORE_PORT", "8000"))
    uvicorn.run("physicore.api.server:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    run()

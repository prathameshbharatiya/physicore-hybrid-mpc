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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from physicore import PhysiCore, PhysiCoreConfig, PLATFORM_DYNAMICS


# ── Global engine state ────────────────────────────────────────────────────────

class EngineState:
    def __init__(self):
        self.engine: Optional[PhysiCore] = None
        self.platform: str = "none"
        self.running: bool = False
        self.last_step_time: float = 0.0
        self.telemetry_clients: List[WebSocket] = []
        self.last_diagnostics: dict = {}
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
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

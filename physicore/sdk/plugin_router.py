"""
physicore/sdk/plugin_router.py — Dynamic FastAPI router for plugin routes
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from physicore.sdk.plugin_loader import PluginLoader


# ── Request / Response models ────────────────────────────────────────────────

class ReloadRequest(BaseModel):
    engine_id: Optional[str] = None


# ── Factory ──────────────────────────────────────────────────────────────────

def build_plugin_router(loader: PluginLoader, engine=None) -> APIRouter:
    """
    Build and return an APIRouter that:
      GET  /plugins/                           list all loaded plugins
      GET  /plugins/{plugin_id}/manifest       full manifest dict
      GET  /plugins/{plugin_id}/status         sandbox status (errors, disabled)
      POST /plugins/{plugin_id}/reload         hot-reload the plugin
      GET  /plugins/{plugin_id}/panels         list dashboard panel specs
      GET  /plugins/{plugin_id}/{panel_id}/data   call plugin.get_panel_data(panel_id)
    """
    router = APIRouter(prefix="/plugins", tags=["plugins"])

    # ── /plugins/ ────────────────────────────────────────────────────────────

    @router.get("/")
    def list_plugins() -> list:
        return loader.list_loaded()

    # ── /plugins/{plugin_id}/manifest ────────────────────────────────────────

    @router.get("/{plugin_id}/manifest")
    def get_manifest(plugin_id: str) -> dict:
        manifest = loader.get_manifest(plugin_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Plugin {plugin_id!r} not loaded")
        return manifest.to_dict()

    # ── /plugins/{plugin_id}/status ──────────────────────────────────────────

    @router.get("/{plugin_id}/status")
    def get_status(plugin_id: str) -> dict:
        sandbox = loader.get_sandbox(plugin_id)
        if sandbox is None:
            raise HTTPException(status_code=404, detail=f"Plugin {plugin_id!r} not loaded")
        return sandbox.status

    # ── /plugins/{plugin_id}/reload ──────────────────────────────────────────

    @router.post("/{plugin_id}/reload")
    def reload_plugin(plugin_id: str) -> dict:
        if loader.get_manifest(plugin_id) is None:
            raise HTTPException(status_code=404, detail=f"Plugin {plugin_id!r} not loaded")
        try:
            loader.reload(plugin_id, engine)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"status": "reloaded", "plugin_id": plugin_id}

    # ── /plugins/{plugin_id}/panels ──────────────────────────────────────────

    @router.get("/{plugin_id}/panels")
    def get_panels(plugin_id: str) -> list:
        manifest = loader.get_manifest(plugin_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Plugin {plugin_id!r} not loaded")
        return [p.to_dict() for p in manifest.panels]

    # ── /plugins/{plugin_id}/{panel_id}/data ─────────────────────────────────

    @router.get("/{plugin_id}/{panel_id}/data")
    def get_panel_data(plugin_id: str, panel_id: str) -> Any:
        manifest = loader.get_manifest(plugin_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Plugin {plugin_id!r} not loaded")

        panel_ids = {p.panel_id for p in manifest.panels}
        if panel_id not in panel_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Panel {panel_id!r} not found in plugin {plugin_id!r}"
            )

        ext = loader.get_extension(plugin_id)
        if ext is None:
            raise HTTPException(status_code=503, detail="Plugin extension not available")

        # Call get_panel_data if the plugin implements it
        get_data = getattr(ext, "get_panel_data", None)
        if callable(get_data):
            try:
                return get_data(panel_id)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))

        # Fallback: call a method named after the panel_id
        panel_method = getattr(ext, panel_id, None)
        if callable(panel_method):
            try:
                return panel_method()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))

        raise HTTPException(
            status_code=501,
            detail=f"Plugin {plugin_id!r} does not implement get_panel_data"
        )

    return router


# ── Plugin API Router class ───────────────────────────────────────────────────

class PluginAPIRouter:
    """
    Higher-level wrapper that owns a PluginLoader and exposes its router.
    Attach to a FastAPI app via:

        par = PluginAPIRouter()
        par.load_all(engine)
        app.include_router(par.router)
    """

    def __init__(
        self,
        loader: Optional[PluginLoader] = None,
        search_paths=None,
    ):
        self._loader = loader or PluginLoader(search_paths=search_paths)
        self._engine = None
        self._router: Optional[APIRouter] = None

    @property
    def loader(self) -> PluginLoader:
        return self._loader

    @property
    def router(self) -> APIRouter:
        if self._router is None:
            self._router = build_plugin_router(self._loader, self._engine)
        return self._router

    def load_all(self, engine=None) -> list[str]:
        self._engine = engine
        return self._loader.load_all(engine)

    def unload(self, plugin_id: str) -> bool:
        return self._loader.unload(plugin_id)

    def reload(self, plugin_id: str) -> bool:
        return self._loader.reload(plugin_id, self._engine)

    def start_hot_reload(self, interval_s: float = 2.0):
        self._loader.start_hot_reload(self._engine, interval_s)

    def stop_hot_reload(self):
        self._loader.stop_hot_reload()

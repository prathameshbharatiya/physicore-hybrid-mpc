"""
physicore/sdk/plugin_manifest.py — Plugin Manifest definitions and loaders
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

VALID_CHART_TYPES = {"line", "bar", "gauge", "value", "heatmap", "custom"}
VALID_PERMISSIONS  = {"read_state", "write_action", "read_telemetry", "network", "filesystem"}


@dataclass
class DashboardPanelSpec:
    panel_id:      str
    title:         str
    chart_type:    str                      # line | bar | gauge | value | heatmap | custom
    data_endpoint: str                      # relative path under /plugins/{id}/
    refresh_hz:    float  = 2.0
    position:      dict   = field(default_factory=lambda: {"row": 0, "col": 0, "w": 6, "h": 4})
    extra:         dict   = field(default_factory=dict)

    def __post_init__(self):
        if self.chart_type not in VALID_CHART_TYPES:
            raise ValueError(f"chart_type must be one of {VALID_CHART_TYPES}, got {self.chart_type!r}")
        if self.refresh_hz <= 0:
            raise ValueError("refresh_hz must be positive")

    def to_dict(self) -> dict:
        return {
            "panel_id":      self.panel_id,
            "title":         self.title,
            "chart_type":    self.chart_type,
            "data_endpoint": self.data_endpoint,
            "refresh_hz":    self.refresh_hz,
            "position":      self.position,
            "extra":         self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DashboardPanelSpec":
        return cls(
            panel_id      = d["panel_id"],
            title         = d["title"],
            chart_type    = d["chart_type"],
            data_endpoint = d["data_endpoint"],
            refresh_hz    = float(d.get("refresh_hz", 2.0)),
            position      = d.get("position", {"row": 0, "col": 0, "w": 6, "h": 4}),
            extra         = d.get("extra", {}),
        )


@dataclass
class PluginManifest:
    plugin_id:    str
    name:         str
    version:      str
    description:  str                = ""
    author:       str                = ""
    entry_point:  str                = "main"          # module attribute with plugin class
    permissions:  list[str]          = field(default_factory=list)
    panels:       list[DashboardPanelSpec] = field(default_factory=list)
    hooks:        list[str]          = field(default_factory=list)
    tags:         list[str]          = field(default_factory=list)
    min_physicore_version: str       = "0.0.0"
    source_path:  Optional[Path]     = field(default=None, compare=False)

    def __post_init__(self):
        if not _SEMVER_RE.match(self.version):
            raise ValueError(f"version must be semver (X.Y.Z), got {self.version!r}")
        bad_perms = set(self.permissions) - VALID_PERMISSIONS
        if bad_perms:
            raise ValueError(f"Unknown permissions: {bad_perms}")
        bad_hooks = set(self.hooks) - {"pre_step", "post_step", "on_fault", "on_telemetry"}
        if bad_hooks:
            raise ValueError(f"Unknown hooks: {bad_hooks}")

    def to_dict(self) -> dict:
        return {
            "plugin_id":             self.plugin_id,
            "name":                  self.name,
            "version":               self.version,
            "description":           self.description,
            "author":                self.author,
            "entry_point":           self.entry_point,
            "permissions":           self.permissions,
            "panels":                [p.to_dict() for p in self.panels],
            "hooks":                 self.hooks,
            "tags":                  self.tags,
            "min_physicore_version": self.min_physicore_version,
        }

    @classmethod
    def from_dict(cls, d: dict, source_path: Optional[Path] = None) -> "PluginManifest":
        panels = [DashboardPanelSpec.from_dict(p) for p in d.get("panels", [])]
        return cls(
            plugin_id    = d["plugin_id"],
            name         = d["name"],
            version      = d["version"],
            description  = d.get("description", ""),
            author       = d.get("author", ""),
            entry_point  = d.get("entry_point", "main"),
            permissions  = d.get("permissions", []),
            panels       = panels,
            hooks        = d.get("hooks", []),
            tags         = d.get("tags", []),
            min_physicore_version = d.get("min_physicore_version", "0.0.0"),
            source_path  = source_path,
        )


def validate_manifest(data: Any) -> PluginManifest:
    """
    Validate raw dict (from JSON / PLUGIN_MANIFEST) and return a PluginManifest.
    Raises ValueError with a descriptive message on any validation error.
    """
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a dict")
    for required in ("plugin_id", "name", "version"):
        if required not in data:
            raise ValueError(f"Manifest missing required field: {required!r}")
    return PluginManifest.from_dict(data)


def load_manifest_from_dir(plugin_dir: Path) -> PluginManifest:
    """
    Load a PluginManifest from a plugin directory.
    Looks for plugin.json first, then plugin_manifest.json.
    """
    plugin_dir = Path(plugin_dir)
    for candidate in ("plugin.json", "plugin_manifest.json"):
        manifest_file = plugin_dir / candidate
        if manifest_file.exists():
            raw = json.loads(manifest_file.read_text(encoding="utf-8"))
            return validate_manifest({**raw, "__source": str(plugin_dir)})
    raise FileNotFoundError(f"No plugin.json found in {plugin_dir}")


def load_manifest_from_file(py_file: Path) -> PluginManifest:
    """
    Load a PluginManifest from a single .py file that contains a PLUGIN_MANIFEST dict.
    """
    import importlib.util, sys

    py_file = Path(py_file)
    spec = importlib.util.spec_from_file_location(f"_pm_{py_file.stem}", py_file)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = getattr(mod, "PLUGIN_MANIFEST", None)
    if raw is None:
        raise AttributeError(f"{py_file} has no PLUGIN_MANIFEST dict")
    return validate_manifest({**raw, "source_path": py_file})

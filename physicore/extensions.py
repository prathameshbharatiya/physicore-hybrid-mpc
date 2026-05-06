"""
PhysiCore Extension System
==========================
Base class and registry for user-defined extensions.

Extensions hook into the control loop via:
  - pre_step(state, x_ref, engine)  → (state, x_ref)  [can modify inputs]
  - post_step(step, engine)         → None              [observe/log outputs]
  - on_fault(fault, engine)         → None              [react to Sentinel faults]

Usage:
  1. Subclass PhysiCoreExtension
  2. Override the hooks you need
  3. Drop the file into ~/.physicore/extensions/
  4. The bridge auto-loads on startup

Author: Prathamesh Shirbhate — physicore.ai
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from physicore.core.engine import PhysiCore, ControlStep


@dataclass
class ExtensionMeta:
    name:        str
    version:     str   = "1.0.0"
    description: str   = ""
    author:      str   = ""
    hooks:       list  = field(default_factory=list)  # ['pre_step','post_step','on_fault']


class PhysiCoreExtension:
    """
    Base class for all PhysiCore extensions.
    Subclass this and implement the hooks you need.
    """

    meta: ExtensionMeta = ExtensionMeta(name="unnamed")

    def setup(self, engine: "PhysiCore") -> None:
        """Called once when the extension is loaded. Use to initialize resources."""
        pass

    def teardown(self) -> None:
        """Called when the bridge shuts down cleanly."""
        pass

    def pre_step(self, state: Any, x_ref: Any, engine: "PhysiCore"):
        """
        Called before each control step.
        Return (state, x_ref) — you may modify both.
        """
        return state, x_ref

    def post_step(self, step: "ControlStep", engine: "PhysiCore") -> None:
        """Called after each control step. Use to log, visualize, or react to outputs."""
        pass

    def on_fault(self, failure_type: str, engine: "PhysiCore") -> None:
        """Called when the Sentinel detects a fault. Use to trigger safe behaviors."""
        pass


# ── Extension Registry ────────────────────────────────────────────────────────

class ExtensionRegistry:
    """Manages all loaded extensions and dispatches hook calls."""

    def __init__(self):
        self._extensions: list[PhysiCoreExtension] = []

    def register(self, ext: PhysiCoreExtension, engine: "PhysiCore") -> None:
        ext.setup(engine)
        self._extensions.append(ext)
        print(f"[Extensions] Loaded: {ext.meta.name} v{ext.meta.version}")

    def teardown_all(self) -> None:
        for ext in self._extensions:
            try:
                ext.teardown()
            except Exception as e:
                print(f"[Extensions] teardown error in {ext.meta.name}: {e}")

    def run_pre_step(self, state, x_ref, engine):
        for ext in self._extensions:
            try:
                state, x_ref = ext.pre_step(state, x_ref, engine)
            except Exception as e:
                print(f"[Extensions] pre_step error in {ext.meta.name}: {e}")
        return state, x_ref

    def run_post_step(self, step, engine) -> None:
        for ext in self._extensions:
            try:
                ext.post_step(step, engine)
            except Exception as e:
                print(f"[Extensions] post_step error in {ext.meta.name}: {e}")

    def run_on_fault(self, failure_type: str, engine) -> None:
        for ext in self._extensions:
            try:
                ext.on_fault(failure_type, engine)
            except Exception as e:
                print(f"[Extensions] on_fault error in {ext.meta.name}: {e}")

    @property
    def loaded(self) -> list[dict]:
        return [{"name": e.meta.name, "version": e.meta.version,
                 "description": e.meta.description, "hooks": e.meta.hooks}
                for e in self._extensions]


# ── Auto-loader ───────────────────────────────────────────────────────────────

def load_extensions_from_dir(directory, engine) -> ExtensionRegistry:
    """
    Scan `directory` for .py files, import each, find the first
    PhysiCoreExtension subclass, and register it.
    """
    import importlib.util, pathlib, sys

    registry = ExtensionRegistry()
    ext_dir  = pathlib.Path(directory)
    if not ext_dir.exists():
        ext_dir.mkdir(parents=True, exist_ok=True)
        return registry

    for py_file in sorted(ext_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for attr in vars(mod).values():
                if (isinstance(attr, type) and
                        issubclass(attr, PhysiCoreExtension) and
                        attr is not PhysiCoreExtension):
                    registry.register(attr(), engine)
                    break
        except Exception as e:
            print(f"[Extensions] Failed to load {py_file.name}: {e}")

    return registry

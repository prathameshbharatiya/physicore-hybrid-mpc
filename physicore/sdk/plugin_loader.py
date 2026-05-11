"""
physicore/sdk/plugin_loader.py — Plugin discovery, loading, sandboxing, hot-reload
"""
from __future__ import annotations

import importlib.util
import sys
import time
import threading
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from physicore.extensions import PhysiCoreExtension, ExtensionRegistry
from physicore.sdk.plugin_manifest import (
    PluginManifest,
    load_manifest_from_dir,
    load_manifest_from_file,
    validate_manifest,
)

_DEFAULT_SEARCH_PATHS: list[Path] = [
    Path.home() / ".physicore" / "plugins",
    Path("plugins"),
]

_MAX_CONSECUTIVE_ERRORS = 10


class PluginSandbox:
    """
    Wraps a PhysiCoreExtension instance; counts consecutive errors
    and auto-disables the plugin after _MAX_CONSECUTIVE_ERRORS.
    """

    def __init__(self, extension: PhysiCoreExtension, manifest: PluginManifest):
        self.extension   = extension
        self.manifest    = manifest
        self.error_count = 0
        self.disabled    = False
        self.last_error: Optional[str] = None

    def _call(self, method: str, *args, **kwargs) -> Any:
        if self.disabled:
            return None
        try:
            result = getattr(self.extension, method)(*args, **kwargs)
            self.error_count = 0
            return result
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            if self.error_count >= _MAX_CONSECUTIVE_ERRORS:
                self.disabled = True
                print(
                    f"[PluginLoader] Plugin {self.manifest.plugin_id!r} auto-disabled "
                    f"after {_MAX_CONSECUTIVE_ERRORS} consecutive errors. Last: {exc}"
                )
            else:
                print(
                    f"[PluginLoader] Plugin {self.manifest.plugin_id!r} error "
                    f"({self.error_count}/{_MAX_CONSECUTIVE_ERRORS}) in {method}: {exc}"
                )
            return None

    def pre_step(self, state, x_ref, engine):
        result = self._call("pre_step", state, x_ref, engine)
        return result if result is not None else (state, x_ref)

    def post_step(self, step, engine):
        self._call("post_step", step, engine)

    def on_fault(self, fault, engine):
        self._call("on_fault", fault, engine)

    def setup(self, engine):
        self._call("setup", engine)

    def teardown(self):
        self._call("teardown")

    @property
    def status(self) -> dict:
        return {
            "plugin_id":   self.manifest.plugin_id,
            "name":        self.manifest.name,
            "version":     self.manifest.version,
            "disabled":    self.disabled,
            "error_count": self.error_count,
            "last_error":  self.last_error,
        }


class PluginLoader:
    """
    Discovers, loads, unloads, and hot-reloads PhysiCore plugins.

    Plugin discovery order:
      1. ~/.physicore/plugins/  (each sub-dir or .py file)
      2. ./plugins/             (same structure)
      3. .physicore-plugin zip archives in those directories

    Each plugin must expose either:
      - A directory with plugin.json + main.py (or entry_point module)
      - A single .py file with PLUGIN_MANIFEST dict
      - A .physicore-plugin zip containing the above
    """

    def __init__(
        self,
        search_paths: Optional[List[Path]] = None,
        registry: Optional[ExtensionRegistry] = None,
    ):
        self._search_paths: List[Path] = [
            Path(p) for p in (search_paths or _DEFAULT_SEARCH_PATHS)
        ]
        self._registry  = registry or ExtensionRegistry()
        self._sandboxes: Dict[str, PluginSandbox] = {}
        self._manifests: Dict[str, PluginManifest] = {}
        self._modules:   Dict[str, Any]             = {}
        self._lock       = threading.Lock()
        self._temp_dirs: Dict[str, Path] = {}   # plugin_id → temp extract dir

    # ── Discovery ────────────────────────────────────────────────────────────

    def scan(self) -> List[PluginManifest]:
        """Return manifests for all discoverable (not yet loaded) plugins."""
        found: Dict[str, PluginManifest] = {}
        for base in self._search_paths:
            if not base.exists():
                continue
            # Directory plugins
            for entry in sorted(base.iterdir()):
                if entry.is_dir() and not entry.name.startswith("_"):
                    try:
                        m = load_manifest_from_dir(entry)
                        if m.plugin_id not in found:
                            found[m.plugin_id] = m
                    except Exception:
                        pass
            # Single-file plugins
            for py_file in sorted(base.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                try:
                    m = load_manifest_from_file(py_file)
                    if m.plugin_id not in found:
                        found[m.plugin_id] = m
                except Exception:
                    pass
            # Zip plugins
            for zip_file in sorted(base.glob("*.physicore-plugin")):
                try:
                    m = self._manifest_from_zip(zip_file)
                    if m.plugin_id not in found:
                        found[m.plugin_id] = m
                except Exception:
                    pass
        return list(found.values())

    def _manifest_from_zip(self, zip_path: Path) -> PluginManifest:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            json_candidates = [n for n in names if n.endswith("plugin.json")]
            if not json_candidates:
                raise FileNotFoundError(f"No plugin.json in {zip_path}")
            raw = zf.read(json_candidates[0]).decode()
            import json
            data = json.loads(raw)
            return validate_manifest(data)

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_all(self, engine=None) -> List[str]:
        """Scan and load all discoverable plugins. Returns list of loaded plugin_ids."""
        loaded = []
        for manifest in self.scan():
            if manifest.plugin_id not in self._sandboxes:
                try:
                    self._load_one(manifest, engine)
                    loaded.append(manifest.plugin_id)
                except Exception as exc:
                    print(f"[PluginLoader] Failed to load {manifest.plugin_id}: {exc}")
        return loaded

    def load_one(self, plugin_id: str, engine=None) -> bool:
        """Load a specific plugin by id. Returns True on success."""
        manifests = {m.plugin_id: m for m in self.scan()}
        if plugin_id not in manifests:
            raise KeyError(f"Plugin {plugin_id!r} not found in search paths")
        self._load_one(manifests[plugin_id], engine)
        return True

    def _load_one(self, manifest: PluginManifest, engine=None) -> PluginSandbox:
        with self._lock:
            if manifest.plugin_id in self._sandboxes:
                return self._sandboxes[manifest.plugin_id]

            ext_instance = self._import_extension(manifest)
            sandbox = PluginSandbox(ext_instance, manifest)

            if engine is not None:
                sandbox.setup(engine)

            self._sandboxes[manifest.plugin_id] = sandbox
            self._manifests[manifest.plugin_id] = manifest
            print(
                f"[PluginLoader] Loaded plugin: {manifest.name} "
                f"v{manifest.version} ({manifest.plugin_id})"
            )
            return sandbox

    def _import_extension(self, manifest: PluginManifest) -> PhysiCoreExtension:
        """Import the extension class from the plugin source."""
        source = manifest.source_path

        if source is None:
            # Try to locate on search paths
            for base in self._search_paths:
                candidate_dir = base / manifest.plugin_id
                if candidate_dir.is_dir():
                    source = candidate_dir
                    break
                candidate_py = base / f"{manifest.plugin_id}.py"
                if candidate_py.is_file():
                    source = candidate_py
                    break

        if source is None:
            raise FileNotFoundError(f"Cannot locate source for plugin {manifest.plugin_id!r}")

        source = Path(source)

        if source.suffix == ".py":
            return self._import_from_py(source, manifest)
        elif source.is_dir():
            return self._import_from_dir(source, manifest)
        elif source.suffix == ".physicore-plugin":
            return self._import_from_zip(source, manifest)
        else:
            raise ValueError(f"Unsupported plugin source: {source}")

    def _import_from_py(self, py_file: Path, manifest: PluginManifest) -> PhysiCoreExtension:
        module_name = f"_physicore_plugin_{manifest.plugin_id}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        self._modules[manifest.plugin_id] = mod

        # Find PhysiCoreExtension subclass
        for attr in vars(mod).values():
            if (isinstance(attr, type)
                    and issubclass(attr, PhysiCoreExtension)
                    and attr is not PhysiCoreExtension):
                return attr()

        # Fallback: look for `plugin` attr that is an instance
        plugin_attr = getattr(mod, manifest.entry_point, None)
        if isinstance(plugin_attr, PhysiCoreExtension):
            return plugin_attr
        if isinstance(plugin_attr, type) and issubclass(plugin_attr, PhysiCoreExtension):
            return plugin_attr()

        raise AttributeError(
            f"No PhysiCoreExtension subclass found in {py_file}"
        )

    def _import_from_dir(self, plugin_dir: Path, manifest: PluginManifest) -> PhysiCoreExtension:
        main_py = plugin_dir / "main.py"
        if not main_py.exists():
            # Try entry_point as module name
            main_py = plugin_dir / f"{manifest.entry_point}.py"
        if not main_py.exists():
            raise FileNotFoundError(f"No main.py in {plugin_dir}")
        # Add plugin dir to sys.path temporarily
        str_dir = str(plugin_dir)
        if str_dir not in sys.path:
            sys.path.insert(0, str_dir)
        return self._import_from_py(main_py, manifest)

    def _import_from_zip(self, zip_path: Path, manifest: PluginManifest) -> PhysiCoreExtension:
        tmp = Path(tempfile.mkdtemp(prefix="physicore_plugin_"))
        self._temp_dirs[manifest.plugin_id] = tmp
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        return self._import_from_dir(tmp, manifest)

    # ── Unload / Reload ──────────────────────────────────────────────────────

    def unload(self, plugin_id: str) -> bool:
        with self._lock:
            sandbox = self._sandboxes.pop(plugin_id, None)
            if sandbox is None:
                return False
            sandbox.teardown()
            self._manifests.pop(plugin_id, None)
            # Remove from sys.modules
            mod_name = f"_physicore_plugin_{plugin_id}"
            sys.modules.pop(mod_name, None)
            self._modules.pop(plugin_id, None)
            # Clean up temp dir if any
            tmp = self._temp_dirs.pop(plugin_id, None)
            if tmp and tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            print(f"[PluginLoader] Unloaded plugin: {plugin_id}")
            return True

    def reload(self, plugin_id: str, engine=None) -> bool:
        """Hot-reload: unload then re-load a plugin."""
        manifest = self._manifests.get(plugin_id)
        if manifest is None:
            raise KeyError(f"Plugin {plugin_id!r} is not loaded")
        source_path = manifest.source_path
        self.unload(plugin_id)
        # Re-discover manifest (may have changed on disk)
        try:
            if source_path and Path(source_path).is_file():
                new_manifest = load_manifest_from_file(Path(source_path))
            elif source_path and Path(source_path).is_dir():
                new_manifest = load_manifest_from_dir(Path(source_path))
            else:
                new_manifest = manifest
        except Exception:
            new_manifest = manifest
        self._load_one(new_manifest, engine)
        return True

    def start_hot_reload(self, engine=None, interval_s: float = 2.0):
        """
        Start a background thread that watches for file changes and
        auto-reloads any plugin whose source has been modified.
        """
        self._hot_reload_stop  = threading.Event()
        self._hot_reload_mtimes: Dict[str, float] = {}

        def _watcher():
            while not self._hot_reload_stop.is_set():
                with self._lock:
                    for pid, manifest in list(self._manifests.items()):
                        src = manifest.source_path
                        if src is None:
                            continue
                        src = Path(src)
                        try:
                            mtime = src.stat().st_mtime if src.is_file() else max(
                                f.stat().st_mtime for f in src.rglob("*.py") if f.is_file()
                            )
                        except Exception:
                            continue
                        prev = self._hot_reload_mtimes.get(pid, 0.0)
                        if mtime > prev + 0.01:
                            self._hot_reload_mtimes[pid] = mtime
                            if prev > 0:
                                try:
                                    self.reload(pid, engine)
                                    print(f"[PluginLoader] Hot-reloaded: {pid}")
                                except Exception as exc:
                                    print(f"[PluginLoader] Hot-reload failed for {pid}: {exc}")
                            else:
                                self._hot_reload_mtimes[pid] = mtime
                self._hot_reload_stop.wait(interval_s)

        self._hot_reload_thread = threading.Thread(target=_watcher, daemon=True, name="plugin-hot-reload")
        self._hot_reload_thread.start()

    def stop_hot_reload(self):
        if hasattr(self, "_hot_reload_stop"):
            self._hot_reload_stop.set()

    # ── Queries ──────────────────────────────────────────────────────────────

    def list_loaded(self) -> List[dict]:
        return [sb.status for sb in self._sandboxes.values()]

    def get_manifest(self, plugin_id: str) -> Optional[PluginManifest]:
        return self._manifests.get(plugin_id)

    def get_sandbox(self, plugin_id: str) -> Optional[PluginSandbox]:
        return self._sandboxes.get(plugin_id)

    def get_extension(self, plugin_id: str) -> Optional[PhysiCoreExtension]:
        sb = self._sandboxes.get(plugin_id)
        return sb.extension if sb else None

    def run_pre_step(self, state, x_ref, engine):
        for sb in list(self._sandboxes.values()):
            if "pre_step" in sb.manifest.hooks:
                state, x_ref = sb.pre_step(state, x_ref, engine)
        return state, x_ref

    def run_post_step(self, step, engine):
        for sb in list(self._sandboxes.values()):
            if "post_step" in sb.manifest.hooks:
                sb.post_step(step, engine)

    def run_on_fault(self, fault, engine):
        for sb in list(self._sandboxes.values()):
            if "on_fault" in sb.manifest.hooks:
                sb.on_fault(fault, engine)

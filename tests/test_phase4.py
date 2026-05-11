"""tests/test_phase4.py — Phase 4 Plugin SDK Tests"""

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# PART A: PluginManifest tests
# ═══════════════════════════════════════════════════════════════════════

class TestPluginManifest:

    def _valid_raw(self, **overrides):
        base = {
            "plugin_id":   "test_plugin",
            "name":        "Test Plugin",
            "version":     "1.0.0",
            "description": "A test plugin",
            "author":      "Tester",
            "permissions": ["read_state"],
            "hooks":       ["post_step"],
            "panels": [
                {
                    "panel_id":      "main",
                    "title":         "Main Panel",
                    "chart_type":    "line",
                    "data_endpoint": "main/data",
                    "refresh_hz":    2.0,
                }
            ],
        }
        base.update(overrides)
        return base

    def test_01_valid_manifest_round_trips(self):
        """PluginManifest serialises and deserialises correctly."""
        from physicore.sdk.plugin_manifest import validate_manifest
        m = validate_manifest(self._valid_raw())
        d = m.to_dict()
        assert d["plugin_id"] == "test_plugin"
        assert d["version"]   == "1.0.0"
        assert len(d["panels"]) == 1
        assert d["panels"][0]["chart_type"] == "line"

    def test_02_bad_semver_raises(self):
        """Non-semver version raises ValueError."""
        from physicore.sdk.plugin_manifest import validate_manifest
        with pytest.raises(ValueError, match="semver"):
            validate_manifest(self._valid_raw(version="1.0"))

    def test_03_unknown_permission_raises(self):
        """Unknown permission raises ValueError."""
        from physicore.sdk.plugin_manifest import validate_manifest
        with pytest.raises(ValueError, match="permission"):
            validate_manifest(self._valid_raw(permissions=["sudo"]))

    def test_04_unknown_hook_raises(self):
        """Unknown hook raises ValueError."""
        from physicore.sdk.plugin_manifest import validate_manifest
        with pytest.raises(ValueError, match="hook"):
            validate_manifest(self._valid_raw(hooks=["on_destroy"]))

    def test_05_invalid_chart_type_raises(self):
        """Invalid chart_type on a panel raises ValueError."""
        from physicore.sdk.plugin_manifest import validate_manifest, DashboardPanelSpec
        with pytest.raises(ValueError, match="chart_type"):
            DashboardPanelSpec(
                panel_id="x", title="X", chart_type="unknown_chart",
                data_endpoint="x/data"
            )

    def test_06_missing_required_field_raises(self):
        """Missing plugin_id raises ValueError."""
        from physicore.sdk.plugin_manifest import validate_manifest
        raw = self._valid_raw()
        del raw["plugin_id"]
        with pytest.raises(ValueError, match="plugin_id"):
            validate_manifest(raw)

    def test_07_load_manifest_from_dir(self, tmp_path):
        """load_manifest_from_dir reads plugin.json from a directory."""
        from physicore.sdk.plugin_manifest import load_manifest_from_dir
        (tmp_path / "plugin.json").write_text(
            json.dumps(self._valid_raw()), encoding="utf-8"
        )
        m = load_manifest_from_dir(tmp_path)
        assert m.plugin_id == "test_plugin"

    def test_08_load_manifest_from_file(self, tmp_path):
        """load_manifest_from_file reads PLUGIN_MANIFEST from a .py file."""
        from physicore.sdk.plugin_manifest import load_manifest_from_file
        raw = self._valid_raw()
        py_src = f"PLUGIN_MANIFEST = {raw!r}\n"
        py_file = tmp_path / "myplugin.py"
        py_file.write_text(py_src, encoding="utf-8")
        m = load_manifest_from_file(py_file)
        assert m.plugin_id == "test_plugin"

    def test_09_panel_spec_from_dict(self):
        """DashboardPanelSpec.from_dict works for all chart types."""
        from physicore.sdk.plugin_manifest import DashboardPanelSpec, VALID_CHART_TYPES
        for ct in VALID_CHART_TYPES:
            p = DashboardPanelSpec.from_dict({
                "panel_id":      "p1",
                "title":         "T",
                "chart_type":    ct,
                "data_endpoint": "p1/data",
            })
            assert p.chart_type == ct


# ═══════════════════════════════════════════════════════════════════════
# PART B: PluginLoader tests
# ═══════════════════════════════════════════════════════════════════════

def _write_single_file_plugin(tmp_path: Path, plugin_id: str = "dummy_plugin") -> Path:
    """Helper: write a minimal single-file plugin to tmp_path."""
    src = f"""
import numpy as np
from physicore.extensions import PhysiCoreExtension, ExtensionMeta

PLUGIN_MANIFEST = {{
    "plugin_id":   "{plugin_id}",
    "name":        "Dummy Plugin",
    "version":     "0.1.0",
    "description": "Test plugin",
    "author":      "Test",
    "permissions": ["read_state"],
    "hooks":       ["post_step"],
    "panels": [
        {{
            "panel_id":      "main",
            "title":         "Main",
            "chart_type":    "line",
            "data_endpoint": "main/data",
            "refresh_hz":    1.0,
        }}
    ],
}}

class DummyExt(PhysiCoreExtension):
    meta = ExtensionMeta(name="Dummy", version="0.1.0")
    def __init__(self):
        self.call_count = 0
    def post_step(self, step, engine):
        self.call_count += 1
    def get_panel_data(self, panel_id):
        return {{"count": self.call_count}}

main = DummyExt
"""
    py_file = tmp_path / f"{plugin_id}.py"
    py_file.write_text(src, encoding="utf-8")
    return py_file


class TestPluginLoader:

    def test_01_scan_finds_single_file_plugin(self, tmp_path):
        """scan() discovers a single-file plugin."""
        from physicore.sdk.plugin_loader import PluginLoader
        _write_single_file_plugin(tmp_path, "scan_plugin")
        loader = PluginLoader(search_paths=[tmp_path])
        manifests = loader.scan()
        ids = [m.plugin_id for m in manifests]
        assert "scan_plugin" in ids

    def test_02_load_all_loads_plugin(self, tmp_path):
        """load_all() loads discovered plugins and returns their ids."""
        from physicore.sdk.plugin_loader import PluginLoader
        _write_single_file_plugin(tmp_path, "load_all_plugin")
        loader = PluginLoader(search_paths=[tmp_path])
        loaded = loader.load_all()
        assert "load_all_plugin" in loaded
        assert "load_all_plugin" in [s["plugin_id"] for s in loader.list_loaded()]

    def test_03_load_one_by_id(self, tmp_path):
        """load_one() loads a specific plugin by plugin_id."""
        from physicore.sdk.plugin_loader import PluginLoader
        _write_single_file_plugin(tmp_path, "specific_plugin")
        loader = PluginLoader(search_paths=[tmp_path])
        loader.load_one("specific_plugin")
        assert loader.get_manifest("specific_plugin") is not None

    def test_04_unload_removes_plugin(self, tmp_path):
        """unload() removes a loaded plugin."""
        from physicore.sdk.plugin_loader import PluginLoader
        _write_single_file_plugin(tmp_path, "unload_plugin")
        loader = PluginLoader(search_paths=[tmp_path])
        loader.load_all()
        assert loader.unload("unload_plugin")
        assert loader.get_manifest("unload_plugin") is None

    def test_05_get_extension_returns_instance(self, tmp_path):
        """get_extension() returns the PhysiCoreExtension instance."""
        from physicore.sdk.plugin_loader import PluginLoader
        from physicore.extensions import PhysiCoreExtension
        _write_single_file_plugin(tmp_path, "ext_plugin")
        loader = PluginLoader(search_paths=[tmp_path])
        loader.load_all()
        ext = loader.get_extension("ext_plugin")
        assert isinstance(ext, PhysiCoreExtension)

    def test_06_directory_plugin_loads(self, tmp_path):
        """Directory-based plugin (plugin.json + main.py) loads correctly."""
        from physicore.sdk.plugin_loader import PluginLoader
        plugin_dir = tmp_path / "dir_plugin"
        plugin_dir.mkdir()

        manifest_data = {
            "plugin_id":   "dir_plugin",
            "name":        "Dir Plugin",
            "version":     "1.0.0",
            "permissions": ["read_state"],
            "hooks":       ["post_step"],
            "panels":      [],
        }
        (plugin_dir / "plugin.json").write_text(
            json.dumps(manifest_data), encoding="utf-8"
        )
        main_src = """
from physicore.extensions import PhysiCoreExtension, ExtensionMeta
class DirExt(PhysiCoreExtension):
    meta = ExtensionMeta(name="DirPlugin", version="1.0.0")
main = DirExt
"""
        (plugin_dir / "main.py").write_text(main_src, encoding="utf-8")
        (plugin_dir / "__init__.py").write_text("", encoding="utf-8")

        loader = PluginLoader(search_paths=[tmp_path])
        loaded = loader.load_all()
        assert "dir_plugin" in loaded


# ═══════════════════════════════════════════════════════════════════════
# PART C: PluginSandbox tests
# ═══════════════════════════════════════════════════════════════════════

class TestPluginSandbox:

    def _make_sandbox(self, fail_after: int = 999):
        from physicore.sdk.plugin_loader import PluginSandbox
        from physicore.sdk.plugin_manifest import validate_manifest
        from physicore.extensions import PhysiCoreExtension, ExtensionMeta

        class BrokenExt(PhysiCoreExtension):
            meta = ExtensionMeta(name="Broken", version="0.1.0")
            def __init__(self, fail_after=999):
                self.calls = 0
                self._fail_after = fail_after
            def post_step(self, step, engine):
                self.calls += 1
                if self.calls > self._fail_after:
                    raise RuntimeError("Simulated plugin crash")

        raw = {
            "plugin_id": "broken", "name": "Broken",
            "version": "0.1.0", "permissions": [], "hooks": ["post_step"], "panels": [],
        }
        manifest = validate_manifest(raw)
        ext = BrokenExt(fail_after=fail_after)
        return PluginSandbox(ext, manifest)

    def test_01_sandbox_counts_errors(self):
        """PluginSandbox counts errors without crashing the caller."""
        sb = self._make_sandbox(fail_after=0)
        for _ in range(3):
            sb.post_step(object(), None)
        assert sb.error_count > 0

    def test_02_sandbox_auto_disables_after_10_errors(self):
        """Sandbox auto-disables after 10 consecutive errors."""
        from physicore.sdk.plugin_loader import _MAX_CONSECUTIVE_ERRORS
        sb = self._make_sandbox(fail_after=0)
        for _ in range(_MAX_CONSECUTIVE_ERRORS + 2):
            sb.post_step(object(), None)
        assert sb.disabled

    def test_03_disabled_sandbox_does_not_call_extension(self):
        """Disabled sandbox stops calling the extension."""
        sb = self._make_sandbox(fail_after=0)
        from physicore.sdk.plugin_loader import _MAX_CONSECUTIVE_ERRORS
        for _ in range(_MAX_CONSECUTIVE_ERRORS + 2):
            sb.post_step(object(), None)

        # Now disabled; extension calls should stop incrementing
        calls_before = sb.extension.calls
        sb.post_step(object(), None)
        assert sb.extension.calls == calls_before

    def test_04_pre_step_returns_original_on_error(self):
        """pre_step returns (state, x_ref) unchanged when plugin errors."""
        from physicore.sdk.plugin_loader import PluginSandbox
        from physicore.sdk.plugin_manifest import validate_manifest
        from physicore.extensions import PhysiCoreExtension, ExtensionMeta

        class AlwaysFail(PhysiCoreExtension):
            meta = ExtensionMeta(name="Fail", version="0.1.0")
            def pre_step(self, state, x_ref, engine):
                raise RuntimeError("forced")

        raw = {
            "plugin_id": "fail", "name": "Fail",
            "version": "0.1.0", "permissions": [], "hooks": ["pre_step"], "panels": [],
        }
        sb = PluginSandbox(AlwaysFail(), validate_manifest(raw))
        state  = np.array([1.0, 2.0])
        x_ref  = np.array([0.0])
        s2, x2 = sb.pre_step(state, x_ref, None)
        assert np.array_equal(s2, state)
        assert np.array_equal(x2, x_ref)

    def test_05_status_dict_has_required_keys(self):
        """sandbox.status contains all required keys."""
        sb = self._make_sandbox()
        s = sb.status
        for key in ("plugin_id", "name", "version", "disabled", "error_count", "last_error"):
            assert key in s


# ═══════════════════════════════════════════════════════════════════════
# PART D: PluginAPIRouter / plugin_router tests
# ═══════════════════════════════════════════════════════════════════════

class TestPluginRouter:

    def _make_loader_with_plugin(self, tmp_path):
        from physicore.sdk.plugin_loader import PluginLoader
        _write_single_file_plugin(tmp_path, "router_plugin")
        loader = PluginLoader(search_paths=[tmp_path])
        loader.load_all()
        return loader

    def test_01_list_plugins_endpoint(self, tmp_path):
        """GET /plugins/ returns list of loaded plugin statuses."""
        from physicore.sdk.plugin_router import build_plugin_router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        loader = self._make_loader_with_plugin(tmp_path)
        app = FastAPI()
        app.include_router(build_plugin_router(loader))
        client = TestClient(app)

        resp = client.get("/plugins/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert any(p["plugin_id"] == "router_plugin" for p in data)

    def test_02_manifest_endpoint(self, tmp_path):
        """GET /plugins/{id}/manifest returns the manifest dict."""
        from physicore.sdk.plugin_router import build_plugin_router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        loader = self._make_loader_with_plugin(tmp_path)
        app = FastAPI()
        app.include_router(build_plugin_router(loader))
        client = TestClient(app)

        resp = client.get("/plugins/router_plugin/manifest")
        assert resp.status_code == 200
        d = resp.json()
        assert d["plugin_id"] == "router_plugin"

    def test_03_status_endpoint(self, tmp_path):
        """GET /plugins/{id}/status returns sandbox status."""
        from physicore.sdk.plugin_router import build_plugin_router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        loader = self._make_loader_with_plugin(tmp_path)
        app = FastAPI()
        app.include_router(build_plugin_router(loader))
        client = TestClient(app)

        resp = client.get("/plugins/router_plugin/status")
        assert resp.status_code == 200
        s = resp.json()
        assert "disabled" in s
        assert "error_count" in s

    def test_04_panels_endpoint(self, tmp_path):
        """GET /plugins/{id}/panels returns list of panel specs."""
        from physicore.sdk.plugin_router import build_plugin_router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        loader = self._make_loader_with_plugin(tmp_path)
        app = FastAPI()
        app.include_router(build_plugin_router(loader))
        client = TestClient(app)

        resp = client.get("/plugins/router_plugin/panels")
        assert resp.status_code == 200
        panels = resp.json()
        assert isinstance(panels, list)
        assert len(panels) == 1
        assert panels[0]["panel_id"] == "main"

    def test_05_panel_data_endpoint(self, tmp_path):
        """GET /plugins/{id}/{panel_id}/data returns plugin data."""
        from physicore.sdk.plugin_router import build_plugin_router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        loader = self._make_loader_with_plugin(tmp_path)
        app = FastAPI()
        app.include_router(build_plugin_router(loader))
        client = TestClient(app)

        resp = client.get("/plugins/router_plugin/main/data")
        assert resp.status_code == 200

    def test_06_unknown_plugin_returns_404(self, tmp_path):
        """GET /plugins/nonexistent/manifest returns 404."""
        from physicore.sdk.plugin_router import build_plugin_router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        loader = self._make_loader_with_plugin(tmp_path)
        app = FastAPI()
        app.include_router(build_plugin_router(loader))
        client = TestClient(app)

        resp = client.get("/plugins/nonexistent/manifest")
        assert resp.status_code == 404

    def test_07_reload_endpoint(self, tmp_path):
        """POST /plugins/{id}/reload reloads the plugin."""
        from physicore.sdk.plugin_router import build_plugin_router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        loader = self._make_loader_with_plugin(tmp_path)
        app = FastAPI()
        app.include_router(build_plugin_router(loader))
        client = TestClient(app)

        resp = client.post("/plugins/router_plugin/reload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reloaded"


# ═══════════════════════════════════════════════════════════════════════
# PART E: Plugin Template Generator tests
# ═══════════════════════════════════════════════════════════════════════

class TestPluginTemplate:

    def test_01_generate_directory_plugin(self, tmp_path):
        """generate_plugin creates plugin.json and main.py in a directory."""
        from physicore.sdk.plugin_template import generate_plugin
        out = generate_plugin("my_plugin", out_dir=tmp_path)
        assert (out / "plugin.json").exists()
        assert (out / "main.py").exists()

    def test_02_generated_manifest_is_valid(self, tmp_path):
        """Generated plugin.json parses into a valid PluginManifest."""
        from physicore.sdk.plugin_template import generate_plugin
        from physicore.sdk.plugin_manifest import load_manifest_from_dir
        generate_plugin("valid_gen", out_dir=tmp_path)
        m = load_manifest_from_dir(tmp_path / "valid_gen")
        assert m.plugin_id == "valid_gen"
        assert m.version   == "0.1.0"

    def test_03_generate_single_file_plugin(self, tmp_path):
        """generate_plugin with as_single_file=True creates a .py file."""
        from physicore.sdk.plugin_template import generate_plugin
        out = generate_plugin("sf_plugin", out_dir=tmp_path, as_single_file=True)
        assert out.suffix == ".py"
        assert out.exists()

    def test_04_single_file_has_plugin_manifest(self, tmp_path):
        """Single-file plugin contains PLUGIN_MANIFEST dict."""
        from physicore.sdk.plugin_template import generate_plugin
        out = generate_plugin("sf_check", out_dir=tmp_path, as_single_file=True)
        src = out.read_text()
        assert "PLUGIN_MANIFEST" in src

    def test_05_generated_plugin_loads(self, tmp_path):
        """Generated directory plugin loads via PluginLoader."""
        from physicore.sdk.plugin_template import generate_plugin
        from physicore.sdk.plugin_loader import PluginLoader
        generate_plugin("loadable_gen", name="Loadable", out_dir=tmp_path)
        loader = PluginLoader(search_paths=[tmp_path])
        loaded = loader.load_all()
        assert "loadable_gen" in loaded


# ═══════════════════════════════════════════════════════════════════════
# PART F: Example plugins tests
# ═══════════════════════════════════════════════════════════════════════

class TestExamplePlugins:

    def _examples_dir(self):
        return Path(__file__).parent.parent / "plugins" / "examples"

    def test_01_terrain_classifier_manifest_valid(self):
        """terrain_classifier plugin.json is a valid PluginManifest."""
        from physicore.sdk.plugin_manifest import load_manifest_from_dir
        d = self._examples_dir() / "terrain_classifier"
        if not d.exists():
            pytest.skip("terrain_classifier not found")
        m = load_manifest_from_dir(d)
        assert m.plugin_id == "terrain_classifier"
        assert len(m.panels) >= 1

    def test_02_energy_monitor_manifest_valid(self):
        """energy_monitor plugin.json is a valid PluginManifest."""
        from physicore.sdk.plugin_manifest import load_manifest_from_dir
        d = self._examples_dir() / "energy_monitor"
        if not d.exists():
            pytest.skip("energy_monitor not found")
        m = load_manifest_from_dir(d)
        assert m.plugin_id == "energy_monitor"
        assert len(m.panels) >= 1

    def test_03_terrain_classifier_loads(self):
        """terrain_classifier loads and its extension is a PhysiCoreExtension."""
        from physicore.sdk.plugin_loader import PluginLoader
        from physicore.extensions import PhysiCoreExtension
        d = self._examples_dir()
        if not (d / "terrain_classifier").exists():
            pytest.skip("terrain_classifier not found")
        loader = PluginLoader(search_paths=[d])
        loader.load_all()
        ext = loader.get_extension("terrain_classifier")
        assert isinstance(ext, PhysiCoreExtension)

    def test_04_energy_monitor_loads(self):
        """energy_monitor loads and its extension is a PhysiCoreExtension."""
        from physicore.sdk.plugin_loader import PluginLoader
        from physicore.extensions import PhysiCoreExtension
        d = self._examples_dir()
        if not (d / "energy_monitor").exists():
            pytest.skip("energy_monitor not found")
        loader = PluginLoader(search_paths=[d])
        loader.load_all()
        ext = loader.get_extension("energy_monitor")
        assert isinstance(ext, PhysiCoreExtension)

    def test_05_terrain_classifier_panel_data(self):
        """terrain_classifier.get_panel_data returns expected keys."""
        from physicore.sdk.plugin_loader import PluginLoader
        d = self._examples_dir()
        if not (d / "terrain_classifier").exists():
            pytest.skip("terrain_classifier not found")
        loader = PluginLoader(search_paths=[d])
        loader.load_all()
        ext = loader.get_extension("terrain_classifier")
        data = ext.get_panel_data("terrain_confidence")
        assert "labels" in data
        assert "values" in data
        assert len(data["labels"]) == len(data["values"])

    def test_06_energy_monitor_panel_data(self):
        """energy_monitor.get_panel_data returns gauge data with value key."""
        from physicore.sdk.plugin_loader import PluginLoader
        d = self._examples_dir()
        if not (d / "energy_monitor").exists():
            pytest.skip("energy_monitor not found")
        loader = PluginLoader(search_paths=[d])
        loader.load_all()
        ext = loader.get_extension("energy_monitor")
        data = ext.get_panel_data("energy_gauge")
        assert "value" in data
        assert "max"   in data
        assert "unit"  in data

    def test_07_post_step_updates_terrain_confidences(self):
        """terrain_classifier confidences change after post_step calls."""
        from physicore.sdk.plugin_loader import PluginLoader
        d = self._examples_dir()
        if not (d / "terrain_classifier").exists():
            pytest.skip("terrain_classifier not found")
        loader = PluginLoader(search_paths=[d])
        loader.load_all()
        ext = loader.get_extension("terrain_classifier")

        class FakeStep:
            action = np.ones(4) * 2.0
            state  = np.array([0.5, 0.1, 0.3, 0.0])

        for _ in range(10):
            ext.post_step(FakeStep(), None)

        data = ext.get_panel_data("terrain_confidence")
        # At least one confidence should be > 0.1
        assert max(data["values"]) > 0.1

    def test_08_post_step_accumulates_energy(self):
        """energy_monitor total energy increases after post_step calls."""
        from physicore.sdk.plugin_loader import PluginLoader
        d = self._examples_dir()
        if not (d / "energy_monitor").exists():
            pytest.skip("energy_monitor not found")
        loader = PluginLoader(search_paths=[d])
        loader.load_all()
        ext = loader.get_extension("energy_monitor")

        class FakeStep:
            action = np.ones(4) * 5.0
            state  = np.array([1.0, 0.5, 0.2, 0.1])

        for _ in range(20):
            time.sleep(0.001)
            ext.post_step(FakeStep(), None)

        data = ext.get_panel_data("energy_gauge")
        assert data["value"] > 0.0

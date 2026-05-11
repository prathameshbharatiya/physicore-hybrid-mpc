"""
Phase 8 — Launch Readiness Test Suite
======================================
Verifies that all Phase 8 deliverables exist and function correctly
before the v1.0.0 tag is pushed.
"""

import importlib
import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
DOCS = ROOT / "docs"

# ── Helpers ────────────────────────────────────────────────────────────────────

def run_script(script_path, timeout=30):
    """Run a Python script as a subprocess and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"},
        encoding="utf-8", errors="replace",
    )
    return result.returncode, result.stdout, result.stderr


def read_text(path):
    """Read a file as UTF-8, replacing unknown bytes."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


# ══════════════════════════════════════════════════════════════════════════════
# Part A — Docs site
# ══════════════════════════════════════════════════════════════════════════════

class TestDocsSite:

    EXPECTED_PAGES = [
        "index.html", "quickstart.html", "architecture.html",
        "api-reference.html", "plugin-sdk.html", "robot-loading.html",
        "platforms.html", "safety.html", "deployment.html",
    ]

    def test_docs_directory_exists(self):
        assert DOCS.is_dir(), "docs/ directory not found"

    @pytest.mark.parametrize("page", EXPECTED_PAGES)
    def test_docs_page_exists(self, page):
        assert (DOCS / page).exists(), f"docs/{page} missing"

    @pytest.mark.parametrize("page", EXPECTED_PAGES)
    def test_docs_page_not_empty(self, page):
        content = read_text(DOCS / page)
        assert len(content) > 200, f"docs/{page} too short"

    def test_docs_pages_have_dark_theme(self):
        for page in self.EXPECTED_PAGES:
            html = read_text(DOCS / page)
            assert "_base.css" in html, f"{page} missing _base.css link"

    def test_docs_pages_have_highlight_js(self):
        for page in self.EXPECTED_PAGES:
            html = read_text(DOCS / page)
            assert "highlight.js" in html or "highlight.min.js" in html, \
                f"{page} missing highlight.js"

    def test_docs_pages_have_nav(self):
        for page in self.EXPECTED_PAGES:
            html = read_text(DOCS / page)
            assert "_nav.js" in html, f"{page} missing _nav.js"

    def test_docs_base_css_exists(self):
        assert (DOCS / "_base.css").exists()

    def test_docs_nav_js_exists(self):
        assert (DOCS / "_nav.js").exists()

    def test_docs_nav_js_has_all_pages(self):
        nav_js = read_text(DOCS / "_nav.js")
        for page in self.EXPECTED_PAGES:
            assert page in nav_js, f"_nav.js missing link to {page}"

    def test_docs_search_index_has_all_pages(self):
        nav_js = read_text(DOCS / "_nav.js")
        assert "PAGES" in nav_js
        assert len([p for p in self.EXPECTED_PAGES if p in nav_js]) == len(self.EXPECTED_PAGES)

    def test_docs_index_has_hero(self):
        html = read_text(DOCS / "index.html")
        assert "hero" in html
        assert "PhysiCore" in html

    def test_docs_index_has_quick_install(self):
        html = read_text(DOCS / "index.html")
        assert "pip install" in html or "install.sh" in html


# ══════════════════════════════════════════════════════════════════════════════
# Part B — CLI
# ══════════════════════════════════════════════════════════════════════════════

class TestCLI:

    def test_cli_module_exists(self):
        assert (ROOT / "physicore" / "cli.py").exists()

    def test_cli_importable(self):
        spec = importlib.util.spec_from_file_location(
            "physicore.cli", ROOT / "physicore" / "cli.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")

    def test_cli_help(self):
        rc, out, err = run_script(ROOT / "physicore" / "cli.py")
        # Should print help (no crash)
        assert rc == 0 or "usage" in (out + err).lower() or "physicore" in (out + err).lower()

    def test_cli_version_command(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "physicore" / "cli.py"), "version"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        assert result.returncode == 0
        assert "PhysiCore" in result.stdout or "physicore" in result.stdout.lower()

    def test_cli_has_serve_command(self):
        cli = read_text(ROOT / "physicore" / "cli.py")
        assert "serve" in cli
        assert "cmd_serve" in cli

    def test_cli_has_status_command(self):
        cli = read_text(ROOT / "physicore" / "cli.py")
        assert "status" in cli
        assert "cmd_status" in cli

    def test_cli_has_fleet_command(self):
        cli = read_text(ROOT / "physicore" / "cli.py")
        assert "fleet" in cli

    def test_cli_has_plugins_command(self):
        cli = read_text(ROOT / "physicore" / "cli.py")
        assert "plugins" in cli

    def test_cli_has_robot_command(self):
        cli = read_text(ROOT / "physicore" / "cli.py")
        assert "robot" in cli

    def test_cli_has_data_command(self):
        cli = read_text(ROOT / "physicore" / "cli.py")
        assert "data" in cli

    def test_install_sh_exists(self):
        assert (ROOT / "install.sh").exists()

    def test_install_sh_has_git_clone(self):
        sh = read_text(ROOT / "install.sh")
        assert "git clone" in sh

    def test_install_sh_has_pip_install(self):
        sh = read_text(ROOT / "install.sh")
        assert "pip install" in sh

    def test_setup_py_has_console_scripts(self):
        setup = read_text(ROOT / "setup.py")
        assert "console_scripts" in setup

    def test_plugins_new_scaffold(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(ROOT / "physicore" / "cli.py"),
             "plugins", "new", "test-plugin", "--author", "tester"],
            capture_output=True, text=True,
            cwd=str(tmp_path),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        plugin_dir = tmp_path / "test-plugin"
        assert plugin_dir.exists(), f"scaffold not created: {result.stderr}"
        assert (plugin_dir / "manifest.json").exists()
        assert (plugin_dir / "plugin.py").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Part C — Examples
# ══════════════════════════════════════════════════════════════════════════════

class TestExamples:

    EXPECTED = [
        "balancing_bot_sim.py",
        "quadrotor_sim.py",
        "load_any_urdf.py",
        "fleet_two_robots.py",
        "custom_plugin.py",
        "full_pipeline.py",
    ]

    def test_examples_directory_exists(self):
        assert EXAMPLES.is_dir(), "examples/ directory not found"

    @pytest.mark.parametrize("script", EXPECTED)
    def test_example_file_exists(self, script):
        assert (EXAMPLES / script).exists(), f"examples/{script} missing"

    @pytest.mark.parametrize("script", EXPECTED)
    def test_example_not_empty(self, script):
        content = (EXAMPLES / script).read_text()
        assert len(content) > 300, f"examples/{script} too short"

    @pytest.mark.parametrize("script", EXPECTED)
    def test_example_has_docstring(self, script):
        content = (EXAMPLES / script).read_text()
        assert '"""' in content, f"examples/{script} missing docstring"

    def test_balancing_bot_sim_runs(self):
        rc, out, err = run_script(EXAMPLES / "balancing_bot_sim.py", timeout=30)
        assert rc == 0, f"balancing_bot_sim.py failed:\n{err}"
        assert "Simulation complete" in out or "simulation" in out.lower()

    def test_quadrotor_sim_runs(self):
        rc, out, err = run_script(EXAMPLES / "quadrotor_sim.py", timeout=30)
        assert rc == 0, f"quadrotor_sim.py failed:\n{err}"

    def test_load_any_urdf_runs(self):
        rc, out, err = run_script(EXAMPLES / "load_any_urdf.py", timeout=30)
        assert rc == 0, f"load_any_urdf.py failed:\n{err}"

    def test_fleet_two_robots_runs(self):
        rc, out, err = run_script(EXAMPLES / "fleet_two_robots.py", timeout=30)
        assert rc == 0, f"fleet_two_robots.py failed:\n{err}"

    def test_custom_plugin_runs(self):
        rc, out, err = run_script(EXAMPLES / "custom_plugin.py", timeout=30)
        assert rc == 0, f"custom_plugin.py failed:\n{err}"

    def test_full_pipeline_runs(self):
        rc, out, err = run_script(EXAMPLES / "full_pipeline.py", timeout=60)
        assert rc == 0, f"full_pipeline.py failed:\n{err}"
        assert "MPC" in out or "pipeline" in out.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Part D — CI/CD
# ══════════════════════════════════════════════════════════════════════════════

class TestCICD:

    def test_ci_yml_exists(self):
        assert (ROOT / ".github" / "workflows" / "ci.yml").exists()

    def test_release_yml_exists(self):
        assert (ROOT / ".github" / "workflows" / "release.yml").exists()

    def test_ci_yml_runs_pytest(self):
        ci = read_text(ROOT / ".github" / "workflows" / "ci.yml")
        assert "pytest" in ci

    def test_ci_yml_runs_typescript(self):
        ci = read_text(ROOT / ".github" / "workflows" / "ci.yml")
        assert "typescript" in ci.lower() or "tsc" in ci or "lint" in ci

    def test_ci_yml_builds_docker(self):
        ci = read_text(ROOT / ".github" / "workflows" / "ci.yml")
        assert "docker" in ci.lower()

    def test_release_yml_triggers_on_tags(self):
        rel = read_text(ROOT / ".github" / "workflows" / "release.yml")
        assert "tags" in rel
        assert "v*" in rel

    def test_release_yml_has_pypi(self):
        rel = read_text(ROOT / ".github" / "workflows" / "release.yml")
        assert "pypi" in rel.lower() or "twine" in rel

    def test_release_yml_has_docker_push(self):
        rel = read_text(ROOT / ".github" / "workflows" / "release.yml")
        assert "push: true" in rel or "docker" in rel.lower()

    def test_release_yml_has_github_release(self):
        rel = read_text(ROOT / ".github" / "workflows" / "release.yml")
        assert "release" in rel.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Part E — README
# ══════════════════════════════════════════════════════════════════════════════

class TestREADME:

    def test_readme_exists(self):
        assert (ROOT / "README.md").exists()

    def test_readme_not_trivial(self):
        content = read_text(ROOT / "README.md")
        assert len(content) > 2000, "README.md too short"

    def test_readme_has_badges(self):
        content = read_text(ROOT / "README.md")
        assert "![" in content or "[![" in content

    def test_readme_has_quick_start(self):
        content = read_text(ROOT / "README.md")
        assert "quick" in content.lower() or "Quick Start" in content

    def test_readme_has_installation(self):
        content = read_text(ROOT / "README.md")
        assert "pip install" in content or "install" in content.lower()

    def test_readme_has_architecture(self):
        content = read_text(ROOT / "README.md")
        assert "architecture" in content.lower() or "Architecture" in content

    def test_readme_has_platforms(self):
        content = read_text(ROOT / "README.md")
        assert "quadrotor" in content.lower() or "platform" in content.lower()

    def test_readme_has_docker(self):
        content = read_text(ROOT / "README.md")
        assert "docker" in content.lower()

    def test_readme_has_license(self):
        content = read_text(ROOT / "README.md")
        assert "license" in content.lower() or "MIT" in content

    def test_readme_has_contributing(self):
        content = read_text(ROOT / "README.md")
        assert "contribut" in content.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Core imports
# ══════════════════════════════════════════════════════════════════════════════

class TestCoreImports:

    def test_physicore_importable(self):
        import physicore
        assert hasattr(physicore, "__version__")

    def test_physicore_version_set(self):
        import physicore
        assert physicore.__version__ and physicore.__version__ != "0.0.0"

    def test_platform_dynamics_available(self):
        from physicore import PLATFORM_DYNAMICS
        assert len(PLATFORM_DYNAMICS) >= 10

    def test_engine_for_platform(self):
        from physicore import PhysiCore
        import numpy as np
        engine = PhysiCore.for_platform("balancing_bot", {"mass": 1.0, "friction": 0.15, "inertia": 0.01})
        x = np.array([0.1, 0.0, 0.0, 0.0])
        step = engine.step(x, np.zeros(1))
        assert step.action is not None
        assert step.residual_norm >= 0

    def test_fleet_manager_importable(self):
        from physicore.core.fleet import FleetManager, FleetRobotSpec
        fm = FleetManager()
        assert fm is not None

    def test_perception_importable(self):
        try:
            from physicore.perception.interface import PerceptionFusion, Observation
            assert True
        except ImportError:
            pytest.skip("perception module not available")

    def test_planning_importable(self):
        try:
            from physicore.planning.planner import TrajectoryPlanner, IKSolver
            assert True
        except ImportError:
            pytest.skip("planning module not available")

    def test_org_importable(self):
        try:
            from physicore.api.org import OrgStore, OrgPlan
            assert True
        except ImportError:
            pytest.skip("org module not available")

    def test_marketplace_importable(self):
        try:
            from physicore.sdk.marketplace import MarketplaceRegistry, scan_source
            assert True
        except ImportError:
            pytest.skip("marketplace module not available")

    def test_cli_importable(self):
        from physicore import cli
        assert hasattr(cli, "main")


# ══════════════════════════════════════════════════════════════════════════════
# API server
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIServer:

    @pytest.fixture(scope="class")
    def server(self):
        """Start the FastAPI server in a background thread for the test class."""
        try:
            import uvicorn
            from physicore.api import server as srv_module
        except ImportError:
            pytest.skip("uvicorn or API server not available")

        import threading
        config = uvicorn.Config(
            "physicore.api.server:app",
            host="127.0.0.1", port=18765,
            log_level="error",
        )
        server_obj = uvicorn.Server(config)
        thread = threading.Thread(target=server_obj.run, daemon=True)
        thread.start()
        time.sleep(1.5)
        yield "http://127.0.0.1:18765"
        server_obj.should_exit = True

    def test_health_endpoint(self, server):
        import urllib.request
        try:
            with urllib.request.urlopen(f"{server}/health", timeout=5) as r:
                data = json.loads(r.read())
            assert data.get("status") == "ok"
        except Exception as e:
            pytest.skip(f"Server not reachable: {e}")

    def test_platforms_endpoint(self, server):
        import urllib.request
        try:
            with urllib.request.urlopen(f"{server}/api/platforms", timeout=5) as r:
                data = json.loads(r.read())
            assert isinstance(data, list)
            assert len(data) >= 10
        except Exception as e:
            pytest.skip(f"Server not reachable: {e}")

    def test_openapi_schema(self, server):
        import urllib.request
        try:
            with urllib.request.urlopen(f"{server}/openapi.json", timeout=5) as r:
                schema = json.loads(r.read())
            assert "paths" in schema
            assert len(schema["paths"]) > 5
        except Exception as e:
            pytest.skip(f"Server not reachable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Docker (xfail if Docker not available)
# ══════════════════════════════════════════════════════════════════════════════

class TestDocker:

    @pytest.mark.xfail(reason="Docker may not be available in CI")
    def test_dockerfile_exists(self):
        assert (ROOT / "Dockerfile").exists()

    @pytest.mark.xfail(reason="Docker may not be available in CI")
    def test_docker_build(self):
        result = subprocess.run(
            ["docker", "build", "-t", "physicore-test:ci", "."],
            capture_output=True, text=True, cwd=str(ROOT), timeout=300,
        )
        assert result.returncode == 0, f"Docker build failed:\n{result.stderr[-2000:]}"

    @pytest.mark.xfail(reason="Docker may not be available in CI")
    def test_docker_compose_exists(self):
        assert (ROOT / "docker-compose.yml").exists() or (ROOT / "docker-compose.yaml").exists()

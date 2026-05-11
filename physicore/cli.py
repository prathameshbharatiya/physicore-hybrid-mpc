#!/usr/bin/env python3
"""
PhysiCore CLI — command-line interface for PhysiCore.

Usage:
    physicore <command> [options]

Commands:
    install             Install PhysiCore and all dependencies
    run                 Run a PhysiCore example or script
    serve               Start the API server
    status              Show system status and loaded robots
    plugins list        List installed plugins
    plugins install     Install a plugin from the marketplace
    plugins new         Scaffold a new plugin
    robot new           Create a new robot YAML config scaffold
    robot load          Load and validate a robot YAML file
    fleet status        Show fleet health and all robot states
    data sessions       List telemetry sessions
    data export         Export telemetry session to CSV/JSON
    version             Print version info
    docs                Open docs in browser
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ── Colour helpers ────────────────────────────────────────────────────────────
_NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")

def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"

def ok(msg):    print(f"  {_c('92', '✓')}  {msg}")
def fail(msg):  print(f"  {_c('91', '✗')}  {msg}"); sys.exit(1)
def info(msg):  print(f"  {_c('96', '→')}  {msg}")
def warn(msg):  print(f"  {_c('93', '!')}  {msg}")
def hdr(msg):   print(f"\n{_c('1', msg)}")
def dim(msg):   print(f"  {_c('90', msg)}")


PHYSICORE_DIR = Path.home() / ".physicore"
REGISTRY_DIR  = PHYSICORE_DIR / "registry"
PLUGINS_DIR   = PHYSICORE_DIR / "plugins"
SESSIONS_DIR  = PHYSICORE_DIR / "sessions"


# ── install ───────────────────────────────────────────────────────────────────

def cmd_install(args):
    hdr("PhysiCore Install")
    script = Path(__file__).parent.parent / "install.py"
    if script.exists():
        subprocess.run([sys.executable, str(script)], check=True)
    else:
        info("Running pip install...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(Path(__file__).parent.parent), "-q"], check=True)
        ok("Installed via pip")


# ── run ───────────────────────────────────────────────────────────────────────

def cmd_run(args):
    script_path = Path(args.script)
    if not script_path.exists():
        # Check examples/
        candidate = Path(__file__).parent.parent / "examples" / args.script
        if not candidate.suffix:
            candidate = candidate.with_suffix(".py")
        if candidate.exists():
            script_path = candidate
        else:
            fail(f"Script not found: {args.script}")

    hdr(f"Running {script_path.name}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)
    result = subprocess.run([sys.executable, str(script_path)], env=env)
    sys.exit(result.returncode)


# ── serve ─────────────────────────────────────────────────────────────────────

def cmd_serve(args):
    host = args.host or os.environ.get("PHYSICORE_HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("PHYSICORE_PORT", "8000"))
    workers = args.workers or 1

    hdr("Starting PhysiCore API Server")
    info(f"Host: {host}:{port}")
    info(f"Workers: {workers}")
    info(f"Docs: http://{host}:{port}/docs")
    print()

    cmd = [
        sys.executable, "-m", "uvicorn",
        "physicore.api.server:app",
        "--host", host,
        "--port", str(port),
        "--workers", str(workers),
    ]
    if args.reload:
        cmd.append("--reload")
    subprocess.run(cmd)


# ── status ────────────────────────────────────────────────────────────────────

def cmd_status(args):
    import urllib.request, urllib.error

    host = args.host or "127.0.0.1"
    port = args.port or 8000
    base = f"http://{host}:{port}"

    hdr("PhysiCore Status")

    try:
        with urllib.request.urlopen(f"{base}/health", timeout=3) as r:
            data = json.loads(r.read())
        ok(f"API server reachable at {base}")
        for k, v in data.items():
            dim(f"{k}: {v}")
    except Exception as e:
        warn(f"API server not reachable ({e})")
        info("Start with: physicore serve")

    # Local info
    print()
    hdr("Local Environment")
    try:
        from physicore import __version__, PLATFORM_DYNAMICS
        ok(f"PhysiCore {__version__} installed")
        info(f"Platforms: {', '.join(list(PLATFORM_DYNAMICS.keys())[:6])}...")
    except ImportError as e:
        warn(f"PhysiCore not importable: {e}")

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    ok(f"Registry: {REGISTRY_DIR}")

    plugin_count = len(list(PLUGINS_DIR.glob("*.physicore-plugin"))) if PLUGINS_DIR.exists() else 0
    info(f"Installed plugins: {plugin_count}")

    session_count = len(list(SESSIONS_DIR.glob("*.json"))) if SESSIONS_DIR.exists() else 0
    info(f"Saved sessions: {session_count}")


# ── plugins list ──────────────────────────────────────────────────────────────

def cmd_plugins_list(args):
    hdr("Installed Plugins")
    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    plugins = list(PLUGINS_DIR.glob("*.physicore-plugin"))
    if not plugins:
        dim("No plugins installed.")
        info("Browse the marketplace: physicore plugins install <name>")
        return
    for p in plugins:
        size_kb = p.stat().st_size // 1024
        print(f"  {_c('96', p.stem):<40} {size_kb:>6} KB")


# ── plugins install ───────────────────────────────────────────────────────────

def cmd_plugins_install(args):
    hdr(f"Installing Plugin: {args.name}")
    try:
        import urllib.request, urllib.error
        host = args.host or "127.0.0.1"
        port = args.port or 8000
        url = f"http://{host}:{port}/api/marketplace/{args.name}/install"
        req = urllib.request.Request(url, method="POST")
        req.add_header("X-User-Id", args.user or "cli-user")
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        ok(f"Plugin installed: {data.get('plugin_id', args.name)}")
        info(f"Location: {data.get('install_path', PLUGINS_DIR)}")
    except Exception as e:
        fail(f"Install failed: {e}")


# ── plugins new ───────────────────────────────────────────────────────────────

def cmd_plugins_new(args):
    plugin_dir = Path(args.name)
    if plugin_dir.exists():
        fail(f"Directory already exists: {plugin_dir}")

    hdr(f"Scaffolding Plugin: {args.name}")
    plugin_dir.mkdir(parents=True)

    manifest = {
        "id": args.name,
        "name": args.name.replace("-", " ").title(),
        "version": "0.1.0",
        "description": "A PhysiCore plugin",
        "category": "perception",
        "author": args.author or "unknown",
        "entry_point": "plugin.py",
        "dependencies": [],
    }
    (plugin_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    (plugin_dir / "plugin.py").write_text(textwrap.dedent(f'''\
        """
        {manifest["name"]} — PhysiCore plugin
        """
        from physicore.sdk.plugin_loader import PhysicorePlugin, PluginMeta

        class {args.name.replace("-", "_").title().replace("_", "")}Plugin(PhysicorePlugin):
            """Custom PhysiCore plugin."""

            @property
            def meta(self) -> PluginMeta:
                return PluginMeta(
                    id="{args.name}",
                    name="{manifest["name"]}",
                    version="0.1.0",
                    description="A PhysiCore plugin",
                    author="{manifest.get("author", "unknown")}",
                )

            def on_load(self):
                print(f"[{manifest["name"]}] loaded")

            def on_step(self, state, action, dt: float):
                # Process state/action here
                return {{}}

            def on_unload(self):
                print(f"[{manifest["name"]}] unloaded")
        '''))

    (plugin_dir / "README.md").write_text(f"# {manifest['name']}\n\nA PhysiCore plugin.\n\n## Usage\n\n```python\nfrom plugin import {args.name.replace('-','_').title().replace('_','')}Plugin\n```\n")

    ok(f"Scaffold created: {plugin_dir}/")
    info("Files: manifest.json, plugin.py, README.md")
    info(f"Test: cd {args.name} && physicore run plugin.py")


# ── robot new ─────────────────────────────────────────────────────────────────

def cmd_robot_new(args):
    hdr(f"Scaffolding Robot Config: {args.name}")
    platform = args.platform or "balancing_bot"

    yaml_content = textwrap.dedent(f"""\
        # PhysiCore robot config — {args.name}
        robot:
          name: {args.name}
          platform: {platform}
          urdf: null  # path/to/robot.urdf or null for built-in dynamics

        physics:
          mass: 1.0
          friction: 0.15
          inertia: 0.01

        control:
          state_dim: 4
          action_dim: 1
          dt: 0.02
          horizon: 20

        mpc:
          optimizer: cem
          samples: 512
          elites: 64
          iterations: 5

        bridge:
          type: sim   # sim | serial | mavlink | ros2
          connection: null

        plugins: []
        """)

    out = Path(f"{args.name}.yaml")
    if out.exists() and not args.force:
        fail(f"{out} already exists. Use --force to overwrite.")
    out.write_text(yaml_content)
    ok(f"Config written: {out}")
    info(f"Load with: physicore robot load {out}")


# ── robot load ────────────────────────────────────────────────────────────────

def cmd_robot_load(args):
    hdr(f"Loading Robot: {args.file}")
    config_path = Path(args.file)
    if not config_path.exists():
        fail(f"File not found: {config_path}")

    try:
        import yaml  # type: ignore
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        ok("YAML parsed successfully")
    except ImportError:
        import json as _json
        warn("pyyaml not installed, trying JSON parse...")
        try:
            with open(config_path) as f:
                cfg = _json.load(f)
            ok("JSON parsed successfully")
        except Exception as e:
            fail(f"Parse failed: {e}")
    except Exception as e:
        fail(f"YAML parse error: {e}")

    robot = cfg.get("robot", {})
    info(f"Name: {robot.get('name', '?')}")
    info(f"Platform: {robot.get('platform', '?')}")
    info(f"URDF: {robot.get('urdf', 'none')}")

    if robot.get("urdf"):
        urdf_path = Path(robot["urdf"])
        if not urdf_path.exists():
            warn(f"URDF not found: {urdf_path}")
        else:
            try:
                from physicore.core.urdf_loader import load_robot
                model = load_robot(str(urdf_path))
                ok(f"URDF loaded: {model.robot_name}, {model.n_joints} joints")
            except Exception as e:
                warn(f"URDF load error: {e}")

    ok("Robot config valid")


# ── fleet status ──────────────────────────────────────────────────────────────

def cmd_fleet_status(args):
    import urllib.request, urllib.error

    host = args.host or "127.0.0.1"
    port = args.port or 8000
    base = f"http://{host}:{port}"

    hdr("Fleet Status")
    try:
        with urllib.request.urlopen(f"{base}/api/fleet/health", timeout=3) as r:
            data = json.loads(r.read())

        robots = data if isinstance(data, list) else data.get("robots", [])
        if not robots:
            dim("No robots in fleet.")
            return
        print(f"\n  {'ID':<20} {'Platform':<18} {'Status':<12} {'Residual':>10}")
        print(f"  {'-'*20} {'-'*18} {'-'*12} {'-'*10}")
        for r in robots:
            rid  = str(r.get("robot_id", r.get("id", "?")))[:18]
            plat = str(r.get("platform", "?"))[:16]
            stat = str(r.get("status", "?"))
            res  = r.get("residual_norm", r.get("residual", 0))
            color = "92" if stat == "healthy" else "91"
            print(f"  {rid:<20} {plat:<18} {_c(color, stat):<12} {res:>10.4f}")
    except Exception as e:
        warn(f"Fleet API not reachable: {e}")
        info("Start server first: physicore serve")


# ── data sessions ─────────────────────────────────────────────────────────────

def cmd_data_sessions(args):
    import urllib.request, urllib.error

    host = args.host or "127.0.0.1"
    port = args.port or 8000
    base = f"http://{host}:{port}"

    hdr("Telemetry Sessions")
    try:
        with urllib.request.urlopen(f"{base}/api/telemetry/sessions", timeout=3) as r:
            sessions = json.loads(r.read())
        if not sessions:
            dim("No sessions recorded yet.")
            return
        print(f"\n  {'Session ID':<36} {'Robot':<20} {'Steps':>8} {'Duration':>10}")
        print(f"  {'-'*36} {'-'*20} {'-'*8} {'-'*10}")
        for s in sessions[:args.limit or 20]:
            sid      = str(s.get("session_id", "?"))[:34]
            robot    = str(s.get("robot_id", "?"))[:18]
            steps    = s.get("step_count", 0)
            duration = s.get("duration_s", 0)
            print(f"  {sid:<36} {robot:<20} {steps:>8} {duration:>9.1f}s")
    except Exception as e:
        warn(f"Telemetry API not reachable: {e}")
        info("Start server first: physicore serve")


# ── data export ───────────────────────────────────────────────────────────────

def cmd_data_export(args):
    import urllib.request, urllib.error

    host  = args.host or "127.0.0.1"
    port  = args.port or 8000
    base  = f"http://{host}:{port}"
    fmt   = args.format or "csv"
    out   = args.output or f"{args.session_id}.{fmt}"

    hdr(f"Exporting Session: {args.session_id}")
    try:
        url = f"{base}/api/telemetry/sessions/{args.session_id}/export?format={fmt}"
        with urllib.request.urlopen(url, timeout=10) as r:
            content = r.read()
        Path(out).write_bytes(content)
        ok(f"Exported to {out} ({len(content):,} bytes)")
    except Exception as e:
        fail(f"Export failed: {e}")


# ── version ───────────────────────────────────────────────────────────────────

def cmd_version(args):
    try:
        from physicore import __version__
        print(f"PhysiCore {__version__}")
    except ImportError:
        print("PhysiCore (version unknown)")
    print(f"Python {sys.version.split()[0]}")


# ── docs ──────────────────────────────────────────────────────────────────────

def cmd_docs(args):
    docs_dir = Path(__file__).parent.parent / "docs"
    page = args.page or "index"
    html_file = docs_dir / f"{page}.html"

    if html_file.exists():
        import webbrowser
        webbrowser.open(html_file.as_uri())
        ok(f"Opened {html_file}")
    else:
        info("Docs not found locally. Opening online docs...")
        import webbrowser
        webbrowser.open("https://github.com/prathameshbharatiya/physicore-hybrid-mpc#readme")


# ── Argument parsing ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="physicore",
        description=_c("1", "PhysiCore — Hybrid Uncertainty-Aware Sim-to-Real Engine"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              physicore serve --port 8000
              physicore run examples/balancing_bot_sim.py
              physicore status
              physicore robot new my_robot --platform quadrotor
              physicore plugins new my-plugin --author alice
              physicore fleet status
              physicore data sessions
              physicore data export <session-id> --format csv
        """),
    )

    sub = p.add_subparsers(dest="command", metavar="<command>")

    # install
    sub.add_parser("install", help="Install PhysiCore and all dependencies")

    # run
    run_p = sub.add_parser("run", help="Run a script or example")
    run_p.add_argument("script", help="Script path or example name")

    # serve
    srv = sub.add_parser("serve", help="Start the API server")
    srv.add_argument("--host", default=None)
    srv.add_argument("--port", type=int, default=None)
    srv.add_argument("--workers", type=int, default=1)
    srv.add_argument("--reload", action="store_true", help="Auto-reload on code changes")

    # status
    sta = sub.add_parser("status", help="Show system status")
    sta.add_argument("--host", default=None)
    sta.add_argument("--port", type=int, default=None)

    # plugins
    pl = sub.add_parser("plugins", help="Manage plugins")
    pl_sub = pl.add_subparsers(dest="subcommand", metavar="<subcommand>")

    pl_sub.add_parser("list", help="List installed plugins")

    pl_inst = pl_sub.add_parser("install", help="Install a plugin")
    pl_inst.add_argument("name")
    pl_inst.add_argument("--host", default=None)
    pl_inst.add_argument("--port", type=int, default=None)
    pl_inst.add_argument("--user", default=None)

    pl_new = pl_sub.add_parser("new", help="Scaffold a new plugin")
    pl_new.add_argument("name")
    pl_new.add_argument("--author", default=None)

    # robot
    rb = sub.add_parser("robot", help="Manage robot configs")
    rb_sub = rb.add_subparsers(dest="subcommand", metavar="<subcommand>")

    rb_new = rb_sub.add_parser("new", help="Create a robot config scaffold")
    rb_new.add_argument("name")
    rb_new.add_argument("--platform", default="balancing_bot")
    rb_new.add_argument("--force", action="store_true")

    rb_load = rb_sub.add_parser("load", help="Load and validate a robot config")
    rb_load.add_argument("file")

    # fleet
    fl = sub.add_parser("fleet", help="Fleet management")
    fl_sub = fl.add_subparsers(dest="subcommand", metavar="<subcommand>")
    fl_sta = fl_sub.add_parser("status", help="Show fleet status")
    fl_sta.add_argument("--host", default=None)
    fl_sta.add_argument("--port", type=int, default=None)

    # data
    da = sub.add_parser("data", help="Telemetry data tools")
    da_sub = da.add_subparsers(dest="subcommand", metavar="<subcommand>")

    da_ses = da_sub.add_parser("sessions", help="List sessions")
    da_ses.add_argument("--host", default=None)
    da_ses.add_argument("--port", type=int, default=None)
    da_ses.add_argument("--limit", type=int, default=20)

    da_exp = da_sub.add_parser("export", help="Export session")
    da_exp.add_argument("session_id")
    da_exp.add_argument("--format", choices=["csv", "json"], default="csv")
    da_exp.add_argument("--output", default=None)
    da_exp.add_argument("--host", default=None)
    da_exp.add_argument("--port", type=int, default=None)

    # version
    sub.add_parser("version", help="Print version info")

    # docs
    doc_p = sub.add_parser("docs", help="Open documentation")
    doc_p.add_argument("page", nargs="?", default="index")

    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "install": cmd_install,
        "run":     cmd_run,
        "serve":   cmd_serve,
        "status":  cmd_status,
        "version": cmd_version,
        "docs":    cmd_docs,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
        return

    if args.command == "plugins":
        if not hasattr(args, "subcommand") or args.subcommand is None:
            parser.parse_args(["plugins", "--help"])
        elif args.subcommand == "list":
            cmd_plugins_list(args)
        elif args.subcommand == "install":
            cmd_plugins_install(args)
        elif args.subcommand == "new":
            cmd_plugins_new(args)

    elif args.command == "robot":
        if not hasattr(args, "subcommand") or args.subcommand is None:
            parser.parse_args(["robot", "--help"])
        elif args.subcommand == "new":
            cmd_robot_new(args)
        elif args.subcommand == "load":
            cmd_robot_load(args)

    elif args.command == "fleet":
        if not hasattr(args, "subcommand") or args.subcommand is None:
            parser.parse_args(["fleet", "--help"])
        elif args.subcommand == "status":
            cmd_fleet_status(args)

    elif args.command == "data":
        if not hasattr(args, "subcommand") or args.subcommand is None:
            parser.parse_args(["data", "--help"])
        elif args.subcommand == "sessions":
            cmd_data_sessions(args)
        elif args.subcommand == "export":
            cmd_data_export(args)


if __name__ == "__main__":
    main()

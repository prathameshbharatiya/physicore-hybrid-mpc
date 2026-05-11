"""
physicore/sdk/plugin_template.py — Plugin scaffold generator + CLI

Usage:
    python -m physicore.sdk.plugin_template my_plugin --author "Jane Doe" --out ./plugins
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_MAIN_PY_TEMPLATE = '''\
"""
{name} — PhysiCore Plugin
"""
from physicore.extensions import PhysiCoreExtension, ExtensionMeta

PLUGIN_MANIFEST = {{
    "plugin_id":   "{plugin_id}",
    "name":        "{name}",
    "version":     "0.1.0",
    "description": "{description}",
    "author":      "{author}",
    "entry_point": "main",
    "permissions": ["read_state", "read_telemetry"],
    "hooks":       ["post_step"],
    "tags":        [],
    "panels": [
        {{
            "panel_id":      "main_panel",
            "title":         "{name} Panel",
            "chart_type":    "line",
            "data_endpoint": "main_panel/data",
            "refresh_hz":    2.0,
            "position":      {{"row": 0, "col": 0, "w": 6, "h": 4}}
        }}
    ]
}}


class {class_name}(PhysiCoreExtension):
    meta = ExtensionMeta(
        name        = "{name}",
        version     = "0.1.0",
        description = "{description}",
        author      = "{author}",
        hooks       = ["post_step"],
    )

    def __init__(self):
        self._history: list = []

    def setup(self, engine) -> None:
        pass

    def teardown(self) -> None:
        self._history.clear()

    def post_step(self, step, engine) -> None:
        self._history.append({{
            "time":   getattr(step, "time", 0.0),
            "effort": float(getattr(step, "control_effort", 0.0)),
        }})
        if len(self._history) > 200:
            self._history = self._history[-200:]

    def get_panel_data(self, panel_id: str):
        if panel_id == "main_panel":
            return {{
                "series": [
                    {{
                        "name":   "effort",
                        "points": self._history[-50:],
                    }}
                ]
            }}
        return {{}}


# Module-level instance so PluginLoader can find it by entry_point name
main = {class_name}
'''

_PLUGIN_JSON_TEMPLATE = {
    "plugin_id":   "{plugin_id}",
    "name":        "{name}",
    "version":     "0.1.0",
    "description": "{description}",
    "author":      "{author}",
    "entry_point": "main",
    "permissions": ["read_state", "read_telemetry"],
    "hooks":       ["post_step"],
    "tags":        [],
    "panels": [
        {
            "panel_id":      "main_panel",
            "title":         "{name} Panel",
            "chart_type":    "line",
            "data_endpoint": "main_panel/data",
            "refresh_hz":    2.0,
            "position":      {"row": 0, "col": 0, "w": 6, "h": 4},
        }
    ],
}


def _to_class_name(plugin_id: str) -> str:
    return "".join(part.title() for part in plugin_id.replace("-", "_").split("_")) + "Plugin"


def generate_plugin(
    plugin_id:   str,
    name:        str        = "",
    description: str        = "",
    author:      str        = "",
    out_dir:     Path       = Path("plugins"),
    as_single_file: bool    = False,
) -> Path:
    """
    Generate a plugin scaffold. Returns the path to the created file/directory.

    If as_single_file=True, creates a single .py file with PLUGIN_MANIFEST.
    Otherwise, creates a directory with plugin.json + main.py.
    """
    name        = name        or plugin_id.replace("_", " ").title()
    description = description or f"A PhysiCore plugin: {name}"
    author      = author      or "Unknown"
    class_name  = _to_class_name(plugin_id)

    ctx = dict(
        plugin_id   = plugin_id,
        name        = name,
        description = description,
        author      = author,
        class_name  = class_name,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if as_single_file:
        out_file = out_dir / f"{plugin_id}.py"
        out_file.write_text(_MAIN_PY_TEMPLATE.format(**ctx), encoding="utf-8")
        print(f"[plugin_template] Created: {out_file}")
        return out_file

    plugin_dir = out_dir / plugin_id
    plugin_dir.mkdir(exist_ok=True)

    # plugin.json
    raw_json = {k: (v.format(**ctx) if isinstance(v, str) else v)
                for k, v in _PLUGIN_JSON_TEMPLATE.items()}
    raw_json["panels"] = [
        {
            "panel_id":      "main_panel",
            "title":         f"{name} Panel",
            "chart_type":    "line",
            "data_endpoint": "main_panel/data",
            "refresh_hz":    2.0,
            "position":      {"row": 0, "col": 0, "w": 6, "h": 4},
        }
    ]
    (plugin_dir / "plugin.json").write_text(
        json.dumps(raw_json, indent=2), encoding="utf-8"
    )

    # main.py  (without PLUGIN_MANIFEST — manifest comes from plugin.json)
    main_py = '''\
"""
{name} — PhysiCore Plugin
"""
from physicore.extensions import PhysiCoreExtension, ExtensionMeta


class {class_name}(PhysiCoreExtension):
    meta = ExtensionMeta(
        name        = "{name}",
        version     = "0.1.0",
        description = "{description}",
        author      = "{author}",
        hooks       = ["post_step"],
    )

    def __init__(self):
        self._history: list = []

    def setup(self, engine) -> None:
        pass

    def teardown(self) -> None:
        self._history.clear()

    def post_step(self, step, engine) -> None:
        self._history.append({{
            "time":   getattr(step, "time", 0.0),
            "effort": float(getattr(step, "control_effort", 0.0)),
        }})
        if len(self._history) > 200:
            self._history = self._history[-200:]

    def get_panel_data(self, panel_id: str):
        if panel_id == "main_panel":
            return {{
                "series": [
                    {{
                        "name":   "effort",
                        "points": self._history[-50:],
                    }}
                ]
            }}
        return {{}}


main = {class_name}
'''.format(**ctx)

    (plugin_dir / "main.py").write_text(main_py, encoding="utf-8")
    (plugin_dir / "__init__.py").write_text("", encoding="utf-8")

    print(f"[plugin_template] Created directory plugin: {plugin_dir}")
    return plugin_dir


# ── CLI entry point ───────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        prog="python -m physicore.sdk.plugin_template",
        description="Scaffold a new PhysiCore plugin",
    )
    parser.add_argument("plugin_id",              help="Snake_case plugin identifier")
    parser.add_argument("--name",        default="", help="Human-readable name")
    parser.add_argument("--description", default="", help="Short description")
    parser.add_argument("--author",      default="", help="Author name")
    parser.add_argument("--out",         default="plugins", help="Output directory")
    parser.add_argument(
        "--single-file", action="store_true",
        help="Generate a single .py file instead of a directory"
    )
    args = parser.parse_args()

    generate_plugin(
        plugin_id      = args.plugin_id,
        name           = args.name,
        description    = args.description,
        author         = args.author,
        out_dir        = Path(args.out),
        as_single_file = args.single_file,
    )


if __name__ == "__main__":
    _main()

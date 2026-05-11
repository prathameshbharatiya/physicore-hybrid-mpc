"""
PhysiCore SDK
=============
from physicore.sdk import PhysicoreClient, PhysicoreSimulator, PhysicoreAnalyzer
from physicore.sdk import PluginManifest, PluginLoader, PluginAPIRouter
"""

from .client          import PhysicoreClient
from .simulate        import PhysicoreSimulator
from .analyze         import PhysicoreAnalyzer
from .plugin_manifest import PluginManifest, DashboardPanelSpec, validate_manifest
from .plugin_loader   import PluginLoader, PluginSandbox
from .plugin_router   import PluginAPIRouter, build_plugin_router
from .plugin_template import generate_plugin

__all__ = [
    "PhysicoreClient", "PhysicoreSimulator", "PhysicoreAnalyzer",
    "PluginManifest", "DashboardPanelSpec", "validate_manifest",
    "PluginLoader", "PluginSandbox",
    "PluginAPIRouter", "build_plugin_router",
    "generate_plugin",
]

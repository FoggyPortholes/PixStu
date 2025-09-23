"""Plugin discovery helpers for Mini Travy."""

import importlib
import importlib.metadata
import pathlib
import sys
from typing import List, Tuple

from .api import CLIPlugin, Context, HookPlugin


def load_entrypoint_plugins() -> list:
    """Load plugins registered via ``minitravy.plugins`` entry points."""

    plugins = []
    try:
        for entry in importlib.metadata.entry_points(group="minitravy.plugins"):
            try:
                plugins.append(entry.load())
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] failed to load plugin entry point {entry.name}: {exc}")
    except Exception:
        # Entry points not available on some interpreters (e.g. standalone).
        pass
    return plugins


def load_local_plugins(root: str = "plugins") -> list:
    """Import plugin modules from a local ``plugins`` folder."""

    plugins = []
    base = pathlib.Path(root)
    if not base.exists():
        return plugins

    sys.path.insert(0, str(base.resolve()))
    for pkg in base.iterdir():
        if not (pkg / "plugin.py").exists():
            continue
        try:
            plugins.append(importlib.import_module(f"{pkg.name}.plugin"))
        except Exception as exc:
            print(f"[warn] failed to import plugin {pkg.name}: {exc}")
    return plugins


def collect(ctx: Context) -> Tuple[List[CLIPlugin], List[HookPlugin]]:
    """Collect CLI and hook plugins from entry points and local directories."""

    modules = load_entrypoint_plugins() + load_local_plugins()
    cli_plugins: List[CLIPlugin] = []
    hook_plugins: List[HookPlugin] = []
    for module in modules:
        plugin = getattr(module, "PLUGIN", None)
        if plugin is None:
            continue
        if hasattr(plugin, "register_subcommands") and hasattr(plugin, "handle"):
            cli_plugins.append(plugin)
        if any(hasattr(plugin, name) for name in ("on_pack_pre", "on_pack_post", "on_validate_post")):
            hook_plugins.append(plugin)
    return cli_plugins, hook_plugins

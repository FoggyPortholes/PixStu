"""Public API definitions for Mini Travy plugins."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol

API_VERSION = "1.0"


@dataclass
class Context:
    """Execution context shared with plugins."""

    api_version: str = API_VERSION
    config: Dict[str, Any] = None
    log: Callable[[str], None] = print


class CLIPlugin(Protocol):
    """Protocol for command-line plugins."""

    name: str

    def register_subcommands(self, subparsers) -> None:
        """Allow the plugin to register CLI subcommands."""

    def handle(self, args, ctx: Context) -> bool:
        """Handle parsed arguments, returning ``True`` if processed."""


class HookPlugin(Protocol):
    """Protocol for lifecycle hook plugins."""

    name: str

    def on_pack_pre(self, frames: List[str], args, ctx: Context) -> List[str]:
        """Return the frame list to pack just before packing begins."""

    def on_pack_post(self, sheet_path: str, args, ctx: Context) -> None:
        """Run after the sheet has been saved."""

    def on_validate_post(self, report: Dict[str, Any], args, ctx: Context) -> Dict[str, Any]:
        """Run after validation is complete, returning an updated report."""

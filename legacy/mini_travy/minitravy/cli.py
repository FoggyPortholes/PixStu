"""Command-line entry point for Mini Travy using the plugin system."""

import argparse

from .api import Context
from .plugin_manager import collect


def build_cli():
    parser = argparse.ArgumentParser("minitravy")
    subparsers = parser.add_subparsers(dest="cmd")
    return parser, subparsers


def main(argv=None):
    ctx = Context()
    parser, subparsers = build_cli()
    cli_plugins, _ = collect(ctx)

    for plugin in cli_plugins:
        try:
            plugin.register_subcommands(subparsers)
        except Exception as exc:
            ctx.log(f"[warn] register_subcommands failed in {getattr(plugin, 'name', '?')}: {exc}")

    args = parser.parse_args(argv)

    for plugin in cli_plugins:
        try:
            if plugin.handle(args, ctx):
                return
        except Exception as exc:
            ctx.log(f"[error] plugin {getattr(plugin, 'name', '?')} failed: {exc}")

    parser.print_help()


if __name__ == "__main__":
    main()

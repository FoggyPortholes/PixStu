"""Command line helpers for inspecting and maintaining the PixStu cache."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cache import Cache


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def _build_cache(args: argparse.Namespace) -> Cache:
    root = Path(args.root) if args.root else None
    return Cache(
        namespace=args.namespace,
        root=root,
        ttl=args.ttl,
        max_entries=args.max_entries,
        max_bytes=args.max_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PixStu cache utility")
    parser.add_argument("command", choices=["list", "clear", "prune", "stats"], help="Action to execute")
    parser.add_argument("--namespace", default="default", help="Cache namespace (default: default)")
    parser.add_argument("--root", help="Override cache root directory")
    parser.add_argument("--ttl", type=int, help="Override TTL in seconds")
    parser.add_argument("--max-entries", type=int, help="Override maximum number of entries")
    parser.add_argument("--max-bytes", type=int, help="Override cache size in bytes")

    args = parser.parse_args(argv)
    cache = _build_cache(args)

    if args.command == "clear":
        cache.clear()
        print("Cache cleared")
        return 0

    if args.command == "prune":
        cache.prune()
        print("Cache pruned")
        return 0

    entries = cache.index()

    if args.command == "list":
        if not entries:
            print("Cache empty")
            return 0
        for entry in entries:
            print(
                f"{entry.key}\n  path: {entry.path}\n  size: {_format_bytes(entry.size)}\n"
                f"  created: {entry.created:.0f}\n  last_access: {entry.last_access:.0f}"
            )
        return 0

    if args.command == "stats":
        total = sum(entry.size for entry in entries)
        print(f"Entries: {len(entries)}")
        print(f"Size: {_format_bytes(total)}")
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

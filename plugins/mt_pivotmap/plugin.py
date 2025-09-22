"""Hook plugin that loads per-frame pivot metadata before packing."""

import json
import os

from minitravy.api import Context, HookPlugin


class PivotMap:
    """Inject a ``pivot_map`` into the shared context if present."""

    name = "mt_pivotmap"

    def on_pack_pre(self, frames, args, ctx: Context):
        path = getattr(args, "pivot_map", None) or getattr(args, "pivotmap", None)
        if path and os.path.exists(path):
            ctx.config = ctx.config or {}
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    ctx.config["pivot_map"] = json.load(handle)
            except Exception as exc:
                ctx.log(f"[pivotmap] failed to read {path}: {exc}")
        return frames


PLUGIN: HookPlugin = PivotMap()

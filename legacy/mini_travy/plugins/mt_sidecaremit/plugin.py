"""Hook plugin that emits JSON metadata after packing sprite sheets."""

import json
import pathlib
import time

from minitravy.api import Context, HookPlugin


class SidecarEmit:
    """Write a JSON sidecar describing the generated sheet."""

    name = "mt_sidecaremit"

    def on_pack_post(self, sheet_path: str, args, ctx: Context) -> None:
        root = pathlib.Path(sheet_path)
        meta = {
            "sheet_name": root.stem,
            "tile_size": list(args.tile) if hasattr(args, "tile") else None,
            "grid": list(args.sheet) if hasattr(args, "sheet") else None,
            "pivot_norm": list(args.pivot) if hasattr(args, "pivot") else None,
            "generated_at": int(time.time()),
        }
        out_path = root.with_suffix(".json")
        try:
            out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            ctx.log(f"[sidecar] wrote {out_path}")
        except Exception as exc:
            ctx.log(f"[sidecar] failed to write {out_path}: {exc}")


PLUGIN: HookPlugin = SidecarEmit()

"""CLI plugin that explodes sprite sheets into individual frames."""

import os

from PIL import Image

from minitravy.api import CLIPlugin, Context


class SliceCLI:
    """Provide the ``slice`` command to break sheets into frames."""

    name = "mt_slice"

    def register_subcommands(self, subparsers) -> None:
        parser = subparsers.add_parser("slice", help="Explode a sheet into frames")
        parser.add_argument("image")
        parser.add_argument("--tile", type=int, nargs=2, default=[16, 16])
        parser.add_argument("--sheet", type=int, nargs=2, required=True)
        parser.add_argument("-o", "--outdir", default="frames")

    def handle(self, args, ctx: Context) -> bool:
        if getattr(args, "cmd", "") != "slice":
            return False

        img = Image.open(args.image).convert("RGBA")
        tile_w, tile_h = args.tile
        rows, cols = args.sheet
        os.makedirs(args.outdir, exist_ok=True)
        for r in range(rows):
            for c in range(cols):
                box = (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
                frame = img.crop(box)
                frame.save(os.path.join(args.outdir, f"r{r}_c{c}.png"))
        ctx.log(f"[slice] wrote frames to {args.outdir}")
        return True


PLUGIN: CLIPlugin = SliceCLI()

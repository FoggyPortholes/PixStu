"""Lightweight integration checks for PixStu core workflows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from chargen import inpaint, lora_blend, studio
from chargen.reference_gallery import cleanup_gallery, list_gallery


class _DummyPipeline:
    def __call__(self, *, prompt, image, mask_image, guidance_scale, num_inference_steps):
        del prompt, guidance_scale, num_inference_steps
        base = image.copy()
        overlay = Image.new("RGB", base.size, (255, 0, 0))
        base.paste(overlay, mask=mask_image.convert("L"))
        return SimpleNamespace(images=[base])


def _test_inpaint_and_gallery(tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cleanup_gallery()

    original_pipeline = getattr(inpaint, "_PIPELINE", None)
    original_device = getattr(inpaint, "_PIPELINE_DEVICE", None)
    inpaint._PIPELINE = _DummyPipeline()
    inpaint._PIPELINE_DEVICE = "cpu"
    if hasattr(inpaint, "_CACHE"):
        inpaint._CACHE.clear()

    try:
        base = Image.new("RGB", (32, 32), color="blue")
        mask = Image.new("L", (32, 32), color=0)
        mask.paste(255, (8, 8, 24, 24))

        result = inpaint.inpaint_region(base, mask, prompt="integration")

        gallery_listing = studio._save_output_to_gallery(result, "integration")
        assert gallery_listing, "Gallery should contain the saved result"
        saved_path = Path(gallery_listing[0])
        assert saved_path.exists(), "Saved gallery image missing"
        with Image.open(saved_path) as handle:
            sample = handle.copy()
        assert sample.size == (32, 32)
        loaded_gallery = list_gallery()
        assert saved_path.as_posix() in loaded_gallery
    finally:
        inpaint._PIPELINE = original_pipeline
        inpaint._PIPELINE_DEVICE = original_device


def _test_lora_blend_repro(tmp_dir: Path) -> None:
    blend_name = "integration"
    lora_blend.delete_set(blend_name)
    rows = [[tmp_dir / "a.safetensors", 0.75], [tmp_dir / "b.safetensors", 0.25]]
    lora_blend.save_set(blend_name, rows)
    loaded = lora_blend.blend_to_rows(lora_blend.get_set(blend_name))

    preset = {
        "loras": [
            {"path": str(tmp_dir / "a.safetensors"), "weight": 0.1},
            {"path": str(tmp_dir / "c.safetensors"), "weight": 0.9},
        ]
    }

    first = deepcopy(preset)
    second = deepcopy(preset)
    lora_blend.apply_blend(first, rows)
    lora_blend.apply_blend(second, loaded)
    assert first == second
    lora_blend.delete_set(blend_name)


def main() -> int:
    tmp_dir = Path(".pixstu") / "integration_tmp"
    gallery_dir = tmp_dir / "gallery"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    from chargen import reference_gallery

    original_gallery = Path(reference_gallery.GALLERY_PATH)
    reference_gallery.GALLERY_PATH = gallery_dir
    try:
        _test_inpaint_and_gallery(gallery_dir)
        _test_lora_blend_repro(tmp_dir)
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        reference_gallery.GALLERY_PATH = original_gallery
    print("Integration checks passed")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())

"""Standalone Gradio UI for building mapped sprite sheets from a single sprite."""
from __future__ import annotations

import os
from typing import List

import gradio as gr

from .builder import build_sprite_sheet, list_presets

PRESETS = list_presets()
PRESET_CHOICES = [preset.name for preset in PRESETS]
PRESET_DESCRIPTIONS = {preset.name: preset.description for preset in PRESETS}
DEFAULT_PRESET = PRESET_CHOICES[0] if PRESET_CHOICES else ""


def _resolve_background(choice: str, custom: str | None) -> str:
    custom = (custom or "").strip()
    if custom:
        return custom
    return choice or "transparent"


def _build_sheet(
    sprite_file,
    preset: str,
    tile_size: int,
    padding: int,
    background_choice: str,
    custom_background: str,
):
    if sprite_file is None:
        return None, None, None, [], "Upload a sprite to continue."

    try:
        sprite_path = sprite_file.name if hasattr(sprite_file, "name") else sprite_file
    except AttributeError:
        sprite_path = sprite_file

    try:
        sheet_path, mapping_path, zip_path, frames, mapping = build_sprite_sheet(
            sprite_path,
            preset_name=preset,
            tile_size=int(tile_size),
            padding=int(padding),
            background=_resolve_background(background_choice, custom_background),
        )
    except Exception as exc:  # pragma: no cover - surfaced to UI
        return None, None, None, [], f"Error: {exc}"

    message = (
        f"Created sprite sheet with {len(mapping['frames'])} frames using preset '{mapping['preset']}'."
        f"\nSheet: {os.path.basename(sheet_path)} | Mapping: {os.path.basename(mapping_path)}"
    )
    return sheet_path, mapping_path, zip_path, frames, message


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("## Sprite Sheet Studio")
    gr.Markdown(
        "Import a single sprite and automatically build a mapped sheet using the predefined layouts."
    )

    with gr.Row():
        sprite_input = gr.File(label="Source Sprite", file_types=[".png", ".webp", ".bmp"], type="filepath")
        preset_dropdown = gr.Dropdown(
            PRESET_CHOICES,
            label="Preset layout",
            value=DEFAULT_PRESET,
            info="Choose how frames are arranged and which variants are generated.",
        )
        tile_slider = gr.Slider(
            32,
            256,
            value=128,
            step=8,
            label="Tile size (px)",
            info="Each cell in the sheet will be square with this dimension.",
        )
        padding_slider = gr.Slider(0, 32, value=4, step=1, label="Padding (px)")

    with gr.Row():
        background_choice = gr.Radio(
            ["transparent", "black", "white", "gray"],
            value="transparent",
            label="Background",
        )
        custom_background = gr.Textbox(
            label="Custom background (#RRGGBB[AA])",
            placeholder="#00000000",
            info="Leave empty to use the selection above.",
        )

    with gr.Accordion("Preset details", open=False):
        gr.Markdown(
            "\n".join(
                f"**{preset.name}** — {preset.description}" for preset in PRESETS
            )
        )

    build_btn = gr.Button("Build sprite sheet", variant="primary")

    with gr.Row():
        sheet_file = gr.File(label="Sprite sheet image")
        mapping_file = gr.File(label="Frame mapping JSON")
        zip_file = gr.File(label="Bundled ZIP")
    preview_gallery = gr.Gallery(label="Preview frames", height=240)
    status_box = gr.Textbox(label="Status", interactive=False)

    build_btn.click(
        _build_sheet,
        inputs=[
            sprite_input,
            preset_dropdown,
            tile_slider,
            padding_slider,
            background_choice,
            custom_background,
        ],
        outputs=[sheet_file, mapping_file, zip_file, preview_gallery, status_box],
    )

__all__ = ["demo"]


if __name__ == "__main__":  # pragma: no cover - convenience launch
    from tools.device import pick_device

    print(f"[PixStu] Using device: {pick_device()}")
    demo.launch(
        share=False,
        inbrowser=True,
        server_name=os.getenv("PCS_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PCS_SPRITE_SHEET_PORT", "7865")),
        show_error=True,
    )

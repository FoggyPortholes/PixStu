from __future__ import annotations

from typing import Optional

import gradio as gr

from ..editor import apply_edit
from ..presets import Presets
from .base import Plugin, UIContext

_PRESETS = Presets()


class AIEditPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__()
        self.image_input: Optional[gr.Image] = None
        self.mask_input: Optional[gr.Image] = None
        self.prompt_box: Optional[gr.Textbox] = None
        self.strength_slider: Optional[gr.Slider] = None
        self.auto_mask_toggle: Optional[gr.Checkbox] = None
        self.target_dropdown: Optional[gr.Dropdown] = None
        self.apply_button: Optional[gr.Button] = None
        self.output_image: Optional[gr.Image] = None
        self.output_status: Optional[gr.Textbox] = None
        self.meta_text: Optional[gr.Textbox] = None
        self.preset_component: Optional[gr.Dropdown] = None

    def setup_ui(self, ui: UIContext) -> list[gr.Component]:
        self.preset_component = ui.components.get("preset")  # type: ignore[assignment]

        with ui.container.expander("AI Edit (Experimental)"):
            gr.Markdown(
                "Upload the generated character and optionally a mask to run targeted inpainting."
            )
            self.image_input = gr.Image(label="Image to Edit", type="filepath")
            self.mask_input = gr.Image(label="Mask (optional)", type="filepath")
            self.prompt_box = gr.Textbox(label="Edit Prompt", lines=2)
            with gr.Row():
                self.strength_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.45,
                    step=0.05,
                    label="Edit Strength",
                    info="Higher values apply stronger changes.",
                )
                self.auto_mask_toggle = gr.Checkbox(label="Auto-mask", value=True)
                self.target_dropdown = gr.Dropdown(
                    ["entire", "upper", "lower", "left", "right"],
                    value="entire",
                    label="Target Region",
                )
            self.apply_button = gr.Button("Apply Edit", elem_classes=["chargen-primary"])
            self.output_image = gr.Image(label="Edited Output", interactive=False)
            self.output_status = gr.Textbox(label="Edit Status", interactive=False)
            self.meta_text = gr.Textbox(label="Edit Metadata JSON", interactive=False)

        def _run_edit(image_path, mask_path, prompt, strength, auto_mask, target_region, preset_name):
            try:
                preset = _PRESETS.get(preset_name)
                if not preset:
                    raise ValueError("Select a preset before applying edits.")
                result = apply_edit(
                    preset,
                    image_path,
                    mask_path,
                    prompt,
                    strength=float(strength or 0.5),
                    auto_mask_enabled=bool(auto_mask),
                    target_region=target_region or "entire",
                )
                return (
                    result["output_path"],
                    f"Edit applied. Saved to {result['output_path']}",
                    result["metadata_path"],
                )
            except Exception as exc:  # pragma: no cover - surface to UI
                return (None, f"Edit failed: {exc}", "")

        if self.apply_button and self.preset_component:
            self.apply_button.click(
                _run_edit,
                inputs=[
                    self.image_input,
                    self.mask_input,
                    self.prompt_box,
                    self.strength_slider,
                    self.auto_mask_toggle,
                    self.target_dropdown,
                    self.preset_component,
                ],
                outputs=[self.output_image, self.output_status, self.meta_text],
            )

        self._outputs = []
        return self._outputs

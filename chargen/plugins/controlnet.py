from __future__ import annotations

from typing import List, Optional

import gradio as gr
from PIL import Image

from .. import model_setup
from .base import GenerationSession, Plugin, UIContext


class ControlNetPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__()
        self.enable: Optional[gr.Checkbox] = None
        self.model_dropdown: Optional[gr.Dropdown] = None
        self.scale_slider: Optional[gr.Slider] = None
        self.image_input: Optional[gr.Image] = None
        self.status_box: Optional[gr.Textbox] = None
        self._status_message: str = "ControlNet disabled."

    def setup_ui(self, ui: UIContext) -> List[gr.Component]:
        records = model_setup.list_controlnets()
        options = [record.name for record in records]
        default = options[0] if options else None

        with ui.container:
            with gr.Accordion("ControlNet", open=False):
                self.enable = gr.Checkbox(label="Enable ControlNet", value=False)
                self.model_dropdown = gr.Dropdown(
                    options,
                    value=default,
                    label="ControlNet Model",
                info="Choose which ControlNet to apply.",
            )
            self.scale_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.05,
                label="Control Weight",
                info="Scale applied to the ControlNet conditioning.",
            )
            self.image_input = gr.Image(
                label="Control Conditioning Image",
                type="filepath",
                info="Upload a preprocessed map (e.g., canny edges, depth).",
            )
                self.status_box = gr.Textbox(label="ControlNet Status", interactive=False)

        if self.enable:
            self.register_input(self.enable)
        if self.model_dropdown:
            self.register_input(self.model_dropdown)
        if self.scale_slider:
            self.register_input(self.scale_slider)
        if self.image_input:
            self.register_input(self.image_input)

        self._outputs = [self.status_box] if self.status_box else []
        return self._outputs

    def prepare_session(self, session: GenerationSession, values: List) -> None:
        enabled = bool(values[0]) if values else False
        model_name = values[1] if len(values) > 1 else None
        scale = float(values[2]) if len(values) > 2 and values[2] is not None else 1.0
        image_path = values[3] if len(values) > 3 else None

        session.storage.pop("controlnet", None)

        if not enabled:
            self._status_message = "ControlNet disabled."
            return

        if not model_name:
            self._status_message = "Select a ControlNet model."
            return

        if not image_path:
            self._status_message = "Upload a conditioning image for ControlNet."
            return

        record = model_setup.find_controlnet(model_name)
        if record is None:
            self._status_message = f"Unknown ControlNet: {model_name}"
            return

        try:
            with Image.open(image_path) as img:
                conditioning_image = img.convert("RGB")
        except Exception as exc:
            self._status_message = f"Failed to load conditioning image: {exc}"
            return

        session.storage["controlnet"] = {
            "record": record,
            "image": conditioning_image,
            "scale": scale,
        }
        self._status_message = f"ControlNet armed: {record.name}"

    def on_generation_start(self, session: GenerationSession) -> List[gr.Update]:
        return [gr.update(value=self._status_message)] if self.status_box else []

    def on_preview(self, session: GenerationSession, step: int, total: int, image) -> List[gr.Update]:
        if not self.status_box:
            return []
        status = session.storage.get("controlnet")
        if not status:
            return [gr.update()]
        return [gr.update(value=f"ControlNet active (step {step + 1}/{total}).")]

    def on_generation_complete(
        self,
        session: GenerationSession,
        image,
        image_path: str,
        metadata_path: str,
    ) -> List[gr.Update]:
        if not self.status_box:
            return []
        if session.storage.get("controlnet"):
            return [gr.update(value="ControlNet applied successfully.")]
        return [gr.update()]

    def on_error(self, session: GenerationSession, message: str) -> List[gr.Update]:
        if not self.status_box:
            return []
        if session.storage.get("controlnet"):
            return [gr.update(value=f"ControlNet error: {message}")]
        return [gr.update()]

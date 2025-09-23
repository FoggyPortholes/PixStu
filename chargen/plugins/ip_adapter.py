from __future__ import annotations

from typing import List, Optional

import gradio as gr
from PIL import Image

from .. import model_setup
from .base import GenerationSession, Plugin, UIContext


class IPAdapterPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__()
        self.enable: Optional[gr.Checkbox] = None
        self.model_dropdown: Optional[gr.Dropdown] = None
        self.scale_slider: Optional[gr.Slider] = None
        self.image_input: Optional[gr.Image] = None
        self.status_box: Optional[gr.Textbox] = None
        self._status_message: str = "IP-Adapter disabled."

    def setup_ui(self, ui: UIContext) -> List[gr.Component]:
        records = model_setup.list_ip_adapters()
        options = [record.name for record in records]
        default = options[0] if options else None

        with ui.container:
            with gr.Accordion("IP-Adapter", open=False):
                self.enable = gr.Checkbox(label="Enable IP-Adapter", value=False)
                self.model_dropdown = gr.Dropdown(
                    options,
                    value=default,
                    label="IP-Adapter Weights",
                info="Select the adapter weights to load.",
            )
            self.scale_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.6,
                step=0.05,
                label="Adapter Scale",
                info="Blend IP-Adapter features with the base prompt.",
            )
            self.image_input = gr.Image(
                label="Adapter Reference Image",
                type="filepath",
                info="Upload a reference image used by IP-Adapter.",
            )
                self.status_box = gr.Textbox(label="IP-Adapter Status", interactive=False)

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
        scale = float(values[2]) if len(values) > 2 and values[2] is not None else 0.6
        image_path = values[3] if len(values) > 3 else None

        session.storage.pop("ip_adapter", None)

        if not enabled:
            self._status_message = "IP-Adapter disabled."
            return

        if not model_name:
            self._status_message = "Select IP-Adapter weights."
            return

        if not image_path:
            self._status_message = "Upload a reference image for IP-Adapter."
            return

        record = model_setup.find_ip_adapter(model_name)
        if record is None:
            self._status_message = f"Unknown IP-Adapter: {model_name}"
            return

        try:
            with Image.open(image_path) as img:
                reference = img.convert("RGB")
        except Exception as exc:
            self._status_message = f"Failed to load reference image: {exc}"
            return

        session.storage["ip_adapter"] = {
            "record": record,
            "image": reference,
            "scale": scale,
        }
        self._status_message = f"IP-Adapter armed: {record.name}"

    def on_generation_start(self, session: GenerationSession) -> List[gr.Update]:
        return [gr.update(value=self._status_message)] if self.status_box else []

    def on_preview(self, session: GenerationSession, step: int, total: int, image) -> List[gr.Update]:
        if not self.status_box:
            return []
        if session.storage.get("ip_adapter"):
            return [gr.update(value=f"IP-Adapter active (step {step + 1}/{total}).")]
        return [gr.update()]

    def on_generation_complete(
        self,
        session: GenerationSession,
        image,
        image_path: str,
        metadata_path: str,
    ) -> List[gr.Update]:
        if not self.status_box:
            return []
        if session.storage.get("ip_adapter"):
            return [gr.update(value="IP-Adapter applied successfully.")]
        return [gr.update()]

    def on_error(self, session: GenerationSession, message: str) -> List[gr.Update]:
        if not self.status_box:
            return []
        if session.storage.get("ip_adapter"):
            return [gr.update(value=f"IP-Adapter error: {message}")]
        return [gr.update()]

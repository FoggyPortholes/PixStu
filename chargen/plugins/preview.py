from __future__ import annotations

import gradio as gr

from .base import GenerationSession, Plugin, UIContext


class LivePreviewPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__()
        self.preview: gr.Image | None = None
        self._latest_image = None

    def setup_ui(self, ui: UIContext) -> list[gr.Component]:
        with ui.container:
            self.preview = gr.Image(label="Live Preview", interactive=False)
        self._outputs = [self.preview]
        return self._outputs

    def on_generation_start(self, session: GenerationSession) -> list[gr.Update]:
        self._latest_image = None
        return [gr.update(value=None)]

    def on_preview(self, session: GenerationSession, step: int, total: int, image) -> list[gr.Update]:
        self._latest_image = image
        return [gr.update(value=image)]

    def on_generation_complete(
        self,
        session: GenerationSession,
        image,
        image_path: str,
        metadata_path: str,
    ) -> list[gr.Update]:
        final = image or self._latest_image
        return [gr.update(value=final)]

    def on_error(self, session: GenerationSession, message: str) -> list[gr.Update]:
        return [gr.update(value=None)]

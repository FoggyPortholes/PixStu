from __future__ import annotations

import gradio as gr

from ..metadata import update_rating
from .base import GenerationSession, Plugin, UIContext


class RatingPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__()
        self.rating: gr.Radio | None = None
        self.status: gr.Textbox | None = None
        self._latest_meta: str | None = None

    def setup_ui(self, ui: UIContext) -> list[gr.Component]:
        with ui.container:
            self.rating = gr.Radio(
                ["1", "2", "3", "4", "5"],
                label="Rate Output",
                interactive=False,
                info="Select a rating once the image finishes rendering.",
            )
            self.status = gr.Textbox(label="Rating Status", interactive=False)

        def _apply_rating(value: str) -> str:
            if not value:
                return "Rating cleared."
            if not self._latest_meta:
                return "No metadata available yet."
            try:
                update_rating(self._latest_meta, int(value))
                return f"Saved rating {value}."
            except Exception as exc:  # pragma: no cover - user feedback only
                return f"Rating failed: {exc}"

        self.rating.change(_apply_rating, inputs=[self.rating], outputs=[self.status])
        self._outputs = [self.rating, self.status]
        return self._outputs

    def on_generation_start(self, session: GenerationSession) -> list[gr.Update]:
        self._latest_meta = None
        return [gr.update(value=None, interactive=False), gr.update(value="")]

    def on_preview(self, session: GenerationSession, step: int, total: int, image) -> list[gr.Update]:
        return [gr.update(), gr.update()]

    def on_generation_complete(
        self,
        session: GenerationSession,
        image,
        image_path: str,
        metadata_path: str,
    ) -> list[gr.Update]:
        self._latest_meta = metadata_path
        return [gr.update(interactive=True), gr.update(value="Rate this output.")]

    def on_error(self, session: GenerationSession, message: str) -> list[gr.Update]:
        self._latest_meta = None
        return [gr.update(value=None, interactive=False), gr.update(value=f"Rating unavailable: {message}")]

from __future__ import annotations

import os
from typing import List

import gradio as gr

from ..logging_config import get_log_file
from .base import GenerationSession, Plugin, UIContext


class DiagnosticsPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__()
        self.toggle: gr.Checkbox | None = None
        self.log_path: gr.Textbox | None = None
        self.log_output: gr.Textbox | None = None
        self._show_logs = False

    def setup_ui(self, ui: UIContext) -> List[gr.Component]:
        with ui.container:
            self.toggle = gr.Checkbox(label="Show Diagnostics", value=False)
            self.log_path = gr.Textbox(label="Log File", interactive=False, visible=False)
            self.log_output = gr.Textbox(label="Log Tail", lines=10, interactive=False, visible=False)

        def _handle_toggle(show: bool):
            self._show_logs = bool(show)
            if not self._show_logs:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            return (
                gr.update(value=get_log_file(), visible=True),
                gr.update(value=self._tail_log(), visible=True),
            )

        self.toggle.change(_handle_toggle, inputs=[self.toggle], outputs=[self.log_path, self.log_output])
        self._outputs = [self.log_path, self.log_output]
        return self._outputs

    def on_generation_start(self, session: GenerationSession) -> List[gr.Update]:
        if not self._show_logs:
            return [gr.update(visible=False), gr.update(visible=False)]
        return [
            gr.update(value=get_log_file(), visible=True),
            gr.update(value=self._tail_log(), visible=True),
        ]

    def on_preview(self, session: GenerationSession, step: int, total: int, image) -> List[gr.Update]:
        if not self._show_logs:
            return [gr.update(visible=False), gr.update(visible=False)]
        return [
            gr.update(value=get_log_file(), visible=True),
            gr.update(value=self._tail_log(), visible=True),
        ]

    def on_generation_complete(
        self,
        session: GenerationSession,
        image,
        image_path: str,
        metadata_path: str,
    ) -> List[gr.Update]:
        if not self._show_logs:
            return [gr.update(visible=False), gr.update(visible=False)]
        return [
            gr.update(value=get_log_file(), visible=True),
            gr.update(value=self._tail_log(), visible=True),
        ]

    def on_error(self, session: GenerationSession, message: str) -> List[gr.Update]:
        if not self._show_logs:
            return [gr.update(visible=False), gr.update(visible=False)]
        return [
            gr.update(value=get_log_file(), visible=True),
            gr.update(value=f"{self._tail_log()}\nError: {message}", visible=True),
        ]

    def _tail_log(self, lines: int = 80) -> str:
        path = get_log_file()
        if not os.path.exists(path):
            return "Log file not found."
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = handle.readlines()
            return "".join(data[-lines:])
        except Exception as exc:  # pragma: no cover - diagnostics only
            return f"Unable to read log file: {exc}"

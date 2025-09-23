"""Placeholder AI edit module. Will be expanded with inpainting capabilities."""

from typing import Optional

import gradio as gr


def build_editor_tab(container: gr.Blocks, info: Optional[str] = None) -> None:
    with gr.Column(elem_classes=["chargen-panel"]):
        gr.Markdown("### AI Edit (Inpainting)", elem_classes=["chargen-group-title"])
        gr.Markdown(
            info
            or "AI Edit tools are coming soon. This placeholder keeps the layout stable for future updates."
        )

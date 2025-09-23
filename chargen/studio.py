from __future__ import annotations

import gradio as gr

from .character_studio import build_ui
from .logging_config import configure_logging


def build_app() -> gr.Blocks:
    configure_logging()
    demo = build_ui()
    demo.share = False
    return demo

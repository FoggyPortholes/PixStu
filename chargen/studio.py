from __future__ import annotations

import gradio as gr

from .character_studio import build_ui
from .logging_config import configure_logging


def build_app() -> gr.Blocks:
    configure_logging()
    demo = build_ui()
    try:
        from .ui_guard import check_ui  # lazy import to avoid optional dependency issues

        for message in check_ui(demo):
            print(message)
    except Exception as exc:  # pragma: no cover - defensive logging only
        print("[UI] Drift check skipped:", exc)
    demo.share = False
    return demo

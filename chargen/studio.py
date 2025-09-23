from __future__ import annotations

import gradio as gr

from .character_studio import build_ui
from .logging_config import configure_logging


def build_app() -> gr.Blocks:
    configure_logging()
    demo = build_ui()
    print('[SELFTEST] Starting quick checks…')
    try:
        from .presets import get_preset_names

        names = get_preset_names()
        assert names, 'No presets found'
        print('[SELFTEST] Presets:', len(names))
    except Exception as exc:
        print('[SELFTEST] Preset load error:', exc)

    try:
        from .ui_guard import check_ui

        issues = check_ui(demo)
        if issues:
            print('[SELFTEST] UI drift found:')
            for issue in issues:
                print(' -', issue)
        else:
            print('[SELFTEST] UI OK')
    except Exception as exc:
        print('[SELFTEST] UI check skipped:', exc)
    demo.share = False
    return demo

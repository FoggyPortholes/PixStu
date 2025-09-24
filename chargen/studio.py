from __future__ import annotations

import os
import socket

import gradio as gr

from .character_studio import build_ui
from .logging_config import configure_logging


def _env_port() -> int | None:
    for key in ("PCS_PORT", "GRADIO_SERVER_PORT"):
        value = os.environ.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def _pick_port(host: str) -> int | None:
    env_port = _env_port()
    if env_port:
        return env_port
    default_port = 7860
    return default_port if _port_available(host, default_port) else None


def build_app() -> gr.Blocks:
    configure_logging()
    demo = build_ui()
    print('[SELFTEST] Starting quick checks...')
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


if __name__ == "__main__":
    demo = build_app()
    server_name = os.getenv("PCS_SERVER_NAME", "127.0.0.1")
    port = _pick_port(server_name)
    open_browser = os.getenv("PCS_OPEN_BROWSER", "0").lower() in {"1", "true", "yes", "on"}
    try:
        demo.launch(
            share=False,
            inbrowser=open_browser,
            server_name=server_name,
            server_port=port,
            show_error=True,
        )
    except OSError as exc:
        message = str(exc)
        if port and "Cannot find empty port" in message:
            demo.launch(
                share=False,
                inbrowser=open_browser,
                server_name=server_name,
                server_port=None,
                show_error=True,
            )
        else:
            raise

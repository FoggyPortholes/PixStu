from __future__ import annotations

from .preview import LivePreviewPlugin
from .controlnet import ControlNetPlugin
from .ip_adapter import IPAdapterPlugin
from .ai_edit import AIEditPlugin
from .rating import RatingPlugin
from .diagnostics import DiagnosticsPlugin

PLUGIN_CLASSES = [
    LivePreviewPlugin,
    ControlNetPlugin,
    IPAdapterPlugin,
    AIEditPlugin,
    RatingPlugin,
    DiagnosticsPlugin,
]

from __future__ import annotations

from .preview import LivePreviewPlugin
from .rating import RatingPlugin
from .diagnostics import DiagnosticsPlugin

PLUGIN_CLASSES = [
    LivePreviewPlugin,
    RatingPlugin,
    DiagnosticsPlugin,
]

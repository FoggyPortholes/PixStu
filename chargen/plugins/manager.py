from __future__ import annotations

from typing import Iterable, List

import gradio as gr

from .base import GenerationSession, Plugin, UIContext


class PluginManager:
    def __init__(self, plugins: Iterable[Plugin]) -> None:
        self.plugins: List[Plugin] = list(plugins)
        self._outputs: List[gr.Component] = []

    def setup_ui(self, ui_context: UIContext) -> List[gr.Component]:
        outputs: List[gr.Component] = []
        for plugin in self.plugins:
            outputs.extend(plugin.setup_ui(ui_context))
        self._outputs = outputs
        return outputs

    def create_session(
        self,
        preset_name: str,
        seed: int,
        size: int,
        ref_image: str | None,
        ref_strength: float | None,
    ) -> GenerationSession:
        return GenerationSession(
            preset_name=preset_name,
            seed=seed,
            size=size,
            ref_image=ref_image,
            ref_strength=ref_strength,
        )

    def on_generation_start(self, session: GenerationSession) -> List[gr.Update]:
        updates: List[gr.Update] = []
        for plugin in self.plugins:
            updates.extend(plugin.on_generation_start(session))
        return updates

    def on_preview(self, session: GenerationSession, step: int, total: int, image) -> List[gr.Update]:
        updates: List[gr.Update] = []
        for plugin in self.plugins:
            updates.extend(plugin.on_preview(session, step, total, image))
        return updates

    def on_generation_complete(
        self,
        session: GenerationSession,
        image,
        image_path: str,
        metadata_path: str,
    ) -> List[gr.Update]:
        updates: List[gr.Update] = []
        for plugin in self.plugins:
            updates.extend(plugin.on_generation_complete(session, image, image_path, metadata_path))
        return updates

    def on_error(self, session: GenerationSession, message: str) -> List[gr.Update]:
        updates: List[gr.Update] = []
        for plugin in self.plugins:
            updates.extend(plugin.on_error(session, message))
        return updates

    @property
    def outputs(self) -> List[gr.Component]:
        return self._outputs


_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    from .registry import PLUGIN_CLASSES

    global _plugin_manager
    if _plugin_manager is None:
        plugins = [cls() for cls in PLUGIN_CLASSES]
        _plugin_manager = PluginManager(plugins)
    return _plugin_manager

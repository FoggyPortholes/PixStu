from __future__ import annotations

from typing import Iterable, List

import gradio as gr

from .base import GenerationSession, Plugin, UIContext


class PluginManager:
    def __init__(self, plugins: Iterable[Plugin]) -> None:
        self.plugins: List[Plugin] = list(plugins)
        self._outputs: List[gr.Component] = []
        self._inputs: List[gr.Component] = []
        self._inputs_map: List[tuple[Plugin, List[gr.Component]]] = []

    def setup_ui(self, ui_context: UIContext) -> List[gr.Component]:
        outputs: List[gr.Component] = []
        inputs: List[gr.Component] = []
        mapping: List[tuple[Plugin, List[gr.Component]]] = []
        for plugin in self.plugins:
            outputs.extend(plugin.setup_ui(ui_context))
            plugin_inputs = plugin.inputs
            if plugin_inputs:
                inputs.extend(plugin_inputs)
                mapping.append((plugin, plugin_inputs))
        self._outputs = outputs
        self._inputs = inputs
        self._inputs_map = mapping
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

    def prepare_session(self, session: GenerationSession, values: List) -> None:
        idx = 0
        for plugin, components in self._inputs_map:
            count = len(components)
            plugin_values = values[idx : idx + count]
            plugin.prepare_session(session, plugin_values)
            idx += count

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

    @property
    def inputs(self) -> List[gr.Component]:
        return self._inputs


_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    from .registry import PLUGIN_CLASSES

    global _plugin_manager
    if _plugin_manager is None:
        plugins = [cls() for cls in PLUGIN_CLASSES]
        _plugin_manager = PluginManager(plugins)
    return _plugin_manager

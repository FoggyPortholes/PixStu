from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import gradio as gr


@dataclass
class UIContext:
    """Context passed to plugins when constructing UI components."""

    gradio: object
    container: gr.Blocks
    components: Dict[str, gr.Component]


@dataclass
class GenerationSession:
    """Runtime information for a single generation request."""

    preset_name: str
    seed: int
    size: int
    ref_image: Optional[str] = None
    ref_strength: Optional[float] = None
    metadata_path: Optional[str] = None
    storage: Dict[str, object] = field(default_factory=dict)


class Plugin:
    """Base plugin interface. Subclasses may override any hook."""

    def __init__(self) -> None:
        self._outputs: List[gr.Component] = []
        self._inputs: List[gr.Component] = []
        self._status: str | None = None

    # -- UI -----------------------------------------------------------------
    def setup_ui(self, ui: UIContext) -> List[gr.Component]:
        """Create plugin UI inside the provided container and return outputs."""

        self._outputs = []
        self._inputs = []
        return self._outputs

    # -- Helpers -------------------------------------------------------------
    def _blank(self) -> List[gr.Update]:
        return [gr.update()] * len(self._outputs)

    def register_input(self, component: gr.Component) -> None:
        self._inputs.append(component)

    # -- Generation lifecycle hooks ----------------------------------------
    def on_generation_start(self, session: GenerationSession) -> List[gr.Update]:
        return self._blank()

    def on_preview(self, session: GenerationSession, step: int, total: int, image) -> List[gr.Update]:
        return self._blank()

    def on_generation_complete(
        self,
        session: GenerationSession,
        image,
        image_path: str,
        metadata_path: str,
    ) -> List[gr.Update]:
        session.metadata_path = metadata_path
        return self._blank()

    def on_error(self, session: GenerationSession, message: str) -> List[gr.Update]:
        return self._blank()

    def prepare_session(self, session: GenerationSession, values: List) -> None:
        """Capture UI state ahead of a generation run."""
        session.storage[self.__class__.__name__] = values

    # ----------------------------------------------------------------------
    @property
    def outputs(self) -> List[gr.Component]:
        return self._outputs

    @property
    def inputs(self) -> List[gr.Component]:
        return self._inputs

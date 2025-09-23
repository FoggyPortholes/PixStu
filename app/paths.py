"""Compatibility shim exposing path constants from the new chargen package."""

from chargen.model_setup import (  # noqa: F401
    ROOT,
    CONFIGS,
    OUTPUTS,
    GALLERY,
    MODELS,
    LORAS,
    CONTROLNET_DIR,
    IP_ADAPTER_DIR,
)

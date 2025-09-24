"""Wan2.2 video wrapper with runtime guards.

This module keeps the UI responsive even when the optional Wan2.2
dependency is missing.  When the package is installed we attempt to call a
compatible generation entry point while gracefully handling signature
mismatches across different releases.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple

from .wan_install import ensure_wan22_installed

WanResult = Tuple[Optional[str], str]


def _invoke_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call ``fn`` while filtering unsupported keyword arguments."""

    signature = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return fn(**filtered)


def _normalise_output(output: Any) -> WanResult:
    """Convert Wan2.2 outputs to ``(path, status)`` tuples."""

    status_ok = "[Wan2.2] Generation complete."
    if isinstance(output, (str, Path)):
        return str(output), status_ok
    if isinstance(output, Iterable):
        for item in output:
            if isinstance(item, (str, Path)):
                return str(item), status_ok
    return None, "[Wan2.2] Generator returned no file path."


def txt2vid_wan_guarded(
    prompt: str,
    *,
    n_frames: int = 16,
    seed: Optional[int] = None,
    fps: int = 12,
    width: int = 576,
    height: int = 320,
) -> WanResult:
    """Generate a video via Wan2.2 when available.

    Returns a tuple of ``(path, status_message)``.  The path is ``None`` when
    generation is not available.
    """

    install_note: Optional[str] = None

    def _with_install_note(result: WanResult) -> WanResult:
        path, status = result
        if install_note:
            status = f"{status}\n{install_note}"
        return path, status

    try:
        wan_module = importlib.import_module("wan22")
    except ModuleNotFoundError:
        available, message, installed_now = ensure_wan22_installed()
        if not available:
            return _with_install_note((None, f"[Wan2.2] {message}"))
        if installed_now:
            install_note = f"[Wan2.2] {message}"
        try:
            wan_module = importlib.import_module("wan22")
        except Exception as exc:  # pragma: no cover - runtime failures
            return _with_install_note((None, f"[Wan2.2] import failed after installation: {exc}"))
    except Exception as exc:  # pragma: no cover - runtime failures
        return _with_install_note((None, f"[Wan2.2] import failed: {exc}"))

    kwargs = dict(
        prompt=prompt,
        seed=seed,
        num_frames=n_frames,
        n_frames=n_frames,
        frames=n_frames,
        fps=fps,
        width=width,
        height=height,
    )

    # Look for direct function style APIs first
    for attr in ("txt2video", "txt2vid", "generate_video"):
        candidate = getattr(wan_module, attr, None)
        if callable(candidate):
            try:
                return _with_install_note(
                    _normalise_output(_invoke_with_supported_kwargs(candidate, **kwargs))
                )
            except TypeError:
                continue
            except Exception as exc:  # pragma: no cover - vendor specific errors
                return _with_install_note((None, f"[Wan2.2] Generation failed: {exc}"))

    # Fall back to pipeline-style APIs
    for attr in ("WanVideoPipeline", "Wan22Pipeline"):
        pipeline_cls = getattr(wan_module, attr, None)
        if pipeline_cls is None:
            continue
        try:
            pipeline = pipeline_cls()
        except Exception as exc:  # pragma: no cover - vendor specific errors
            return _with_install_note((None, f"[Wan2.2] Pipeline initialisation failed: {exc}"))
        for method in ("generate", "txt2vid", "__call__"):
            candidate = getattr(pipeline, method, None)
            if not callable(candidate):
                continue
            try:
                return _with_install_note(
                    _normalise_output(_invoke_with_supported_kwargs(candidate, **kwargs))
                )
            except TypeError:
                continue
            except Exception as exc:  # pragma: no cover - vendor specific errors
                return _with_install_note((None, f"[Wan2.2] Generation failed: {exc}"))

    return _with_install_note((None, "[Wan2.2] Installed but compatible generator not found."))


__all__ = ["txt2vid_wan_guarded"]

"""Utilities for installing the optional Wan2.2 dependency."""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Tuple

WAN_MODULE = "wan22"
WAN_SPEC = "git+https://github.com/Wan-Video/Wan2.2.git#egg=wan22"


def ensure_wan22_installed() -> Tuple[bool, str, bool]:
    """Ensure the Wan2.2 package is available.

    Returns a tuple of ``(available, message, installed_now)`` where:
    - ``available`` indicates whether the module can be imported after running.
    - ``message`` contains a human-readable status message without the ``Wan2.2`` prefix.
    - ``installed_now`` is ``True`` when the package was installed during this call.
    """

    try:
        importlib.import_module(WAN_MODULE)
    except ModuleNotFoundError:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            WAN_SPEC,
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            output = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            return False, f"install failed: {output}", False
        except Exception as exc:  # pragma: no cover - runtime failures
            return False, f"install failed: {exc}", False
        importlib.invalidate_caches()
        try:
            importlib.import_module(WAN_MODULE)
        except Exception as exc:  # pragma: no cover - runtime failures
            return False, f"installed but import failed: {exc}", True
        output = result.stdout.strip().splitlines()
        tail = output[-1] if output else "Installation complete."
        return True, f"installed: {tail}", True
    except Exception as exc:  # pragma: no cover - runtime failures
        return False, f"check failed: {exc}", False
    return True, "already installed.", False


__all__ = ["ensure_wan22_installed"]

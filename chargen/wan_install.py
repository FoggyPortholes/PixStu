"""Utilities for installing the optional Wan2.2 dependency."""

from __future__ import annotations

import importlib
import subprocess
import sys
import re
from typing import Iterable, Tuple

WAN_MODULE = "wan22"
WAN_SPEC = "git+https://github.com/Wan-Video/Wan2.2.git#egg=wan22"


def _normalise_lines(output: str) -> Iterable[str]:
    """Yield trimmed, non-empty lines from the provided ``output`` string."""

    for line in output.splitlines():
        stripped = line.strip()
        if stripped:
            yield stripped


def _summarise_install_failure(output: str) -> str:
    """Return a concise, user-focused explanation for a pip install failure."""

    lines = list(_normalise_lines(output))
    if not lines:
        return "install failed but pip produced no output."

    lowered = [line.lower() for line in lines]

    for line, lower in zip(lines, lowered):
        if "fatal: unable to access" in lower:
            if "403" in lower or "forbidden" in lower:
                return (
                    "GitHub denied access to the Wan2.2 repository (HTTP 403). "
                    "Check your proxy or network permissions."
                )
            if "could not resolve host" in lower:
                return (
                    "GitHub hostname could not be resolved while downloading Wan2.2. "
                    "Check your internet connection or DNS configuration."
                )
            if "connection timed out" in lower:
                return (
                    "Connection to GitHub timed out while downloading Wan2.2. "
                    "Verify your network stability."
                )
            return line

        if "git" in lower and ("not found" in lower or "not recognized" in lower):
            return "Git does not appear to be installed. Install Git and retry."

        if "ssl" in lower and "certificate" in lower:
            return (
                "SSL verification failed when contacting GitHub. Check your certificate "
                "trust store or proxy configuration."
            )

        if "python" in lower and "(you have" in lower:
            version_match = re.search(
                r"python\s+([0-9.]+)\s*\(you have\s*([0-9.]+)\)",
                lower,
            )
            if version_match:
                required, current = version_match.groups()
                return (
                    "Your Python version is "
                    f"{current}, but the dependency expects {required}. "
                    "Upgrade Python or install a matching build (for example, reinstall xformers)."
                )
            return (
                "The dependency requires a different Python version than the one "
                "currently running. Upgrade Python or install a compatible build."
            )

        if "reinstall xformers" in lower:
            return (
                "xformers reported an installation issue. Reinstall xformers "
                "matching your Python version, then retry the Wan2.2 setup."
            )

        if "inconsistent name" in lower and "metadata has" in lower:
            expected_match = re.search(r"expected '([^']+)'", line)
            actual_match = re.search(r"metadata has '([^']+)'", line)
            if expected_match and actual_match:
                expected_name = expected_match.group(1)
                actual_name = actual_match.group(1)
                return (
                    "Repository metadata reports package name "
                    f"'{actual_name}' instead of '{expected_name}'. "
                    "Install using the reported name or update the specification."
                )
            return (
                "Package name reported by repository metadata does not match the "
                "requested Wan2.2 name. Install using the metadata name or update "
                "the specification."
            )

    return lines[-1]


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
            combined_output = "\n".join(
                part.strip()
                for part in (exc.stderr, exc.stdout)
                if part and part.strip()
            )
            summary = _summarise_install_failure(combined_output or str(exc))
            return False, f"install failed: {summary}", False
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

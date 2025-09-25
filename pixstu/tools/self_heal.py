"""
Self-healing decorator for common issues.
"""
from __future__ import annotations

import re
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

LOG = Path(".pixstu/self_heal.log")
LOG.parent.mkdir(exist_ok=True)

PATS = {
    "missing": re.compile("Missing assets"),
    "import": re.compile("sys.modules"),
    "dtype": re.compile("fp16|HALF", re.IGNORECASE),
    "preset": re.compile("invalid preset|KeyError: 'lora'"),
    "sqlite": re.compile("sqlite3.DatabaseError"),
}

F = TypeVar("F", bound=Callable[..., Any])


def _log(name: str, err: Exception) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.now()}] {name}: {err}\n{traceback.format_exc()}\n")


def self_heal(name: str) -> Callable[[F], F]:
    def deco(fn: F) -> F:
        @wraps(fn)
        def wrap(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive logic
                _log(name, exc)
                text = str(exc)
                if PATS["missing"].search(text):
                    raise RuntimeError("Self-heal: Missing assets.") from exc
                if PATS["import"].search(text):
                    import sys

                    sys.modules.pop("pixstu.app.studio", None)
                    return fn(*args, **kwargs)
                if PATS["dtype"].search(text):
                    kwargs["dtype"] = "float32"
                    return fn(*args, **kwargs)
                if PATS["preset"].search(text):
                    return []
                if PATS["sqlite"].search(text):
                    from .cache import CACHE_DB

                    CACHE_DB.unlink(missing_ok=True)
                    raise
                raise

        return wrap  # type: ignore[return-value]

    return deco

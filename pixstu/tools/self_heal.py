"""
Self-healing decorator for recurring issues.
"""
from __future__ import annotations
import re, traceback
from pathlib import Path
from functools import wraps
from datetime import datetime

LOG = Path(".pixstu/self_heal.log"); LOG.parent.mkdir(parents=True, exist_ok=True)

PATS = {
    "missing": re.compile(r"Missing assets"),
    "import":  re.compile(r"found in sys.modules"),
    "dtype":   re.compile(r"fp16|HALF|dtype"),
    "preset":  re.compile(r"invalid preset|KeyError: 'lora'"),
    "sqlite":  re.compile(r"sqlite3.DatabaseError|database disk image is malformed"),
}


def append_log(name: str, err: Exception, fix: str):
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {name} → {fix}\n{traceback.format_exc()}\n---\n")
    except Exception:
        pass


def self_heal(name: str):
    def deco(fn):
        @wraps(fn)
        def wrap(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as e:
                s = str(e)

                if PATS["missing"].search(s):
                    append_log(name, e, "Advise Downloads tab / PIXSTU_AUTO_DOWNLOAD")
                    raise RuntimeError("Self-heal: Missing assets → use Downloads tab or set PIXSTU_AUTO_DOWNLOAD=1")

                if PATS["import"].search(s):
                    import sys
                    sys.modules.pop("pixstu.app.studio", None)
                    append_log(name, e, "Cleared sys.modules; retry once")
                    return fn(*a, **kw)

                if PATS["dtype"].search(s):
                    append_log(name, e, "Force dtype=float32; retry once")
                    kw["dtype"] = "float32"
                    return fn(*a, **kw)

                if PATS["preset"].search(s):
                    append_log(name, e, "Ignored malformed preset; returned empty list")
                    return []

                if PATS["sqlite"].search(s):
                    from .cache import CACHE_DB
                    try:
                        if CACHE_DB.exists(): CACHE_DB.unlink(missing_ok=True)
                        append_log(name, e, "Cache DB recreated")
                    except Exception:
                        append_log(name, e, "Cache DB recreate failed")
                    raise RuntimeError("Self-heal: Cache reset; re-run.")

                raise
        return wrap
    return deco

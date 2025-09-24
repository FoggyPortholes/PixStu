#!/usr/bin/env python3
import re, traceback
from pathlib import Path
from functools import wraps
from datetime import datetime

LOG_DIR = Path(".pixstu"); LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "self_heal.log"

ERROR_PATTERNS = {
    "missing_asset": re.compile(r"Missing assets"),
    "import_drift": re.compile(r"found in sys.modules after import"),
    "dtype_error": re.compile(r"HALF|fp16|dtype"),
    "preset_format": re.compile(r"invalid preset|KeyError: 'lora'"),
    "sqlite_corrupt": re.compile(r"database disk image is malformed|sqlite3.DatabaseError"),
}

def log_recovery(name: str, error: Exception, fix: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {name} → {fix}\n{traceback.format_exc()}\n---\n")

def self_heal(name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = f"{e}"
                if ERROR_PATTERNS["missing_asset"].search(msg):
                    log_recovery(name, e, "Advise Downloads tab / PIXSTU_AUTO_DOWNLOAD");
                    raise RuntimeError("Self-heal: Missing assets → use Downloads tab or set PIXSTU_AUTO_DOWNLOAD=1")
                if ERROR_PATTERNS["import_drift"].search(msg):
                    import sys; sys.modules.pop("chargen.studio", None)
                    log_recovery(name, e, "Cleared sys.modules; retry once");
                    return func(*args, **kwargs)
                if ERROR_PATTERNS["dtype_error"].search(msg):
                    log_recovery(name, e, "Forced dtype=float32; retry once"); kwargs["dtype"] = "float32"; return func(*args, **kwargs)
                if ERROR_PATTERNS["preset_format"].search(msg):
                    log_recovery(name, e, "Ignored malformed preset; returned empty set"); return []
                if ERROR_PATTERNS["sqlite_corrupt"].search(msg):
                    from tools.cache import CACHE_DB
                    try:
                        if CACHE_DB.exists(): CACHE_DB.unlink(missing_ok=True)
                        log_recovery(name, e, "Cache DB recreated")
                    except Exception: log_recovery(name, e, "Cache DB recreate failed")
                    raise RuntimeError("Self-heal: Cache reset; re-run the operation")
                raise
        return wrapper
    return decorator

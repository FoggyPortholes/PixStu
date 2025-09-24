#!/usr/bin/env python3
"""
Minimal persistent cache for PixStu
- Thread-safe via RLock
- SQLite-backed key/value store living at .pixstu/cache.sqlite
- Convenience helpers for PIL Images
"""
from __future__ import annotations
from pathlib import Path
import sqlite3
import threading
from typing import Optional
from PIL import Image
import io

CACHE_DIR = Path(".pixstu")
CACHE_DB = CACHE_DIR / "cache.sqlite"

class Cache:
    _lock = threading.RLock()

    def __init__(self, namespace: str = "default"):
        self.ns = namespace
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
        self._init()

    def _init(self):
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS kv (ns TEXT, k TEXT, v BLOB, PRIMARY KEY(ns, k))"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.conn.close()

    def get(self, key: str) -> Optional[bytes]:
        with Cache._lock, self.conn:
            cur = self.conn.execute("SELECT v FROM kv WHERE ns=? AND k=?", (self.ns, key))
            row = cur.fetchone()
            return row[0] if row else None

    def put(self, key: str, value: bytes) -> None:
        with Cache._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO kv(ns,k,v) VALUES(?,?,?)",
                (self.ns, key, value),
            )

    # Image helpers
    def get_image(self, key: str) -> Optional[Image.Image]:
        data = self.get(key)
        if data is None:
            return None
        return Image.open(io.BytesIO(data)).convert("RGB")

    def put_image(self, key: str, img: Image.Image) -> None:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.put(key, buf.getvalue())

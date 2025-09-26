"""
SQLite cache with self-heal, TTL/LRU stubs, image helpers.
"""
from __future__ import annotations
from pathlib import Path
import sqlite3, threading, time, io
from typing import Optional
from PIL import Image

CACHE_DIR = Path(".pixstu"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB = CACHE_DIR / "cache.sqlite"


class Cache:
    _lock = threading.RLock()
    def __init__(self, ns="default", max_bytes=int(2e9), ttl=0):
        self.ns, self.max_bytes, self.ttl = ns, max_bytes, ttl
        self.conn = self._connect(); self._init(); self._tune()

    def _connect(self):
        try:
            return sqlite3.connect(CACHE_DB, check_same_thread=False)
        except Exception:
            if CACHE_DB.exists(): CACHE_DB.unlink(missing_ok=True)
            return sqlite3.connect(CACHE_DB, check_same_thread=False)

    def _tune(self):
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")

    def _init(self):
        with self.conn:
            self.conn.execute("""CREATE TABLE IF NOT EXISTS kv(
                ns TEXT, k TEXT, v BLOB, ts INTEGER, sz INTEGER, PRIMARY KEY(ns,k))""")

    def _now(self): return int(time.time())

    def get(self, key: str) -> Optional[bytes]:
        try:
            with Cache._lock, self.conn:
                row = self.conn.execute("SELECT v FROM kv WHERE ns=? AND k=?", (self.ns, key)).fetchone()
                return None if not row else row[0]
        except sqlite3.DatabaseError:
            # Self-heal cache corruption
            try:
                self.conn.close()
            except Exception:
                pass
            if CACHE_DB.exists():
                CACHE_DB.unlink(missing_ok=True)
            self.conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
            self._init()
            return None

    def put(self, key: str, blob: bytes):
        ts, sz = self._now(), len(blob)
        with Cache._lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO kv(ns,k,v,ts,sz) VALUES(?,?,?,?,?)",
                              (self.ns, key, blob, ts, sz))

    def put_image(self, key: str, img: Image.Image):
        buf = io.BytesIO(); img.save(buf, format="PNG")
        self.put(key, buf.getvalue())

    def get_image(self, key: str) -> Optional[Image.Image]:
        b = self.get(key)
        if not b: return None
        try:
            return Image.open(io.BytesIO(b)).convert("RGBA")
        except Exception:
            # Remove bad entry
            with Cache._lock, self.conn:
                self.conn.execute("DELETE FROM kv WHERE ns=? AND k=?", (self.ns, key))
            return None


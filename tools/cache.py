#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sqlite3, threading, time, io
from typing import Optional, Tuple
from PIL import Image

CACHE_DIR = Path(".pixstu"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB = CACHE_DIR / "cache.sqlite"
MAX_BYTES_DEFAULT = int(2e9)
TTL_SECONDS_DEFAULT = 0

class Cache:
    _lock = threading.RLock()
    def __init__(self, namespace: str = "default", max_bytes: int = MAX_BYTES_DEFAULT, ttl_seconds: int = TTL_SECONDS_DEFAULT):
        self.ns, self.max_bytes, self.ttl = namespace, max_bytes, ttl_seconds
        self.conn = self._connect(); self._init(); self._tune()
    def _connect(self):
        try:
            return sqlite3.connect(CACHE_DB, check_same_thread=False)
        except Exception:
            CACHE_DB.unlink(missing_ok=True)
            return sqlite3.connect(CACHE_DB, check_same_thread=False)
    def _tune(self):
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA temp_store=MEMORY;")
            self.conn.execute("PRAGMA mmap_size=268435456;")
    def _init(self):
        try:
            with self.conn:
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kv(ns TEXT,k TEXT,v BLOB,ts INTEGER,sz INTEGER,PRIMARY KEY(ns,k))
                    """
                )
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_kv_ns_ts ON kv(ns, ts)")
        except sqlite3.DatabaseError:
            self.conn.close()
            CACHE_DB.unlink(missing_ok=True)
            self.conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
            with self.conn:
                self.conn.execute("CREATE TABLE kv(ns TEXT,k TEXT,v BLOB,ts INTEGER,sz INTEGER,PRIMARY KEY(ns,k))")
                self.conn.execute("CREATE INDEX idx_kv_ns_ts ON kv(ns, ts)")
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.conn.close()
    def _now(self):
        return int(time.time())
    def _lru_evict_if_needed(self):
        if self.max_bytes <= 0:
            return
        with Cache._lock, self.conn:
            total = (
                self.conn.execute("SELECT COALESCE(SUM(sz),0) FROM kv WHERE ns=?", (self.ns,)).fetchone()[0] or 0
            )
            if total <= self.max_bytes:
                return
            to_free, freed = total - self.max_bytes, 0
            for k, sz in self.conn.execute("SELECT k, sz FROM kv WHERE ns=? ORDER BY ts ASC", (self.ns,)):
                self.conn.execute("DELETE FROM kv WHERE ns=? AND k=?", (self.ns, k))
                freed += (sz or 0)
                if freed >= to_free:
                    break
            self.conn.execute("VACUUM")
    def _ttl_prune_if_needed(self):
        if self.ttl <= 0:
            return
        cutoff = self._now() - self.ttl
        with Cache._lock, self.conn:
            self.conn.execute("DELETE FROM kv WHERE ns=? AND ts<?", (self.ns, cutoff))
    def get(self, key: str) -> Optional[bytes]:
        try:
            with Cache._lock, self.conn:
                row = self.conn.execute("SELECT v, ts FROM kv WHERE ns=? AND k=?", (self.ns, key)).fetchone()
                return None if not row else row[0]
        except sqlite3.DatabaseError:
            self._init()
            return None
    def put(self, key: str, value: bytes) -> None:
        ts, sz = self._now(), len(value)
        try:
            with Cache._lock, self.conn:
                self.conn.execute("INSERT OR REPLACE INTO kv(ns,k,v,ts,sz) VALUES(?,?,?,?,?)", (self.ns, key, value, ts, sz))
        except sqlite3.DatabaseError:
            self._init()
            with Cache._lock, self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO kv(ns,k,v,ts,sz) VALUES(?,?,?,?,?)",
                    (self.ns, key, value, ts, sz),
                )
        self._ttl_prune_if_needed()
        self._lru_evict_if_needed()
    def get_image(self, key: str) -> Optional[Image.Image]:
        data = self.get(key)
        if data is None:
            return None
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            with Cache._lock, self.conn:
                self.conn.execute("DELETE FROM kv WHERE ns=? AND k=?", (self.ns, key))
            return None
    def put_image(self, key: str, img: Image.Image) -> None:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.put(key, buf.getvalue())
    def stats(self) -> Tuple[int, int]:
        with Cache._lock, self.conn:
            c, s = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(sz),0) FROM kv WHERE ns=?", (self.ns,)).fetchone()
            return int(c or 0), int(s or 0)
    def prune(self, bytes_target: int) -> int:
        with Cache._lock, self.conn:
            total = (
                self.conn.execute("SELECT COALESCE(SUM(sz),0) FROM kv WHERE ns=?", (self.ns,)).fetchone()[0] or 0
            )
            if total <= bytes_target:
                return 0
            to_free, freed = total - bytes_target, 0
            for k, sz in self.conn.execute("SELECT k, sz FROM kv WHERE ns=? ORDER BY ts ASC", (self.ns,)):
                self.conn.execute("DELETE FROM kv WHERE ns=? AND k=?", (self.ns, k))
                freed += (sz or 0)
                if freed >= to_free:
                    break
            self.conn.execute("VACUUM")
            return freed

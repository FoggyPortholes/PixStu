"""
SQLite cache with self-heal, TTL/LRU, image helpers.
"""
from __future__ import annotations

import io
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from PIL import Image

CACHE_DIR = Path(".pixstu")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB = CACHE_DIR / "cache.sqlite"


class Cache:
    _lock = threading.RLock()

    def __init__(self, ns: str = "default", max_bytes: int = int(2e9), ttl: int = 0):
        self.ns = ns
        self.max_bytes = max(0, int(max_bytes))
        self.ttl = max(0, int(ttl))
        self.conn = self._connect()
        self._init()
        self._tune()

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.conn.close()
        finally:
            self.conn = None  # type: ignore[assignment]

    # -- sqlite helpers ---------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        try:
            return sqlite3.connect(CACHE_DB, check_same_thread=False)
        except Exception:
            if CACHE_DB.exists():
                CACHE_DB.unlink(missing_ok=True)
            return sqlite3.connect(CACHE_DB, check_same_thread=False)

    def _tune(self) -> None:
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")

    def _init(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    ns TEXT,
                    k TEXT,
                    v BLOB,
                    ts INTEGER,
                    sz INTEGER,
                    PRIMARY KEY (ns, k)
                )
                """
            )

    # -- utility ----------------------------------------------------------
    def _now(self) -> int:
        return int(time.time())

    def _purge_expired(self, cur: sqlite3.Cursor) -> None:
        if not self.ttl:
            return
        cutoff = self._now() - self.ttl
        cur.execute("DELETE FROM kv WHERE ns=? AND ts<?", (self.ns, cutoff))

    def _enforce_size(self, cur: sqlite3.Cursor) -> None:
        if not self.max_bytes:
            return
        total = cur.execute("SELECT COALESCE(SUM(sz), 0) FROM kv WHERE ns=?", (self.ns,)).fetchone()[0]
        if total <= self.max_bytes:
            return
        # remove least recently used (smallest ts) until under budget
        while total > self.max_bytes:
            row = cur.execute(
                "SELECT k, sz FROM kv WHERE ns=? ORDER BY ts ASC LIMIT 1",
                (self.ns,),
            ).fetchone()
            if not row:
                break
            cur.execute("DELETE FROM kv WHERE ns=? AND k=?", (self.ns, row[0]))
            total -= row[1]

    # -- public API -------------------------------------------------------
    def get(self, k: str) -> Optional[bytes]:
        try:
            with Cache._lock, self.conn:
                cur = self.conn.cursor()
                self._purge_expired(cur)
                row = cur.execute(
                    "SELECT v, ts FROM kv WHERE ns=? AND k=?",
                    (self.ns, k),
                ).fetchone()
                if not row:
                    return None
                value, ts = row
                if self.ttl and self._now() - ts > self.ttl:
                    cur.execute("DELETE FROM kv WHERE ns=? AND k=?", (self.ns, k))
                    return None
                # update timestamp for LRU purposes
                cur.execute(
                    "UPDATE kv SET ts=? WHERE ns=? AND k=?",
                    (self._now(), self.ns, k),
                )
                return value
        except sqlite3.DatabaseError:
            CACHE_DB.unlink(missing_ok=True)
            self._init()
            return None

    def put(self, k: str, v: bytes) -> None:
        ts = self._now()
        sz = len(v)
        with Cache._lock, self.conn:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO kv(ns, k, v, ts, sz) VALUES (?, ?, ?, ?, ?)",
                (self.ns, k, sqlite3.Binary(v), ts, sz),
            )
            self._purge_expired(cur)
            self._enforce_size(cur)

    def delete(self, k: str) -> None:
        with Cache._lock, self.conn:
            self.conn.execute("DELETE FROM kv WHERE ns=? AND k=?", (self.ns, k))

    def clear(self) -> None:
        with Cache._lock, self.conn:
            self.conn.execute("DELETE FROM kv WHERE ns=?", (self.ns,))

    def put_image(self, k: str, img: Image.Image) -> None:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.put(k, buf.getvalue())

    def get_image(self, k: str) -> Optional[Image.Image]:
        payload = self.get(k)
        if not payload:
            return None
        try:
            return Image.open(io.BytesIO(payload)).convert("RGBA")
        except Exception:
            return None


@contextmanager
def cache(ns: str, **kw):
    c = Cache(ns, **kw)
    try:
        yield c
    finally:
        c.__exit__(None, None, None)

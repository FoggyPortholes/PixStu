"""Persistent on-disk cache with TTL and LRU eviction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from pixstu_config import PIXSTU_DIR, load_config


def _is_disabled() -> bool:
    flag = os.getenv("PIXSTU_CACHE_OFF", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class CacheEntry:
    key: str
    path: Path
    size: int
    created: float
    last_access: float


class Cache:
    """Filesystem backed cache supporting TTL, size caps and LRU eviction."""

    def __init__(
        self,
        namespace: str = "default",
        *,
        root: Optional[Path] = None,
        ttl: Optional[int] = None,
        max_entries: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> None:
        config = load_config().get("cache", {})
        if ttl is None:
            ttl = int(config.get("ttl_seconds", 0) or 0)
        if max_entries is None:
            max_entries = int(config.get("max_entries", 0) or 0)
        if max_bytes is None:
            max_bytes = int(config.get("max_bytes", 0) or 0)

        base_root = Path(root) if root is not None else PIXSTU_DIR / "cache"
        self.root = base_root / namespace
        self.root.mkdir(parents=True, exist_ok=True)

        self.ttl = ttl if ttl and ttl > 0 else None
        self.max_entries = max_entries if max_entries and max_entries > 0 else None
        self.max_bytes = max_bytes if max_bytes and max_bytes > 0 else None

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _entry_dir(self, key: str) -> Path:
        return self.root / self._hash_key(key)

    def _meta_path(self, key: str) -> Path:
        return self._entry_dir(key) / "meta.json"

    def _data_path(self, key: str) -> Path:
        return self._entry_dir(key) / "payload"

    def _load_meta(self, key: str) -> Optional[dict]:
        meta_path = self._meta_path(key)
        if not meta_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return None

    def _write_meta(self, key: str, meta: dict) -> None:
        entry_dir = self._entry_dir(key)
        entry_dir.mkdir(parents=True, exist_ok=True)
        with self._meta_path(key).open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _delete_entry(self, key: str) -> None:
        shutil.rmtree(self._entry_dir(key), ignore_errors=True)

    def _entry_from_meta(self, key: str, meta: dict) -> CacheEntry:
        path = self._data_path(key)
        size = path.stat().st_size if path.exists() else 0
        created = float(meta.get("created", 0.0)) or path.stat().st_mtime
        last_access = float(meta.get("last_access", 0.0)) or created
        return CacheEntry(key=key, path=path, size=size, created=created, last_access=last_access)

    def _iter_entries(self) -> Iterable[CacheEntry]:
        for entry_dir in self.root.glob("*"):
            if not entry_dir.is_dir():
                continue
            meta_path = entry_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with meta_path.open("r", encoding="utf-8") as handle:
                    meta = json.load(handle)
            except json.JSONDecodeError:
                continue
            key = meta.get("key")
            if not key:
                continue
            yield self._entry_from_meta(str(key), meta)

    def _prune_expired(self) -> None:
        if self.ttl is None:
            return
        cutoff = time.time() - self.ttl
        for entry in list(self._iter_entries()):
            if entry.last_access < cutoff:
                self._delete_entry(entry.key)

    def _prune_limits(self) -> None:
        entries = sorted(self._iter_entries(), key=lambda e: e.last_access)
        total_bytes = sum(entry.size for entry in entries)

        while self.max_entries is not None and len(entries) > self.max_entries:
            victim = entries.pop(0)
            self._delete_entry(victim.key)

        while self.max_bytes is not None and total_bytes > self.max_bytes and entries:
            victim = entries.pop(0)
            total_bytes -= victim.size
            self._delete_entry(victim.key)

    def prune(self) -> None:
        if _is_disabled():
            return
        self._prune_expired()
        self._prune_limits()

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def index(self) -> List[CacheEntry]:
        return sorted(self._iter_entries(), key=lambda entry: entry.last_access, reverse=True)

    def get(self, key: str) -> Optional[bytes]:
        if _is_disabled():
            return None
        meta = self._load_meta(key)
        if meta is None:
            return None
        entry = self._entry_from_meta(key, meta)
        if self.ttl is not None and (time.time() - entry.created) > self.ttl:
            self._delete_entry(key)
            return None
        data_path = self._data_path(key)
        if not data_path.exists():
            self._delete_entry(key)
            return None
        payload = data_path.read_bytes()
        meta["last_access"] = time.time()
        self._write_meta(key, meta)
        os.utime(data_path, None)
        return payload

    def set(self, key: str, payload: bytes) -> None:
        if _is_disabled():
            return
        entry_dir = self._entry_dir(key)
        entry_dir.mkdir(parents=True, exist_ok=True)
        data_path = self._data_path(key)
        data_path.write_bytes(payload)
        now = time.time()
        meta = {
            "key": key,
            "created": now,
            "last_access": now,
        }
        self._write_meta(key, meta)
        self.prune()


__all__ = ["Cache", "CacheEntry"]

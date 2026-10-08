"""Content-addressed disk cache shared by all provider calls.

The key covers everything that can change the answer: namespace, model, prompt, generation
config, response schema, and the *bytes* of every input (hashed). Changing any of them is a miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def make_key(namespace: str, **parts: Any) -> str:
    """Stable key from JSON-able parts. Pass input bytes as pre-computed sha256 strings."""
    return sha256_bytes(canonical_json({"ns": namespace, **parts}).encode("utf-8"))


class DiskCache:
    """Tiny JSON-on-disk cache. Thread-safe counters; atomic writes; corrupt entries are misses."""

    def __init__(self, root: str | Path, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            with open(path, encoding="utf-8") as fh:
                entry = json.load(fh)
            value = entry["value"]
        except (OSError, ValueError, KeyError):
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return value

    def put(self, key: str, value: Any, meta: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "meta": meta or {}, "value": value},
                          fh, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses}

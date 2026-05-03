"""SHA-256 hashing utilities for the DLS audit system.

All functions are stateless and reusable across modules.
"""
import hashlib
import json
from typing import Any

# Sentinel used as the previous_hash of the very first log entry,
# mirroring the Bitcoin genesis-block convention.
GENESIS_HASH = "0" * 64


def hash_file(file_path: str) -> str:
    """Returns SHA-256 hex digest of a file's binary contents.

    Streams in 64 KB chunks so arbitrarily large files never cause OOM.

    Raises:
        ValueError: if the file is missing or unreadable.
    """
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError) as exc:
        raise ValueError(f"Cannot hash file '{file_path}': {exc}") from exc


def hash_text(text: str, encoding: str = "utf-8") -> str:
    """Returns SHA-256 hex digest of a text string.

    Raises:
        ValueError: on encoding failures.
    """
    try:
        return hashlib.sha256(text.encode(encoding)).hexdigest()
    except (UnicodeEncodeError, LookupError) as exc:
        raise ValueError(f"Cannot hash text: {exc}") from exc


def hash_json(data: Any) -> str:
    """Returns SHA-256 hex digest of deterministically serialized JSON.

    Canonical form: sort_keys=True, compact separators, ASCII-safe.
    This ensures the same dict always produces the same hash regardless
    of insertion order or Python version.

    Raises:
        ValueError: if data contains non-JSON-serializable values.
    """
    try:
        canonical = json.dumps(
            data, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot hash non-serializable data: {exc}") from exc

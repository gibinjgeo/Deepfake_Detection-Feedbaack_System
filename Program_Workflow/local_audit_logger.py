"""
Append-only, hash-chained audit logger — DLS Phase 1.

Storage decision: JSONL over SQLite
  - Append-only is a natural file operation; no SQL constraints needed.
  - Human-readable: tail, grep, and jq work directly on the log.
  - Records are immutable after write, so there is no benefit to a
    relational schema.
  - The future Solana bridge reads records sequentially for anchoring,
    not via relational queries — JSONL is a better structural fit.

Hash chain formula:
  current_hash = SHA256(JSON({event_type, payload_hash, previous_hash, timestamp}))

  Structured JSON is used instead of raw string concatenation to prevent
  length-extension ambiguity where "abc"+"def" would equal "ab"+"cdef".
  Keys are sorted so the serialization is fully deterministic.

Thread safety:
  log_event() holds an exclusive advisory flock for the read-then-write
  of the previous_hash/current_hash pair, so concurrent writers from
  multiple threads or processes serialize correctly.
"""

import fcntl
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hash_utils import GENESIS_HASH, hash_json

LOG_PATH = Path(__file__).parent / "dls_audit_log.jsonl"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_entry_hash(
    event_type: str,
    payload_hash: str,
    timestamp: str,
    previous_hash: str,
) -> str:
    """Derives the chaining hash for a single audit entry."""
    return hash_json({
        "event_type": event_type,
        "payload_hash": payload_hash,
        "previous_hash": previous_hash,
        "timestamp": timestamp,
    })


def _tail_hash(f: Any, size: int) -> str:
    """Returns current_hash from the last JSON line in an open binary file.

    Reads at most 8 KB from the end — O(1) regardless of log size.
    A typical entry is ~350 bytes, so 8 KB covers any reasonable record.
    """
    if size == 0:
        return GENESIS_HASH
    chunk_size = min(8192, size)
    f.seek(-chunk_size, 2)
    chunk = f.read(chunk_size).decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    if not lines:
        return GENESIS_HASH
    try:
        return json.loads(lines[-1]).get("current_hash", GENESIS_HASH)
    except json.JSONDecodeError:
        return GENESIS_HASH


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_event(event_type: str, payload_dict: dict[str, Any]) -> dict[str, Any]:
    """Appends a new event to the audit log and returns the written entry.

    The read of previous_hash and the write of the new entry are both
    performed under an exclusive file lock, making the operation atomic
    with respect to other processes or threads using this function.

    Args:
        event_type:   Short uppercase identifier, e.g. "DETECTION_RESULT".
        payload_dict: Arbitrary JSON-serializable event data.  The dict
                      itself is NOT written to the log — only its hash is,
                      keeping the log compact and privacy-friendly.

    Returns:
        The full entry dict that was appended to the log.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    payload_hash = hash_json(payload_dict)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Open in append+read binary mode so the file is created if absent and
    # both seek-for-read and append-write work on a single file descriptor.
    with open(LOG_PATH, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, 2)
            previous_hash = _tail_hash(f, f.tell())

            current_hash = _compute_entry_hash(
                event_type, payload_hash, timestamp, previous_hash
            )

            entry: dict[str, Any] = {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "payload_hash": payload_hash,
                "timestamp": timestamp,
                "previous_hash": previous_hash,
                "current_hash": current_hash,
            }

            # Append mode guarantees the OS seeks to EOF before each write.
            f.write((json.dumps(entry) + "\n").encode("utf-8"))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    return entry


def get_last_hash() -> str:
    """Returns the current_hash of the most recent log entry.

    Returns GENESIS_HASH if the log is absent or empty.
    """
    if not LOG_PATH.exists():
        return GENESIS_HASH
    try:
        with open(LOG_PATH, "rb") as f:
            f.seek(0, 2)
            return _tail_hash(f, f.tell())
    except OSError:
        return GENESIS_HASH


def verify_chain() -> tuple[bool, int, str]:
    """Validates the integrity of the entire audit log.

    Recomputes every current_hash and verifies that each previous_hash
    correctly links to the prior entry's current_hash.  A single
    corrupted byte in any entry will cause the chain to break.

    Returns:
        (is_valid, entries_checked, message)
        - is_valid:        True if the chain is fully intact.
        - entries_checked: Entries verified before the first failure,
                           or the total count on success.
        - message:         Human-readable result string.
    """
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return True, 0, "Log is empty — nothing to verify."

    expected_previous = GENESIS_HASH
    checked = 0

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    return False, checked, (
                        f"TAMPER DETECTED at line {lineno}: malformed JSON — {exc}"
                    )

                event_num = checked + 1

                if entry.get("previous_hash") != expected_previous:
                    return False, checked, (
                        f"TAMPER DETECTED at event {event_num} (line {lineno}): "
                        f"previous_hash mismatch"
                    )

                try:
                    recomputed = _compute_entry_hash(
                        entry["event_type"],
                        entry["payload_hash"],
                        entry["timestamp"],
                        entry["previous_hash"],
                    )
                except KeyError as exc:
                    return False, checked, (
                        f"TAMPER DETECTED at event {event_num} (line {lineno}): "
                        f"missing required field {exc}"
                    )

                if recomputed != entry.get("current_hash"):
                    return False, checked, (
                        f"TAMPER DETECTED at event {event_num} (line {lineno}): "
                        f"current_hash does not match recomputed value"
                    )

                expected_previous = entry["current_hash"]
                checked += 1

    except OSError as exc:
        return False, checked, f"Cannot read audit log: {exc}"

    return True, checked, f"CHAIN VALID — {checked} entries verified."

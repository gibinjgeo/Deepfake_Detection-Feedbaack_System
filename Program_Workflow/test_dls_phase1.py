#!/usr/bin/env python3
"""
Phase 1 DLS integration test.

Logs four representative events, verifies chain integrity, then demonstrates
tamper detection by corrupting a field in memory and re-running verification
against the on-disk log (which remains untouched).

Run from Program_Workflow/:
    python test_dls_phase1.py
"""

import json

from local_audit_logger import LOG_PATH, log_event, verify_chain


def _hr(title: str = "") -> None:
    print(("─" * 60) if not title else f"\n{'─' * 4} {title} {'─' * (53 - len(title))}")


def main() -> None:
    _hr()
    print("  DLS Phase 1 — Audit System Demo")
    _hr()
    print(f"  Log: {LOG_PATH}\n")

    # ------------------------------------------------------------------ #
    # Log four representative events                                       #
    # ------------------------------------------------------------------ #
    events = [
        ("MODEL_LOADED", {
            "model": "image_classifier",
            "checkpoint": "best_model.pt",
            "architecture": "resnet18",
        }),
        ("DETECTION_RESULT", {
            "video_file": "sample.mp4",
            "audio_verdict": "fake",
            "audio_confidence": 0.91,
            "image_verdict": "fake",
            "image_confidence": 0.87,
            "final_verdict": "fake",
        }),
        ("FEEDBACK_SUBMITTED", {
            "video_file": "sample.mp4",
            "user_label": "real",
            "model_verdict": "fake",
            "corrected": True,
        }),
        ("MODEL_RETRAINED", {
            "model": "image_classifier",
            "trigger": "threshold_reached",
            "samples_used": 42,
            "val_accuracy_before": 0.82,
            "val_accuracy_after": 0.89,
            "new_checkpoint": "best_model.pt",
        }),
    ]

    logged = []
    for i, (etype, payload) in enumerate(events, start=1):
        entry = log_event(etype, payload)
        logged.append(entry)
        print(f"[{i}] {etype}")
        print(f"     event_id      : {entry['event_id']}")
        print(f"     payload_hash  : {entry['payload_hash'][:32]}...")
        print(f"     previous_hash : {entry['previous_hash'][:32]}...")
        print(f"     current_hash  : {entry['current_hash'][:32]}...")
        print(f"     timestamp     : {entry['timestamp']}")

    # ------------------------------------------------------------------ #
    # Verify the chain                                                     #
    # ------------------------------------------------------------------ #
    _hr("Chain verification")
    is_valid, count, message = verify_chain()
    print(message)
    assert is_valid, "Chain should be valid after clean writes"

    # ------------------------------------------------------------------ #
    # Show the raw log entry for the detection event                       #
    # ------------------------------------------------------------------ #
    _hr("Example JSONL record (DETECTION_RESULT)")
    print(json.dumps(logged[1], indent=2))

    # ------------------------------------------------------------------ #
    # Tamper detection demo — corrupt the on-disk log, then re-verify     #
    # ------------------------------------------------------------------ #
    _hr("Tamper detection demo")
    print("Corrupting event 2's current_hash on disk...")

    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    original_line = lines[1]
    tampered = json.loads(original_line)
    tampered["current_hash"] = "deadbeef" * 8  # 64-char garbage hash
    lines[1] = json.dumps(tampered)
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    is_valid2, count2, message2 = verify_chain()
    print(message2)
    assert not is_valid2, "Tampered chain should be detected"

    # Restore the log
    print("\nRestoring original log...")
    lines[1] = original_line
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    is_valid3, count3, message3 = verify_chain()
    print(message3)
    assert is_valid3, "Restored chain should pass again"

    _hr()
    print("  All assertions passed.")
    _hr()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DLS audit log chain verification tool.

Usage:
    python verify_dls_audit.py
    python verify_dls_audit.py --log /custom/path/dls_audit_log.jsonl

Exit codes:
    0  — chain valid (or log is empty)
    1  — tamper detected or unreadable log
"""

import argparse
import sys
from pathlib import Path

import local_audit_logger as _aud


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify integrity of the DLS audit log hash chain."
    )
    parser.add_argument(
        "--log",
        metavar="PATH",
        help="Path to JSONL audit log (default: dls_audit_log.jsonl beside this script)",
    )
    args = parser.parse_args()

    if args.log:
        _aud.LOG_PATH = Path(args.log)

    print("DLS Audit Log Verifier")
    print(f"Log : {_aud.LOG_PATH}")
    print("-" * 60)

    is_valid, count, message = _aud.verify_chain()
    print(message)

    if is_valid and count > 0:
        last = _aud.get_last_hash()
        print(f"Tip : {last[:32]}...{last[-8:]}")

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

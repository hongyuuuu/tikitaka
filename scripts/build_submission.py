#!/usr/bin/env python3
"""Build the deterministic M6 participant archive outside the repository."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m6_release import build_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development only; final evidence rejects dirty revisions",
    )
    arguments = parser.parse_args()
    report = build_submission(
        arguments.output,
        catalog=arguments.catalog,
        allow_dirty=arguments.allow_dirty,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

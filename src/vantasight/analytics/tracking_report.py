"""Structured JSON export for sampled offline tracking runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_tracking_report(path: Path, report: dict[str, Any]) -> None:
    """Write a UTF-8 JSON report, rejecting non-finite numeric values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")

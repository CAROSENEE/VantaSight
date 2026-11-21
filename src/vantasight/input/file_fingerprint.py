"""Content fingerprints for local source-media files used by offline reports."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a local file."""

    source_path = Path(path).expanduser()
    digest = hashlib.sha256()
    try:
        with source_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise OSError(f"Could not fingerprint source media: {source_path}") from exc
    return digest.hexdigest()

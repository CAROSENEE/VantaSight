"""Stable fingerprints for decoded source frames used by manual annotations."""

from __future__ import annotations

import hashlib
import re

import numpy as np


def frame_sha256(pixels: np.ndarray) -> str:
    """Hash decoded pixel shape, dtype, and bytes for one source frame."""
    contiguous = np.ascontiguousarray(pixels)
    digest = hashlib.sha256()
    digest.update(f"{contiguous.dtype.str}:{contiguous.shape}".encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def verify_frame_fingerprint(record: dict, pixels: np.ndarray, frame_index: int) -> None:
    """Verify an optional annotation fingerprint; legacy annotations remain supported."""
    expected = record.get("source_frame_sha256")
    if expected is None:
        return
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"Invalid source_frame_sha256 at frame {frame_index}")
    actual = frame_sha256(pixels)
    if actual != expected:
        raise ValueError(f"Source frame fingerprint mismatch at frame {frame_index}")

"""What every source file reader returns (§5.2, §5.4, §5.5)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class SourceError(ValueError):
    """The file cannot be a source image: §5.5 rules it out, or this
    library cannot decode it. `reason` names which."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# Reasons
NOT_SUPPORTED = "not_supported"
"""Not a PNG, JPEG, TIFF or .scn file, or a variant this library does not
decode (TIFF compression, 12-bit JPEG …)."""
DAMAGED = "damaged"
NOT_A_SOURCE_IMAGE = "not_a_source_image"
"""§5.5 rules the file out (palette, CMYK, transparent pixels, bit depth,
signed or floating values, several images …)."""


@dataclass
class Decoded:
    format: str
    pixels: np.ndarray
    """(h, w) grey or (h, w, 3) RGB, uint8 or uint16, oriented and with 0 as
    black (§5.5)."""
    metadata: tuple[dict, bytes] | None = None
    notes: list[str] = field(default_factory=list)

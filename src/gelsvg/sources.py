"""Reading a source file into a source image (§5.2, §5.4, §5.5): its
channel images, named as §5.2 says, and its metadata in the formal format.
PNG, JPEG, TIFF and Bio-Rad Image Lab `.scn` files are read."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np

from . import _formal, _hashes, _source_jpeg, _source_png, _source_scn, _source_tiff
from ._sourcebase import DAMAGED, NOT_A_SOURCE_IMAGE, NOT_SUPPORTED, SourceError

__all__ = ["DAMAGED", "NOT_A_SOURCE_IMAGE", "NOT_SUPPORTED", "SourceError", "SourceImage", "read_source"]


@dataclass
class SourceImage:
    format: str
    """`png`, `jpeg`, `tiff` or `scn`."""
    file_name: str | None
    file_hash: str
    pixels: np.ndarray
    """(h, w) grey or (h, w, 3) RGB, uint8 or uint16: the values the file
    records, turned the way the file asks and with 0 as black (§5.5)."""
    metadata: tuple[dict, bytes] | None
    """The source metadata (catalog entry, payload), or None when the file
    cannot have it (§5.4; `notes` says why)."""
    notes: list[str] = field(default_factory=list)

    @property
    def dtype(self) -> str:
        return "uint8" if self.pixels.dtype == np.uint8 else "uint16"

    @property
    def shape(self) -> list[int]:
        return list(self.pixels.shape[:2])

    @property
    def channels(self) -> dict[str, np.ndarray]:
        """The channel images by name, in file order (§5.2)."""

        if self.pixels.ndim == 2:
            return {"gray": self.pixels}
        return {name: np.ascontiguousarray(self.pixels[..., i]) for i, name in enumerate(("red", "green", "blue"))}

    @property
    def channel_images(self) -> list[np.ndarray]:
        return list(self.channels.values())

    @property
    def source_metadata(self) -> list[tuple[dict, bytes]]:
        return [self.metadata] if self.metadata is not None else []

    def to_json(self, source_id: str) -> dict:
        """The source image as an element of `core.sources` (§5.2)."""

        out: dict = {"id": source_id}
        if self.file_name is not None:
            out["file_name"] = self.file_name
        out["file_hash"] = self.file_hash
        out["dtype"], out["shape"] = self.dtype, self.shape
        out["channels"] = [{"name": name, "pixel_hash": _hashes.pixel_hash(image)} for name, image in self.channels.items()]
        if self.metadata is not None:
            out["source_metadata"] = _formal.key(self.metadata[0])
        return out


def read_source(source: str | os.PathLike | bytes, file_name: str | None = None) -> SourceImage:
    """Read a source file, given as a path or as its bytes. Raises
    SourceError when it cannot be a source image."""

    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    else:
        path = Path(source)
        data = path.read_bytes()
        file_name = file_name if file_name is not None else path.name
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        decoded = _source_png.read(data)
    elif data[:3] == b"\xff\xd8\xff":
        decoded = _source_jpeg.read(data)
    elif data[:4] in (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+"):
        decoded = _source_tiff.read(data)
    elif _source_scn.is_scn(data):
        decoded = _source_scn.read(data)
    else:
        raise SourceError(NOT_SUPPORTED, "The file is not a PNG, JPEG, TIFF or Bio-Rad Image Lab .scn file.")
    return SourceImage(decoded.format, file_name, _hashes.sha256(data), decoded.pixels, decoded.metadata,
                       list(decoded.notes))

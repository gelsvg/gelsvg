"""The orientation a source file asks for (§5.5), and turning pixels to it."""

from __future__ import annotations

import re

import numpy as np

from . import _tiff

ORIENTATION_TAG = 274
_XMP_ORIENTATION = re.compile(rb'tiff:Orientation(?:="|>)([0-9])')

# Rows run down, columns right (§5.5).
_SHOWN = {
    1: lambda a: a,
    2: lambda a: a[:, ::-1],
    3: lambda a: a[::-1, ::-1],
    4: lambda a: a[::-1, :],
    5: lambda a: np.swapaxes(a, 0, 1),
    6: lambda a: np.rot90(a, k=-1),
    7: lambda a: np.swapaxes(a, 0, 1)[::-1, ::-1],
    8: lambda a: np.rot90(a, k=1),
}


class Unreadable(Exception):
    """The EXIF block is damaged: no orientation is applied, and XMP is not
    looked at (§5.5)."""


def valid(value) -> int:
    return value if value in _SHOWN else 1


def apply(pixels: np.ndarray, orientation: int) -> np.ndarray:
    """The stored pixels, (h, w) or (h, w, c), turned the way the
    orientation value says to show them."""

    return np.ascontiguousarray(_SHOWN[valid(orientation)](pixels))


def from_ifd0(tiff: _tiff.Tiff, entries: list[_tiff.Entry]) -> int | None:
    """The value of IFD0's 274, or None when it has none. A 274 that is not
    one SHORT counts as 1."""

    entry = _tiff.first(entries, ORIENTATION_TAG)
    if entry is None:
        return None
    if entry.type != 3 or entry.count != 1 or entry.value is None:
        return 1
    return valid(tiff.ints(entry)[0])


def from_exif(block: bytes) -> int | None:
    """274 of an EXIF block (the TIFF-shaped bytes, after any `Exif\\0\\0`).
    Raises Unreadable when the block is damaged."""

    if block[:4] not in (b"II*\0", b"MM\0*"):
        raise Unreadable("the EXIF block does not start with a TIFF header")
    try:
        tiff = _tiff.Tiff(block, bigtiff=False)
        entries, _ = tiff.ifd(tiff.first_ifd)
    except _tiff.TiffError as exc:
        raise Unreadable(str(exc)) from None
    return from_ifd0(tiff, entries)


def from_xmp(text: bytes | None) -> int | None:
    if not text:
        return None
    match = _XMP_ORIENTATION.search(text)
    return None if match is None else valid(int(match.group(1)))


def decide(exif: bytes | None, xmp: bytes | None) -> int:
    """§5.5: EXIF's 274, else XMP's tiff:Orientation, else 1; nothing when
    the EXIF block is damaged."""

    if exif is not None:
        try:
            value = from_exif(exif)
        except Unreadable:
            return 1
        if value is not None:
            return value
    value = from_xmp(xmp)
    return 1 if value is None else value

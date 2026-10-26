"""A JPEG source file (§5.4 `jpeg`, §5.5). Pillow decodes the pixels; the
segments are walked here."""

from __future__ import annotations

import io
import struct

import numpy as np
from PIL import JpegImagePlugin

from . import _formal, _orientation, _tiff
from ._sourcebase import DAMAGED, NOT_A_SOURCE_IMAGE, NOT_SUPPORTED, Decoded, SourceError

_KEPT = {**{0xE0 + i: f"APP{i}" for i in range(16)}, 0xFE: "COM"}
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_XMP_HEAD = b"http://ns.adobe.com/xap/1.0/\0"


def _walk(data: bytes) -> tuple[list[tuple[int, bytes]], bool]:
    """(marker, contents) of each segment from SOI to the first EOI, and
    whether the walk got there (§5.4's way of walking)."""

    segments, pos, size = [], 2, len(data)
    while True:
        if pos < size and data[pos] != 0xFF:
            found = data.find(b"\xff", pos)  # stray bytes between segments
            pos = size if found < 0 else found
        while pos < size and data[pos] == 0xFF:
            pos += 1
        if pos >= size:
            return segments, False
        marker = data[pos]
        pos += 1
        if marker == 0xD9:  # EOI
            return segments, True
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # TEM, RSTn: no length
            continue
        if pos + 2 > size:
            return segments, False
        length = struct.unpack_from(">H", data, pos)[0]
        if length < 2 or pos + length > size:
            return segments, False
        segments.append((marker, data[pos + 2 : pos + length]))
        pos += length
        if marker == 0xDA:  # SOS: skip the scan to the next marker
            while True:
                found = data.find(b"\xff", pos)
                if found < 0 or found + 1 >= size:
                    return segments, False
                following = data[found + 1]
                if following == 0x00 or 0xD0 <= following <= 0xD7:
                    pos = found + 2
                elif following == 0xFF:
                    pos = found + 1
                else:
                    pos = found
                    break


def _more_than_one_image(segments) -> bool:
    """An APP2 MPF block pointing to a second image (§5.5)."""

    for marker, body in segments:
        if marker != 0xE2 or not body.startswith(b"MPF\0"):
            continue
        try:
            tiff = _tiff.Tiff(body[4:], bigtiff=False)
            entries, _ = tiff.ifd(tiff.first_ifd)
        except _tiff.TiffError:
            continue
        number = _tiff.first(entries, 0xB001)
        values = tiff.ints(number) if number is not None else None
        listing = _tiff.first(entries, 0xB002)
        if (values and values[0] > 1) or (listing is not None and listing.count // 16 > 1):
            return True
    return False


def _orientation_of(segments) -> int:
    exif = next((body[6:] for m, body in segments if m == 0xE1 and body.startswith(b"Exif\0\0")), None)
    xmp = next((body[len(_XMP_HEAD):] for m, body in segments if m == 0xE1 and body.startswith(_XMP_HEAD)), None)
    return _orientation.decide(exif, xmp)


def read(data: bytes) -> Decoded:
    segments, complete = _walk(data)
    frame = next((body for m, body in segments if m in _SOF), None)
    if frame is not None and len(frame) >= 6:
        if frame[5] == 4:
            raise SourceError(NOT_A_SOURCE_IMAGE, "The JPEG file is CMYK; its values are not brightness.")
        if frame[0] != 8:
            raise SourceError(NOT_SUPPORTED, f"The JPEG file has {frame[0]}-bit samples, which this library cannot decode.")
    if _more_than_one_image(segments):
        raise SourceError(NOT_A_SOURCE_IMAGE, "The JPEG file holds more than one image (MPF).")
    try:
        image = JpegImagePlugin.JpegImageFile(io.BytesIO(data))
        image.load()
    except (OSError, SyntaxError, ValueError) as exc:
        raise SourceError(DAMAGED, f"The JPEG file cannot be decoded: {exc}.") from None
    if image.mode not in ("L", "RGB"):
        raise SourceError(NOT_A_SOURCE_IMAGE, f"The JPEG file's colours ({image.mode}) are not grey or RGB.")
    pixels = np.asarray(image)
    pixels = _orientation.apply(pixels, _orientation_of(segments))
    notes = []
    metadata = None
    if not complete:
        notes.append("The JPEG file's segments cannot be walked to its end marker, so its metadata is not kept (§5.4).")
    else:
        metadata = _formal.build("jpeg", [({"marker": _KEPT[m]}, body) for m, body in segments if m in _KEPT])
    return Decoded("jpeg", pixels, metadata, notes)

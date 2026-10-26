"""A TIFF source file (§5.4 `tiff`, §5.5).

The strips or tiles are taken apart here. Pillow only undoes LZW and
PackBits: each strip or tile is handed to it inside a small TIFF of 8-bit
grey rows, so what comes back is the decompressed bytes as they are, and
the byte order, predictor, sample layout, alpha, orientation and white-is-
zero are all applied here, by the specification.
"""

from __future__ import annotations

import io
import math
import struct
import zlib

import numpy as np
from PIL import TiffImagePlugin

from . import _formal, _orientation, _tiff
from ._sourcebase import DAMAGED, NOT_A_SOURCE_IMAGE, NOT_SUPPORTED, Decoded, SourceError

WIDTH, LENGTH, BITS, COMPRESSION, PHOTOMETRIC, FILL_ORDER = 256, 257, 258, 259, 262, 266
STRIP_OFFSETS, SAMPLES, ROWS_PER_STRIP, STRIP_COUNTS, PLANAR = 273, 277, 278, 279, 284
PREDICTOR, TILE_WIDTH, TILE_LENGTH, TILE_OFFSETS, TILE_COUNTS = 317, 322, 323, 324, 325
SUB_IFDS, EXTRA_SAMPLES, SAMPLE_FORMAT, XMP = 330, 338, 339, 700
EXIF_POINTER, GPS_POINTER, INTEROP_POINTER = 34665, 34853, 40965
# §5.4: entries that only say where the pixels lie.
LAYOUT_TAGS = frozenset({273, 279, 288, 289, 324, 325, 513, 514})
COMPRESSIONS = {1: "none", 5: "LZW", 8: "Deflate", 32946: "Deflate", 32773: "PackBits"}


def _pillow_decompress(chunk: bytes, compression: int, row_bytes: int, rows: int) -> bytes:
    entries = [(WIDTH, 4, row_bytes), (LENGTH, 4, rows), (BITS, 3, 8), (COMPRESSION, 3, compression),
               (PHOTOMETRIC, 3, 1), (STRIP_OFFSETS, 4, 8 + 2 + 12 * 9 + 4), (SAMPLES, 3, 1),
               (ROWS_PER_STRIP, 4, rows), (STRIP_COUNTS, 4, len(chunk))]
    ifd = struct.pack("<H", len(entries)) + b"".join(
        struct.pack("<HHI", tag, kind, 1) + (struct.pack("<HH", value, 0) if kind == 3 else struct.pack("<I", value))
        for tag, kind, value in entries) + struct.pack("<I", 0)
    image = TiffImagePlugin.TiffImageFile(io.BytesIO(b"II*\0" + struct.pack("<I", 8) + ifd + chunk))
    image.load()
    return image.tobytes()


def _decompress(chunk: bytes, compression: int, row_bytes: int, rows: int) -> bytes:
    expected = row_bytes * rows
    try:
        if compression == 1:
            out = chunk
        elif compression in (8, 32946):
            out = zlib.decompressobj().decompress(chunk, expected)
        else:
            out = _pillow_decompress(chunk, compression, row_bytes, rows)
    except (zlib.error, OSError, ValueError, SyntaxError) as exc:
        raise SourceError(DAMAGED, f"A strip or tile of the TIFF file cannot be decompressed ({exc}).") from None
    if len(out) < expected:
        raise SourceError(DAMAGED, "A strip or tile of the TIFF file is shorter than its image needs.")
    return out[:expected]


def _samples(raw: bytes, rows: int, width: int, per_pixel: int, bits: int, endian: str) -> np.ndarray:
    """(rows, width, per_pixel) samples of one strip or tile."""

    if bits == 16:
        values = np.frombuffer(raw, dtype=endian + "u2").astype(np.uint16)
        return values.reshape(rows, width, per_pixel)
    data = np.frombuffer(raw, dtype=np.uint8).reshape(rows, -1)
    if bits == 8:
        return data.reshape(rows, width, per_pixel)
    shifts = np.arange(8 - bits, -1, -bits, dtype=np.uint8)
    values = (data[:, :, None] >> shifts) & ((1 << bits) - 1)
    return values.reshape(rows, -1)[:, :width, None].astype(np.uint8)


class _Tags:
    def __init__(self, tiff: _tiff.Tiff, entries):
        self.tiff, self.entries = tiff, entries

    def get(self, tag: int, default=None) -> list[int] | None:
        entry = _tiff.first(self.entries, tag)
        if entry is None:
            return default
        values = self.tiff.ints(entry)
        if values is None or not values:
            raise SourceError(DAMAGED, f"The TIFF tag {tag} does not hold whole numbers inside the file.")
        return values

    def one(self, tag: int, default=None) -> int | None:
        values = self.get(tag)
        return default if values is None else values[0]


def _pixels(tiff: _tiff.Tiff, tags: _Tags) -> tuple[np.ndarray, int]:
    """The samples, 0 as black and alpha dropped, before orientation; and
    the bit depth."""

    width, height = tags.one(WIDTH), tags.one(LENGTH)
    if not width or not height:
        raise SourceError(DAMAGED, "The TIFF file has no image width or height.")
    per_pixel = tags.one(SAMPLES, 1)
    photometric = tags.one(PHOTOMETRIC)
    if photometric is None:
        raise SourceError(DAMAGED, "The TIFF file has no PhotometricInterpretation.")
    if photometric not in (0, 1, 2):
        raise SourceError(NOT_A_SOURCE_IMAGE, f"The TIFF file's colours (PhotometricInterpretation {photometric}) "
                          "are not grey or RGB.")
    extra = tags.get(EXTRA_SAMPLES, [])
    base = 3 if photometric == 2 else 1
    if per_pixel != base + len(extra) or len(extra) > 1 or any(e not in (1, 2) for e in extra):
        raise SourceError(NOT_A_SOURCE_IMAGE, "The TIFF file has samples other than grey or RGB and one alpha.")
    bits_list = tags.get(BITS, [1])
    bits_list = bits_list * per_pixel if len(bits_list) == 1 else bits_list
    if len(set(bits_list)) != 1 or len(bits_list) != per_pixel:
        raise SourceError(NOT_A_SOURCE_IMAGE, "The TIFF file's samples have different bit depths.")
    bits = bits_list[0]
    if bits not in ((1, 2, 4, 8, 16) if per_pixel == 1 else (8, 16)):
        raise SourceError(NOT_A_SOURCE_IMAGE, f"The TIFF file has {bits}-bit samples.")
    formats = tags.get(SAMPLE_FORMAT, [1])
    if any(f != 1 for f in formats):
        raise SourceError(NOT_A_SOURCE_IMAGE, "The TIFF file's values are signed or floating point.")
    compression = tags.one(COMPRESSION, 1)
    if compression not in COMPRESSIONS:
        raise SourceError(NOT_SUPPORTED, f"This library cannot decompress TIFF compression {compression}.")
    predictor = tags.one(PREDICTOR, 1)
    if predictor not in (1, 2) or (predictor == 2 and bits < 8):
        raise SourceError(NOT_SUPPORTED, f"This library cannot undo TIFF predictor {predictor} for {bits}-bit samples.")
    if tags.one(FILL_ORDER, 1) != 1:
        raise SourceError(NOT_SUPPORTED, "This library cannot read TIFF bits in reversed fill order.")
    planar = tags.one(PLANAR, 1)
    if planar not in (1, 2):
        raise SourceError(DAMAGED, f"The TIFF file has PlanarConfiguration {planar}.")
    planes = per_pixel if planar == 2 else 1
    chunky = per_pixel if planar == 1 else 1

    if _tiff.first(tags.entries, TILE_WIDTH) is not None:
        tile_w, tile_h = tags.one(TILE_WIDTH), tags.one(TILE_LENGTH)
        offsets, counts = tags.get(TILE_OFFSETS), tags.get(TILE_COUNTS)
        if not tile_w or not tile_h:
            raise SourceError(DAMAGED, "The TIFF file's tile size is zero.")
        across, down = math.ceil(width / tile_w), math.ceil(height / tile_h)
        places = [(r * tile_h, c * tile_w, tile_h, tile_w) for r in range(down) for c in range(across)]
    else:
        rows_per = min(tags.one(ROWS_PER_STRIP, 2**32 - 1), height)
        offsets, counts = tags.get(STRIP_OFFSETS), tags.get(STRIP_COUNTS)
        if not rows_per:
            raise SourceError(DAMAGED, "The TIFF file's RowsPerStrip is zero.")
        places = [(top, 0, min(rows_per, height - top), width) for top in range(0, height, rows_per)]
    if offsets is None or len(offsets) != len(places) * planes:
        raise SourceError(DAMAGED, "The TIFF file's strip or tile offsets do not match its image.")
    if counts is None and compression != 1:
        raise SourceError(DAMAGED, "The compressed TIFF file has no strip or tile byte counts.")
    if counts is not None and len(counts) != len(offsets):
        raise SourceError(DAMAGED, "The TIFF file's strip or tile byte counts do not match its offsets.")

    dtype = np.uint16 if bits == 16 else np.uint8
    out = np.zeros((height, width, per_pixel), dtype=dtype)
    endian = tiff.endian
    for plane in range(planes):
        for i, (top, left, rows, cols) in enumerate(places):
            k = plane * len(places) + i
            row_bytes = (cols * chunky * bits + 7) // 8
            size = counts[k] if counts is not None else row_bytes * rows
            start = offsets[k]
            if start + size > len(tiff.data):
                raise SourceError(DAMAGED, "A strip or tile of the TIFF file lies past the end of the file.")
            raw = _decompress(tiff.data[start : start + size], compression, row_bytes, rows)
            block = _samples(raw, rows, cols, chunky, bits, endian)
            if predictor == 2:
                block = np.cumsum(block, axis=1, dtype=dtype)
            h, w = min(rows, height - top), min(cols, width - left)
            channel = slice(None) if planar == 1 else slice(plane, plane + 1)
            out[top : top + h, left : left + w, channel] = block[:h, :w]

    if extra:
        top_value = (1 << bits) - 1
        if (out[..., -1] != top_value).any():
            raise SourceError(NOT_A_SOURCE_IMAGE, "The TIFF file has transparent pixels.")
        out = out[..., :-1]
    if bits < 8:
        out = out * np.uint8(255 // (2**bits - 1))
    if photometric == 0:
        out = (np.iinfo(dtype).max - out).astype(dtype)
    pixels = out[..., 0] if out.shape[2] == 1 else out
    return pixels, bits


def _metadata(tiff: _tiff.Tiff, ifd0) -> tuple[dict, bytes] | None:
    """§5.4 `tiff` version 1, or None when the file is not one it can be
    used for."""

    collected: dict[str, list] = {}
    pointers: dict[str, int] = {}
    visited = {tiff.first_ifd}

    def collect(name: str, entries, pointer_tags: dict[int, str]) -> bool:
        kept = []
        for e in entries:
            if not e.known_type or not e.inside:
                return False
            if e.tag in pointer_tags:
                if e.count != 1 or _tiff.TYPE_SIZES[e.type] not in (4, 8):
                    return False
                target = struct.unpack(tiff.endian + ("I" if len(e.value) == 4 else "Q"), e.value)[0]
                if target:
                    pointers[pointer_tags[e.tag]] = target
                continue
            if e.tag not in LAYOUT_TAGS:
                kept.append(({"ifd": name, "tag": e.tag, "type": e.type, "count": e.count}, e.value))
        collected[name] = kept
        return True

    def follow(name: str, pointer_tags: dict[int, str]) -> bool:
        if name not in pointers:
            return True
        offset = pointers[name]
        if offset in visited:
            return False
        visited.add(offset)
        try:
            entries, _ = tiff.ifd(offset)
        except _tiff.TiffError:
            return False
        return collect(name, entries, pointer_tags)

    if not (collect("ifd0", ifd0, {EXIF_POINTER: "exif", GPS_POINTER: "gps"})
            and follow("exif", {INTEROP_POINTER: "interop"}) and follow("interop", {}) and follow("gps", {})):
        return None
    segments = [part for name in ("ifd0", "exif", "interop", "gps") for part in collected.get(name, ())]
    return _formal.build("tiff", segments, byte_order=tiff.byte_order, tiff_version=tiff.version)


def read(data: bytes) -> Decoded:
    try:
        tiff = _tiff.Tiff(data)
        if tiff.first_ifd == 0:
            raise SourceError(DAMAGED, "The TIFF file holds no image.")
        entries, next_ifd = tiff.ifd(tiff.first_ifd)
    except _tiff.TiffError as exc:
        raise SourceError(DAMAGED, f"The TIFF file's structure cannot be read ({exc}).") from None
    if next_ifd != 0:
        raise SourceError(NOT_A_SOURCE_IMAGE, "The TIFF file holds more than one image (pages).")
    if _tiff.first(entries, SUB_IFDS) is not None:
        raise SourceError(NOT_A_SOURCE_IMAGE, "The TIFF file holds more than one image (SubIFDs).")
    tags = _Tags(tiff, entries)
    pixels, _ = _pixels(tiff, tags)

    orientation = _orientation.from_ifd0(tiff, entries)
    if orientation is None:
        xmp = _tiff.first(entries, XMP)
        orientation = _orientation.from_xmp(xmp.value if xmp is not None else None) or 1
    pixels = _orientation.apply(pixels, orientation)
    metadata = _metadata(tiff, entries)
    notes = [] if metadata else ["The TIFF file has entries this format cannot keep as they are (an unknown field "
                                 "type, a value outside the file, or IFDs that loop), so its metadata is not kept (§5.4)."]
    return Decoded("tiff", pixels, metadata, notes)

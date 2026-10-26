"""Writing PNGs: chunks, filtered scanlines, channel images and the parts of
a carrier PNG (§1.3, §1.4, §5.3)."""

from __future__ import annotations

import hashlib
import struct
import zlib

import numpy as np

from . import _png


def chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + chunk_type + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF))


def ihdr(width: int, height: int, bit_depth: int, color_type: int) -> bytes:
    return chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0))


IEND = chunk(b"IEND", b"")


def _filter_block(raw: np.ndarray, above: np.ndarray, unit: int) -> np.ndarray:
    """Filter rows given the unfiltered row above the first one, choosing a
    filter per row by the smallest sum of absolute differences (the usual
    heuristic). Returns the scanlines with their filter bytes."""

    height = raw.shape[0]
    zero_col = np.zeros((height, unit), dtype=np.uint8)
    up = np.vstack([above[None, :], raw[:-1]])
    left = np.hstack([zero_col, raw[:, :-unit]])
    upper_left = np.hstack([zero_col, up[:, :-unit]])
    a16, b16, c16 = left.astype(np.int16), up.astype(np.int16), upper_left.astype(np.int16)
    p = a16 + b16 - c16
    pa, pb, pc = np.abs(p - a16), np.abs(p - b16), np.abs(p - c16)
    paeth = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, upper_left))
    candidates = np.stack([
        raw,
        raw - left,
        raw - up,
        raw - ((left.astype(np.uint16) + up) >> 1).astype(np.uint8),
        raw - paeth,
    ])
    scores = np.abs(candidates.view(np.int8).astype(np.int16)).sum(axis=2)
    choice = scores.argmin(axis=0)
    rows = candidates[choice, np.arange(height)]
    return np.hstack([choice.astype(np.uint8)[:, None], rows])


def _filtered_rows(raw: np.ndarray, unit: int, block: int = 256) -> bytes:
    above = np.zeros(raw.shape[1], dtype=np.uint8)
    out = []
    for start in range(0, raw.shape[0], block):
        part = raw[start : start + block]
        out.append(_filter_block(part, above, unit).tobytes())
        above = part[-1]
    return b"".join(out)


def encode_image(pixels: np.ndarray) -> tuple[bytes, bytes]:
    """The IHDR chunk and the zlib stream of a grey or RGB image of 8 or 16
    bits (no interlace)."""

    if pixels.dtype not in (np.uint8, np.uint16) or pixels.ndim not in (2, 3):
        raise ValueError("pixels must be uint8 or uint16, grey (h, w) or RGB (h, w, 3)")
    if pixels.ndim == 3 and pixels.shape[2] != 3:
        raise ValueError("an RGB image has three values per pixel")
    height, width = pixels.shape[:2]
    channels = 1 if pixels.ndim == 2 else 3
    depth = 8 if pixels.dtype == np.uint8 else 16
    if depth == 8:
        raw = np.ascontiguousarray(pixels).reshape(height, width * channels)
    else:
        raw = np.ascontiguousarray(pixels.astype(">u2")).view(np.uint8).reshape(height, width * channels * 2)
    unit = channels * depth // 8
    stream = zlib.compress(_filtered_rows(raw, unit), 9)
    return ihdr(width, height, depth, 0 if channels == 1 else 2), stream


def idat_chunks(stream: bytes, size: int = 1 << 20) -> bytes:
    return b"".join(chunk(b"IDAT", stream[i : i + size]) for i in range(0, len(stream), size)) or chunk(b"IDAT", b"")


def channel_png(pixels: np.ndarray) -> bytes:
    """A channel image PNG: greyscale, IHDR, IDAT and IEND only (§5.3)."""

    if pixels.ndim != 2:
        raise ValueError("a channel image has one value per pixel")
    header, stream = encode_image(pixels)
    return _png.PNG_SIGNATURE + header + idat_chunks(stream) + IEND


def itxt(text: bytes, keyword: bytes = b"GelSVG", compressed: bool = True,
         language: bytes = b"", translated: bytes = b"") -> bytes:
    body = zlib.compress(text, 9) if compressed else text
    return chunk(b"iTXt", keyword + b"\0" + bytes([1 if compressed else 0, 0]) + language + b"\0"
                 + translated + b"\0" + body)


def gels(payload: bytes, codec: int) -> bytes:
    header = (_png.GELS_MAGIC + bytes([_png.GELS_HEADER_VERSION, codec]) + struct.pack(">HI", 0, len(payload))
              + hashlib.sha256(payload).digest())
    return chunk(_png.GELS_TYPE, header + payload)


def gels_chunks(channel_payloads, metadata_payloads) -> bytes:
    """All geLs chunks in the recommended order: codec 1 by payload SHA-256,
    then codec 2 the same way (§1.4)."""

    def ordered(payloads):
        unique = {hashlib.sha256(p).digest(): p for p in payloads}
        return [unique[k] for k in sorted(unique)]

    return (b"".join(gels(p, 1) for p in ordered(channel_payloads))
            + b"".join(gels(p, 2) for p in ordered(metadata_payloads)))

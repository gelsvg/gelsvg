"""Build PNGs, carrier PNGs and SVGs byte by byte, so tests can make exactly
the good and broken files they need."""

from __future__ import annotations

import base64
import hashlib
import struct
import zlib

import numpy as np

SIG = b"\x89PNG\r\n\x1a\n"
ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def chunk(chunk_type: bytes, data: bytes, crc: int | None = None) -> bytes:
    if crc is None:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def ihdr(width: int, height: int, depth: int = 8, color_type: int = 0, interlace: int = 0) -> bytes:
    return chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, interlace))


IEND = chunk(b"IEND", b"")


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def filter_rows(rows: list[bytes], unit: int, filters: list[int]) -> bytes:
    out = bytearray()
    previous = bytes(len(rows[0])) if rows else b""
    for row, kind in zip(rows, filters):
        encoded = bytearray(len(row))
        for i, value in enumerate(row):
            a = row[i - unit] if i >= unit else 0
            b = previous[i]
            c = previous[i - unit] if i >= unit else 0
            prediction = (0, a, b, (a + b) // 2, _paeth(a, b, c))[kind]
            encoded[i] = (value - prediction) & 0xFF
        out.append(kind)
        out += encoded
        previous = row
    return bytes(out)


def _row_bytes(pixels: np.ndarray, depth: int) -> list[bytes]:
    if depth == 16:
        return [row.astype(">u2").tobytes() for row in pixels]
    return [row.astype(np.uint8).tobytes() for row in pixels]


def scanlines(pixels: np.ndarray, depth: int = 8, filters=None, interlace: int = 0) -> bytes:
    channels = 1 if pixels.ndim == 2 else pixels.shape[2]
    unit = max(1, depth * channels // 8)
    if interlace == 0:
        rows = _row_bytes(pixels, depth)
        kinds = filters if filters is not None else [0] * len(rows)
        return filter_rows(rows, unit, list(kinds))
    out = b""
    height, width = pixels.shape[:2]
    for number, (x0, y0, dx, dy) in enumerate(ADAM7):
        part = pixels[y0::dy, x0::dx]
        if part.shape[0] == 0 or part.shape[1] == 0:
            continue
        rows = _row_bytes(part, depth)
        kinds = [(number + r) % 5 for r in range(len(rows))] if filters == "cycle" else [0] * len(rows)
        out += filter_rows(rows, unit, kinds)
    return out


def image_png(pixels: np.ndarray, depth: int = 8, filters=None, interlace: int = 0,
              before_idat=(), after_idat=(), idat_parts: int = 1, raw=None) -> bytes:
    height, width = pixels.shape[:2]
    color_type = 0 if pixels.ndim == 2 else 2
    data = raw if raw is not None else zlib.compress(scanlines(pixels, depth, filters, interlace))
    size = max(1, -(-len(data) // idat_parts))
    idats = b"".join(chunk(b"IDAT", data[i : i + size]) for i in range(0, len(data), size))
    return (SIG + ihdr(width, height, depth, color_type, interlace) + b"".join(before_idat)
            + idats + b"".join(after_idat) + IEND)


def itxt(text: bytes, keyword: bytes = b"GelSVG", compressed: bool = False) -> bytes:
    body = zlib.compress(text) if compressed else text
    return chunk(b"iTXt", keyword + b"\0" + bytes([1 if compressed else 0, 0]) + b"\0" + b"\0" + body)


def gels_data(payload: bytes, codec: int = 1, version: int = 1, flags: int = 0,
              magic: bytes = b"GSVG", payload_length: int | None = None, sha: bytes | None = None) -> bytes:
    if payload_length is None:
        payload_length = len(payload)
    if sha is None:
        sha = hashlib.sha256(payload).digest()
    return magic + bytes([version, codec]) + struct.pack(">HI", flags, payload_length) + sha + payload


def gels(payload: bytes, **kwargs) -> bytes:
    return chunk(b"geLs", gels_data(payload, **kwargs))


def small_pixels(height: int = 3, width: int = 4, rgb: bool = False, depth: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    top = 255 if depth == 8 else 65535
    shape = (height, width, 3) if rgb else (height, width)
    return rng.integers(0, top + 1, size=shape, dtype=np.uint16 if depth == 16 else np.uint8)


def carrier(json_text: bytes = b"{}", pixels: np.ndarray | None = None, depth: int = 8,
            before=(), after=(), compressed: bool = False) -> bytes:
    if pixels is None:
        pixels = small_pixels(depth=depth)
    return image_png(pixels, depth=depth, before_idat=(itxt(json_text, compressed=compressed), *before),
                     after_idat=after)


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def svg(*images: str, xlink: bool = False, prolog: str = "") -> bytes:
    attr = "xlink:href" if xlink else "href"
    body = "".join(f'<image {attr}="{href}" width="10" height="10"/>' for href in images)
    ns = ' xmlns:xlink="http://www.w3.org/1999/xlink"' if xlink else ""
    return (prolog + f'<svg xmlns="http://www.w3.org/2000/svg"{ns}>{body}</svg>').encode("utf-8")


def flip_bit(data: bytes, offset: int, bit: int = 0) -> bytes:
    out = bytearray(data)
    out[offset] ^= 1 << bit
    return bytes(out)

"""Source files built byte by byte (TIFF, PNG, .scn), for the cases Pillow
cannot write: 16-bit colour, tiles, planes, predictors, odd tags."""

from __future__ import annotations

import io
import math
import struct
import zlib

import numpy as np
from PIL import Image

import builders as B

_FORMATS = {1: "B", 3: "H", 4: "I", 7: "B", 9: "i", 13: "I", 16: "Q", 18: "Q"}
SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8}


def value(endian: str, kind: int, values) -> tuple[int, bytes]:
    """(count, bytes) of an entry's value."""

    if isinstance(values, (bytes, bytearray)):
        return len(values) // SIZES.get(kind, 1), bytes(values)
    return len(values), struct.pack(f"{endian}{len(values)}{_FORMATS[kind]}", *values)


class TiffWriter:
    def __init__(self, endian: str = "<", big: bool = False):
        self.endian, self.big = endian, big
        mark = b"II" if endian == "<" else b"MM"
        if big:
            self.buf = bytearray(mark + struct.pack(endian + "HHHQ", 43, 8, 0, 0))
        else:
            self.buf = bytearray(mark + struct.pack(endian + "HI", 42, 0))

    def add(self, data: bytes) -> int:
        if len(self.buf) % 2:
            self.buf += b"\0"
        offset = len(self.buf)
        self.buf += data
        return offset

    def ifd(self, entries: list[tuple], next_offset: int = 0) -> int:
        """entries: (tag, type, values) or (tag, type, count, raw bytes)."""

        e, big = self.endian, self.big
        inline = 8 if big else 4
        packed = []
        for item in sorted(entries, key=lambda t: t[0]):
            if len(item) == 4:
                tag, kind, count, raw = item
            else:
                tag, kind, values = item
                count, raw = value(e, kind, values)
            if len(raw) > inline:
                field = struct.pack(e + ("Q" if big else "I"), self.add(raw))
            else:
                field = raw.ljust(inline, b"\0")
            packed.append(struct.pack(e + ("HHQ" if big else "HHI"), tag, kind, count) + field)
        body = struct.pack(e + ("Q" if big else "H"), len(packed)) + b"".join(packed)
        return self.add(body + struct.pack(e + ("Q" if big else "I"), next_offset))

    def finish(self, first_ifd: int) -> bytes:
        if self.big:
            self.buf[8:16] = struct.pack(self.endian + "Q", first_ifd)
        else:
            self.buf[4:8] = struct.pack(self.endian + "I", first_ifd)
        return bytes(self.buf)


def pillow_compress(data: bytes, compression: str, row_bytes: int) -> bytes:
    """LZW or PackBits bytes of `data`, made by Pillow writing them as an
    8-bit grey TIFF one strip long."""

    rows = len(data) // row_bytes
    image = Image.frombytes("L", (row_bytes, rows), data)
    buf = io.BytesIO()
    image.save(buf, format="TIFF", compression=compression, tiffinfo={278: rows})
    tiff = Image.open(io.BytesIO(buf.getvalue()))
    (offset,), (count,) = tiff.tag_v2[273], tiff.tag_v2[279]
    return buf.getvalue()[offset : offset + count]


def _encode(raw: bytes, compression: int, row_bytes: int) -> bytes:
    if compression == 1:
        return raw
    if compression in (8, 32946):
        return zlib.compress(raw)
    return pillow_compress(raw, {5: "tiff_lzw", 32773: "packbits"}[compression], row_bytes)


def _row_bytes(samples: np.ndarray, bits: int, endian: str) -> bytes:
    """One block of samples (rows, width, per_pixel) as file bytes."""

    if bits > 8:  # 16, or a depth only refusals are tested with
        return samples.astype(endian + "u2").tobytes()
    if bits == 8:
        return samples.astype(np.uint8).tobytes()
    rows, width, _ = samples.shape
    per_byte = 8 // bits
    padded = np.zeros((rows, math.ceil(width / per_byte) * per_byte), dtype=np.uint8)
    padded[:, :width] = samples[..., 0]
    groups = padded.reshape(rows, -1, per_byte)
    shifts = np.arange(8 - bits, -1, -bits, dtype=np.uint8)
    return (groups << shifts).sum(axis=2, dtype=np.uint16).astype(np.uint8).tobytes()


def tiff(samples: np.ndarray, *, photometric: int | None = None, bits: int | None = None, endian: str = "<",
         big: bool = False, compression: int = 1, predictor: int = 1, planar: int = 1, rows_per_strip=None,
         tile=None, extra_samples=None, tags=(), exif=None, gps=None, interop=None, next_ifd: bool = False,
         omit=()) -> bytes:
    """A TIFF of `samples` (h, w) or (h, w, c), stored exactly as given."""

    if samples.ndim == 2:
        samples = samples[..., None]
    height, width, per_pixel = samples.shape
    bits = bits or (16 if samples.dtype == np.uint16 else 8)
    photometric = photometric if photometric is not None else (2 if per_pixel - len(extra_samples or ()) == 3 else 1)
    w = TiffWriter(endian, big)
    planes = per_pixel if planar == 2 else 1
    chunky = per_pixel if planar == 1 else 1

    def block_bytes(block: np.ndarray) -> bytes:
        if predictor == 2:
            diff = block.astype(np.int64)
            diff[:, 1:] -= block[:, :-1].astype(np.int64)
            block = (diff % (1 << bits)).astype(block.dtype)
        return _row_bytes(block, bits, endian)

    if tile:
        tw, tl = tile
        places = [(r, c) for r in range(0, height, tl) for c in range(0, width, tw)]
    else:
        rps = rows_per_strip or height
        places = [(r, 0) for r in range(0, height, rps)]
    offsets, counts = [], []
    for plane in range(planes):
        part = samples if planar == 1 else samples[..., plane : plane + 1]
        for top, left in places:
            if tile:
                block = np.zeros((tl, tw, chunky), dtype=samples.dtype)
                piece = part[top : top + tl, left : left + tw]
                block[: piece.shape[0], : piece.shape[1]] = piece
            else:
                block = part[top : top + rps]
            raw = block_bytes(block)
            data = _encode(raw, compression, len(raw) // block.shape[0])
            offsets.append(w.add(data))
            counts.append(len(data))

    long = 16 if big else 4
    entries = [(256, 4, [width]), (257, 4, [height]), (258, 3, [bits] * per_pixel), (259, 3, [compression]),
               (262, 3, [photometric]), (277, 3, [per_pixel]), (284, 3, [planar])]
    if predictor != 1:
        entries.append((317, 3, [predictor]))
    if extra_samples:
        entries.append((338, 3, list(extra_samples)))
    if tile:
        entries += [(322, 3, [tile[0]]), (323, 3, [tile[1]]), (324, long, offsets), (325, long, counts)]
    else:
        entries += [(273, long, offsets), (278, 4, [rps]), (279, long, counts)]
    pointer = 16 if big else 4
    for name, tag in (("gps", 34853), ("exif", 34665)):
        sub = {"gps": gps, "exif": exif}[name]
        if sub is not None:
            sub_entries = list(sub)
            if name == "exif" and interop is not None:
                sub_entries.append((40965, pointer, [w.ifd(list(interop))]))
            entries.append((tag, pointer, [w.ifd(sub_entries)]))
    entries = [e for e in entries if e[0] not in omit] + list(tags)
    second = w.ifd([(256, 4, [1]), (257, 4, [1])]) if next_ifd else 0
    return w.finish(w.ifd(entries, second))


# --- PNG ---------------------------------------------------------------------------------------


def png(samples: np.ndarray, color_type: int, depth: int = 8, before=(), after=(), interlace: int = 0) -> bytes:
    """A PNG of the stored samples (h, w) or (h, w, c), filter 0."""

    if samples.ndim == 2:
        samples = samples[..., None]
    height, width, _ = samples.shape
    if depth < 8:
        rows = _row_bytes(samples, depth, ">")
        row_len = len(rows) // height
        raw = b"".join(b"\0" + rows[i * row_len : (i + 1) * row_len] for i in range(height))
    else:
        pixels = samples if samples.shape[2] > 1 else samples[..., 0]
        raw = B.scanlines(pixels, depth, interlace=interlace)
    head = B.chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, interlace))
    return B.SIG + head + b"".join(before) + B.chunk(b"IDAT", zlib.compress(raw)) + b"".join(after) + B.IEND


def exif_block(orientation: int | None, endian: str = "<", kind: int = 3, extra=()) -> bytes:
    """A small EXIF block: IFD0 with 274, and IFD1 (a thumbnail) after it."""

    w = TiffWriter(endian)
    entries = list(extra) + ([(274, kind, [orientation])] if orientation is not None else [])
    ifd1 = w.ifd([(256, 4, [1])])
    return w.finish(w.ifd(entries or [(271, 2, b"cam\0")], ifd1))


def xmp(orientation: str) -> bytes:
    return (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" '
            + orientation.encode() + b"/></x:xmpmeta>")


# --- .scn ---------------------------------------------------------------------------------------


def scn(pixels: np.ndarray, endian: str = "little", max_value: int | None = 65535, channel_count=None,
        extra_pixel_bytes: bytes = b"", trailer: bytes = b"--BOUND--\r\n") -> bytes:
    height, width = pixels.shape
    xml = (f'<?xml version="1.0"?><scan><size_pix width="{width}" height="{height}"/>'
           + (f'<scanner max_value="{max_value}"/>' if max_value is not None else "")
           + f"<endian>{endian}</endian>"
           + (f"<channel_count>{channel_count}</channel_count>" if channel_count is not None else "")
           + "</scan>").encode()
    data = pixels.astype((">" if endian == "big" else "<") + ("u1" if pixels.dtype == np.uint8 else "u2")).tobytes()
    data += extra_pixel_bytes
    return (b"MIME-Version: 1.0 Generated by Image Lab 6.1\r\n"
            b'Content-Type: multipart/mixed; boundary="BOUND"\r\n\r\n'
            b"--BOUND\r\nContent-Type: text/xml\r\nContent-Length: " + str(len(xml)).encode() + b"\r\n\r\n" + xml
            + b"\r\n--BOUND\r\nContent-Type: application/octet-stream\r\nContent-Length: " + str(len(data)).encode()
            + b"\r\n\r\n" + data + b"\r\n" + trailer)

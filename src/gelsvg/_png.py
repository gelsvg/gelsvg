"""PNG chunks, the `geLs` chunk, and decoding pixels (§1.2–§1.4, §5.3).

Everything here works on bytes and never calls into a PNG library for the
structure: the specification decides exactly which PNGs are valid, which
damage is rescued and how far the chunks are walked, and a library's own
leniency would make the answer depend on its version. Only undoing the
filters of scanlines that have passed every check is handed to Pillow, when
it can, as it is far faster; that arithmetic has one answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import struct
import warnings
import zlib

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
GELS_TYPE = b"geLs"
GELS_MAGIC = b"GSVG"
GELS_HEADER_SIZE = 44
GELS_HEADER_VERSION = 1
GELS_CODECS = (1, 2)
VALID_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"
PERMITTED_CRITICAL = (b"IHDR", b"PLTE", b"IDAT", b"IEND")

_ADAM7 = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_ALLOWED_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
_MAX_DIMENSION = 2**31 - 1


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def is_letters(chunk_type: bytes) -> bool:
    return len(chunk_type) == 4 and all(
        0x41 <= c <= 0x5A or 0x61 <= c <= 0x7A for c in chunk_type
    )


def is_critical(chunk_type: bytes) -> bool:
    """Critical when the first byte is an upper-case ASCII letter (§1.3)."""

    return len(chunk_type) == 4 and 0x41 <= chunk_type[0] <= 0x5A


@dataclass
class Chunk:
    type: bytes
    offset: int
    """Offset of the length field from the start of the PNG."""
    length: int
    """The value of the length field (`i` in §1.4)."""
    data: bytes
    """The data part, as far as the PNG holds it."""
    crc_ok: bool
    """False when the CRC does not match or cannot be read."""
    runs_past_end: bool = False
    """The length field points past the end of the data."""


@dataclass
class GelsChunk:
    """One `geLs` chunk and what §1.4 made of it."""

    chunk: Chunk
    status: str
    """One of "ok", "rescued", "damaged", "writer_error", "unknown_header",
    "duplicate"."""
    detail: str = ""
    version: int | None = None
    codec: int | None = None
    flags: int | None = None
    payload: bytes | None = None
    """The payload for "ok" and "rescued" chunks."""
    payload_sha256: bytes | None = None
    """The SHA-256 the header records (raw 32 bytes), when readable."""
    rescued_by: str | None = None
    """"i" or "ii": which length gave back the payload of a damaged chunk."""

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "rescued")


@dataclass
class Walk:
    """The chunks reached by walking a PNG as §1.2 step 3 and §1.4 say."""

    signature_ok: bool
    chunks: list[Chunk] = field(default_factory=list)
    """Every chunk reached, in order, `geLs` chunks included."""
    gels: list[GelsChunk] = field(default_factory=list)
    ended_at_iend: bool = False
    bytes_after_iend: bool = False
    tail_lost_from: int | None = None
    """Offset from which the bytes were treated as absent because the walk
    could not go on (a length running past the end, a damaged `geLs` that
    could not be rescued, data ending before IEND)."""
    final_iend_ok: bool = False
    """The last 12 bytes of the data are an IEND with length 0 and a good
    CRC."""

    @property
    def truncated(self) -> bool:
        return self.tail_lost_from is not None


def _readable_chunk_at(png: bytes, pos: int) -> bool:
    n = len(png)
    if pos + 8 > n:
        return False
    length = _u32(png, pos)
    chunk_type = png[pos + 4 : pos + 8]
    if not is_letters(chunk_type):
        return False
    end = pos + 12 + length
    if end > n:
        return False
    data = png[pos + 8 : pos + 8 + length]
    return zlib.crc32(chunk_type + data) & 0xFFFFFFFF == _u32(png, pos + 8 + length)


def _gels_header(png: bytes, data_start: int):
    if data_start + GELS_HEADER_SIZE > len(png):
        return None
    magic = png[data_start : data_start + 4]
    version = png[data_start + 4]
    codec = png[data_start + 5]
    flags = struct.unpack_from(">H", png, data_start + 6)[0]
    payload_length = _u32(png, data_start + 8)
    sha = png[data_start + 12 : data_start + 44]
    return magic, version, codec, flags, payload_length, sha


def _read_gels(png: bytes, chunk: Chunk) -> tuple[GelsChunk, int | None]:
    """Apply the three steps of §1.4 to one `geLs` chunk. Returns the result
    and where the walk goes on, or None when the rest is lost."""

    n = len(png)
    data_start = chunk.offset + 8
    length = chunk.length
    normal_next = data_start + length + 4

    if chunk.crc_ok:
        data = chunk.data
        if length < GELS_HEADER_SIZE or data[:4] != GELS_MAGIC:
            return GelsChunk(chunk, "writer_error", "header too short or wrong marker"), normal_next
        _, version, codec, flags, payload_length, sha = _gels_header(png, data_start)
        result = GelsChunk(chunk, "ok", version=version, codec=codec, flags=flags, payload_sha256=sha)
        if version != GELS_HEADER_VERSION or codec not in GELS_CODECS or flags != 0:
            result.status = "unknown_header"
            result.detail = f"header version {version}, codec {codec}, flags {flags}"
            return result, normal_next
        if length != GELS_HEADER_SIZE + payload_length:
            result.status = "writer_error"
            result.detail = "chunk length and payload length disagree"
            return result, normal_next
        payload = data[GELS_HEADER_SIZE:]
        if hashlib.sha256(payload).digest() != sha:
            result.status = "writer_error"
            result.detail = "payload does not match its SHA-256"
            return result, normal_next
        result.payload = bytes(payload)
        return result, normal_next

    # Damaged: rescue only what the header's SHA-256 confirms.
    header = _gels_header(png, data_start)
    result = GelsChunk(chunk, "damaged")
    matched = None
    payload_length = None
    if header is None:
        result.detail = "header cannot be read"
    else:
        magic, version, codec, flags, payload_length, sha = header
        result.version, result.codec, result.flags = version, codec, flags
        result.payload_sha256 = sha
        if (
            magic != GELS_MAGIC
            or version != GELS_HEADER_VERSION
            or codec not in GELS_CODECS
            or flags != 0
        ):
            result.detail = "header does not look like a GSVG header"
        else:
            if length == GELS_HEADER_SIZE + payload_length:
                candidates = [("i", payload_length)]
            else:
                candidates = []
                if length >= GELS_HEADER_SIZE:
                    candidates.append(("i", length - GELS_HEADER_SIZE))
                candidates.append(("ii", payload_length))
            start = data_start + GELS_HEADER_SIZE
            for which, size in candidates:
                end = start + size
                if end > n:
                    continue
                if hashlib.sha256(png[start:end]).digest() == sha:
                    matched = which
                    result.payload = bytes(png[start:end])
                    break
            if matched:
                result.status = "rescued"
                result.rescued_by = matched
            else:
                result.detail = "no length gives a payload matching the SHA-256"

    if matched == "ii" and length != GELS_HEADER_SIZE + payload_length:
        return result, data_start + GELS_HEADER_SIZE + payload_length + 4
    if matched == "i":
        return result, normal_next
    if _readable_chunk_at(png, normal_next):
        return result, normal_next
    return result, None


def walk(png: bytes) -> Walk:
    """Walk the chunks from the signature to IEND by their length fields."""

    if not png.startswith(PNG_SIGNATURE):
        return Walk(signature_ok=False)
    result = Walk(signature_ok=True)
    n = len(png)
    pos = len(PNG_SIGNATURE)
    while True:
        if pos + 8 > n:
            result.tail_lost_from = min(pos, n)
            break
        length = _u32(png, pos)
        chunk_type = bytes(png[pos + 4 : pos + 8])
        data_start = pos + 8
        data_end = data_start + length
        if data_end + 4 > n:
            chunk = Chunk(
                chunk_type, pos, length, bytes(png[data_start : min(data_end, n)]),
                crc_ok=False, runs_past_end=True,
            )
        else:
            data = bytes(png[data_start:data_end])
            crc_ok = zlib.crc32(chunk_type + data) & 0xFFFFFFFF == _u32(png, data_end)
            chunk = Chunk(chunk_type, pos, length, data, crc_ok=crc_ok)
        result.chunks.append(chunk)

        if chunk_type == GELS_TYPE:
            gels, next_pos = _read_gels(png, chunk)
            result.gels.append(gels)
            if next_pos is None:
                result.tail_lost_from = pos
                break
            pos = next_pos
            continue
        if chunk.runs_past_end:
            result.tail_lost_from = pos
            break
        if chunk_type == b"IEND":
            result.ended_at_iend = True
            result.bytes_after_iend = data_end + 4 < n
            break
        pos = data_end + 4

    result.final_iend_ok = n >= 12 + len(PNG_SIGNATURE) and png[-12:] == VALID_IEND
    _mark_duplicates(result.gels)
    return result


def _mark_duplicates(gels: list[GelsChunk]) -> None:
    seen = set()
    for item in gels:
        if not item.usable:
            continue
        key = (item.codec, hashlib.sha256(item.payload).digest())
        if key in seen:
            item.status = "duplicate"
            item.detail = "same codec and payload as an earlier chunk"
            item.payload = None
        else:
            seen.add(key)


# --- IHDR, scanlines and pixels -------------------------------------------------


@dataclass(frozen=True)
class Ihdr:
    width: int
    height: int
    bit_depth: int
    color_type: int
    compression: int
    filter_method: int
    interlace: int

    @classmethod
    def parse(cls, data: bytes) -> "Ihdr | None":
        if len(data) != 13:
            return None
        return cls(*struct.unpack(">IIBBBBB", data))

    def problems(self) -> list[str]:
        out = []
        if not (1 <= self.width <= _MAX_DIMENSION and 1 <= self.height <= _MAX_DIMENSION):
            out.append("width or height out of range")
        if self.color_type not in _ALLOWED_DEPTHS:
            out.append(f"colour type {self.color_type} unknown")
        elif self.bit_depth not in _ALLOWED_DEPTHS[self.color_type]:
            out.append(f"bit depth {self.bit_depth} not allowed for colour type {self.color_type}")
        if self.compression != 0:
            out.append("compression method not 0")
        if self.filter_method != 0:
            out.append("filter method not 0")
        if self.interlace not in (0, 1):
            out.append("interlace method not 0 or 1")
        return out

    @property
    def bits_per_pixel(self) -> int:
        return self.bit_depth * _CHANNELS[self.color_type]

    @property
    def filter_unit(self) -> int:
        """Bytes per complete pixel, at least 1 (the `bpp` of PNG filters)."""

        return max(1, self.bits_per_pixel // 8)

    def passes(self) -> list[tuple[int, int, int, int, int, int]]:
        """(x0, y0, dx, dy, width, height) for each pass that has pixels."""

        if self.interlace == 0:
            return [(0, 0, 1, 1, self.width, self.height)]
        out = []
        for x0, y0, dx, dy in _ADAM7:
            w = (self.width - x0 + dx - 1) // dx if self.width > x0 else 0
            h = (self.height - y0 + dy - 1) // dy if self.height > y0 else 0
            if w and h:
                out.append((x0, y0, dx, dy, w, h))
        return out

    def row_bytes(self, width: int) -> int:
        return (width * self.bits_per_pixel + 7) // 8

    def expected_length(self) -> int:
        return sum(h * (1 + self.row_bytes(w)) for *_, w, h in self.passes())


def inflate_complete(data: bytes) -> bytes | None:
    """The inflated bytes of one complete zlib stream (header, Adler-32 and
    all), or None when the stream is incomplete, damaged, or followed by
    more bytes."""

    obj = zlib.decompressobj()
    try:
        out = obj.decompress(data)
        out += obj.flush()
    except zlib.error:
        return None
    if not obj.eof or obj.unused_data:
        return None
    return out


def scanline_problem(ihdr: Ihdr, raw: bytes) -> str | None:
    """Why the inflated image data cannot be pixels of this IHDR, or None."""

    if len(raw) != ihdr.expected_length():
        return "inflated length does not match the IHDR"
    pos = 0
    for *_, w, h in ihdr.passes():
        step = 1 + ihdr.row_bytes(w)
        block = np.frombuffer(raw, dtype=np.uint8, count=h * step, offset=pos)
        filters = block[::step]
        if (filters > 4).any():
            return "a row has a filter type other than 0-4"
        pos += h * step
    return None


def _paeth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    a16, b16, c16 = a.astype(np.int16), b.astype(np.int16), c.astype(np.int16)
    p = a16 + b16 - c16
    pa, pb, pc = np.abs(p - a16), np.abs(p - b16), np.abs(p - c16)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def _unfilter_pass(raw: np.ndarray, width: int, height: int, unit: int) -> np.ndarray:
    """Reconstruct one pass of 8- or 16-bit pixels (row bytes a multiple of
    `unit`). Pixels are rebuilt along anti-diagonals: each depends only on
    its left, upper and upper-left neighbours, so every pixel on one
    diagonal can be computed at once whatever filter its row uses."""

    step = 1 + width * unit
    rows = raw.reshape(height, step)
    types = rows[:, 0]
    filtered = rows[:, 1:].reshape(height, width, unit)
    out = np.zeros((height, width, unit), dtype=np.uint8)

    simple = np.isin(types, (0, 1, 2)).all()
    if simple:
        # Only None, Sub and Up: whole rows at a time.
        previous = np.zeros((width, unit), dtype=np.uint8)
        for r in range(height):
            line = filtered[r]
            t = types[r]
            if t == 0:
                recon = line.copy()
            elif t == 1:
                recon = np.cumsum(line, axis=0, dtype=np.uint8)
            else:
                recon = line + previous
            out[r] = recon
            previous = recon
        return out

    for d in range(height + width - 1):
        r = np.arange(max(0, d - width + 1), min(height - 1, d) + 1)
        c = d - r
        x = filtered[r, c]
        a = np.where((c > 0)[:, None], out[r, np.maximum(c - 1, 0)], 0).astype(np.uint8)
        b = np.where((r > 0)[:, None], out[np.maximum(r - 1, 0), c], 0).astype(np.uint8)
        cc = np.where(((r > 0) & (c > 0))[:, None], out[np.maximum(r - 1, 0), np.maximum(c - 1, 0)], 0).astype(np.uint8)
        t = types[r][:, None]
        prediction = np.select(
            [t == 0, t == 1, t == 2, t == 3],
            [np.zeros_like(x), a, b, ((a.astype(np.uint16) + b) >> 1).astype(np.uint8)],
            default=_paeth(a, b, cc),
        )
        out[r, c] = x + prediction
    return out


def _unpack_bits(rows: np.ndarray, bit_depth: int, width: int) -> np.ndarray:
    """Samples of fewer than 8 bits, first sample in the high bits."""

    per_byte = 8 // bit_depth
    shifts = np.arange(8 - bit_depth, -1, -bit_depth, dtype=np.uint8)
    values = (rows[:, :, None] >> shifts) & ((1 << bit_depth) - 1)
    return values.reshape(rows.shape[0], rows.shape[1] * per_byte)[:, :width].astype(np.uint8)


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)


_PILLOW_FORMS = {(0, 8): ("L",), (0, 16): ("I;16", "I;16B", "I"), (2, 8): ("RGB",)}


def _pillow_samples(ihdr: Ihdr, raw: bytes) -> np.ndarray | None:
    """The samples Pillow rebuilds from scanlines that passed
    `scanline_problem`, handed to it in a PNG of just those scanlines; or None
    when it cannot give them exactly (other forms, interlaced, Pillow
    missing, too large for it), so the caller rebuilds them itself."""

    forms = _PILLOW_FORMS.get((ihdr.color_type, ihdr.bit_depth))
    if forms is None or ihdr.interlace != 0 or len(raw) > 2**31 - 100:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    header = struct.pack(">IIBBBBB", ihdr.width, ihdr.height, ihdr.bit_depth, ihdr.color_type, 0, 0, 0)
    png = PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(raw, 0)) + _chunk(b"IEND", b"")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(png), formats=("PNG",)) as image:
                if image.mode not in forms or image.size != (ihdr.width, ihdr.height):
                    return None
                image.load()
                pixels = np.asarray(image)
    except Exception:  # Pillow's own limits: rebuild it here instead
        return None
    channels = _CHANNELS[ihdr.color_type]
    if pixels.shape != ((ihdr.height, ihdr.width) if channels == 1 else (ihdr.height, ihdr.width, channels)):
        return None
    dtype = np.uint16 if ihdr.bit_depth == 16 else np.uint8
    if pixels.dtype != dtype and (pixels.dtype.kind not in "iu" or pixels.min() < 0 or pixels.max() > np.iinfo(dtype).max):
        return None
    return np.ascontiguousarray(pixels.astype(dtype, copy=False).reshape(ihdr.height, ihdr.width, channels))


def decode_samples(ihdr: Ihdr, raw: bytes) -> np.ndarray:
    """The stored samples of any colour type and bit depth, shape (height,
    width, samples per pixel): uint16 for 16 bits, else uint8 (fewer than 8
    bits are not widened). `raw` must already have passed
    `scanline_problem`."""

    fast = _pillow_samples(ihdr, raw)
    return fast if fast is not None else _rebuilt_samples(ihdr, raw)


def _rebuilt_samples(ihdr: Ihdr, raw: bytes) -> np.ndarray:
    """decode_samples, every filter undone here."""

    channels = _CHANNELS[ihdr.color_type]
    data = np.frombuffer(raw, dtype=np.uint8)
    pos = 0
    if ihdr.bit_depth < 8:
        out = np.zeros((ihdr.height, ihdr.width, 1), dtype=np.uint8)
        for x0, y0, dx, dy, w, h in ihdr.passes():
            row_bytes = ihdr.row_bytes(w)
            size = h * (1 + row_bytes)
            part = _unfilter_pass(data[pos : pos + size], row_bytes, h, 1)[..., 0]
            out[y0::dy, x0::dx, 0] = _unpack_bits(part, ihdr.bit_depth, w)
            pos += size
        return out
    unit = ihdr.filter_unit
    full = np.zeros((ihdr.height, ihdr.width, unit), dtype=np.uint8)
    for x0, y0, dx, dy, w, h in ihdr.passes():
        size = h * (1 + w * unit)
        part = _unfilter_pass(data[pos : pos + size], w, h, unit)
        full[y0::dy, x0::dx] = part
        pos += size
    if ihdr.bit_depth == 8:
        return full.reshape(ihdr.height, ihdr.width, channels)
    pairs = full.reshape(ihdr.height, ihdr.width, channels, 2)
    return np.ascontiguousarray((pairs[..., 0].astype(np.uint16) << 8) | pairs[..., 1])


def decode_pixels(ihdr: Ihdr, raw: bytes) -> np.ndarray:
    """Pixels of a grey or RGB image with bit depth 8 or 16, as uint8 or
    uint16, shape (height, width) or (height, width, 3). `raw` must already
    have passed `scanline_problem`."""

    if ihdr.color_type not in (0, 2) or ihdr.bit_depth not in (8, 16):
        raise ValueError("only grey and RGB images with 8 or 16 bits are decoded")
    pixels = decode_samples(ihdr, raw)
    if ihdr.color_type == 0:
        pixels = pixels[..., 0]
    return np.ascontiguousarray(pixels)

"""A PNG source file (§5.4 `png`, §5.5)."""

from __future__ import annotations

import re
import struct
import zlib

import numpy as np

from . import _formal, _orientation, _png
from ._sourcebase import DAMAGED, NOT_A_SOURCE_IMAGE, Decoded, SourceError

_WHITESPACE = re.compile(rb"[ \t\r\n\f\v]")
_HEX = re.compile(rb"[0-9A-Fa-f]*")


def _chunks(data: bytes) -> list[tuple[bytes, bytes, bool]]:
    """(type, data, CRC matches) of each chunk up to IEND."""

    out, pos = [], len(_png.PNG_SIGNATURE)
    while True:
        if pos + 12 > len(data):
            raise SourceError(DAMAGED, "The PNG file ends before its IEND chunk.")
        length = struct.unpack_from(">I", data, pos)[0]
        ctype = data[pos + 4 : pos + 8]
        end = pos + 8 + length
        if end + 4 > len(data):
            raise SourceError(DAMAGED, "A PNG chunk runs past the end of the file.")
        body = data[pos + 8 : end]
        crc_ok = zlib.crc32(ctype + body) & 0xFFFFFFFF == struct.unpack_from(">I", data, end)[0]
        out.append((ctype, body, crc_ok))
        pos = end + 4
        if ctype == b"IEND":
            return out


def _text(ctype: bytes, body: bytes) -> tuple[bytes, bytes] | None:
    """(keyword, text) of a tEXt, zTXt or iTXt chunk; None if it is not one
    or its text cannot be taken out."""

    keyword, sep, rest = body.partition(b"\0")
    if not sep:
        return None
    try:
        if ctype == b"tEXt":
            return keyword, rest
        if ctype == b"zTXt":
            return keyword, zlib.decompress(rest[1:])
        if ctype == b"iTXt":
            flag = rest[0]
            _, _, rest = rest[2:].partition(b"\0")
            _, _, text = rest.partition(b"\0")
            return keyword, zlib.decompress(text) if flag else text
    except (zlib.error, IndexError):
        return None
    return None


def _raw_profile_exif(text: bytes) -> bytes:
    digits = _WHITESPACE.sub(b"", b"".join(text.split(b"\n")[3:]))
    if len(digits) % 2 or not _HEX.fullmatch(digits):
        raise _orientation.Unreadable("the raw profile's hex digits are not whole bytes")
    block = bytes.fromhex(digits.decode("ascii"))
    return block[6:] if block.startswith(b"Exif\0\0") else block


def _orientation_of(chunks) -> int:
    exif = xmp = raw_profile = None
    raw_profile_bad = xmp_seen = False
    for ctype, body, _ in chunks:
        if ctype == b"eXIf" and exif is None:
            exif = body[6:] if body.startswith(b"Exif\0\0") else body
        elif ctype in (b"tEXt", b"zTXt", b"iTXt"):
            text = _text(ctype, body)
            keyword = body.partition(b"\0")[0]
            if keyword == b"Raw profile type exif" and raw_profile is None and not raw_profile_bad:
                if text is None:
                    raw_profile_bad = True
                else:
                    raw_profile = text[1]
            elif ctype == b"iTXt" and keyword == b"XML:com.adobe.xmp" and not xmp_seen:
                xmp_seen = True
                xmp = text[1] if text is not None else None
    if exif is None and (raw_profile is not None or raw_profile_bad):
        if raw_profile_bad:
            return 1
        try:
            exif = _raw_profile_exif(raw_profile)
        except _orientation.Unreadable:
            return 1
    return _orientation.decide(exif, xmp)


def _metadata(chunks) -> tuple[dict, bytes] | None:
    if any(not crc_ok or not (ctype.isascii() and ctype.isalpha()) for ctype, _, crc_ok in chunks):
        return None
    segments, after = [], False
    for ctype, body, _ in chunks:
        if ctype == b"IDAT":
            after = True
        elif ctype != _png.GELS_TYPE:
            segments.append(({"chunk_type": ctype.decode("ascii"), "after_pixels": after}, body))
    return _formal.build("png", segments)


def read(data: bytes) -> Decoded:
    chunks = _chunks(data)
    types = [c[0] for c in chunks]
    for ctype, _, crc_ok in chunks:
        critical = ctype[:1].isupper()
        if critical and ctype not in _png.PERMITTED_CRITICAL:
            raise SourceError(DAMAGED, f"The PNG file has a critical chunk {ctype!r} no decoder knows.")
        if critical and not crc_ok:
            raise SourceError(DAMAGED, f"The PNG file's {ctype.decode('latin-1')} chunk is damaged (CRC).")
    if b"acTL" in types:
        raise SourceError(NOT_A_SOURCE_IMAGE, "The PNG file is animated: it holds more than one image.")
    ihdr = _png.Ihdr.parse(chunks[0][1]) if types[0] == b"IHDR" else None
    if ihdr is None or ihdr.problems() or types.count(b"IHDR") != 1:
        raise SourceError(DAMAGED, "The PNG file's IHDR chunk is missing or not valid.")
    if ihdr.color_type == 3:
        raise SourceError(NOT_A_SOURCE_IMAGE, "The PNG file is a palette image; its values are not brightness.")
    idat = [i for i, t in enumerate(types) if t == b"IDAT"]
    if not idat or idat[-1] - idat[0] + 1 != len(idat):
        raise SourceError(DAMAGED, "The PNG file's image data is missing or split up.")
    raw = _png.inflate_complete(b"".join(chunks[i][1] for i in idat))
    if raw is None or _png.scanline_problem(ihdr, raw):
        raise SourceError(DAMAGED, "The PNG file's image data cannot be decoded.")
    samples = _png.decode_samples(ihdr, raw)

    if ihdr.color_type in (4, 6):
        top = 65535 if ihdr.bit_depth == 16 else 255
        if (samples[..., -1] != top).any():
            raise SourceError(NOT_A_SOURCE_IMAGE, "The PNG file has transparent pixels.")
        samples = samples[..., :-1]
    trns = next((body for ctype, body, _ in chunks if ctype == b"tRNS"), None)
    if trns is not None and ihdr.color_type in (0, 2) and len(trns) == 2 * samples.shape[2]:
        key = np.array(struct.unpack(f">{samples.shape[2]}H", trns))
        if (samples == key).all(axis=2).any():
            raise SourceError(NOT_A_SOURCE_IMAGE, "The PNG file has transparent pixels.")
    if ihdr.bit_depth < 8:
        samples = samples * np.uint8(255 // (2**ihdr.bit_depth - 1))
    pixels = samples[..., 0] if samples.shape[2] == 1 else samples
    pixels = _orientation.apply(pixels, _orientation_of(chunks))
    metadata = _metadata(chunks)
    notes = [] if metadata else ["The PNG file has a damaged or misnamed chunk, so its metadata is not kept (§5.4)."]
    return Decoded("png", pixels, metadata, notes)

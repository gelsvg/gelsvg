"""The payload of source metadata in a formal format, version 1 (§5.4):
the checks a reader can make without the source file."""

from __future__ import annotations

import struct

from . import _json

JPEG_MARKERS = tuple(f"APP{i}" for i in range(16)) + ("COM",)
TIFF_IFDS = ("ifd0", "exif", "gps", "interop")
TIFF_ORDER = {"ifd0": 0, "exif": 1, "interop": 2, "gps": 3}
TIFF_TYPE_SIZE = {1: 1, 2: 1, 6: 1, 7: 1, 3: 2, 8: 2, 4: 4, 9: 4, 11: 4, 13: 4, 5: 8, 10: 8, 12: 8, 16: 8, 17: 8, 18: 8}
SCN_PARTS = ("before_pixels", "after_pixels")


def _letters(value) -> bool:
    return type(value) is str and len(value) == 4 and value.isascii() and value.isalpha()


def _uint(value) -> bool:
    return type(value) is int and value >= 0


def check_formal(fmt: str, payload: bytes) -> tuple[dict | None, str | None]:
    """The payload's index, or why the source metadata is broken."""

    if len(payload) < 4:
        return None, "the payload is shorter than its index length"
    index_length = struct.unpack_from(">I", payload, 0)[0]
    if 4 + index_length > len(payload):
        return None, "4 + the index length exceeds 'byte_length'"
    index_bytes = payload[4 : 4 + index_length]
    try:
        index = _json.parse(index_bytes)
    except _json.JsonError as exc:
        return None, f"the index is not JSON this format reads ({exc})"
    if _json.canonical(index) != index_bytes:
        return None, "the index is not in RFC 8785 canonical form"

    tiff = fmt == "tiff"
    allowed = {"segments", "byte_order", "tiff_version"} if tiff else {"segments"}
    extra = set(index) - allowed
    if extra:
        return None, "the index has keys it may not have: " + ", ".join(sorted(map(repr, extra)))
    tiff_version = None
    if tiff:
        if index.get("byte_order") not in ("II", "MM"):
            return None, "'byte_order' is missing or not II or MM"
        tiff_version = index.get("tiff_version")
        if tiff_version not in (42, 43) or type(tiff_version) is not int:
            return None, "'tiff_version' is missing or not 42 or 43"
    segments = index.get("segments")
    if type(segments) is not list or not segments:
        return None, "'segments' is missing or empty"

    own = {"png": {"chunk_type", "after_pixels"}, "jpeg": {"marker"},
           "tiff": {"ifd", "tag", "type", "count"}, "scn": {"part"}}[fmt]
    expected_offset = 0
    last_ifd_rank = -1
    parts_seen: list[str] = []
    for i, segment in enumerate(segments):
        where = f"segment {i}"
        if type(segment) is not dict or set(segment) != own | {"offset", "length"}:
            return None, f"{where} does not have exactly the keys {sorted(own | {'offset', 'length'})}"
        offset, length = segment["offset"], segment["length"]
        if not (_uint(offset) and _uint(length)):
            return None, f"{where} has an offset or length that is not a whole number"
        if offset != expected_offset:
            return None, f"{where} leaves a gap or overlaps the previous one"
        expected_offset = offset + length
        if fmt == "png":
            if not _letters(segment["chunk_type"]) or segment["chunk_type"] in ("IDAT", "geLs"):
                return None, f"{where} has a chunk type that may not be kept"
            if type(segment["after_pixels"]) is not bool:
                return None, f"{where} has an 'after_pixels' that is not true or false"
        elif fmt == "jpeg":
            if segment["marker"] not in JPEG_MARKERS:
                return None, f"{where} has a marker other than APP0-APP15 and COM"
        elif fmt == "tiff":
            ifd, tag, kind, count = segment["ifd"], segment["tag"], segment["type"], segment["count"]
            if ifd not in TIFF_IFDS:
                return None, f"{where} has an unknown IFD"
            if not (type(tag) is int and 0 <= tag <= 65535):
                return None, f"{where} has a tag outside 0-65535"
            if type(kind) is not int or kind not in TIFF_TYPE_SIZE or (kind >= 16 and tiff_version != 43):
                return None, f"{where} has a TIFF type that is not allowed"
            if not _uint(count) or length != count * TIFF_TYPE_SIZE[kind]:
                return None, f"{where} has a length that is not count × the type's size"
            rank = TIFF_ORDER[ifd]
            if rank < last_ifd_rank:
                return None, "the segments are not in the order ifd0, exif, interop, gps"
            last_ifd_rank = rank
        else:
            part = segment["part"]
            if part not in SCN_PARTS or part in parts_seen or (parts_seen and part == "before_pixels"):
                return None, f"{where} has a part that is unknown, repeated or out of order"
            if length == 0:
                return None, f"{where} is empty"
            parts_seen.append(part)
    if expected_offset != len(payload) - 4 - index_length:
        return None, "the segments do not cover the data to its end"
    return index, None

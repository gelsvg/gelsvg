"""Building source metadata in a formal format (§5.4, `version` 1): the
segments a source file is cut into, joined behind their index."""

from __future__ import annotations

import struct

from . import _hashes, _json, _metadata


def build(fmt: str, segments: list[tuple[dict, bytes]], **index_keys) -> tuple[dict, bytes] | None:
    """(catalog entry, payload), or None when there is no segment (§5.4:
    then no source metadata is written)."""

    if not segments:
        return None
    listed, offset = [], 0
    for fields, data in segments:
        listed.append({**fields, "offset": offset, "length": len(data)})
        offset += len(data)
    index = _json.canonical({"segments": listed, **index_keys})
    payload = struct.pack(">I", len(index)) + index + b"".join(data for _, data in segments)
    _, problem = _metadata.check_formal(fmt, payload)
    if problem is not None:  # what this builds must read back (§5.4)
        raise AssertionError(f"built {fmt} metadata does not read back: {problem}")
    entry = {"format": fmt, "version": 1, "byte_length": len(payload), "payload_hash": _hashes.sha256(payload)}
    return entry, payload


def key(entry: dict) -> str:
    """The source metadata's hash: its catalog key (§5.4)."""

    return _hashes.sha256(_json.canonical(entry))

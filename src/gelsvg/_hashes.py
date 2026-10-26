"""The hashes the format records (§2.7, §5.3, §5.4, §6.3)."""

from __future__ import annotations

import copy
import hashlib
import struct

import numpy as np

from . import _json

PIXEL_PREFIX = b"MyWB decoded scalar plane v1\0"
RENDERED_PREFIX = b"GelSVG rendered image v1\0"

# §2.7: what the hash for quantification leaves out, beyond `embedded` and
# `sources[].source_metadata`.
_QUANT_TEXT = {
    ("samples", "conditions", "name"),
    ("samples", "lanes", "number"),
    ("samples", "lanes", "details"),
    ("samples", "lanes", "values"),
    ("rows", "name"),
    ("markers", "label"),
    ("markers", "size"),
    ("blot", "invert"),
    ("blot", "contrast"),
    ("blot", "color"),
}
_QUANT_TOP = ("annotation", "note", "output_bit_depth")
_QUANT_MARKER = ("invert", "contrast")


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _drop_in_array(container, array_key: str, key: str) -> None:
    items = container.get(array_key) if isinstance(container, dict) else None
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                item.pop(key, None)


def core_for_hash(core) -> object:
    """The core as written, without what says whether pixels and metadata
    travel in the document (§2.7)."""

    if not isinstance(core, dict):
        return copy.deepcopy(core)
    out = copy.deepcopy(core)
    out.pop("embedded", None)
    _drop_in_array(out, "sources", "source_metadata")
    return out


def core_hash(core) -> str:
    return sha256(_json.canonical(core_for_hash(core)))


def quant_core_hash(core) -> str:
    out = core_for_hash(core)
    if isinstance(out, dict):
        for path in _QUANT_TEXT:
            if len(path) == 2:
                _drop_in_array(out, path[0], path[1])
            else:
                samples = out.get(path[0])
                if isinstance(samples, dict):
                    _drop_in_array(samples, path[1], path[2])
        for key in _QUANT_TOP:
            out.pop(key, None)
        marker = out.get("marker")
        if isinstance(marker, dict):
            for key in _QUANT_MARKER:
                marker.pop(key, None)
    return sha256(_json.canonical(out))


def _values_bytes(pixels: np.ndarray) -> bytes:
    if pixels.dtype == np.uint8:
        return pixels.tobytes()
    return pixels.astype(">u2").tobytes()


def pixel_hash(pixels: np.ndarray) -> str:
    """§5.3: one channel image's canonical form."""

    if pixels.ndim != 2:
        raise ValueError("a channel image has one value per pixel")
    mark = b"u8\0" if pixels.dtype == np.uint8 else b"u16"
    height, width = pixels.shape
    return sha256(PIXEL_PREFIX + mark + struct.pack(">II", height, width) + _values_bytes(pixels))


def rendered_hash(pixels: np.ndarray) -> str:
    """§6.3: the rendered image's canonical form."""

    colour = b"gray\0" if pixels.ndim == 2 else b"rgb\0"
    mark = b"u8\0" if pixels.dtype == np.uint8 else b"u16"
    height, width = pixels.shape[:2]
    return sha256(RENDERED_PREFIX + colour + mark + struct.pack(">II", height, width) + _values_bytes(pixels))

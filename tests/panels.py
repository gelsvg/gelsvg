"""A small, valid panel, and ways to change it before it is written."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

import builders as B
from gelsvg import _hashes

FORMAT_URL = "https://mybioapps.com/gelsvg/panel"


def source_pixels() -> np.ndarray:
    return (np.arange(48, dtype=np.uint8).reshape(6, 8) * 5)


def base_panel():
    """(panel JSON, channel image PNG, rendered pixels)."""

    pixels = source_pixels()
    channel_png = B.image_png(pixels)
    pixel = _hashes.pixel_hash(pixels)
    core = {
        "sources": [{
            "id": "s1", "file_name": "blot.tif", "file_hash": "sha256:" + "ab" * 32,
            "dtype": "uint8", "shape": [6, 8], "channels": [{"name": "gray", "pixel_hash": pixel}],
        }],
        "embedded": {"channel_images": {pixel: {
            "kind": "decoded_scalar_plane", "media_type": "image/png", "dtype": "uint8", "shape": [6, 8],
            "byte_length": len(channel_png), "png_hash": "sha256:" + hashlib.sha256(channel_png).hexdigest(),
        }}},
        "blot": [{"id": "c1", "source": "s1", "source_channel": "gray", "invert": False, "contrast": [0, 255]}],
        "orientation": {"turn": 0, "flip": False, "rotation": 0},
        "output_bit_depth": 8,
        "samples": {
            "conditions": [{"id": "t1", "name": "Genotype"}],
            "lanes": [
                {"id": "1", "number": 1, "x": 2.0, "y": 1.0, "details": "WT", "values": {"t1": "WT"}},
                {"id": "n1"},
            ],
        },
        "rows": [{"id": "r1", "name": "p-ERK"}],
        "bands": [
            {"lane": "1", "row": "r1", "shape": "rect", "x": 1, "y": 2, "w": 2, "h": 1},
            {"lane": "n1", "row": "r1", "shape": "point", "x": 6, "y": 2.5},
        ],
        "markers": [{"id": "m1", "x": 0.5, "y": 3, "label": "50", "size": {"value": 50, "unit": "kDa"}}],
    }
    rendered = pixels.copy()
    panel = {
        "format": FORMAT_URL, "schema_version": 1, "schema_revision": 0,
        "saved_at": "2026-10-26T10:00:00-04:00",
        "written_by": {"name": "test", "version": "1"},
        "core": core,
        "quantifications": [{
            "mode": "lane", "pixels": "aligned", "method": {"name": "sum", "by": "test 1"},
            "inverted": False, "computed_on": _hashes.quant_core_hash(core),
            "values": [{"lane": "1", "region": {"shape": "rect", "x": 0, "y": 0, "w": 4, "h": 6}, "raw": 100, "signal": 90}],
        }],
        "figure": {"crop": {"x": 0, "y": 0, "w": 8, "h": 6},
                   "rendered_hash": _hashes.rendered_hash(rendered), "drawn_from": _hashes.core_hash(core)},
    }
    return panel, channel_png, rendered


def carrier_png(panel: dict, channel_png: bytes | None, rendered: np.ndarray, extra_after=(), text: bytes | None = None) -> bytes:
    body = text if text is not None else json.dumps(panel).encode("utf-8")
    after = [B.gels(channel_png, codec=1)] if channel_png is not None else []
    return B.image_png(rendered, before_idat=[B.itxt(body)], after_idat=[*after, *extra_after])


def build(change=None, rehash: bool = True, **kwargs) -> bytes:
    """A carrier PNG of the base panel after `change(panel)`. With rehash,
    drawn_from and computed_on follow the changed core."""

    panel, channel_png, rendered = base_panel()
    if change is not None:
        change(panel)
    if rehash and isinstance(panel.get("core"), dict):
        panel["figure"]["drawn_from"] = _hashes.core_hash(panel["core"])
        for q in panel.get("quantifications", []):
            q["computed_on"] = _hashes.quant_core_hash(panel["core"])
    return carrier_png(panel, channel_png, rendered, **kwargs)


def svg_of(*pngs: bytes) -> bytes:
    return B.svg(*(B.data_uri(p) for p in pngs))

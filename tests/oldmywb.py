"""Old MyWB documents (schema 4–6), built byte by byte the way the old
specification describes them."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

import builders as B
import panels as P
from gelsvg import _hashes, _svghash


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mywb_chunk(payload: bytes, codec: int) -> bytes:
    return B.chunk(b"myWb", B.gels_data(payload, codec=codec, magic=b"MWAS"))


def plane(pixels: np.ndarray) -> tuple[str, dict, bytes]:
    png = B.image_png(pixels, depth=16 if pixels.dtype == np.uint16 else 8)
    pixel = _hashes.pixel_hash(pixels).split(":")[1]
    asset = {"kind": "decoded_scalar_plane", "media_type": "image/png",
             "dtype": "uint16" if pixels.dtype == np.uint16 else "uint8", "shape": list(pixels.shape),
             "byte_length": len(png), "png_sha256": sha(png), "pixel_sha256": pixel}
    return "sha256:" + pixel, asset, png


METADATA_PAYLOAD = b"Cam\0" + b"\x01\x00"
METADATA_RECORD = {"format": "tiff", "byte_order": "II", "tiff_version": 42, "byte_length": len(METADATA_PAYLOAD),
                   "payload_sha256": sha(METADATA_PAYLOAD),
                   "units": [{"ifd": "ifd0", "tag": 271, "type": 2, "count": 4, "offset": 0, "length": 4},
                             {"ifd": "ifd0", "tag": 274, "type": 3, "count": 1, "offset": 4, "length": 2}]}


def base(schema: int = 6) -> tuple[dict, dict]:
    """(document JSON, payloads) of a schema 6 document with an 8-bit grey
    blot and no marker image."""

    pixels = P.source_pixels()
    asset_id, asset, png = plane(pixels)
    bundle_id = "sha256:" + "bb" * 32
    record_id = "sha256:" + "cc" * 32
    doc = {
        "schema_version": schema, "datetime": "2026-10-13 12:34:56.789012",
        "source_image_info": {"marker_file_name": "", "marker_sha256": "", "marker_rgb_channel": "",
                              "marker_image_shape": [], "blot_file_name": "blot.tif", "blot_sha256": "ab" * 32,
                              "blot_rgb_channel": "", "blot_image_shape": [6, 8]},
        "filter_data_m": {"rotation_offset": 0, "resize": None, "flip": 0, "rotation": 0, "invert": 0, "contrast": [0, 255]},
        "filter_data_b": {"rotation_offset": 0, "resize": None, "flip": 0, "rotation": 0, "invert": 0, "contrast": [0, 255]},
        "roi_info": {"roi_x": 1, "roi_y": 1, "roi_w": 6, "roi_h": 4},
        "misc_info": {"sample_label_type": "basic", "output_bit_depth": "8", "note": "a note",
                      "annotation_text": "line 1\\nline 2"},
        "svg_style": {"font_family": "Helvetica", "image_scale_percent": 100},
        "sample_label_info": {"1": {"x_spinbox": 2.5, "y_spinbox": 0.5, "label": "WT\\nrep 1"},
                              "2": {"x_spinbox": 5.5, "y_spinbox": 0.5, "label": ""}},
        "marker_label_info": {"1": {"x_spinbox": 0.5, "y_spinbox": 3, "label": "50 kDa"},
                              "2": {"x_spinbox": 0.5, "y_spinbox": 4, "label": "37"}},
        "band_info": [{"x": 2, "y": 2, "w": 1, "h": 1, "lane": 1, "row": 0},
                      {"x": 5, "y": 2, "w": 0, "h": 0, "lane": 2, "row": 0}],
        "band_row_names": ["p-ERK"],
        "embedded_png_info": {"source_assets": {
            "assets": {asset_id: asset},
            "bundles": {bundle_id: {"color_format": "grayscale", "dtype": "uint8", "shape": [6, 8],
                                    "planes": {"gray": asset_id}}},
            "role_views": {"blot": {"bundle_id": bundle_id, "selected_channel": "gray",
                                    "selected_pixel_sha256": asset["pixel_sha256"], "file_metadata_id": record_id}},
            "file_metadata": {record_id: copy.deepcopy(METADATA_RECORD)},
        }},
    }
    return doc, {1: [png], 2: [METADATA_PAYLOAD]}


def picture(doc: dict) -> np.ndarray:
    """The old picture of `base()`'s blot: its ROI at full contrast."""

    roi = doc["roi_info"]
    return P.source_pixels()[roi["roi_y"] : roi["roi_y"] + roi["roi_h"], roi["roi_x"] : roi["roi_x"] + roi["roi_w"]].copy()


def carrier(doc: dict, payloads: dict, rendered: np.ndarray) -> bytes:
    text = json.dumps(doc, ensure_ascii=False).encode("utf-8")
    after = [mywb_chunk(p, codec) for codec in (1, 2) for p in payloads.get(codec, [])]
    depth = 16 if rendered.dtype == np.uint16 else 8
    return B.image_png(rendered, depth=depth, before_idat=[B.itxt(text, keyword=b"MyWB", compressed=True)], after_idat=after)


def svg_with(png: bytes, extra: str = "", attrs: str = "") -> bytes:
    return (f'<svg xmlns="http://www.w3.org/2000/svg"><g id="layer1"><image x="0" y="0"{attrs} '
            f'href="{B.data_uri(png)}"/>{extra}</g></svg>').encode()


def signed(doc: dict, payloads: dict, rendered: np.ndarray, extra: str = "") -> bytes:
    """The SVG of a schema 6 document with its Canonical XML integrity hash
    (old §11)."""

    depth = 16 if rendered.dtype == np.uint16 else 8
    unsigned = svg_with(B.image_png(rendered, depth=depth), extra)
    doc = copy.deepcopy(doc)
    doc["svg_integrity"] = {"algorithm": "sha256", "target": "svg_c14n2_without_mywb_data",
                            "hash": _svghash.svg_hash(unsigned).split(":")[1], "generated_by": "MyWB",
                            "app_version": "0.6.0", "build_id": "build-1", "render_schema_version": 5}
    return svg_with(carrier(doc, payloads, rendered), extra)


def schema4(doc: dict, rendered: np.ndarray, fallback: str | None = None) -> bytes:
    """Schema 3–4: the JSON in a metadata element inside the image."""

    text = json.dumps(doc, ensure_ascii=False).replace("&", "&amp;").replace("<", "&lt;")
    attrs = f' data-mywb-fallback-mode="{fallback}"' if fallback else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg"><g id="layer1"><image x="0" y="0"{attrs} '
            f'href="{B.data_uri(B.image_png(rendered))}"><metadata data-mywb-role="settings" '
            f'type="application/json">{text}</metadata></image><text>A</text></g></svg>').encode()

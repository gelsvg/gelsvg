"""Converting old MyWB documents (schema 3–6) to GelSVG panels (Appendix B).

Each old document becomes one GelSVG panel. Its rendered image is kept:
the carrier PNG keeps every chunk but the old JSON and the old `myWb`
chunks, so the SVG around it (labels drawn by the old MyWB) still matches,
and so does an old `svg_hash` computed with Canonical XML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

from . import _carrier, _check as C, _datauri, _formal, _hashes, _json, _png, _splice, _svghash, _svgxml
from ._core import ANNOTATION_SIDES
from .container import read_container
from .writer import Content, Tool, WriteError, _build, _Kept, _replace_json_member

MYWB_EXTENSION = "https://mybioapps.com/mywb"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ROLES = ("blot", "marker")


class LegacyError(ValueError):
    """The old document cannot be converted. `notices`, when the converted
    panel was refused by its read-back, says why."""

    def __init__(self, message: str, notices=None):
        super().__init__(message)
        self.notices = list(notices or [])


class NoOldDocument(LegacyError):
    """The file holds no old MyWB document to convert."""


@dataclass
class ConvertedPanel:
    schema_version: int
    """The old document's schema."""
    png: bytes
    """The GelSVG carrier PNG."""
    notes: list[str] = field(default_factory=list)
    """What the conversion dropped or could not carry (Appendix B's
    "知らせるとよい")."""


@dataclass
class Converted:
    svg: bytes
    panels: list[ConvertedPanel]


# --- small readers -----------------------------------------------------------------------------


def _text(value: str, backslash_n: bool = True) -> str:
    """Old text: the two characters `\\n` were a line break too; CR LF and
    CR become LF (§2.5)."""

    if backslash_n:
        value = value.replace("\\n", "\n")
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _flag(value, what: str) -> bool:
    if type(value) is not int or value not in (0, 1):
        raise LegacyError(f"'{what}' is not 0 or 1.")
    return value == 1


def _turn(value, what: str) -> int:
    if not C.is_number(value) or float(value) not in (0.0, 90.0, 180.0, 270.0):
        raise LegacyError(f"'{what}' is not 0, 90, 180 or 270.")
    return int(value)


def _contrast(value, what: str) -> list[int]:
    if type(value) is not list or len(value) != 2 or not all(type(v) is int for v in value):
        raise LegacyError(f"'{what}' is not two integers.")
    return sorted(value)  # the old reader swapped a reversed pair


def _hex_hash(value) -> str | None:
    return "sha256:" + value if isinstance(value, str) and _HEX64.fullmatch(value) else None


def _dihedral(image: np.ndarray, turn: int, flip: bool) -> np.ndarray:
    turned = np.rot90(image, k=-(turn // 90))
    return turned[:, ::-1] if flip else turned


def _marker_turn_flip(turn_m: int, flip_m: bool, turn_b: int, flip_b: bool) -> tuple[int, bool]:
    """The old marker image was turned and flipped on its own; GelSVG turns
    it towards the reference channel and then by the panel's orientation
    (§4.2). The pair that gives the same picture."""

    probe = np.arange(6).reshape(2, 3)
    want = _dihedral(probe, turn_m, flip_m)
    for turn in (0, 90, 180, 270):
        for flip in (False, True):
            got = _dihedral(_dihedral(probe, turn, flip), turn_b, flip_b)
            if got.shape == want.shape and (got == want).all():
                return turn, flip
    raise AssertionError("every turn and flip is one of the eight")


def _gels_like(chunk) -> tuple[int, bytes] | None:
    """(codec, payload) of an intact old `myWb` chunk (magic `MWAS`)."""

    data = chunk.data
    if not chunk.crc_ok or len(data) < 44 or data[:4] != b"MWAS" or data[4] != 1 or data[6:8] != b"\0\0":
        return None
    length = struct.unpack(">I", data[8:12])[0]
    payload = data[44:]
    if len(payload) != length or hashlib.sha256(payload).digest() != data[12:44]:
        return None
    return data[5], payload


# --- one old document --------------------------------------------------------------------------


@dataclass
class _Old:
    data: dict
    png: bytes
    image: _svgxml.Element
    """The `image` element holding the PNG."""
    metadata: _svgxml.Element | None = None
    """Schema 3–4: the `metadata` element holding the JSON."""


def _sources(old: dict, schema: int, payloads: dict, notes: list[str]):
    """`core.sources`, the payloads they point to, and per role (source
    id, source channel)."""

    info = old.get("source_image_info") or {}
    assets_root = (old.get("embedded_png_info") or {}).get("source_assets") or {}
    views = assets_root.get("role_views") or {}
    bundles, assets = assets_root.get("bundles") or {}, assets_root.get("assets") or {}
    records = assets_root.get("file_metadata") or {}
    sources, roles, keys = [], {}, {}
    images: dict = {}
    metadata: dict = {}

    for role in _ROLES:
        view = views.get(role)
        present = role == "blot" or view is not None or info.get(f"{role}_sha256") or info.get(f"{role}_file_name")
        if not present:
            continue
        record: dict = {}
        if info.get(f"{role}_file_name"):
            record["file_name"] = info[f"{role}_file_name"]
        file_hash = _hex_hash(info.get(f"{role}_sha256"))
        if file_hash:
            record["file_hash"] = file_hash
        if view is not None and view.get("bundle_id") in bundles:
            bundle = bundles[view["bundle_id"]]
            names = ["gray"] if bundle.get("color_format") == "grayscale" else ["red", "green", "blue"]
            channels = []
            for name in names:
                asset = assets.get((bundle.get("planes") or {}).get(name)) or {}
                channel = {"name": name}
                if _hex_hash(asset.get("pixel_sha256")):
                    channel["pixel_hash"] = "sha256:" + asset["pixel_sha256"]
                    png = payloads.get((1, asset.get("png_sha256")))
                    if png is not None:
                        images[channel["pixel_hash"]] = {
                            "dtype": asset["dtype"], "shape": asset["shape"], "byte_length": len(png),
                            "png_hash": "sha256:" + asset["png_sha256"], "payload": png}
                    else:
                        notes.append(f"The {role} image's {name} pixels were missing or damaged in the old document; "
                                     "they are not embedded.")
                channels.append(channel)
            record["dtype"], record["shape"] = bundle.get("dtype"), bundle.get("shape")
            record["source_channels"] = channels
            selected = view.get("selected_channel") or "gray"
            meta_id = view.get("file_metadata_id")
            if meta_id is not None:
                converted = _metadata(records.get(meta_id), payloads, notes, role)
                if converted is not None:
                    entry, payload = converted
                    key = _formal.key(entry)
                    metadata[key] = (entry, payload)
                    record["source_metadata"] = key
            share = ("bundle", view["bundle_id"])
        else:
            selected = info.get(f"{role}_rgb_channel") or "gray"
            names = ["gray"] if selected == "gray" else ["red", "green", "blue"]
            record["source_channels"] = [{"name": n} for n in names]
            shape = info.get(f"{role}_image_shape")
            if C.is_shape(shape):
                record["shape"] = list(shape)
            if schema >= 5:
                notes.append(f"The old document did not embed the {role} image's pixels, so the panel has no pixel "
                             "hash for it until its source file is read again (Appendix B).")
            share = ("file", file_hash or record.get("file_name") or role)
        if share in keys:
            roles[role] = (keys[share], selected)
            continue
        source_id = f"s{len(sources) + 1}"
        keys[share] = source_id
        ordered = {"id": source_id}
        for key in ("file_name", "file_hash", "dtype", "shape", "source_channels", "source_metadata"):
            if record.get(key) is not None:
                ordered[key] = record[key]
        sources.append(ordered)
        roles[role] = (source_id, selected)
    return sources, roles, images, metadata


def _metadata(record, payloads: dict, notes: list[str], role: str):
    """An old `file_metadata` record as source metadata of `version` 1."""

    def lost(why: str):
        notes.append(f"The {role} file's metadata could not be carried over ({why}).")
        return None

    if type(record) is not dict:
        return lost("its record is missing")
    payload = payloads.get((2, record.get("payload_sha256")))
    if payload is None:
        return lost("its bytes are missing or damaged")
    segments, offset = [], 0
    for unit in record.get("units") or []:
        if type(unit) is not dict or unit.get("offset") != offset or not C.is_int(unit.get("length")):
            return lost("its units do not cover its bytes")
        fields = {k: v for k, v in unit.items() if k not in ("offset", "length")}
        segments.append((fields, payload[offset : offset + unit["length"]]))
        offset += unit["length"]
    if offset != len(payload):
        return lost("its units do not cover its bytes")
    keys = {k: record[k] for k in ("byte_order", "tiff_version") if k in record}
    try:
        return _formal.build(record.get("format"), segments, **keys)
    except AssertionError as exc:
        return lost(str(exc))


@dataclass
class _Labels:
    """What the old sample labels say of how they were drawn: whether the
    old MyWB could draw the table or rotated form at all, the table's
    divider (as the old MyWB read it) and its number of rows, and the angle
    of each rotated lane (Appendix B)."""

    drawable: bool = False
    separator: int | None = None
    row_count: int = 0
    angles: list[float] = field(default_factory=list)
    merged: bool | None = None
    """Whether the table drew the same values of lanes next to each other
    once; None when its lane did not say (drawn by the default, not moved)."""


# The table's `cell_render_mode`s that drew the same values once, across the
# lanes (§6.7 `merge_same_values`); any other value drew each lane's.
_MERGING_MODES = {"underline_span": "underline", "bracket_span": "bracket"}


def _in_roi_x(x, roi: dict) -> bool:
    """Whether a lane is within the ROI's breadth, as the old MyWB drew a
    table only of those lanes."""

    roi_x, roi_w = roi.get("roi_x"), roi.get("roi_w")
    if not (C.is_number(roi_x) and C.is_number(roi_w)):
        return True
    return min(roi_x, roi_x + roi_w) <= x < max(roi_x, roi_x + roi_w)


def _old_float(value) -> float | None:
    """A number as the old MyWB read it (Python's float()), or None."""

    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _old_separator(value, row_count: int) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    number = _old_float(value)
    if number is None or not np.isfinite(number) or not number.is_integer():
        return None
    return min(max(0, int(number)), row_count)


def _old_angle(value) -> float:
    angle = _old_float(value)
    return min(90.0, max(-90.0, 45.0 if angle is None else angle))


def _samples(old: dict, label_type: str, notes: list[str], extension: dict):
    """`core.samples` lanes and conditions from `sample_label_info`, and
    what the labels say of how they were drawn (`_Labels`)."""

    labels = old.get("sample_label_info") or {}
    readable = []
    for key, item in labels.items():
        if (type(item) is not dict or not C.is_number(item.get("x_spinbox")) or not C.is_number(item.get("y_spinbox"))
                or not isinstance(item.get("label"), str)):
            notes.append(f"The sample label {key!r} could not be read and was dropped.")
            continue
        readable.append((key, item))

    parsed: dict[str, dict | None] = {}
    if label_type in ("table", "rotated"):
        for key, item in readable:
            try:
                value = _json.parse(item["label"].encode("utf-8"))
            except _json.JsonError:
                value = None
            ok = isinstance(value, dict) and value.get("version") == 1 and (
                (label_type == "table" and type(value.get("rows")) is list
                 and all(type(r) is list and len(r) == 2 and all(isinstance(c, str) for c in r) for r in value["rows"]))
                or (label_type == "rotated" and "angle" in value))
            parsed[key] = value if ok else None
            if not ok:
                notes.append(f"The {label_type} label of lane {key!r} could not be read; its text is kept as details.")

    conditions, values = [], {key: {} for key, _ in readable}
    drawn = _Labels()
    if label_type == "table" and any(parsed.values()):
        by_x = sorted((k for k, _ in readable if parsed.get(k)), key=lambda k: labels[k]["x_spinbox"])
        count = max(len(parsed[k]["rows"]) for k in by_x)
        drawn.drawable, drawn.row_count = True, count
        # The old MyWB took the divider of the leftmost lane that had one.
        drawn.separator = next((sep for sep in (_old_separator(parsed[k].get("row_separator_index"), len(parsed[k]["rows"]))
                                                for k in by_x) if sep is not None), None)
        if drawn.separator is not None:
            drawn.separator = min(drawn.separator, count)
        # It drew the table by the settings of its leftmost lane within the ROI.
        roi = old.get("roi_info") if type(old.get("roi_info")) is dict else {}
        inside = [k for k in by_x if _in_roi_x(labels[k]["x_spinbox"], roi)]
        mode = parsed[(inside or by_x)[0]].get("cell_render_mode")
        drawn.merged = (mode in _MERGING_MODES) if mode in ("per_lane", *_MERGING_MODES) else None
        for i in range(count):
            heading = next((parsed[k]["rows"][i][0] for k in by_x if i < len(parsed[k]["rows"]) and parsed[k]["rows"][i][0]), "")
            others = {parsed[k]["rows"][i][0] for k in by_x if i < len(parsed[k]["rows"])} - {heading, ""}
            if others:
                notes.append(f"Row {i + 1} of the sample table had other headings ({', '.join(sorted(others))}); "
                             f"{heading!r} was kept.")
            condition = {"id": f"t{i + 1}"}
            if heading:
                condition["name"] = _text(heading)
            conditions.append(condition)
            for k in by_x:
                if i < len(parsed[k]["rows"]) and parsed[k]["rows"][i][1]:
                    values[k][condition["id"]] = _text(parsed[k]["rows"][i][1])
    elif label_type == "rotated" and readable:
        conditions.append({"id": "r1"})
        # A label the old MyWB could not read was drawn rotated by 45° (its default).
        drawn.drawable = True
        drawn.angles = [_old_angle(parsed[k]["angle"]) if parsed.get(k) else 45.0 for k, _ in readable]
        for key, item in readable:
            p = parsed.get(key)
            text = p.get("rotated_label", p.get("label", "")) if p else ""
            if isinstance(text, str) and text:
                values[key]["r1"] = _text(text)

    settings = {}
    lanes = []
    for key, item in readable:
        lane: dict = {"id": key}
        p = parsed.get(key)
        details = p.get("label", "") if p else item["label"]
        if p:
            # The divider is figure.style's split (Appendix B); the angle goes there when every lane has one.
            settings[key] = {k: v for k, v in p.items()
                             if k not in ("version", "label", "rows", "rotated_label", "row_separator_index",
                                          "cell_render_mode")}
            # Whether to merge is figure.style's; the shape of the mark, MyWB's (Appendix B).
            if _MERGING_MODES.get(p.get("cell_render_mode")) == "bracket":
                settings[key]["span_shape"] = "bracket"
        if not re.fullmatch(r"[1-9][0-9]*", key):
            notes.append(f"The sample label {key!r} has no lane number; it was dropped.")
            continue
        else:
            lane.update(number=int(key), x=item["x_spinbox"], y=item["y_spinbox"])
            if isinstance(details, str) and details:
                lane["details"] = _text(details)
            if values[key]:
                lane["values"] = values[key]
        lanes.append(lane)
    if label_type == "rotated" and drawn.angles and len(set(drawn.angles)) == 1:
        for item in settings.values():
            item.pop("angle", None)
    settings = {k: v for k, v in settings.items() if v}
    if settings:
        extension["sample_labels"] = settings
    return lanes, conditions, drawn


def _bands(old: dict, lanes: list[dict], notes: list[str]):
    names = old.get("band_row_names") or []
    raw = old.get("band_info") or []
    good = []
    for item in raw if type(raw) is list else []:
        try:
            x, y, w, h = (float(item[k]) for k in ("x", "y", "w", "h"))
            lane, row = item["lane"], item["row"]
            ok = (all(np.isfinite([x, y, w, h])) and w >= 0 and h >= 0 and type(lane) is int and lane >= 1
                  and type(row) is int and row >= 0)
        except (KeyError, TypeError, ValueError):
            ok = False
        if ok:
            good.append(item)
    if len(good) != len(raw if type(raw) is list else []):
        notes.append(f"{len(raw) - len(good)} band(s) that the old reader also skipped were dropped.")

    numbers = sorted({b["row"] for b in good} | {i for i, n in enumerate(names) if isinstance(n, str) and n})
    row_ids = {n: f"r{i + 1}" for i, n in enumerate(numbers)}
    rows = []
    for n in numbers:
        row = {"id": row_ids[n]}
        if n < len(names) and isinstance(names[n], str) and names[n]:
            row["name"] = _text(names[n])
        rows.append(row)

    unlabelled = {n: f"n{i + 1}" for i, n in enumerate(sorted({b["lane"] for b in good if b["lane"] >= 10000}))}
    lane_ids = {lane["id"] for lane in lanes}
    bands, cells = [], set()
    for b in good:
        lane = unlabelled.get(b["lane"], str(b["lane"]))
        if lane not in lane_ids:
            lanes.append({"id": lane})
            lane_ids.add(lane)
        cell = (lane, row_ids[b["row"]])
        if cell in cells:
            notes.append(f"A second band in lane {lane!r}, row {cell[1]!r} was dropped.")
            continue
        cells.add(cell)
        if b["w"] > 0 and b["h"] > 0:
            bands.append({"lane": lane, "row": cell[1], "shape": "rect", "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]})
        else:
            bands.append({"lane": lane, "row": cell[1], "shape": "point",
                          "x": b["x"] + b["w"] / 2, "y": b["y"] + b["h"] / 2})
    return rows, bands, row_ids


def _markers(old: dict, notes: list[str]) -> list[dict]:
    out = []
    for key, item in (old.get("marker_label_info") or {}).items():
        if (type(item) is not dict or not C.is_number(item.get("x_spinbox")) or not C.is_number(item.get("y_spinbox"))
                or not isinstance(item.get("label"), str) or not C.is_id(key)):
            notes.append(f"The marker label {key!r} could not be read and was dropped.")
            continue
        out.append({"id": key, "x": item["x_spinbox"], "y": item["y_spinbox"], "label": _text(item["label"])})
    return out


def _side(value) -> str | None:
    return value if value in ("left", "right") else None


def annotation_side(old: dict, picture_size: tuple[int, int] | None = None) -> str:
    """The side, left or right, an old document drew its one annotation on:
    its `annotation_pos` when that is set; else the side opposite the marker
    labels -- opposite `marker_pos` when that is set, and with each label
    placed by itself, the side with fewer of them drawn, a tie going right.
    A label is drawn when it is level with the picture, on the left when it
    is left of the picture's middle. `picture_size` (width, height) is the
    picture's; without it, the ROI's, and without that the labels are not
    counted."""

    style = old.get("svg_style") if type(old.get("svg_style")) is dict else {}
    misc = old.get("misc_info") if type(old.get("misc_info")) is dict else {}
    chosen = _side(style.get("annotation_pos"))
    if chosen:
        return chosen
    schema = old.get("schema_version")
    marker_pos = _side(misc.get("marker_pos") if type(schema) is int and schema <= 3 else style.get("marker_pos"))
    if marker_pos:
        return "right" if marker_pos == "left" else "left"
    roi = old.get("roi_info") if type(old.get("roi_info")) is dict else {}
    x0, y0 = (roi.get(k) if C.is_number(roi.get(k)) else 0 for k in ("roi_x", "roi_y"))
    size = picture_size or (roi.get("roi_w"), roi.get("roi_h"))
    if not all(C.is_number(v) for v in size):
        return "right"
    width, height = size
    left = right = 0
    for item in (old.get("marker_label_info") or {}).values() if type(old.get("marker_label_info")) is dict else ():
        if type(item) is not dict or not C.is_number(item.get("x_spinbox")) or not C.is_number(item.get("y_spinbox")):
            continue
        if not 0 <= item["y_spinbox"] - y0 <= height:
            continue
        if item["x_spinbox"] - x0 < width / 2:
            left += 1
        else:
            right += 1
    return "left" if right > left else "right"


def _rendered(png: bytes):
    walk = _png.walk(png)
    ihdr = _png.Ihdr.parse(walk.chunks[0].data) if walk.chunks and walk.chunks[0].type == b"IHDR" else None
    idat = [c for c in walk.chunks if c.type == b"IDAT"]
    raw = _png.inflate_complete(b"".join(c.data for c in idat)) if idat and all(c.crc_ok for c in idat) else None
    if ihdr is None or ihdr.problems() or raw is None or _png.scanline_problem(ihdr, raw) \
            or ihdr.color_type not in (0, 2) or ihdr.bit_depth not in (8, 16):
        raise LegacyError("The old document's picture cannot be decoded.")
    return walk, ihdr, _png.decode_pixels(ihdr, raw)


@dataclass
class Mapped:
    """A schema 6 JSON (an old MyWB document's, or MyWB's own while it holds a
    document) as a GelSVG panel: its core, the MyWB extension's content
    (without `based_on`), the payloads they point to, and `style`, the
    figure's style (`figure.style`, §6.7) as the old MyWB drew it, empty
    when nothing was written that says."""

    schema_version: int
    core: dict
    extension: dict
    source_channel_pngs: list[bytes]
    source_metadata: list[tuple[dict, bytes]]
    notes: list[str]
    roi: dict | None
    old_svg_hash: str | None
    style: dict = field(default_factory=dict)


# Appendix B: the keys of `svg_style` that §6.7 covers. They move to
# `figure.style` with the values the old MyWB drew with, and leave the
# extension; `annotation_pos` only decides the side in `core.annotations`.
_FONT_SIZES = {"sample_label_font_size": "sample_label", "sample_details_font_size": "details_list",
               "marker_label_font_size": "marker_label", "band_label_font_size": "row_name",
               "annotation_font_size": "annotation"}
_STYLE_KEYS = (*_FONT_SIZES, "font_family", "image_scale_percent", "image_border_stroke_width", "sample_pos",
               "hide_sample_details", "marker_pos", "band_label_pos", "band_label_visible", "annotation_pos")


def _old_bool(value) -> bool:
    """A setting turned on or off, as the old MyWB read it."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _old_clamped(value, low: float) -> float | None:
    number = _old_float(value)
    if number is None:
        return None  # drawn with the default: not moved
    number = max(low, number)
    return number if np.isfinite(number) else None


def _old_side(value, sides: tuple[str, ...]) -> str | None:
    """A side the old MyWB was set to draw on, or None for one it decided
    itself (`auto`, and absent or unreadable): where it then drew is not
    known here (Appendix B)."""

    return value if value in sides else None


def _style(old: dict, schema, misc: dict, label_type, labels: _Labels, row_ids: dict) -> dict:
    """`figure.style` from the old settings (Appendix B): only the keys
    written, with the values the old MyWB drew with."""

    svg_style = dict(old["svg_style"]) if type(old.get("svg_style")) is dict else {}
    for key in ("marker_pos", "sample_pos"):  # schema 3 kept them in misc_info
        if type(schema) is int and schema <= 3 and isinstance(misc.get(key), str):
            svg_style[key] = misc[key]
    style: dict = {}

    def put(group: str | None, key: str, value) -> None:
        if value is not None:
            (style.setdefault(group, {}) if group else style)[key] = value

    if isinstance(svg_style.get("font_family"), str):
        put(None, "font_family", svg_style["font_family"].strip().strip("\"'") or None)
    if "image_scale_percent" in svg_style:
        percent = _old_clamped(svg_style["image_scale_percent"], 0)
        put(None, "image_scale", None if percent is None else max(percent, 0.001) / 100)
    if "image_border_stroke_width" in svg_style:
        put(None, "image_border_width", _old_clamped(svg_style["image_border_stroke_width"], 0))

    form = {"basic": "numbers", "table": "table", "rotated": "rotated"}.get(label_type) if isinstance(label_type, str) else None
    if form in ("table", "rotated") and not labels.drawable:
        form = "numbers"  # the old MyWB drew lane numbers instead
    put("sample_label", "form", form)
    side = _old_side(svg_style.get("sample_pos"), ("top", "bottom"))
    automatic = None if side is None else False
    divider, count = labels.separator, labels.row_count
    if form == "table" and divider is not None and count:
        before, after = divider > 0, divider < count
        if side is None:
            # Left automatic (absent or unreadable is automatic too), the old MyWB drew the rows
            # before the divider above the image and the rest below; an empty part was not drawn.
            side = "split" if before and after else ("bottom" if after else "top")
            automatic = True
            if side == "split":
                put("sample_label", "bottom_from", f"t{divider + 1}")
        elif before and after:
            put("sample_label", "bottom_from", f"t{divider + 1}")  # kept, not used (§6.7)
    put("sample_label", "side", side)
    put("sample_label", "side_automatic", automatic)
    if form == "table":
        put("sample_label", "merge_same_values", labels.merged)
    if form == "rotated" and len(set(labels.angles)) == 1:
        put("sample_label", "angle", labels.angles[0])

    for key, group in _FONT_SIZES.items():
        if key in svg_style:
            put(group, "font_size", _old_clamped(svg_style[key], 0.5))
    if "hide_sample_details" in svg_style:
        put("details_list", "visible", not _old_bool(svg_style["hide_sample_details"]))
    for group, key in (("marker_label", "marker_pos"), ("row_name", "band_label_pos")):
        side = _old_side(svg_style.get(key), ("left", "right"))
        put(group, "side", side)
        put(group, "side_automatic", None if side is None else False)
    hidden = [row_ids[n] for n in misc.get("band_label_hidden_rows") or [] if type(n) is int and n in row_ids]
    if "band_label_visible" in svg_style and not _old_bool(svg_style["band_label_visible"]):
        hidden = list(row_ids.values())  # no row named: every row (§6.7 has no `visible`)
    put("row_name", "hidden_rows", list(dict.fromkeys(hidden)) or None)
    return style


def converted_from(old: dict) -> dict:
    """What the MyWB extension records of the old document a panel was
    converted from: its schema, and the MyWB that wrote it."""

    integrity = old.get("svg_integrity") if type(old.get("svg_integrity")) is dict else {}
    written = {k: integrity[o] for k, o in (("name", "generated_by"), ("version", "app_version"), ("build", "build_id"))
               if isinstance(integrity.get(o), str) and integrity[o]}
    return {"schema_version": old.get("schema_version"), **({"written_by": written} if written else {})}


def from_schema6(old: dict, payloads: dict, *, picture_depth: int | None = None,
                 picture_size: tuple[int, int] | None = None, converted: bool = True) -> Mapped:
    """Map a schema 3–6 JSON by Appendix B. `payloads` maps (codec, SHA-256
    hex) to the old `myWb` payloads (codec 1 plane PNGs, codec 2 metadata).
    `picture_depth` is the old picture's bit depth when there is one: it
    decides `output_bit_depth`; `picture_size` (width, height), the side an
    old annotation was drawn on (`annotation_side`). With `converted`, the
    extension records the schema and writer converted from."""

    notes: list[str] = []
    schema = old.get("schema_version")
    extension: dict = {}
    if converted:
        extension["converted_from"] = converted_from(old)

    sources, roles, images, metadata = _sources(old, schema, payloads, notes)
    fb, fm = old.get("filter_data_b"), old.get("filter_data_m")
    if type(fb) is not dict or type(fm) is not dict or type(old.get("roi_info")) is not dict:
        raise LegacyError("The old document lacks 'filter_data_b', 'filter_data_m' or 'roi_info'.")
    turn_b, flip_b = _turn(fb.get("rotation_offset", 0), "filter_data_b.rotation_offset"), _flag(fb.get("flip"), "filter_data_b.flip")
    rotation = fb.get("rotation")
    if not C.is_number(rotation) or not -180 <= rotation < 180:
        raise LegacyError("'filter_data_b.rotation' is not a number from −180 up to 180.")

    # Appendix B: the blot is the one drawn panel channel and the reference
    # channel; the marker image becomes a marker channel, not drawn.
    source_b, channel_b = roles["blot"]
    blot = {"id": "blot", "drawn": True, "source": source_b, "source_channel": channel_b,
            "invert": _flag(fb.get("invert"), "filter_data_b.invert"), "contrast": _contrast(fb.get("contrast"), "filter_data_b.contrast")}
    channels = [blot]
    if "marker" in roles:
        source_m, channel_m = roles["marker"]
        turn, flip = _marker_turn_flip(_turn(fm.get("rotation_offset", 0), "filter_data_m.rotation_offset"),
                                       _flag(fm.get("flip"), "filter_data_m.flip"), turn_b, flip_b)
        channels.append({"id": "marker", "role": "marker", "drawn": False, "source": source_m,
                         "source_channel": channel_m, "turn": turn, "flip": flip,
                         "invert": _flag(fm.get("invert"), "filter_data_m.invert"),
                         "contrast": _contrast(fm.get("contrast"), "filter_data_m.contrast")})
    core: dict = {"sources": sources, "channels": channels, "reference_channel": "blot"}
    core["orientation"] = {"turn": turn_b, "flip": flip_b, "rotation": rotation}
    extension["highlight_saturated"] = {r: bool(f.get("highlight_saturated", False)) for r, f in (("blot", fb), ("marker", fm))}

    roi_info = old["roi_info"]
    corners = [roi_info.get(k) for k in ("roi_x", "roi_y", "roi_w", "roi_h")]
    roi = None
    if all(type(v) is int for v in corners):
        x, y, w, h = corners
        if w < 0:
            x, w = x + w, -w
        if h < 0:
            y, h = y + h, -h
        if w and h:
            roi = {"x": x, "y": y, "w": w, "h": h}
            core["roi"] = roi
        else:
            notes.append("The old ROI was empty and was dropped.")
    elif any(v is not None for v in corners):
        notes.append("The old ROI could not be read and was dropped.")

    misc = old.get("misc_info") if type(old.get("misc_info")) is dict else {}
    blot_dtype = next((s.get("dtype") for s in sources if s["id"] == source_b), None)
    stated = {"8": 8, "16": 16}.get(misc.get("output_bit_depth"))
    if picture_depth is not None:
        # The old picture was drawn at the output bit depth (old §1.5); an
        # old reader read "16" for an 8-bit image as 8. The picture decides.
        depth = 8 if picture_depth == 16 and blot_dtype == "uint8" else picture_depth
        if stated is not None and stated != depth:
            notes.append(f"The old output bit depth {misc['output_bit_depth']!r} did not match the picture's "
                         f"{picture_depth} bits; {depth} was used.")
    else:
        # Old §6: missing, unknown, or "16" for an 8-bit image is the image's own depth.
        own = {"uint8": 8, "uint16": 16}.get(blot_dtype)
        depth = stated if stated is not None and not (stated == 16 and own == 8) else own or 8
    core["output_bit_depth"] = depth

    label_type = misc.get("sample_label_type") if misc.get("sample_label_type") in ("basic", "table", "rotated") else "basic"
    lanes, conditions, labels = _samples(old, label_type, notes, extension)
    rows, bands, row_ids = _bands(old, lanes, notes)
    samples = {}
    if conditions:
        samples["conditions"] = conditions
    if lanes:
        samples["lanes"] = lanes
    if samples:
        core["samples"] = samples
    if rows:
        core["rows"] = [{**row, "channel": "blot"} for row in rows]
    if bands:
        core["bands"] = bands
    markers = _markers(old, notes)
    if markers:
        core["markers"] = markers
    unit = misc.get("marker_unit")  # MyWB's own, while it holds a document; no old document has one
    if isinstance(unit, str) and unit.strip():
        core["marker_unit"] = _text(unit, backslash_n=False)
    annotations = {}
    if type(misc.get("annotations")) is dict:  # MyWB's own, while it holds a document
        annotations = {side: _text(misc["annotations"][side]) for side in ANNOTATION_SIDES
                       if isinstance(misc["annotations"].get(side), str) and misc["annotations"][side]}
    elif isinstance(misc.get("annotation_text"), str) and misc["annotation_text"]:
        annotations = {annotation_side(old, picture_size): _text(misc["annotation_text"])}
    if annotations:
        core["annotations"] = annotations
    if isinstance(misc.get("note"), str) and misc["note"]:
        core["note"] = _text(misc["note"], backslash_n=False)

    style = _style(old, schema, misc, misc.get("sample_label_type"), labels, row_ids)
    # What §6.7 covers is in `style` now, not the extension (§8).
    rest = {k: v for k, v in old["svg_style"].items() if k not in _STYLE_KEYS} \
        if type(old.get("svg_style")) is dict else {}
    if rest:
        extension["svg_style"] = rest
    if type(old.get("quantification_state")) is dict:
        extension["quantification_state"] = old["quantification_state"]

    old_svg_hash = None
    integrity = old.get("svg_integrity") if type(old.get("svg_integrity")) is dict else {}
    if integrity.get("algorithm") == "sha256" and integrity.get("target") == "svg_c14n2_without_mywb_data" \
            and _hex_hash(integrity.get("hash")):
        old_svg_hash = "sha256:" + integrity["hash"]
    elif integrity.get("hash") and converted:
        notes.append("The old document's SVG hash was made in a way that cannot be checked again; it was not carried over.")
    return Mapped(schema, core, extension, [e["payload"] for e in images.values()], list(metadata.values()), notes,
                  roi, old_svg_hash, style)


def _convert_one(item: _Old, tool: Tool, saved_at: str | None) -> tuple[ConvertedPanel, str | None]:
    walk, ihdr, rendered = _rendered(item.png)
    payloads, damaged = {}, []
    for chunk in walk.chunks:
        if chunk.type == b"myWb":
            got = _gels_like(chunk)
            if got is None:
                damaged.append("An old myWb chunk was damaged and was not used.")
            else:
                payloads[(got[0], hashlib.sha256(got[1]).hexdigest())] = got[1]
    mapped = from_schema6(item.data, payloads, picture_depth=ihdr.bit_depth, picture_size=(ihdr.width, ihdr.height))
    notes = damaged + mapped.notes
    roi = mapped.roi
    fallback = (item.data.get("embedded_png_info") or {}).get("fallback_mode") or item.image.attributes.get("data-mywb-fallback-mode")
    if fallback == "full-rendered" or roi is None:
        crop = {"x": 0, "y": 0, "w": ihdr.width, "h": ihdr.height}
    else:
        crop = {"x": max(roi["x"], 0), "y": max(roi["y"], 0), "w": ihdr.width, "h": ihdr.height}
    figure = {"crop": crop, "rendered_hash": _hashes.rendered_hash(rendered), "drawn_from": _hashes.core_hash(mapped.core)}
    if mapped.style:
        figure["style"] = mapped.style
    if mapped.old_svg_hash:
        figure.update(svg_hash=mapped.old_svg_hash, svg_hash_method="c14n2_without_panel_data")
    panel = {"core": mapped.core, "figure": figure,
             "extensions": {MYWB_EXTENSION: {"based_on": _hashes.core_hash(mapped.core), **mapped.extension}}}
    try:
        png = _build(Content(panel, source_channel_pngs=mapped.source_channel_pngs, source_metadata=mapped.source_metadata), tool,
                     saved_at, [], index=None, element=item.image, images={}, metadata={},
                     kept=_Kept(item.png, walk, figure, legacy=True))
    except WriteError as exc:
        raise LegacyError(f"The converted panel would not be valid: {exc}", exc.notices) from None
    return ConvertedPanel(mapped.schema_version, png, notes), mapped.old_svg_hash


# --- finding old documents ---------------------------------------------------------------------


def _find(svg: bytes) -> list[_Old]:
    container = read_container(svg)
    if container.tree is None:
        raise LegacyError(f"The file is not an SVG this library reads: {container.not_a_document.detail}.")
    found = []
    for slot in container.legacy:
        chunk = _carrier.find_keyword(_png.walk(slot.png), _carrier.LEGACY_KEYWORD)[0]
        fields, why = _carrier.parse_itxt(chunk.data) if chunk.crc_ok else (None, "its CRC does not match")
        if fields is None:
            raise LegacyError(f"An old MyWB panel's JSON cannot be read ({why}).")
        found.append(_Old(_parse(fields.text, 5), slot.png, slot.element))
    elements = container.tree.elements
    for index, node in enumerate(ET.fromstring(svg).iter()):
        if node.tag.rsplit("}", 1)[-1] != "metadata" or node.get("data-mywb-role") != "settings":
            continue
        element = elements[index]
        parent = element.parent
        png = _datauri.decode_png_data_uri(_datauri.image_href(parent.attributes) or "") if parent is not None \
            and parent.local == "image" else None
        if not isinstance(png, bytes):
            continue
        if _carrier.find_keyword(_png.walk(png), _carrier.LEGACY_KEYWORD):
            continue  # its PNG is the carrier (schema 5–6), already found
        found.append(_Old(_parse((node.text or "").encode("utf-8"), 3, 4), png, parent, element))
    return found


def _parse(text: bytes, low: int, high: int | None = None) -> dict:
    try:
        data = _json.parse(text)
    except _json.JsonError as exc:
        raise LegacyError(f"An old MyWB document's JSON cannot be read ({exc}).") from None
    schema = data.get("schema_version")
    if not C.is_int(schema) or schema < low or (high is not None and schema > high) or schema > 6:
        raise LegacyError(f"An old MyWB document of schema {schema!r} cannot be converted here (schema 3–6 can; the "
                          "JSON in a PNG must be schema 5 or later, in a metadata element schema 4 or earlier).")
    return data


def convert(svg: bytes, tool: Tool, *, saved_at: str | None = None) -> Converted:
    """Convert every old MyWB document in the SVG to a GelSVG panel, put in
    the old one's place; GelSVG panels and every other byte stay as they
    were. Raises LegacyError when there is none or one cannot be
    converted."""

    found = _find(svg)
    if not found:
        raise NoOldDocument("The file holds no old MyWB document.")
    edits, panels, hashes = [], [], []
    for item in found:
        panel, old_hash = _convert_one(item, tool, saved_at)
        panels.append(panel)
        hashes.append(old_hash)
        start, end = _splice.href_span(svg, item.image)
        edits.append((start, end, _splice.png_data_uri(panel.png)))
        if item.metadata is not None:
            edits.append((*_splice.element_span(svg, item.metadata), b""))
    out = _splice.apply(svg, edits)

    if any(hashes):
        try:
            now = _svghash.svg_hash(out)
        except (_svghash.Unavailable, ET.ParseError):
            now = None
        stale = [i for i, h in enumerate(hashes) if h and h != now]
        if stale:
            container = read_container(out)
            fixes = []
            for i in stale:
                panels[i].notes.append("The SVG had been changed after the old MyWB wrote it, so its old SVG hash no "
                                       "longer matched and was not carried over.")
                figure = _json.parse(_carrier.parse_itxt(_carrier.find_keyword(_png.walk(panels[i].png),
                                     _carrier.GELSVG_KEYWORD)[0].data)[0].text)["figure"]
                figure.pop("svg_hash"), figure.pop("svg_hash_method")
                new_png = _replace_json_member(panels[i].png, "figure", figure)
                slot = next(s for s in container.panels if s.carrier.png == panels[i].png)
                panels[i].png = new_png
                fixes.append((slot.element, new_png))
            out = _splice.replace_hrefs(out, fixes)
    return Converted(out, panels)

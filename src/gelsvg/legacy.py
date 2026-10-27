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
from .container import read_container
from .writer import Content, Tool, WriteError, _build, _Kept, _replace_json_member

MYWB_EXTENSION = "https://mybioapps.com/mywb"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SIZE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(kda|kd|bp|kbp|kb|nt)\s*", re.IGNORECASE)
_UNITS = {"kda": "kDa", "kd": "kDa", "bp": "bp", "kbp": "kb", "kb": "kb", "nt": "nt"}
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
    id, channel, element pixel hash)."""

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
        pixel_hash = None
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
            record["channels"] = channels
            selected = view.get("selected_channel") or "gray"
            if selected == "average":
                pixel_hash = _hex_hash(view.get("selected_pixel_sha256"))
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
            record["channels"] = [{"name": n} for n in names]
            shape = info.get(f"{role}_image_shape")
            if C.is_shape(shape):
                record["shape"] = list(shape)
            if schema >= 5:
                notes.append(f"The old document did not embed the {role} image's pixels, so the panel has no pixel "
                             "hash for it until its source file is read again (Appendix B).")
            share = ("file", file_hash or record.get("file_name") or role)
        if share in keys:
            roles[role] = (keys[share], selected, pixel_hash)
            continue
        source_id = f"s{len(sources) + 1}"
        keys[share] = source_id
        ordered = {"id": source_id}
        for key in ("file_name", "file_hash", "dtype", "shape", "channels", "source_metadata"):
            if record.get(key) is not None:
                ordered[key] = record[key]
        sources.append(ordered)
        roles[role] = (source_id, selected, pixel_hash)
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


def _samples(old: dict, label_type: str, notes: list[str], extension: dict):
    """`core.samples` lanes and conditions from `sample_label_info`."""

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
    if label_type == "table" and any(parsed.values()):
        by_x = sorted((k for k, _ in readable if parsed.get(k)), key=lambda k: labels[k]["x_spinbox"])
        count = max(len(parsed[k]["rows"]) for k in by_x)
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
            settings[key] = {k: v for k, v in p.items() if k not in ("version", "label", "rows", "rotated_label")}
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
    if settings:
        extension["sample_labels"] = settings
    return lanes, conditions


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
        label = {"id": key, "x": item["x_spinbox"], "y": item["y_spinbox"], "label": _text(item["label"])}
        match = _SIZE.fullmatch(item["label"])
        if match and float(match.group(1)) > 0:
            number = float(match.group(1))
            label["size"] = {"value": int(number) if number.is_integer() else number, "unit": _UNITS[match.group(2).lower()]}
        out.append(label)
    return out


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
    (without `based_on`), and the payloads they point to."""

    schema_version: int
    core: dict
    extension: dict
    channel_pngs: list[bytes]
    source_metadata: list[tuple[dict, bytes]]
    notes: list[str]
    roi: dict | None
    old_svg_hash: str | None


def from_schema6(old: dict, payloads: dict, *, picture_depth: int | None = None, converted: bool = True) -> Mapped:
    """Map a schema 3–6 JSON by Appendix B. `payloads` maps (codec, SHA-256
    hex) to the old `myWb` payloads (codec 1 plane PNGs, codec 2 metadata).
    `picture_depth` is the old picture's bit depth when there is one: it
    decides `output_bit_depth`. With `converted`, the extension records the
    schema and writer converted from."""

    notes: list[str] = []
    schema = old.get("schema_version")
    extension: dict = {}
    integrity = old.get("svg_integrity") if type(old.get("svg_integrity")) is dict else {}
    if converted:
        written = {k: integrity[o] for k, o in (("name", "generated_by"), ("version", "app_version"), ("build", "build_id"))
                   if isinstance(integrity.get(o), str) and integrity[o]}
        extension["converted_from"] = {"schema_version": schema, **({"written_by": written} if written else {})}

    sources, roles, images, metadata = _sources(old, schema, payloads, notes)
    fb, fm = old.get("filter_data_b"), old.get("filter_data_m")
    if type(fb) is not dict or type(fm) is not dict or type(old.get("roi_info")) is not dict:
        raise LegacyError("The old document lacks 'filter_data_b', 'filter_data_m' or 'roi_info'.")
    turn_b, flip_b = _turn(fb.get("rotation_offset", 0), "filter_data_b.rotation_offset"), _flag(fb.get("flip"), "filter_data_b.flip")
    rotation = fb.get("rotation")
    if not C.is_number(rotation) or not -180 <= rotation < 180:
        raise LegacyError("'filter_data_b.rotation' is not a number from −180 up to 180.")

    source_b, channel_b, hash_b = roles["blot"]
    blot = {"id": "blot", "source": source_b, "source_channel": channel_b,
            "invert": _flag(fb.get("invert"), "filter_data_b.invert"), "contrast": _contrast(fb.get("contrast"), "filter_data_b.contrast")}
    if hash_b:
        blot["pixel_hash"] = hash_b
    core: dict = {"sources": sources, "blot": [blot]}
    if "marker" in roles:
        source_m, channel_m, hash_m = roles["marker"]
        turn, flip = _marker_turn_flip(_turn(fm.get("rotation_offset", 0), "filter_data_m.rotation_offset"),
                                       _flag(fm.get("flip"), "filter_data_m.flip"), turn_b, flip_b)
        marker = {"source": source_m, "source_channel": channel_m, "turn": turn, "flip": flip,
                  "invert": _flag(fm.get("invert"), "filter_data_m.invert"), "contrast": _contrast(fm.get("contrast"), "filter_data_m.contrast")}
        if hash_m:
            marker["pixel_hash"] = hash_m
        core["marker"] = marker
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
    extension["sample_label_type"] = label_type
    lanes, conditions = _samples(old, label_type, notes, extension)
    rows, bands, row_ids = _bands(old, lanes, notes)
    samples = {}
    if conditions:
        samples["conditions"] = conditions
    if lanes:
        samples["lanes"] = lanes
    if samples:
        core["samples"] = samples
    if rows:
        core["rows"] = rows
    if bands:
        core["bands"] = bands
    markers = _markers(old, notes)
    if markers:
        core["markers"] = markers
    if isinstance(misc.get("annotation_text"), str) and misc["annotation_text"]:
        core["annotation"] = _text(misc["annotation_text"])
    if isinstance(misc.get("note"), str) and misc["note"]:
        core["note"] = _text(misc["note"], backslash_n=False)

    hidden = [row_ids[n] for n in misc.get("band_label_hidden_rows") or [] if type(n) is int and n in row_ids]
    if hidden:
        extension["band_label_hidden_rows"] = hidden
    for key in ("svg_style", "quantification_state"):
        if type(old.get(key)) is dict:
            extension[key] = old[key]
    for key in ("marker_pos", "sample_pos"):  # schema 3 kept them in misc_info
        if schema <= 3 and isinstance(misc.get(key), str):
            extension.setdefault("svg_style", {})[key] = misc[key]

    old_svg_hash = None
    if integrity.get("algorithm") == "sha256" and integrity.get("target") == "svg_c14n2_without_mywb_data" \
            and _hex_hash(integrity.get("hash")):
        old_svg_hash = "sha256:" + integrity["hash"]
    elif integrity.get("hash") and converted:
        notes.append("The old document's SVG hash was made in a way that cannot be checked again; it was not carried over.")
    return Mapped(schema, core, extension, [e["payload"] for e in images.values()], list(metadata.values()), notes,
                  roi, old_svg_hash)


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
    mapped = from_schema6(item.data, payloads, picture_depth=ihdr.bit_depth)
    notes = damaged + mapped.notes
    roi = mapped.roi
    fallback = (item.data.get("embedded_png_info") or {}).get("fallback_mode") or item.image.attributes.get("data-mywb-fallback-mode")
    if fallback == "full-rendered" or roi is None:
        crop = {"x": 0, "y": 0, "w": ihdr.width, "h": ihdr.height}
    else:
        crop = {"x": max(roi["x"], 0), "y": max(roi["y"], 0), "w": ihdr.width, "h": ihdr.height}
    figure = {"crop": crop, "rendered_hash": _hashes.rendered_hash(rendered), "drawn_from": _hashes.core_hash(mapped.core)}
    if mapped.old_svg_hash:
        figure.update(svg_hash=mapped.old_svg_hash, svg_hash_method="c14n2_without_panel_data")
    panel = {"core": mapped.core, "figure": figure,
             "extensions": {MYWB_EXTENSION: {"based_on": _hashes.core_hash(mapped.core), **mapped.extension}}}
    try:
        png = _build(Content(panel, channel_pngs=mapped.channel_pngs, source_metadata=mapped.source_metadata), tool,
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

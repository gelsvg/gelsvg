"""Reading `core` (§5) by rules 1–4 of §2.6.

The result is a view of the core as read: broken elements dropped, ignored
keys left out, CR read as LF. The core as written is kept by the caller for
hashing and re-saving. Each collection is checked element by element first;
then references are resolved in one pass that drops whatever points to
something dropped, until nothing changes (rule 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import _check as C
from ._check import Context, Obj

DTYPES = ("uint8", "uint16")
TURNS = (0, 90, 180, 270)
RGB_NAMES = {"red", "green", "blue"}
SIZE_UNITS = ("kDa", "bp", "kb", "nt")
BAND_SHAPES = ("rect", "point")
CHANNEL_IMAGE_KIND = "decoded_scalar_plane"
CHANNEL_IMAGE_MEDIA = "image/png"
FORMAL_METADATA = ("png", "jpeg", "tiff", "scn")


@dataclass
class CoreView:
    sources: list[dict] | None = None
    channel_images: dict[str, dict] = field(default_factory=dict)
    source_metadata: dict[str, dict] = field(default_factory=dict)
    blot: list[dict] | None = None
    marker: dict | None = None
    orientation: dict | None = None
    roi: dict | None = None
    output_bit_depth: int | None = None
    conditions: list[dict] = field(default_factory=list)
    lanes: list[dict] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    bands: list[dict] = field(default_factory=list)
    markers: list[dict] = field(default_factory=list)
    annotation: str | None = None
    note: str | None = None
    unreadable: set[str] = field(default_factory=set)
    """Required core keys that could not be read (rule 4)."""
    blot_count_as_written: int = 0
    metadata_keys_written: set[str] = field(default_factory=set)
    """Keys of `embedded.source_metadata` as written, valid or not."""
    ambiguous: dict[str, set[str]] = field(default_factory=dict)
    """IDs used twice: references to them were dropped (rule 3)."""

    def source(self, source_id: str) -> dict | None:
        for s in self.sources or ():
            if s["id"] == source_id:
                return s
        return None

    def lane_ids(self) -> set[str]:
        return {lane["id"] for lane in self.lanes}

    def row_ids(self) -> set[str]:
        return {row["id"] for row in self.rows}

    def blot_ids(self) -> set[str]:
        return {e["id"] for e in self.blot or ()}

    def band_cells(self) -> set[tuple[str, str]]:
        return {(b["lane"], b["row"]) for b in self.bands}


def _broken(ctx: Context, what: str, where: str, problems: list[str]) -> None:
    ctx.broken("element.broken", f"The {what} was dropped: " + "; ".join(problems) + ".", where)


def _array(ctx: Context, core: Obj, key: str) -> list:
    value = core.optional(key, lambda v: type(v) is list, "an array")
    return value or []


def _dedupe_ids(ctx: Context, items: list[tuple[int, dict]], what: str, where: str) -> tuple[list[tuple[int, dict]], set[str]]:
    """Keep the first element of each ID; drop the later ones and make the
    ID unusable as a reference target (rule 3)."""

    seen: dict[str, int] = {}
    ambiguous: set[str] = set()
    kept = []
    for index, item in items:
        ident = item["id"]
        if ident in seen:
            ambiguous.add(ident)
            ctx.broken("id.duplicate", f"A later {what} with the ID {ident!r} was dropped, "
                       "and so is everything pointing to that ID.", f"{where}[{index}]")
            continue
        seen[ident] = index
        kept.append((index, item))
    return kept, ambiguous


# --- sources and embedded ----------------------------------------------------------


def _read_channel(ctx: Context, value, where: str):
    if type(value) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, value, where)
    problems: list[str] = []
    name = o.required("name", lambda v: C.is_str(v) and v != "" and v != "average",
                      "a non-empty name other than 'average'", problems)
    out = {"name": name}
    o.seen.add("pixel_hash")
    if "pixel_hash" in value:
        v = value["pixel_hash"]
        w = f"{where}.pixel_hash"
        parts = C.hash_parts(v)
        if v is None or C.is_empty(v):
            ctx.broken("key.empty", "'pixel_hash' is null or empty and was read as absent.", w)
        elif parts is None:
            ctx.broken("key.invalid", "'pixel_hash' is not a hash and was ignored.", w)
        elif parts[0] != "sha256":
            if ctx.unknown_value(w, v):
                out["pixel_hash"], out["pixel_hash_known"] = v, False
            else:
                ctx.broken("key.invalid", "'pixel_hash' uses a hash kind this version does not define and was ignored.", w)
        else:
            out["pixel_hash"], out["pixel_hash_known"] = v, True
    o.finish()
    return (None, problems) if problems else (out, [])


def _read_source(ctx: Context, value, where: str):
    if type(value) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, value, where)
    problems: list[str] = []
    out: dict = {"id": o.required("id", C.is_id, "an ID", problems)}
    out["file_name"] = o.optional("file_name", C.is_str, "a string")
    out["file_hash"] = o.optional("file_hash", C.is_hash, "a hash")
    out["source_metadata"] = o.optional("source_metadata", C.is_hash, "a hash")

    channels_raw = o.raw("channels")
    channels: list[dict] = []
    if type(channels_raw) is not list or not channels_raw:
        problems.append("'channels' is missing or empty")
    else:
        names: set[str] = set()
        out["ambiguous_channels"] = set()
        for i, item in enumerate(channels_raw):
            channel, why = _read_channel(ctx, item, f"{where}.channels[{i}]")
            if channel is None:
                _broken(ctx, "channel", f"{where}.channels[{i}]", why)
                continue
            if channel["name"] in names:
                out["ambiguous_channels"].add(channel["name"])
                ctx.broken("channel.duplicate", f"A later channel named {channel['name']!r} was dropped, and so is "
                           "every element choosing that name.", f"{where}.channels[{i}]")
                continue
            names.add(channel["name"])
            channels.append(channel)
        if not channels:
            problems.append("no channel can be read")
    out["channels"] = channels

    has_pixels = any("pixel_hash" in c for c in channels)
    dtype = None
    if o.has("dtype"):
        v = o.raw("dtype")
        if v in DTYPES:
            dtype = v
        elif C.is_str(v) and ctx.unknown_value(f"{where}.dtype", v):
            out["dtype_unknown"] = True
        elif has_pixels:
            problems.append("'dtype' is not uint8 or uint16")
        else:
            ctx.broken("key.invalid", "'dtype' is not uint8 or uint16 and was ignored.", f"{where}.dtype")
    elif has_pixels:
        problems.append("'dtype' is missing although a channel has a pixel hash")
    shape = None
    if o.has("shape"):
        v = o.raw("shape")
        if C.is_shape(v):
            shape = list(v)
        elif has_pixels:
            problems.append("'shape' is not [height, width]")
        else:
            ctx.broken("key.invalid", "'shape' is not [height, width] and was ignored.", f"{where}.shape")
    elif has_pixels:
        problems.append("'shape' is missing although a channel has a pixel hash")
    out["dtype"], out["shape"] = dtype, shape
    o.finish()
    if problems:
        return None, problems
    return out, []


def _is_rgb(source: dict) -> bool:
    names = [c["name"] for c in source["channels"]]
    return len(names) == 3 and set(names) == RGB_NAMES


def _read_embedded(ctx: Context, core: Obj, view: CoreView) -> None:
    value = core.optional("embedded", lambda v: type(v) is dict, "an object")
    if value is None:
        return
    o = Obj(ctx, value, "core.embedded")
    images = o.optional("channel_images", lambda v: type(v) is dict, "an object")
    metadata = o.optional("source_metadata", lambda v: type(v) is dict, "an object")
    o.finish()
    for key, entry in (images or {}).items():
        where = f"core.embedded.channel_images[{key!r}]"
        item, why = _read_image_entry(ctx, key, entry, where)
        if item is None:
            if why:
                _broken(ctx, "channel image entry", where, why)
            continue
        view.channel_images[key] = item
    view.metadata_keys_written = set(metadata or {})
    for key, entry in (metadata or {}).items():
        where = f"core.embedded.source_metadata[{key!r}]"
        item, why = _read_metadata_entry(ctx, key, entry, where)
        if item is None:
            if why:
                _broken(ctx, "source metadata entry", where, why)
            continue
        view.source_metadata[key] = item


def _enum(ctx: Context, o: Obj, key: str, allowed, problems: list[str], where: str):
    """A required enumerated key. Returns the value, or None after
    recording either a newer addition (returns the marker "unknown") or a
    value error."""

    v = o.raw(key)
    if v in allowed:
        return v
    if v is None or key not in o.value:
        problems.append(f"{key!r} is missing")
        return None
    if C.is_str(v) and ctx.unknown_value(f"{where}.{key}", v):
        return "unknown"
    problems.append(f"{key!r} has the value {v!r}")
    return None


def _read_image_entry(ctx: Context, key: str, entry, where: str):
    parts = C.hash_parts(key)
    if parts is None:
        return None, ["its key is not a hash"]
    if parts[0] != "sha256":
        if ctx.unknown_value(where, key):
            return None, []
        return None, ["its key has a hash kind this version does not define"]
    if type(entry) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, entry, where, closed=True)
    problems: list[str] = []
    kind = _enum(ctx, o, "kind", (CHANNEL_IMAGE_KIND,), problems, where)
    media = _enum(ctx, o, "media_type", (CHANNEL_IMAGE_MEDIA,), problems, where)
    dtype = _enum(ctx, o, "dtype", DTYPES, problems, where)
    shape = o.required("shape", C.is_shape, "[height, width]", problems)
    length = o.required("byte_length", lambda v: C.is_int(v) and v >= 0, "a byte count", problems)
    png_hash = o.required("png_hash", C.is_hash, "a hash", problems)
    o.finish()
    if o.unknown_in_closed:
        problems.append("it has keys this closed object does not allow")
    if problems:
        return None, problems
    if "unknown" in (kind, media, dtype):
        return None, []  # a newer kind of entry: not used, already reported
    return {"dtype": dtype, "shape": list(shape), "byte_length": length, "png_hash": png_hash}, []


def _read_metadata_entry(ctx: Context, key: str, entry, where: str):
    parts = C.hash_parts(key)
    if parts is None:
        return None, ["its key is not a hash"]
    if parts[0] != "sha256":
        if ctx.unknown_value(where, key):
            return None, []
        return None, ["its key has a hash kind this version does not define"]
    if type(entry) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, entry, where, closed=True)
    problems: list[str] = []
    fmt = o.required("format", C.is_str, "a string", problems)
    version = o.required("version", lambda v: C.is_int(v) and v >= 1, "an integer of 1 or more", problems)
    length = o.required("byte_length", lambda v: C.is_int(v) and v >= 0, "a byte count", problems)
    payload_hash = o.required("payload_hash", C.is_hash, "a hash", problems)
    o.finish()
    if o.unknown_in_closed:
        problems.append("it has keys this closed object does not allow")
    if problems:
        return None, problems
    formal = fmt in FORMAL_METADATA
    free = fmt.startswith("https://")
    usable = True
    if formal and version != 1:
        usable = False
        if not ctx.unknown_value(f"{where}.version", version):
            return None, [f"'version' {version} is not defined for the format {fmt!r}"]
    elif free:
        usable = False  # free-form contents are their owner's (rule 2 regardless of revision)
        ctx.note(C.Kind.UNKNOWN, "metadata.free_form",
                 f"Source metadata in the free-form format {fmt!r} is kept but not read.", where)
    elif not formal:
        usable = False
        if not ctx.unknown_value(f"{where}.format", fmt):
            return None, [f"'format' {fmt!r} is neither a formal format nor an https URL"]
    return {"format": fmt, "version": version, "byte_length": length, "payload_hash": payload_hash,
            "readable": usable, "entry": entry}, []


# --- blot and marker ------------------------------------------------------------------


def _contrast_ok(v) -> bool:
    return type(v) is list and len(v) == 2 and all(type(x) is int for x in v) and v[0] <= v[1]


def _color_ok(v) -> bool:
    return type(v) is list and len(v) == 3 and all(C.is_number(x) and 0 <= x <= 1 for x in v)


def _read_view_element(ctx: Context, value, where: str, first: bool, count: int, marker: bool):
    if type(value) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, value, where)
    problems: list[str] = []
    out: dict = {}
    if not marker:
        out["id"] = o.required("id", C.is_id, "an ID", problems)
    out["source"] = o.required("source", C.is_id, "an ID", problems)
    out["source_channel"] = o.required("source_channel", lambda v: C.is_str(v) and v != "", "a channel name", problems)
    if first:
        o.forbidden("turn", "the first blot element is the reference channel")
        o.forbidden("flip", "the first blot element is the reference channel")
        out["turn"], out["flip"] = 0, False
    else:
        out["turn"] = o.required("turn", lambda v: C.is_int(v) and v in TURNS, "0, 90, 180 or 270", problems)
        out["flip"] = o.required("flip", C.is_bool, "true or false", problems)
    out["invert"] = o.required("invert", C.is_bool, "true or false", problems)
    out["contrast"] = o.required("contrast", _contrast_ok, "[minimum, maximum] integers with minimum ≤ maximum", problems)
    if not marker:
        if count >= 2:
            out["color"] = o.required("color", _color_ok, "three numbers from 0 to 1", problems)
        else:
            out["color"] = o.optional("color", _color_ok, "three numbers from 0 to 1")
    if out.get("source_channel") == "average":
        out["pixel_hash"] = o.optional("pixel_hash", C.is_hash, "a hash")
    else:
        o.forbidden("pixel_hash", "only an 'average' element records a pixel hash")
    o.finish()
    return (None, problems) if problems else (out, [])


# --- samples, rows, bands, markers ---------------------------------------------------


def _point(lane_raw: dict) -> tuple[bool, bool]:
    """(has a point, has half a point) as written: values of the right type
    count, null and wrong types do not (rule 3)."""

    x_ok = C.is_number(lane_raw.get("x"))
    y_ok = C.is_number(lane_raw.get("y"))
    return x_ok and y_ok, x_ok != y_ok


def _read_lane(ctx: Context, value, where: str, banded: set, condition_ids: set, number_owner: dict):
    if type(value) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, value, where)
    problems: list[str] = []
    lane_id = o.required("id", C.is_id, "an ID", problems)
    for key in ("x", "y", "number", "details", "values"):
        o.seen.add(key)  # handled below: they stand or fall with the point
    if problems:
        o.finish()
        return None, problems
    out: dict = {"id": lane_id}
    has_point, half = _point(value)
    lost = None
    if half:
        lost = "only one of 'x' and 'y' is a number"
    elif has_point:
        number = value.get("number")
        if not (C.is_int(number) and number >= 1):
            lost = "'number' is missing or not an integer of 1 or more"
        elif number_owner.get(number) is not value:
            lost = f"'number' {number} is already used by an earlier lane"
    if lost is not None:
        ctx.broken("lane.point", f"The lane's point was dropped ({lost}), with its number, details and values.", where)
        has_point = False
    elif not has_point:
        stray = [k for k in ("x", "y", "number", "details", "values") if k in value]
        if stray:
            ctx.broken("lane.stray", "A lane without a point may not have " + ", ".join(repr(k) for k in stray)
                       + "; they were ignored.", where)
    if has_point:
        out["number"], out["x"], out["y"] = value["number"], value["x"], value["y"]
        o.seen.discard("details")
        details = o.optional("details", C.is_str, "a string")
        if details is not None:
            out["details"] = details
        o.seen.discard("values")
        raw_values = o.optional("values", lambda v: type(v) is dict, "an object")
        kept = {}
        for cond, text in (raw_values or {}).items():
            w = f"{where}.values[{cond!r}]"
            if cond not in condition_ids:
                ctx.broken("reference.missing", f"The value for condition {cond!r}, which cannot be used, was dropped.", w)
            elif text is None or text == "":
                ctx.broken("key.empty", "An empty value was read as absent.", w)
            elif not C.is_str(text):
                ctx.broken("key.invalid", "A value that is not a string was dropped.", w)
            else:
                kept[cond] = C.strip_text(ctx, text, w)
        if kept:
            out["values"] = kept
    elif lane_id not in banded:
        o.finish()
        return None, ["it has neither a point nor a band"]
    o.finish()
    out["has_point"] = has_point
    return out, []


def _read_row(ctx: Context, value, where: str, blot_count: int | None):
    if type(value) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, value, where)
    problems: list[str] = []
    out = {"id": o.required("id", C.is_id, "an ID", problems), "name": o.optional("name", C.is_str, "a string")}
    if blot_count is None:
        out["channel"] = o.raw("channel") if C.is_id(value.get("channel")) else None
    elif blot_count >= 2:
        out["channel"] = o.required("channel", C.is_id, "an ID", problems)
    else:
        o.forbidden("channel", "the blot has one element")
        out["channel"] = None
    o.finish()
    return (None, problems) if problems else (out, [])


def _read_band(ctx: Context, value, where: str):
    if type(value) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, value, where)
    problems: list[str] = []
    out = {"lane": o.required("lane", C.is_id, "an ID", problems),
           "row": o.required("row", C.is_id, "an ID", problems)}
    shape = _enum(ctx, o, "shape", BAND_SHAPES, problems, where)
    out["shape"] = None if shape == "unknown" else shape
    if shape in BAND_SHAPES:
        out["x"] = o.required("x", C.is_number, "a number", problems)
        out["y"] = o.required("y", C.is_number, "a number", problems)
        if shape == "rect":
            out["w"] = o.required("w", lambda v: C.is_number(v) and v > 0, "a number above 0", problems)
            out["h"] = o.required("h", lambda v: C.is_number(v) and v > 0, "a number above 0", problems)
        else:
            o.forbidden("w", "a point band has no size")
            o.forbidden("h", "a point band has no size")
    else:
        for key in ("x", "y", "w", "h"):
            o.seen.add(key)  # their form depends on the unknown shape (rule 2)
    detection = o.optional("detection", lambda v: type(v) is dict, "an object")
    if detection is not None:
        d = Obj(ctx, detection, f"{where}.detection")
        why: list[str] = []
        by = d.required("by", C.is_str, "a string", why)
        score = d.required("score", C.is_number, "a number", why)
        d.finish()
        if why:
            ctx.broken("key.invalid", "'detection' was ignored: " + "; ".join(why) + ".", f"{where}.detection")
        else:
            out["detection"] = {"by": by, "score": score}
    o.finish()
    return (None, problems) if problems else (out, [])


def _read_marker_label(ctx: Context, value, where: str):
    if type(value) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, value, where)
    problems: list[str] = []
    out = {"id": o.required("id", C.is_id, "an ID", problems),
           "x": o.required("x", C.is_number, "a number", problems),
           "y": o.required("y", C.is_number, "a number", problems),
           "label": o.required("label", C.is_str, "a string", problems)}
    size = o.optional("size", lambda v: type(v) is dict, "an object")
    if size is not None:
        s = Obj(ctx, size, f"{where}.size")
        why: list[str] = []
        number = s.required("value", lambda v: C.is_number(v) and v > 0, "a number above 0", why)
        unit = s.raw("unit")
        newer_unit = False
        if unit not in SIZE_UNITS:
            if not C.is_str(unit):
                why.append("'unit' is missing or not a string")
            elif ctx.unknown_value(f"{where}.size.unit", unit):
                newer_unit = True  # the size is not used (rule 2)
            else:
                why.append(f"'unit' {unit!r} is not defined")
        s.finish()
        if why:
            ctx.broken("key.invalid", "'size' was ignored: " + "; ".join(why) + ".", f"{where}.size")
        elif not newer_unit:
            out["size"] = {"value": number, "unit": unit}
    o.finish()
    return (None, problems) if problems else (out, [])


# --- the core ----------------------------------------------------------------------------


def read_core(ctx: Context, raw: dict) -> CoreView:
    view = CoreView()
    core = Obj(ctx, raw, "core")

    # sources
    sources_raw = core.raw("sources")
    ambiguous_sources: set[str] = set()
    if type(sources_raw) is not list:
        view.unreadable.add("sources")
    else:
        items = []
        for i, item in enumerate(sources_raw):
            source, why = _read_source(ctx, item, f"core.sources[{i}]")
            if source is None:
                _broken(ctx, "source image", f"core.sources[{i}]", why)
            else:
                items.append((i, source))
        items, ambiguous_sources = _dedupe_ids(ctx, items, "source image", "core.sources")
        view.sources = [s for _, s in items]
        if not view.sources:
            view.unreadable.add("sources")
            view.sources = None
    if "sources" in view.unreadable:
        ctx.note(C.Kind.UNREADABLE, "core.unreadable", "The source images cannot be read; the panel's coordinates "
                 "cannot be related to them (rule 4).", "core.sources")

    _read_embedded(ctx, core, view)

    # blot
    blot_raw = core.raw("blot")
    ambiguous_blot: set[str] = set()
    if type(blot_raw) is not list or not blot_raw:
        view.unreadable.add("blot")
    else:
        view.blot_count_as_written = len(blot_raw)
        items = []
        for i, item in enumerate(blot_raw):
            element, why = _read_view_element(ctx, item, f"core.blot[{i}]", i == 0, len(blot_raw), marker=False)
            if element is None:
                _broken(ctx, "blot element", f"core.blot[{i}]", why)
                if i == 0:
                    view.unreadable.add("blot")
            else:
                items.append((i, element))
        if "blot" not in view.unreadable:
            items, ambiguous_blot = _dedupe_ids(ctx, items, "blot element", "core.blot")
            view.blot = [e for _, e in items]

    marker_raw = core.optional("marker", lambda v: type(v) is dict, "an object")
    if marker_raw is not None:
        element, why = _read_view_element(ctx, marker_raw, "core.marker", False, 0, marker=True)
        if element is None:
            _broken(ctx, "marker image", "core.marker", why)
        else:
            view.marker = element

    # orientation
    orientation_raw = core.raw("orientation")
    if type(orientation_raw) is not dict:
        view.unreadable.add("orientation")
    else:
        o = Obj(ctx, orientation_raw, "core.orientation")
        why: list[str] = []
        turn = o.required("turn", lambda v: C.is_int(v) and v in TURNS, "0, 90, 180 or 270", why)
        flip = o.required("flip", C.is_bool, "true or false", why)
        rotation = o.required("rotation", lambda v: C.is_number(v) and -180 <= v < 180, "a number from −180 up to 180", why)
        o.finish()
        if why:
            view.unreadable.add("orientation")
            ctx.broken("core.orientation", "The orientation cannot be read: " + "; ".join(why) + ".", "core.orientation")
        else:
            view.orientation = {"turn": turn, "flip": flip, "rotation": rotation}

    roi_raw = core.optional("roi", lambda v: type(v) is dict, "an object")
    if roi_raw is not None:
        o = Obj(ctx, roi_raw, "core.roi")
        why = []
        roi = {k: o.required(k, C.is_int, "an integer", why) for k in ("x", "y")}
        roi.update({k: o.required(k, lambda v: C.is_int(v) and v >= 1, "an integer of 1 or more", why) for k in ("w", "h")})
        o.finish()
        if why:
            ctx.broken("key.invalid", "'roi' was ignored: " + "; ".join(why) + ".", "core.roi")
        else:
            view.roi = roi

    depth = core.raw("output_bit_depth")
    if depth in (8, 16) and C.is_int(depth):
        view.output_bit_depth = depth
    else:
        view.unreadable.add("output_bit_depth")
        ctx.broken("core.output_bit_depth", "'output_bit_depth' is not 8 or 16.", "core.output_bit_depth")

    # samples
    samples = core.optional("samples", lambda v: type(v) is dict, "an object") or {}
    s = Obj(ctx, samples, "core.samples")
    conditions_raw = s.optional("conditions", lambda v: type(v) is list, "an array") or []
    lanes_raw = s.optional("lanes", lambda v: type(v) is list, "an array") or []
    s.finish()
    cond_items = []
    for i, item in enumerate(conditions_raw):
        if type(item) is not dict:
            _broken(ctx, "condition", f"core.samples.conditions[{i}]", ["not an object"])
            continue
        o = Obj(ctx, item, f"core.samples.conditions[{i}]")
        why = []
        cond = {"id": o.required("id", C.is_id, "an ID", why), "name": o.optional("name", C.is_str, "a string")}
        o.finish()
        if why:
            _broken(ctx, "condition", f"core.samples.conditions[{i}]", why)
        else:
            cond_items.append((i, cond))
    cond_items, ambiguous_conditions = _dedupe_ids(ctx, cond_items, "condition", "core.samples.conditions")
    view.conditions = [c for _, c in cond_items]
    condition_ids = {c["id"] for c in view.conditions} - ambiguous_conditions

    bands_raw = core.optional("bands", lambda v: type(v) is list, "an array") or []
    banded = {b.get("lane") for b in bands_raw if type(b) is dict and C.is_id(b.get("lane"))}
    number_owner: dict[int, dict] = {}
    for item in lanes_raw:
        if type(item) is dict and _point(item)[0] and C.is_int(item.get("number")) and item["number"] >= 1:
            number_owner.setdefault(item["number"], item)
    lane_items = []
    for i, item in enumerate(lanes_raw):
        lane, why = _read_lane(ctx, item, f"core.samples.lanes[{i}]", banded, condition_ids, number_owner)
        if lane is None:
            _broken(ctx, "lane", f"core.samples.lanes[{i}]", why)
        else:
            lane_items.append((i, lane))
    lane_items, ambiguous_lanes = _dedupe_ids(ctx, lane_items, "lane", "core.samples.lanes")
    view.lanes = [lane for _, lane in lane_items]

    rows_raw = core.optional("rows", lambda v: type(v) is list, "an array") or []
    blot_count = None if "blot" in view.unreadable else view.blot_count_as_written
    row_items = []
    for i, item in enumerate(rows_raw):
        row, why = _read_row(ctx, item, f"core.rows[{i}]", blot_count)
        if row is None:
            _broken(ctx, "row", f"core.rows[{i}]", why)
        else:
            row_items.append((i, row))
    row_items, ambiguous_rows = _dedupe_ids(ctx, row_items, "row", "core.rows")

    band_items = []
    cells: set[tuple] = set()
    for i, item in enumerate(bands_raw):
        band, why = _read_band(ctx, item, f"core.bands[{i}]")
        if band is None:
            _broken(ctx, "band", f"core.bands[{i}]", why)
            continue
        cell = (band["lane"], band["row"])
        if cell in cells:
            ctx.broken("band.cell", "A later band in the same cell (lane and row) was dropped.", f"core.bands[{i}]")
            continue
        cells.add(cell)
        band_items.append((i, band))

    markers_raw = core.optional("markers", lambda v: type(v) is list, "an array") or []
    marker_items = []
    for i, item in enumerate(markers_raw):
        label, why = _read_marker_label(ctx, item, f"core.markers[{i}]")
        if label is None:
            _broken(ctx, "marker label", f"core.markers[{i}]", why)
        else:
            marker_items.append((i, label))
    marker_items, _ = _dedupe_ids(ctx, marker_items, "marker label", "core.markers")
    view.markers = [m for _, m in marker_items]

    view.annotation = core.optional("annotation", C.is_str, "a string")
    view.note = core.optional("note", C.is_str, "a string")
    core.finish()

    # references: drop what points to something dropped, until nothing changes
    _resolve(ctx, view, ambiguous_sources, ambiguous_blot, ambiguous_lanes, ambiguous_rows,
             row_items, band_items)

    # output_bit_depth needs a readable blot and its sources
    if view.output_bit_depth == 16 and view.blot and view.sources is not None:
        dtypes = []
        for element in view.blot:
            source = view.source(element["source"])
            dtypes.append(source.get("dtype") if source else None)
        if dtypes and all(d == "uint8" for d in dtypes):
            view.output_bit_depth = None
            view.unreadable.add("output_bit_depth")
            ctx.broken("core.output_bit_depth", "'output_bit_depth' is 16 although every blot source is uint8.",
                       "core.output_bit_depth")
    for key in ("blot", "orientation"):
        if key in view.unreadable:
            ctx.note(C.Kind.UNREADABLE, "core.unreadable", f"The core key {key!r} cannot be read (rule 4).", f"core.{key}")
    if "output_bit_depth" in view.unreadable:
        ctx.note(C.Kind.UNREADABLE, "core.unreadable", "The rendered image cannot be drawn again without "
                 "'output_bit_depth' (rule 4).", "core.output_bit_depth")
    return view


def _resolve(ctx, view, ambiguous_sources, ambiguous_blot, ambiguous_lanes, ambiguous_rows, row_items, band_items):
    sources_readable = view.sources is not None
    rows = list(row_items)
    bands = list(band_items)

    def source_ok(element, where):
        if not sources_readable:
            return True
        if element["source"] in ambiguous_sources:
            ctx.broken("reference.ambiguous", "It points to a source image ID used twice; it was dropped.", where)
            return False
        source = view.source(element["source"])
        if source is None:
            ctx.broken("reference.missing", f"Its source image {element['source']!r} does not exist; it was dropped.", where)
            return False
        name = element["source_channel"]
        if name in source.get("ambiguous_channels", ()):
            ctx.broken("reference.ambiguous", f"It chooses the channel name {name!r}, used twice; it was dropped.", where)
            return False
        names = {c["name"] for c in source["channels"]}
        if name == "average":
            if not _is_rgb(source):
                ctx.broken("reference.average", "'average' can only be chosen in an RGB source image; it was dropped.", where)
                return False
        elif name not in names:
            ctx.broken("reference.missing", f"Its channel {name!r} does not exist in the source image; it was dropped.", where)
            return False
        return True

    if view.blot is not None:
        kept = []
        for i, element in enumerate(view.blot):
            if source_ok(element, f"core.blot[{i}]"):
                kept.append(element)
            elif i == 0:
                view.blot = None
                view.unreadable.add("blot")
                break
        else:
            view.blot = kept
    if view.marker is not None and not source_ok(view.marker, "core.marker"):
        view.marker = None

    blot_ids = None if view.blot is None else ({e["id"] for e in view.blot} - ambiguous_blot)
    kept_rows = []
    for i, row in rows:
        if blot_ids is not None and row["channel"] is not None:
            if row["channel"] in ambiguous_blot or row["channel"] not in blot_ids:
                ctx.broken("reference.missing", f"Its channel {row['channel']!r} is not a blot element that can be used; "
                           "it was dropped.", f"core.rows[{i}]")
                continue
        kept_rows.append(row)
    view.rows = kept_rows

    lane_ids = {lane["id"] for lane in view.lanes} - ambiguous_lanes
    row_ids = {row["id"] for row in view.rows} - ambiguous_rows
    kept_bands = []
    for i, band in bands:
        if band["lane"] not in lane_ids:
            ctx.broken("reference.missing", f"Its lane {band['lane']!r} cannot be used; the band was dropped.", f"core.bands[{i}]")
            continue
        if band["row"] not in row_ids:
            ctx.broken("reference.missing", f"Its row {band['row']!r} cannot be used; the band was dropped.", f"core.bands[{i}]")
            continue
        kept_bands.append(band)
    view.bands = kept_bands
    view.ambiguous = {"lanes": ambiguous_lanes, "rows": ambiguous_rows, "blot": ambiguous_blot}

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
BAND_SHAPES = ("rect", "point")
ROLES = ("marker",)
BLENDS = ("add",)
ANNOTATION_SIDES = ("left", "right", "top", "bottom")
CHANNEL_IMAGE_KIND = "decoded_scalar_plane"
CHANNEL_IMAGE_MEDIA = "image/png"
FORMAL_METADATA = ("png", "jpeg", "tiff", "scn")


@dataclass
class CoreView:
    sources: list[dict] | None = None
    source_channel_images: dict[str, dict] = field(default_factory=dict)
    source_metadata: dict[str, dict] = field(default_factory=dict)
    channels: list[dict] | None = None
    """The panel channels (§5.6), or None when `channels` cannot be read (rule 4)."""
    reference_channel: str | None = None
    """The reference channel's ID (§5.7), or None when it cannot be read (rule 4)."""
    blend: str | None = None
    """`blend` as written (§5.6); None when absent (`add`) or of a newer value."""
    blend_unknown: bool = False
    """`blend` has a value of a newer revision: the rendered image cannot be drawn again (rule 2)."""
    orientation: dict | None = None
    roi: dict | None = None
    output_bit_depth: int | None = None
    conditions: list[dict] = field(default_factory=list)
    lanes: list[dict] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    bands: list[dict] = field(default_factory=list)
    markers: list[dict] = field(default_factory=list)
    marker_unit: str | None = None
    """The unit of the marker labels' numbers, as written (§5.14)."""
    annotations: dict[str, str] = field(default_factory=dict)
    """The text of each side that has one (§5.15), in ANNOTATION_SIDES order."""
    note: str | None = None
    unreadable: set[str] = field(default_factory=set)
    """Required core keys that could not be read (rule 4)."""
    drawn_count_as_written: int = 0
    metadata_keys_written: set[str] = field(default_factory=set)
    """Keys of `embedded.source_metadata` as written, valid or not."""
    pointed_to_written: set[str] = field(default_factory=set)
    """Catalog keys that source channels and source images point to as written."""
    payloads_written: dict[int, set[str]] = field(default_factory=lambda: {1: set(), 2: set()})
    """Payload hashes that catalog entries use as written, by codec. With
    `pointed_to_written`, what decides whether an embedding is referred to:
    as written, as in rule 3, so one whose referrer was dropped or not used
    is not reported again (§5.3, §5.4)."""
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

    def channel(self, channel_id: str) -> dict | None:
        for element in self.channels or ():
            if element["id"] == channel_id:
                return element
        return None

    def channel_ids(self) -> set[str]:
        return {e["id"] for e in self.channels or ()}

    def reference(self) -> dict | None:
        """The reference channel, while `reference_channel` can be read."""

        return None if self.reference_channel is None else self.channel(self.reference_channel)

    def drawn(self) -> list[dict]:
        """The panel channels drawn into the rendered image, in their order (§6.2)."""

        return [e for e in self.channels or () if e["drawn"]]

    def marker_channels(self) -> list[dict]:
        return [e for e in self.channels or () if e["role"] == "marker"]

    def band_cells(self) -> set[tuple[str, str]]:
        return {(b["lane"], b["row"]) for b in self.bands}


def _broken(ctx: Context, what: str, where: str, problems: list[str]) -> None:
    ctx.broken("element.broken", f"The {what} was dropped: " + "; ".join(problems) + ".", where)


def _array(ctx: Context, core: Obj, key: str) -> list:
    value = core.optional(key, lambda v: type(v) is list, "an array")
    return value or []


def _ids_as_written(raw: list) -> list:
    """Each element's ID as written, by place (None where none can be read)."""

    return [item.get("id") if type(item) is dict and C.is_id(item.get("id")) else None for item in raw]


def _firsts(keys: list) -> tuple[dict, set]:
    """Where each key is first written, and the keys written more than once."""

    first, twice = {}, set()
    for index, key in enumerate(keys):
        if key is None:
            continue
        if key in first:
            twice.add(key)
        else:
            first[key] = index
    return first, twice


def _dedupe_ids(ctx: Context, items: list[tuple[int, dict]], what: str, where: str,
                written: list) -> tuple[list[tuple[int, dict]], set[str]]:
    """Keep the element first written with each ID; drop the later ones and
    make the ID unusable as a reference target (rule 3). Overlaps are judged
    on the IDs as written (`written`, the array as written), before anything
    is dropped: a later element is dropped even when the earlier one is
    broken, and an ID written twice stays unusable."""

    first, ambiguous = _firsts(_ids_as_written(written))
    kept = []
    for index, item in items:
        ident = item["id"]
        if first.get(ident, index) != index:
            ctx.broken("id.duplicate", f"A later {what} with the ID {ident!r} was dropped, "
                       "and so is everything pointing to that ID.", f"{where}[{index}]")
            continue
        kept.append((index, item))
    return kept, ambiguous


# --- sources and embedded ----------------------------------------------------------


def _read_source_channel(ctx: Context, value, where: str):
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
    for key in ("file_hash", "source_metadata"):
        if out[key] is not None and ctx.hash_error(f"{where}.{key}", out[key]):
            ctx.broken("key.invalid", f"{key!r} uses a hash kind this version does not define and was ignored.",
                       f"{where}.{key}")
            out[key] = None

    channels_raw = o.raw("source_channels")
    channels: list[dict] = []
    if type(channels_raw) is not list or not channels_raw:
        problems.append("'source_channels' is missing or empty")
    else:
        # Names written twice are judged as written, as IDs are (rule 3).
        first, out["ambiguous_channels"] = _firsts([
            item.get("name") if type(item) is dict and C.is_str(item.get("name")) else None for item in channels_raw])
        for i, item in enumerate(channels_raw):
            channel, why = _read_source_channel(ctx, item, f"{where}.source_channels[{i}]")
            if channel is None:
                _broken(ctx, "source channel", f"{where}.source_channels[{i}]", why)
                continue
            if first.get(channel["name"], i) != i:
                ctx.broken("channel.duplicate", f"A later source channel named {channel['name']!r} was dropped, and "
                           "so is every panel channel choosing that name.", f"{where}.source_channels[{i}]")
                continue
            channels.append(channel)
        if not channels:
            problems.append("no source channel can be read")
    out["source_channels"] = channels

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
        problems.append("'dtype' is missing although a source channel has a pixel hash")
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
        problems.append("'shape' is missing although a source channel has a pixel hash")
    out["dtype"], out["shape"] = dtype, shape
    o.finish()
    if problems:
        return None, problems
    return out, []


def _is_rgb(source: dict) -> bool:
    names = [c["name"] for c in source["source_channels"]]
    return len(names) == 3 and set(names) == RGB_NAMES


def _read_annotations(ctx: Context, core: Obj) -> dict[str, str]:
    value = core.optional("annotations", lambda v: type(v) is dict, "an object")
    if value is None:
        return {}
    o = Obj(ctx, value, "core.annotations")
    sides = {}
    for side in ANNOTATION_SIDES:
        text = o.optional(side, C.is_str, "a string")
        if text is not None:
            sides[side] = text
    o.finish()
    return sides


def _read_embedded(ctx: Context, core: Obj, view: CoreView) -> None:
    value = core.optional("embedded", lambda v: type(v) is dict, "an object")
    if value is None:
        return
    o = Obj(ctx, value, "core.embedded")
    images = o.optional("source_channel_images", lambda v: type(v) is dict, "an object")
    metadata = o.optional("source_metadata", lambda v: type(v) is dict, "an object")
    o.finish()
    for codec, entries, key in ((1, images, "png_hash"), (2, metadata, "payload_hash")):
        for entry in (entries or {}).values():
            if type(entry) is dict and C.is_hash(entry.get(key)):
                view.payloads_written[codec].add(entry[key])
    for key, entry in (images or {}).items():
        where = f"core.embedded.source_channel_images[{key!r}]"
        item, why = _read_image_entry(ctx, key, entry, where)
        if item is None:
            if why:
                _broken(ctx, "source channel image entry", where, why)
            continue
        view.source_channel_images[key] = item
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
    # An unknown format or version is rule 2 whatever the revision (§5.4):
    # writers carry unread metadata over, also into panels of an older
    # revision. A free-form format is no unknown format: kept unread, and
    # not reported (§5.4).
    usable = fmt in FORMAL_METADATA and version == 1
    if not usable and not fmt.startswith("https://"):
        ctx.note(C.Kind.UNKNOWN, "metadata.unknown",
                 f"Source metadata in a format or version this reader does not know ({fmt!r}, version {version}) "
                 "is kept but not read.", where)
    return {"format": fmt, "version": version, "byte_length": length, "payload_hash": payload_hash,
            "readable": usable, "entry": entry}, []


# --- panel channels ---------------------------------------------------------------------


def _contrast_ok(v) -> bool:
    return type(v) is list and len(v) == 2 and all(type(x) is int for x in v) and v[0] <= v[1]


def _color_ok(v) -> bool:
    return type(v) is list and len(v) == 3 and all(C.is_number(x) and 0 <= x <= 1 for x in v)


def _read_panel_channel(ctx: Context, value, where: str, reference: bool | None, drawn_count: int):
    """One panel channel (§5.6). `reference` says whether it is the
    reference channel, or is None when `reference_channel` cannot be read:
    then whether `turn` and `flip` are needed is not checked (rule 4).
    `drawn_count` is how many panel channels are drawn, as written."""

    if type(value) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, value, where)
    problems: list[str] = []
    out: dict = {"id": o.required("id", C.is_id, "an ID", problems), "name": o.optional("name", C.is_str, "a string")}
    role = o.optional("role", C.is_str, "a string")
    if role is not None and role not in ROLES and not ctx.unknown_value(f"{where}.role", role):
        ctx.broken("key.invalid", f"'role' has the value {role!r} and was ignored.", f"{where}.role")
        role = None
    out["role"] = role  # a newer role is kept: neither an ordinary nor a marker channel (rule 2)
    out["drawn"] = o.required("drawn", C.is_bool, "true or false", problems)
    out["source"] = o.required("source", C.is_id, "an ID", problems)
    out["source_channel"] = o.required("source_channel", lambda v: C.is_str(v) and v != "", "a source channel name",
                                       problems)
    if reference:
        o.forbidden("turn", "it is the reference channel")
        o.forbidden("flip", "it is the reference channel")
        out["turn"], out["flip"] = 0, False
    elif reference is None:
        out["turn"] = o.optional("turn", lambda v: C.is_int(v) and v in TURNS, "0, 90, 180 or 270")
        out["flip"] = o.optional("flip", C.is_bool, "true or false")
    else:
        out["turn"] = o.required("turn", lambda v: C.is_int(v) and v in TURNS, "0, 90, 180 or 270", problems)
        out["flip"] = o.required("flip", C.is_bool, "true or false", problems)
    out["invert"] = o.required("invert", C.is_bool, "true or false", problems)
    out["contrast"] = o.required("contrast", _contrast_ok, "[minimum, maximum] integers with minimum ≤ maximum", problems)
    if out["drawn"] is True and drawn_count >= 2:
        out["color"] = o.required("color", _color_ok, "three numbers from 0 to 1", problems)
    else:
        out["color"] = o.optional("color", _color_ok, "three numbers from 0 to 1")
    o.finish()
    return (None, problems) if problems else (out, [])


def _read_channels(ctx: Context, core: Obj, view: CoreView) -> tuple[list[tuple[int, dict]], set[str]]:
    """`channels`, `reference_channel` and `blend` (§5.6, §5.7)."""

    channels_raw = core.raw("channels")
    reference_raw = core.raw("reference_channel")
    if type(channels_raw) is not list or not channels_raw:
        view.unreadable.add("channels")
        channels_raw = []
    written_ids = [item.get("id") for item in channels_raw if type(item) is dict]
    # Which element is the reference channel is known only from an ID
    # written once among the elements; otherwise it cannot be read (rule 4).
    reference_id = reference_raw if C.is_id(reference_raw) and written_ids.count(reference_raw) == 1 else None
    if reference_id is None:
        view.unreadable.add("reference_channel")
    drawn = sum(1 for item in channels_raw if type(item) is dict and item.get("drawn") is True)
    view.drawn_count_as_written = drawn
    items = []
    for i, item in enumerate(channels_raw):
        reference = None if reference_id is None else (type(item) is dict and item.get("id") == reference_id)
        element, why = _read_panel_channel(ctx, item, f"core.channels[{i}]", reference, drawn)
        if element is None:
            _broken(ctx, "panel channel", f"core.channels[{i}]", why)
            if reference:
                view.unreadable.add("reference_channel")
        else:
            items.append((i, element))
    items, ambiguous = _dedupe_ids(ctx, items, "panel channel", "core.channels", channels_raw)
    if channels_raw and not drawn:
        ctx.broken("channels.none_drawn", "No panel channel is drawn ('drawn' is true in none of them); the rendered "
                   "image cannot be drawn again.", "core.channels")
    if "channels" not in view.unreadable:
        view.channels = [e for _, e in items]
        if not view.channels:
            view.channels = None
            view.unreadable.add("channels")
    if "reference_channel" not in view.unreadable:
        view.reference_channel = reference_id

    blend = core.optional("blend", C.is_str, "a string")
    if blend in BLENDS:
        view.blend = blend
    elif blend is not None:
        if ctx.unknown_value("core.blend", blend):
            view.blend_unknown = True
        else:
            ctx.broken("key.invalid", f"'blend' has the value {blend!r} and was ignored.", "core.blend")
    return items, ambiguous


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


def _read_row(ctx: Context, value, where: str):
    if type(value) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, value, where)
    problems: list[str] = []
    out = {"id": o.required("id", C.is_id, "an ID", problems), "name": o.optional("name", C.is_str, "a string"),
           "channel": o.required("channel", C.is_id, "an ID", problems)}
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
        for item in sources_raw:
            if type(item) is dict:
                channels = item.get("source_channels") if type(item.get("source_channels")) is list else []
                pointers = [c.get("pixel_hash") for c in channels if type(c) is dict]
                view.pointed_to_written.update(p for p in [*pointers, item.get("source_metadata")] if C.is_hash(p))
        items = []
        for i, item in enumerate(sources_raw):
            source, why = _read_source(ctx, item, f"core.sources[{i}]")
            if source is None:
                _broken(ctx, "source image", f"core.sources[{i}]", why)
            else:
                items.append((i, source))
        items, ambiguous_sources = _dedupe_ids(ctx, items, "source image", "core.sources", sources_raw)
        view.sources = [s for _, s in items]
        if not view.sources:
            view.unreadable.add("sources")
            view.sources = None
    if "sources" in view.unreadable:
        ctx.note(C.Kind.UNREADABLE, "core.unreadable", "The source images cannot be read; the panel's coordinates "
                 "cannot be related to them (rule 4).", "core.sources")

    _read_embedded(ctx, core, view)

    channel_items, ambiguous_channels = _read_channels(ctx, core, view)

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
    cond_items, ambiguous_conditions = _dedupe_ids(ctx, cond_items, "condition", "core.samples.conditions",
                                                   conditions_raw)
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
    lane_items, ambiguous_lanes = _dedupe_ids(ctx, lane_items, "lane", "core.samples.lanes", lanes_raw)
    view.lanes = [lane for _, lane in lane_items]

    rows_raw = core.optional("rows", lambda v: type(v) is list, "an array") or []
    row_items = []
    for i, item in enumerate(rows_raw):
        row, why = _read_row(ctx, item, f"core.rows[{i}]")
        if row is None:
            _broken(ctx, "row", f"core.rows[{i}]", why)
        else:
            row_items.append((i, row))
    row_items, ambiguous_rows = _dedupe_ids(ctx, row_items, "row", "core.rows", rows_raw)

    band_items = []
    # The cells as written: a later band in a cell is dropped even when the earlier one is broken (rule 3).
    first_in_cell, _ = _firsts([
        (item.get("lane"), item.get("row")) if type(item) is dict and C.is_id(item.get("lane"))
        and C.is_id(item.get("row")) else None for item in bands_raw])
    for i, item in enumerate(bands_raw):
        band, why = _read_band(ctx, item, f"core.bands[{i}]")
        if band is None:
            _broken(ctx, "band", f"core.bands[{i}]", why)
            continue
        if first_in_cell.get((band["lane"], band["row"]), i) != i:
            ctx.broken("band.cell", "A later band in the same cell (lane and row) was dropped.", f"core.bands[{i}]")
            continue
        band_items.append((i, band))

    markers_raw = core.optional("markers", lambda v: type(v) is list, "an array") or []
    marker_items = []
    for i, item in enumerate(markers_raw):
        label, why = _read_marker_label(ctx, item, f"core.markers[{i}]")
        if label is None:
            _broken(ctx, "marker label", f"core.markers[{i}]", why)
        else:
            marker_items.append((i, label))
    marker_items, _ = _dedupe_ids(ctx, marker_items, "marker label", "core.markers", markers_raw)
    view.markers = [m for _, m in marker_items]
    view.marker_unit = core.optional("marker_unit", C.is_str, "a string")

    view.annotations = _read_annotations(ctx, core)
    view.note = core.optional("note", C.is_str, "a string")
    core.finish()

    # references: drop what points to something dropped, until nothing changes
    _resolve(ctx, view, ambiguous_sources, ambiguous_channels, ambiguous_lanes, ambiguous_rows,
             channel_items, row_items, band_items)

    # output_bit_depth (§5.10), while the panel channels and sources can be
    # read (rule 4): judged on them as written, before anything was dropped (rule 3)
    if view.output_bit_depth == 16 and view.channels is not None and view.sources is not None:
        dtypes_of: dict[str, set] = {}
        for item in raw.get("sources") or ():
            if type(item) is dict and C.is_id(item.get("id")):
                dtypes_of.setdefault(item["id"], set()).add(item.get("dtype"))
        dtypes = [dtypes_of.get(item.get("source"), {None}) for item in raw.get("channels") or ()
                  if type(item) is dict and item.get("drawn") is True]
        if dtypes and all(d == {"uint8"} for d in dtypes):
            view.output_bit_depth = None
            view.unreadable.add("output_bit_depth")
            ctx.broken("core.output_bit_depth", "'output_bit_depth' is 16 although the source image of every drawn "
                       "panel channel is uint8.", "core.output_bit_depth")
    for key in ("channels", "reference_channel", "orientation"):
        if key in view.unreadable:
            ctx.note(C.Kind.UNREADABLE, "core.unreadable", f"The core key {key!r} cannot be read (rule 4).", f"core.{key}")
    if "output_bit_depth" in view.unreadable:
        ctx.note(C.Kind.UNREADABLE, "core.unreadable", "The rendered image cannot be drawn again without "
                 "'output_bit_depth' (rule 4).", "core.output_bit_depth")
    return view


def _resolve(ctx, view, ambiguous_sources, ambiguous_channels, ambiguous_lanes, ambiguous_rows,
             channel_items, row_items, band_items):
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
            ctx.broken("reference.ambiguous", f"It chooses the source channel name {name!r}, used twice; it was dropped.",
                       where)
            return False
        names = {c["name"] for c in source["source_channels"]}
        if name == "average":
            if not _is_rgb(source):
                ctx.broken("reference.average", "'average' can only be chosen in an RGB source image; it was dropped.", where)
                return False
        elif name not in names:
            ctx.broken("reference.missing", f"Its source channel {name!r} does not exist in the source image; "
                       "it was dropped.", where)
            return False
        return True

    if view.channels is not None:
        kept = []
        for i, element in channel_items:
            if source_ok(element, f"core.channels[{i}]"):
                kept.append(element)
            elif element["id"] == view.reference_channel:
                view.reference_channel = None
                view.unreadable.add("reference_channel")
        view.channels = kept or None
        if not kept:
            view.unreadable.add("channels")

    channel_ids = None if view.channels is None else (view.channel_ids() - ambiguous_channels)
    kept_rows = []
    for i, row in rows:
        if channel_ids is not None:
            if row["channel"] not in channel_ids:
                ctx.broken("reference.missing", f"Its panel channel {row['channel']!r} cannot be used; it was dropped.",
                           f"core.rows[{i}]")
                continue
            if view.channel(row["channel"])["role"] == "marker":
                ctx.broken("reference.marker", f"It points to the marker channel {row['channel']!r}, which a row may "
                           "not do (§5.6); it was read as written.", f"core.rows[{i}]")
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
    view.ambiguous = {"lanes": ambiguous_lanes, "rows": ambiguous_rows, "channels": ambiguous_channels}

"""Reading one panel: the JSON in its carrier PNG, by §2.6 rules 1–5, with
the checks of §5.3, §5.4, §6 and §7.4."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re

import numpy as np

from . import _check as C, _embedded, _geometry, _hashes, _json, _png
from ._carrier import Carrier
from ._check import Context, Obj
from ._core import CoreView, read_core
from ._quant import read_quantifications
from ._svgxml import Element
from .notices import Kind, Notice

FORMAT_URL = "https://mybioapps.com/gelsvg/panel"
SVG_HASH_METHODS = ("c14n2_without_panel_data",)
_RFC3339 = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})")


@dataclass
class Panel:
    index: int
    element: Element
    carrier: Carrier
    notices: list[Notice] = field(default_factory=list)
    raw: dict | None = None
    """The panel's JSON as written."""
    readable: bool = False
    unreadable_reason: str | None = None
    newer_version: bool = False
    schema_version: int | None = None
    schema_revision: int | None = None
    newer_revision: bool = False
    saved_at: str | None = None
    written_by: dict | None = None
    core: CoreView | None = None
    quantifications: list[dict] = field(default_factory=list)
    figure: dict = field(default_factory=dict)
    assembly: dict | None = None
    extensions: dict = field(default_factory=dict)
    channel_images: dict[str, np.ndarray] = field(default_factory=dict)
    """Decoded channel images that passed every check, by pixel hash."""
    rendered: np.ndarray | None = None
    core_hash: str | None = None
    quant_hash: str | None = None
    checks: dict[str, bool | None] = field(default_factory=dict)
    """Results of the comparisons: True (matched), False (differs), None
    (could not be checked)."""

    def to_json(self) -> dict:
        """The panel as read, as JSON to change and write again
        (writer.Content): what reading ignored or dropped is left out, and
        so is `embedded`, which the writer rebuilds."""

        from ._tojson import panel_json
        return panel_json(self)


def _rfc3339(value) -> bool:
    if not C.is_str(value) or _RFC3339.fullmatch(value) is None:
        return False
    text = value.replace("Z", "+00:00").replace("z", "+00:00")
    text = text[:17] + ("59" if text[17:19] == "60" else text[17:19]) + text[19:]  # leap second
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def _check(ctx: Context, panel: Panel, name: str, recorded, actual: str | None, what: str) -> None:
    """Compare a recorded hash with the one computed now and report it."""

    parts = C.hash_parts(recorded) if recorded is not None else None
    if recorded is None or actual is None or parts is None or parts[0] != "sha256":
        panel.checks[name] = None
        ctx.note(Kind.CHECK, f"check.{name}.unknown", f"Whether {what} could not be checked.")
        return
    panel.checks[name] = recorded == actual
    if recorded == actual:
        ctx.note(Kind.CHECK, f"check.{name}.matched", f"Checked whether {what}: it has not.")
    else:
        ctx.note(Kind.CHECK, f"check.{name}.differs", f"Checked whether {what}: it has.")


def read_panel(index: int, element: Element, carrier: Carrier, *, decoded: dict | None = None) -> Panel:
    panel = Panel(index=index, element=element, carrier=carrier, notices=list(carrier.notices))

    def unreadable(reason: str) -> Panel:
        panel.unreadable_reason = reason
        panel.notices.append(Notice(Kind.UNREADABLE, "panel.unreadable", f"The panel cannot be read: {reason}.", panel=index))
        return panel

    if carrier.unreadable:
        return unreadable(carrier.unreadable)
    try:
        raw = _json.parse(carrier.json_text)
    except _json.JsonError as exc:
        return unreadable(str(exc))
    panel.raw = raw
    if raw.get("format") != FORMAT_URL:
        return unreadable("'format' is missing or names another format")
    version = raw.get("schema_version")
    if not C.is_int(version) or version < 1:
        return unreadable("'schema_version' is missing or not a version number")
    panel.schema_version = version
    if version > C.KNOWN_VERSION:
        panel.newer_version = True
        panel.notices.append(Notice(
            Kind.UNKNOWN, "panel.newer_version",
            f"The panel is written in a newer version of the format (schema {version}); it was not read. "
            "Saving keeps it byte for byte.", panel=index))
        return panel
    if type(raw.get("core")) is not dict:
        return unreadable("'core' is missing or not an object")

    revision = raw.get("schema_revision")
    ctx = Context(panel=index, notices=panel.notices)
    if C.is_int(revision) and revision >= 0:
        panel.schema_revision = revision
        ctx.newer_revision = panel.newer_revision = revision > C.KNOWN_REVISION
    else:
        ctx.newer_revision = panel.newer_revision = True
        ctx.broken("top.schema_revision", "'schema_revision' is missing or not an integer; the panel is read as if "
                   "from a newer revision.", "schema_revision")
    panel.readable = True

    top = Obj(ctx, raw, "panel")
    for key in ("format", "schema_version", "schema_revision", "core"):
        top.seen.add(key)
    saved_at = top.raw("saved_at")
    if _rfc3339(saved_at):
        panel.saved_at = saved_at
    else:
        ctx.broken("top.saved_at", "'saved_at' is missing or not an RFC 3339 date and time with an offset; "
                   "when the panel was saved is unknown.", "saved_at")
    written_by = top.raw("written_by")
    if type(written_by) is dict:
        w = Obj(ctx, written_by, "written_by")
        why: list[str] = []
        record = {"name": w.required("name", C.is_str, "a string", why),
                  "version": w.required("version", C.is_str, "a string", why),
                  "build": w.optional("build", C.is_str, "a string")}
        w.finish()
        if not why:
            panel.written_by = record
    if panel.written_by is None:
        ctx.broken("top.written_by", "'written_by' is missing or cannot be read; the tool that wrote the panel is unknown.",
                   "written_by")

    view = read_core(ctx, raw["core"])
    panel.core = view
    panel.channel_images = _embedded.check_embedded(ctx, view, carrier, decoded)
    panel.core_hash = _hashes.core_hash(raw["core"])
    panel.quant_hash = _hashes.quant_core_hash(raw["core"])

    _read_figure(ctx, panel, top.raw("figure"))

    quant = top.optional("quantifications", lambda v: type(v) is list, "an array")
    panel.quantifications = read_quantifications(ctx, quant or [], view)
    for q in panel.quantifications:
        name = f"computed_on[{q['index']}]"
        _check(ctx, panel, name, q["computed_on"], panel.quant_hash, f"the quantification {q['index']} is out of date")
        q["stale"] = None if panel.checks[name] is None else not panel.checks[name]

    _read_assembly(ctx, panel, top)
    _read_extensions(ctx, panel, top)
    top.finish()
    return panel


def _read_figure(ctx: Context, panel: Panel, figure) -> None:
    keys = ("crop", "rendered_hash", "drawn_from", "svg_hash", "svg_hash_method")
    panel.figure = dict.fromkeys(keys)
    if type(figure) is not dict:
        ctx.broken("figure.unreadable", "'figure' is missing or not an object; nothing it records can be checked.", "figure")
        _check(ctx, panel, "drawn_from", None, None, "the data changed after the figure was drawn")
        _check(ctx, panel, "rendered_hash", None, None, "the rendered image changed after it was drawn")
        return
    o = Obj(ctx, figure, "figure")
    crop = o.raw("crop")
    if type(crop) is dict:
        c = Obj(ctx, crop, "figure.crop")
        why: list[str] = []
        rect = {k: c.required(k, C.is_int, "an integer", why) for k in ("x", "y")}
        rect.update({k: c.required(k, lambda v: C.is_int(v) and v >= 1, "an integer of 1 or more", why) for k in ("w", "h")})
        c.finish()
        if why:
            ctx.broken("figure.crop", "'crop' cannot be read (" + "; ".join(why) + "); the rendered image is not related "
                       "to the panel's coordinates.", "figure.crop")
        else:
            panel.figure["crop"] = rect
    else:
        ctx.broken("figure.crop", "'crop' is missing; the rendered image is not related to the panel's coordinates.", "figure.crop")
    for key in ("rendered_hash", "drawn_from"):
        value = o.raw(key)
        if C.is_hash(value):
            panel.figure[key] = value  # of a kind not known, it is checked as unknown
            if ctx.hash_error(f"figure.{key}", value):
                ctx.broken(f"figure.{key}", f"'{key}' uses a hash kind this version does not define; what it "
                           "records cannot be checked.", f"figure.{key}")
        else:
            ctx.broken(f"figure.{key}", f"'{key}' is missing or not a hash.", f"figure.{key}")
    svg_hash = o.optional("svg_hash", C.is_hash, "a hash")
    if svg_hash is not None:
        if ctx.hash_error("figure.svg_hash", svg_hash):
            ctx.broken("figure.svg_hash", "'svg_hash' uses a hash kind this version does not define; whether the "
                       "SVG changed cannot be checked.", "figure.svg_hash")
        method = o.raw("svg_hash_method")
        if method in SVG_HASH_METHODS:
            panel.figure["svg_hash"], panel.figure["svg_hash_method"] = svg_hash, method
        elif C.is_str(method) and ctx.unknown_value("figure.svg_hash_method", method):
            pass
        else:
            ctx.broken("figure.svg_hash_method", "'svg_hash_method' is missing or not defined; 'svg_hash' cannot be checked.",
                       "figure.svg_hash_method")
    else:
        o.seen.add("svg_hash_method")
    o.finish()

    _check(ctx, panel, "drawn_from", panel.figure["drawn_from"], panel.core_hash,
           "the data changed after the figure was drawn")

    carrier = panel.carrier
    ihdr = carrier.ihdr
    if carrier.rendered_usable and carrier.pixel_data is not None and ihdr is not None:
        if ihdr.color_type in (0, 2) and ihdr.bit_depth in (8, 16):
            panel.rendered = _png.decode_pixels(ihdr, carrier.pixel_data)
            _check(ctx, panel, "rendered_hash", panel.figure["rendered_hash"], _hashes.rendered_hash(panel.rendered),
                   "the rendered image changed after it was drawn")
        else:
            panel.checks["rendered_hash"] = False
            ctx.note(Kind.CHECK, "check.rendered_hash.differs", "The rendered image is not a grey or RGB image of 8 or "
                     "16 bits, so it was changed after it was drawn.")
    else:
        _check(ctx, panel, "rendered_hash", None, None, "the rendered image changed after it was drawn")

    crop = panel.figure["crop"]
    if crop is not None and carrier.rendered_usable and ihdr is not None:
        if (ihdr.width, ihdr.height) != (crop["w"], crop["h"]):
            panel.figure["crop"] = None
            if panel.checks.get("rendered_hash") is False:
                ctx.note(Kind.CHECK, "check.crop.size", "The rendered image's size does not match 'crop': it was redrawn "
                         "after the panel was written. Its coordinates are not used.", "figure.crop")
            else:
                ctx.broken("figure.crop", "'crop' does not match the size of the rendered image; its coordinates are not used.",
                           "figure.crop")
    view = panel.core
    if crop is not None and panel.figure["crop"] is not None and panel.checks.get("drawn_from") and view is not None:
        size = _aligned_size(view)
        if size is not None:
            height, width = size
            if crop["x"] < 0 or crop["y"] < 0 or crop["x"] + crop["w"] > width or crop["y"] + crop["h"] > height:
                panel.figure["crop"] = None
                ctx.broken("figure.crop", "'crop' runs outside the aligned image; its coordinates are not used.", "figure.crop")
    if panel.checks.get("drawn_from") and ihdr is not None and view is not None and view.blot is not None:
        expected_type = 2 if any(e.get("color") for e in view.blot) else 0
        if view.output_bit_depth is not None and (ihdr.color_type, ihdr.bit_depth) != (expected_type, view.output_bit_depth):
            # As for the size of 'crop': only a picture that fails its hash shows it was changed (§6.2).
            if panel.checks.get("rendered_hash") is False:
                ctx.note(Kind.CHECK, "check.rendered_form", "The rendered image's PNG form does not match the core it "
                         "was drawn from, so it was changed after it was drawn.")
            else:
                ctx.broken("figure.rendered_form", "The rendered image's PNG form does not match the core it was "
                           "drawn from, and nothing shows it was changed after it was drawn: it was written so.",
                           "figure")


def _aligned_size(view: CoreView):
    if not view.blot or view.orientation is None or view.sources is None:
        return None
    source = view.source(view.blot[0]["source"])
    if source is None or source.get("shape") is None:
        return None
    return _geometry.aligned_size(source["shape"], view.orientation)


def _read_assembly(ctx: Context, panel: Panel, top: Obj) -> None:
    value = top.optional("assembly", lambda v: type(v) is dict, "an object")
    if value is None:
        return
    o = Obj(ctx, value, "assembly")
    why: list[str] = []
    ident = o.required("id", C.is_id, "an ID", why)
    member = o.required("member", C.is_int, "an integer", why)
    members = o.required("members", lambda v: C.is_int(v) and v >= 1, "an integer of 1 or more", why)
    o.finish()
    if not why and not (1 <= member <= members):
        why.append("'member' is outside 1 to 'members'")
    if why:
        ctx.broken("assembly.broken", "'assembly' was read as absent: " + "; ".join(why) + ".", "assembly")
        return
    panel.assembly = {"id": ident, "member": member, "members": members}


def _read_extensions(ctx: Context, panel: Panel, top: Obj) -> None:
    value = top.optional("extensions", lambda v: type(v) is dict, "an object")
    for name, content in (value or {}).items():
        where = f"extensions[{name!r}]"
        if type(content) is not dict:
            ctx.broken("extension.invalid", "An extension that is not an object was ignored.", where)
            continue
        record = {"value": content, "based_on": None}
        if "based_on" in content:
            # §8: one that cannot be read leaves the extension's validity unknown.
            based_on, w = content["based_on"], f"{where}.based_on"
            if not C.is_hash(based_on):
                ctx.broken("extension.based_on", "'based_on' is not a hash; whether the extension still fits the "
                           "core is unknown.", w)
            elif ctx.hash_error(w, based_on):
                ctx.broken("extension.based_on", "'based_on' uses a hash kind this version does not define; whether "
                           "the extension still fits the core is unknown.", w)
            _check(ctx, panel, f"based_on[{name}]", content["based_on"], panel.core_hash,
                   f"the core changed after the extension {name!r} was written")
            record["based_on"] = panel.checks[f"based_on[{name}]"]
        panel.extensions[name] = record

"""Writing panels and documents.

`save` writes a document again (§2.6 rule 6): panels the caller did not
change are written byte for byte; a panel the caller changed is written
again from its JSON, and its new carrier PNG goes into the SVG in place of
the old one; nothing else in the SVG changes. A panel whose assembly record
alone changes keeps every byte but that record (the assembly-only rewrite).

A tool that draws the figure itself builds the carrier PNGs with
`new_panel`, `rewrite_panel` and `assembly_only` (kept panels keep
`panel.carrier.png`), draws its SVG around them, and then records the
SVG's hash in the panels it wrote with `sign` (§6.5). Drawing the figure is
the tool's job, not this module's.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import zlib

import xml.etree.ElementTree as ET

import numpy as np

from . import _carrier, _check as C, _datauri, _encode, _hashes, _json, _png, _splice, _svghash, _svgxml
from ._core import CHANNEL_IMAGE_KIND, CHANNEL_IMAGE_MEDIA
from .container import read_container
from .document import Document
from .notices import Kind, Notice
from .panel import FORMAT_URL, Panel, read_panel

_SVG_HASH_METHOD = "c14n2_without_panel_data"
_TOP = ("format", "schema_version", "schema_revision", "saved_at", "written_by", "core", "quantifications", "figure")


@dataclass(frozen=True)
class Tool:
    """The writer, recorded as `written_by` in each panel it writes (§3)."""

    name: str
    version: str
    build: str | None = None

    def record(self) -> dict:
        out = {"name": self.name, "version": self.version}
        if self.build is not None:
            out["build"] = self.build
        return out


@dataclass
class Content:
    """What a panel is written from.

    `json` is the panel to write: for a changed panel, usually
    `panel.to_json()` after the caller's edits. The writer sets `format`,
    the version, `saved_at` and `written_by`, builds `embedded`, and fills
    `figure` itself; of `figure`, only `crop` is read from `json`.

    `rendered` is a rendered image the caller drew from this core (§6.2)
    over `json["figure"]["crop"]`; `drawn_from` and `rendered_hash` are
    computed from it. A new panel needs one. When a changed panel has none,
    its previous rendered image is kept as it was, and so are `crop`,
    `rendered_hash` and `drawn_from` (§6.4: a changed core then shows the
    figure as older than the data).

    `channel_images` and `source_metadata` are the payloads the JSON points
    to, beyond what a changed panel already holds: channel images as
    pixels, source metadata as (catalog entry, payload)."""

    json: dict
    rendered: np.ndarray | None = None
    channel_images: list[np.ndarray] = field(default_factory=list)
    source_metadata: list[tuple[dict, bytes]] = field(default_factory=list)


@dataclass(frozen=True)
class WriteNote:
    """Something the writer did that the caller may want to pass on."""

    code: str
    message: str
    panel: int | None = None


class WriteError(Exception):
    """The document was not written. `notices` tells what reading the panel
    as it would have been written found wrong."""

    def __init__(self, message: str, panel: int | None = None, notices: list[Notice] | None = None):
        super().__init__(message)
        self.panel = panel
        self.notices = list(notices or [])


@dataclass
class Saved:
    svg: bytes
    notes: list[WriteNote] = field(default_factory=list)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --- the assembly-only rewrite ---------------------------------------------------------------------


def _check_assembly(record, index: int) -> dict | None:
    if record is None:
        return None
    if (type(record) is not dict or set(record) != {"id", "member", "members"} or not C.is_id(record["id"])
            or not C.is_int(record["member"]) or not C.is_int(record["members"])
            or not 1 <= record["member"] <= record["members"]):
        raise WriteError("An assembly record is an object of exactly 'id' (an ID), 'member' and 'members' "
                         "(integers, 1 ≤ member ≤ members).", index)
    return {"id": record["id"], "member": record["member"], "members": record["members"]}


def new_assembly_id() -> str:
    """A new, random assembly ID (§1.7: a UUID version 4)."""

    return str(uuid.uuid4())


def _replace_json_member(png: bytes, key: str, value) -> bytes:
    """The carrier PNG with one top-level member of its JSON replaced (None
    takes it out). Every other byte, and the iTXt's other fields and place,
    stay as they were."""

    (chunk,) = _carrier.find_keyword(_png.walk(png), _carrier.GELSVG_KEYWORD)
    fields, _ = _carrier.parse_itxt(chunk.data)
    text = _splice.replace_top_member(fields.text, key, value)
    body = zlib.compress(text, 9) if fields.compression_flag == 1 else text
    data = (fields.keyword + b"\0" + bytes([fields.compression_flag, fields.compression_method])
            + fields.language_tag + b"\0" + fields.translated_keyword + b"\0" + body)
    return png[: chunk.offset] + _encode.chunk(b"iTXt", data) + png[chunk.offset + 12 + chunk.length :]


def assembly_only(panel: Panel, record: dict | None, allow_loss: bool = False) -> bytes:
    """The panel's carrier PNG with only the top-level `assembly` changed
    (None takes it out)."""

    record = _check_assembly(record, panel.index)
    if not panel.readable or panel.newer_version or panel.schema_version != C.KNOWN_VERSION:
        raise WriteError("Only the assembly record of a readable panel of this version of the format can be "
                         "changed; this panel keeps its record, and so leaves the assembly.", panel.index)
    raw = panel.raw.get("assembly")
    if type(raw) is dict and set(raw) - {"id", "member", "members"} and not allow_loss:
        raise WriteError("The panel's assembly record has keys this writer does not know; changing it drops them. "
                         "Pass allow_loss=True to go ahead.", panel.index)
    if raw == record and ("assembly" in panel.raw) == (record is not None):
        return panel.carrier.png
    return _replace_json_member(panel.carrier.png, "assembly", record)


# --- writing a panel ------------------------------------------------------------------------------


def _kept_image(panel: Panel) -> None:
    """Refuse to keep a rendered image the panel cannot be written with
    (§6.2, §2.6 rule 6)."""

    carrier = panel.carrier
    why = None
    if not carrier.rendered_usable or panel.rendered is None:
        why = "its rendered image cannot be used"
    elif any(panel.figure.get(k) is None for k in ("crop", "rendered_hash", "drawn_from")):
        why = "'figure' lacks a value it must have"
    elif panel.checks.get("rendered_hash") is None:
        why = "its rendered image cannot be checked against 'rendered_hash'"
    elif any(n.code == "check.rendered_form" for n in panel.notices):
        why = "its rendered image's PNG form does not match the core it was drawn from"
    if why:
        raise WriteError(f"The panel cannot be written with its previous rendered image: {why}. "
                         "Draw it again and pass the pixels as Content.rendered.", panel.index)


def _kept_image_png(panel: Panel, itxt: bytes, gels: bytes) -> bytes:
    """The previous carrier PNG with a new JSON and new geLs chunks. Every
    other chunk keeps its bytes, so the SVG's hash input does not change
    (§6.5)."""

    png = panel.carrier.png
    out = [_png.PNG_SIGNATURE]
    for c in panel.carrier.walk.chunks:
        if c.type == b"IEND":
            break
        if c.runs_past_end or not c.crc_ok or c.type == _png.GELS_TYPE:
            continue
        if _carrier._keyword_of(c) == _carrier.GELSVG_KEYWORD:
            continue
        out.append(png[c.offset : c.offset + 12 + c.length])
        if c.type == b"IHDR":
            out.append(itxt)
    out += [gels, _encode.IEND]
    return b"".join(out)


def _with_embedded(core: dict, embedded: dict) -> dict:
    out = {}
    for key, value in core.items():
        out[key] = value
        if key == "sources":
            out["embedded"] = embedded
    if "embedded" not in out:
        out["embedded"] = embedded
    return out


def _problem(notice: Notice) -> bool:
    if notice.kind is Kind.CHECK:
        return notice.code in ("check.rendered_form", "check.crop.size")
    return notice.code not in ("metadata.free_form", "metadata.unknown")  # carried over unread (§5.4)


def _channel_entry(pixels: np.ndarray) -> dict:
    png = _encode.channel_png(pixels)
    return {"dtype": "uint8" if pixels.dtype == np.uint8 else "uint16", "shape": list(pixels.shape),
            "byte_length": len(png), "png_hash": _hashes.sha256(png), "payload": png}


def _build(content: Content, tool: Tool, saved_at: str | None, notes: list[WriteNote], *,
           index: int | None, element, images: dict, metadata: dict, kept: Panel | None) -> bytes:
    """Write the panel's JSON and carrier PNG, and read them back. `images`
    and `metadata` are the payloads already held, by key; `kept` is the
    panel whose rendered image is kept, when no new one is given."""

    data = copy.deepcopy(content.json)
    if type(data) is not dict or type(data.get("core")) is not dict:
        raise WriteError("Content.json must be an object with a 'core' object.", index)
    core = data["core"]
    core.pop("embedded", None)

    images = dict(images)
    for pixels in content.channel_images:
        if pixels.dtype not in (np.uint8, np.uint16) or pixels.ndim != 2 or 0 in pixels.shape:
            raise WriteError("A channel image is a (h, w) image of uint8 or uint16.", index)
        key = _hashes.pixel_hash(pixels)
        if key not in images:  # a payload already held keeps its bytes
            images[key] = _channel_entry(pixels)
    metadata = dict(metadata)
    for entry, payload in content.source_metadata:
        metadata[_hashes.sha256(_json.canonical(entry))] = (entry, payload)

    image_catalog, metadata_catalog = {}, {}
    for source in core.get("sources") if type(core.get("sources")) is list else ():
        if type(source) is not dict:
            continue
        for channel in source.get("channels") if type(source.get("channels")) is list else ():
            key = channel.get("pixel_hash") if type(channel) is dict else None
            if key in images and key not in image_catalog:
                e = images[key]
                image_catalog[key] = {"kind": CHANNEL_IMAGE_KIND, "media_type": CHANNEL_IMAGE_MEDIA, "dtype": e["dtype"],
                                      "shape": e["shape"], "byte_length": e["byte_length"], "png_hash": e["png_hash"]}
        key = source.get("source_metadata")
        if key in metadata and key not in metadata_catalog:
            metadata_catalog[key] = metadata[key][0]
    embedded = {}
    if image_catalog:
        embedded["channel_images"] = image_catalog
    if metadata_catalog:
        embedded["source_metadata"] = metadata_catalog
    if embedded:
        core = _with_embedded(core, embedded)

    # The figure record.
    if content.rendered is None:
        figure = {k: copy.deepcopy(kept.figure[k]) for k in ("crop", "rendered_hash", "drawn_from")}
    else:
        pixels = content.rendered
        if (pixels.dtype not in (np.uint8, np.uint16) or pixels.ndim not in (2, 3)
                or (pixels.ndim == 3 and pixels.shape[2] != 3) or 0 in pixels.shape):
            raise WriteError("Content.rendered must be a grey (h, w) or RGB (h, w, 3) image of uint8 or uint16.", index)
        crop = data["figure"].get("crop") if type(data.get("figure")) is dict else None
        figure = {"crop": crop, "rendered_hash": _hashes.rendered_hash(pixels), "drawn_from": _hashes.core_hash(core)}

    # Quantifications measured on another core are dropped (§7.4).
    quant_hash = _hashes.quant_core_hash(core)
    quants = data.get("quantifications")
    if type(quants) is list:
        kept_quants = []
        for i, q in enumerate(quants):
            if type(q) is dict and q.get("computed_on") != quant_hash:
                notes.append(WriteNote("quantification.stale", f"The quantification {i} was measured on another core "
                                       "and was left out (§7.4).", index))
                continue
            kept_quants.append(q)
        quants = kept_quants or None

    out = {"format": FORMAT_URL, "schema_version": C.KNOWN_VERSION, "schema_revision": C.KNOWN_REVISION,
           "saved_at": saved_at or _now(), "written_by": tool.record(), "core": core}
    if quants is not None:
        out["quantifications"] = quants  # anything but a list is left for the check below to refuse
    out["figure"] = figure
    for key, value in data.items():
        if key not in _TOP:
            out[key] = value
    try:
        text = _json.dumps(out)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise WriteError(f"The panel's JSON cannot be written: {exc}.", index) from None

    itxt = _encode.itxt(text)
    gels = _encode.gels_chunks([images[k]["payload"] for k in image_catalog],
                               [metadata[k][1] for k in metadata_catalog])
    if content.rendered is None:
        png = _kept_image_png(kept, itxt, gels)
    else:
        header, stream = _encode.encode_image(content.rendered)
        png = _png.PNG_SIGNATURE + header + itxt + _encode.idat_chunks(stream) + gels + _encode.IEND

    # Read it back: what would be written must read without a problem.
    walk = _png.walk(png)
    carrier = _carrier.assess(png, walk, _carrier.find_keyword(walk, _carrier.GELSVG_KEYWORD))
    check = read_panel(index if index is not None else 0, element, carrier)
    problems = [n for n in check.notices if _problem(n)]
    if problems:
        raise WriteError("The panel would not be written as the format requires: "
                         + "; ".join(n.message for n in problems[:3]) + ("…" if len(problems) > 3 else ""),
                         index, problems)
    return png


def rewrite_panel(panel: Panel, content: Content, tool: Tool, *, saved_at: str | None = None,
                  allow_loss: bool = False, notes: list[WriteNote] | None = None) -> bytes:
    """The new carrier PNG of a panel the caller changed. Raises WriteError
    when the panel cannot be written as the format requires."""

    notes = notes if notes is not None else []
    index = panel.index
    if not panel.readable or panel.newer_version:
        raise WriteError("The panel was not read (unreadable, or of a newer version of the format), so it can only "
                         "be kept byte for byte.", index)
    if panel.newer_revision and not allow_loss:
        raise WriteError("The panel is of a newer revision; writing it again drops what that revision added. "
                         "Pass allow_loss=True to go ahead.", index)
    if content.rendered is None:
        _kept_image(panel)
    view = panel.core
    images = {k: e for k, e in view.channel_images.items() if e.get("payload") is not None}
    metadata = {k: (r["entry"], r["payload"]) for k, r in view.source_metadata.items() if r.get("payload") is not None}
    png = _build(content, tool, saved_at, notes, index=index, element=panel.element, images=images,
                 metadata=metadata, kept=panel)
    if panel.figure.get("svg_hash") is not None:
        notes.append(WriteNote("svg_hash.removed", "The panel's svg_hash was left out: it recorded the SVG as an "
                               "earlier writer wrote it (§6.5).", index))
    return png


def new_panel(content: Content, tool: Tool, *, saved_at: str | None = None,
              notes: list[WriteNote] | None = None) -> bytes:
    """The carrier PNG of a new panel. Raises WriteError when it cannot be
    written as the format requires."""

    if content.rendered is None:
        raise WriteError("A new panel needs its rendered image (Content.rendered).")
    element = _svgxml.Element(_svgxml.SVG_NS, "image", {}, None)
    return _build(content, tool, saved_at, notes if notes is not None else [], index=None, element=element,
                  images={}, metadata={}, kept=None)


def image_element(png: bytes, x: float = 0, y: float = 0, scale: float = 1) -> str:
    """An SVG `image` element drawing a carrier PNG at (x, y), each pixel
    `scale` units wide (§1.2, §1.6)."""

    ihdr = _png.Ihdr.parse(_png.walk(png).chunks[0].data)
    n = _json.number
    return (f'<image x="{n(x)}" y="{n(y)}" width="{n(ihdr.width * scale)}" height="{n(ihdr.height * scale)}" '
            f'preserveAspectRatio="xMidYMid meet" href="{_splice.png_data_uri(png).decode("ascii")}"/>')


def sign(svg: bytes, pngs) -> Saved:
    """Record the SVG's hash (§6.5) in the panels whose carrier PNG is one
    of `pngs`: the panels this writer wrote, in an SVG it drew itself. The
    hash's input leaves the panels' JSON out, so recording it does not
    change it. When the input cannot be made, nothing is recorded."""

    targets = set(pngs)
    try:
        value = _svghash.svg_hash(svg)
    except (_svghash.Unavailable, ET.ParseError):
        return Saved(svg, [WriteNote("svg_hash.unavailable", "The SVG's hash could not be computed (§6.5), so it "
                                     "was not recorded.")])
    container = read_container(svg)
    replacements, found = [], set()
    for slot in container.panels:
        png = slot.carrier.png
        if png not in targets:
            continue
        found.add(png)
        figure = _json.parse(slot.carrier.json_text)["figure"]
        figure["svg_hash"], figure["svg_hash_method"] = value, _SVG_HASH_METHOD
        replacements.append((slot.element, _replace_json_member(png, "figure", figure)))
    if found != targets:
        raise WriteError(f"{len(targets - found)} of the PNGs to sign are not panels of this SVG.")
    return Saved(_splice.replace_hrefs(svg, replacements))


# --- the document --------------------------------------------------------------------------------


def save(document: Document, tool: Tool, *, rewrite: dict[int, Content] | None = None,
         assembly: dict[int, dict | None] | None = None, allow_loss: bool = False,
         saved_at: str | None = None) -> Saved:
    """Write the document again. `rewrite` maps panel indexes to the panels
    the caller changed; `assembly` maps indexes of panels kept byte for
    byte to their new assembly record (None leaves the assembly). Every
    other panel, and every other byte of the SVG, is written as read.

    Raises WriteError, writing nothing, when a panel cannot be written as
    asked."""

    rewrite = rewrite or {}
    assembly = assembly or {}
    if not document.is_document:
        raise WriteError("This file is not a GelSVG document.")
    both = set(rewrite) & set(assembly)
    if both:
        raise WriteError("A rewritten panel carries its assembly record in its JSON; it cannot also be given one "
                         f"in 'assembly' (panels {sorted(both)}).")
    for index in [*rewrite, *assembly]:
        if not 0 <= index < len(document.panels):
            raise WriteError(f"There is no panel {index}.")

    notes: list[WriteNote] = []
    stamp = saved_at or _now()
    pngs: dict[int, bytes] = {}
    for index, change in sorted(rewrite.items()):
        pngs[index] = rewrite_panel(document.panels[index], change, tool, saved_at=stamp,
                                    allow_loss=allow_loss, notes=notes)
    for index, record in sorted(assembly.items()):
        png = assembly_only(document.panels[index], record, allow_loss=allow_loss)
        if png is not document.panels[index].carrier.png:
            pngs[index] = png
    if not pngs:
        return Saved(document.svg, notes)

    panels = document.panels
    svg = _splice.replace_hrefs(document.svg, [(panels[i].element, png) for i, png in pngs.items()])
    tree = _svgxml.parse(svg)
    for index, png in pngs.items():
        element = tree.elements[panels[index].element.order]
        if _datauri.decode_png_data_uri(_datauri.image_href(element.attributes) or "") != png:
            raise WriteError("The new PNG could not be put into the SVG.", index)
    return Saved(svg, notes)

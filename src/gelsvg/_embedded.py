"""Matching the JSON's catalogs with the `geLs` payloads (§5.3, §5.4) and
checking what they hold."""

from __future__ import annotations

import hashlib

import numpy as np

from . import _check as C, _hashes, _json, _metadata, _png
from ._carrier import Carrier
from ._check import Context
from ._core import CoreView
from .notices import Kind


def _report_chunks(ctx: Context, carrier: Carrier) -> bool:
    """Report geLs chunks written wrongly or of an unknown kind. Returns
    whether any unknown chunk was ignored."""

    unknown = False
    for g in carrier.gels:
        where = f"geLs at byte {g.chunk.offset}"
        if g.status == "writer_error":
            ctx.broken("gels.writer_error", f"A geLs chunk written wrongly was ignored ({g.detail}).", where)
        elif g.status == "duplicate":
            ctx.broken("gels.duplicate", "A geLs chunk repeating an earlier one was ignored.", where)
        elif g.status == "unknown_header":
            unknown = True
            if ctx.newer_revision:
                ctx.note(Kind.UNKNOWN, "gels.unknown", f"A geLs chunk of an unknown kind was not used ({g.detail}).", where)
            else:
                ctx.broken("gels.unknown", f"A geLs chunk of a kind this version does not define was ignored ({g.detail}).", where)
    return unknown


def _missing_body(ctx: Context, carrier: Carrier, unknown_chunks: bool, what: str, where: str) -> None:
    # A chunk whose CRC does not match, critical or not, may be a geLs whose type was damaged (§5.3).
    lost = carrier.walk.truncated or any(g.status == "damaged" for g in carrier.gels) or any(
        n.code in ("png.crc.critical", "png.crc.other", "png.truncated") for n in carrier.notices)
    if lost:
        ctx.note(Kind.DAMAGED, "embedded.body_damaged",
                 f"The {what} could not be found; it may have been in a damaged part of the PNG.", where)
    elif unknown_chunks:
        message = (f"The {what} could not be found; it is broken, or in a geLs chunk written in a way this reader "
                   "does not know.")
        if ctx.newer_revision:
            ctx.note(Kind.UNKNOWN, "embedded.body_unknown", message, where)
        else:
            ctx.broken("embedded.body_unknown", message, where)
    else:
        ctx.broken("embedded.body_missing", f"The {what} is missing from the PNG.", where)


def decode_channel_png(payload: bytes):
    """The pixels of a channel image PNG, or the reason it cannot be one
    (§5.3)."""

    walk = _png.walk(payload)
    if not walk.signature_ok:
        return None, "not a PNG"
    if walk.truncated or not walk.ended_at_iend or walk.bytes_after_iend:
        return None, "not a complete PNG"
    if any(not c.crc_ok for c in walk.chunks):
        return None, "a CRC does not match"
    types = [c.type for c in walk.chunks]
    if any(t not in (b"IHDR", b"IDAT", b"IEND") for t in types):
        return None, "it has chunks other than IHDR, IDAT and IEND"
    if types[0] != b"IHDR" or types.count(b"IHDR") != 1 or types[-1] != b"IEND" or walk.chunks[-1].length:
        return None, "IHDR or IEND is wrong"
    idat = [i for i, t in enumerate(types) if t == b"IDAT"]
    if not idat or idat[-1] - idat[0] + 1 != len(idat):
        return None, "the IDAT chunks are missing or not together"
    ihdr = _png.Ihdr.parse(walk.chunks[0].data)
    if ihdr is None or ihdr.problems():
        return None, "IHDR is not valid"
    if ihdr.color_type != 0 or ihdr.bit_depth not in (8, 16):
        return None, "it is not an 8- or 16-bit greyscale PNG"
    raw = _png.inflate_complete(b"".join(walk.chunks[i].data for i in idat))
    if raw is None:
        return None, "its image data is not one complete zlib stream"
    problem = _png.scanline_problem(ihdr, raw)
    if problem:
        return None, problem
    return _png.decode_pixels(ihdr, raw), None


def check_embedded(ctx: Context, view: CoreView, carrier: Carrier, decoded: dict | None = None) -> dict:
    """Verify every catalog entry against its payload. Returns the decoded
    channel images by pixel hash (only those that passed). `decoded` holds
    channel image PNGs already decoded, by the SHA-256 (hex) of their bytes:
    the writer reading back what it wrote, which need not decode them again."""

    unknown_chunks = _report_chunks(ctx, carrier)
    codec1 = {hashlib.sha256(g.payload).hexdigest(): g for g in carrier.gels if g.usable and g.codec == 1}
    codec2 = {hashlib.sha256(g.payload).hexdigest(): g for g in carrier.gels if g.usable and g.codec == 2}

    users: dict[str, list[dict]] = {}
    for source in view.sources or ():
        for channel in source["channels"]:
            if channel.get("pixel_hash_known"):
                users.setdefault(channel["pixel_hash"], []).append(source)

    images: dict[str, np.ndarray] = {}
    for key, entry in list(view.channel_images.items()):
        where = f"core.embedded.channel_images[{key!r}]"
        if key not in users:
            if key not in view.pointed_to_written:
                ctx.broken("embedded.unreferenced", "A channel image no channel points to was ignored.", where)
            del view.channel_images[key]
            continue
        problems = []
        head, body = C.hash_parts(entry["png_hash"])
        payload = codec1.get(body) if head == "sha256" else None
        if head != "sha256":
            if not ctx.unknown_value(f"{where}.png_hash", entry["png_hash"]):
                problems.append("'png_hash' uses a hash kind this version does not define")
            else:
                del view.channel_images[key]
                continue
        elif payload is None:
            _missing_body(ctx, carrier, unknown_chunks, "channel image", where)
            del view.channel_images[key]
            continue
        if not problems:
            if len(payload.payload) != entry["byte_length"]:
                problems.append("'byte_length' does not match the payload")
        if not problems:
            pixels, why = (decoded[body], None) if decoded and body in decoded else decode_channel_png(payload.payload)
            if pixels is None:
                problems.append(f"its PNG is not a valid channel image ({why})")
            else:
                dtype = "uint8" if pixels.dtype == np.uint8 else "uint16"
                if dtype != entry["dtype"] or list(pixels.shape) != entry["shape"]:
                    problems.append("its PNG does not match the entry's dtype and shape")
                elif _hashes.pixel_hash(pixels) != key:
                    problems.append("its pixels do not match the pixel hash")
                elif any(s.get("dtype") != dtype or s.get("shape") != entry["shape"] for s in users[key]):
                    problems.append("its dtype and shape do not match the source image")
                else:
                    images[key] = pixels
                    entry["payload"] = payload.payload
        if problems:
            ctx.broken("element.broken", "The channel image was dropped: " + "; ".join(problems)
                       + ". Its channel is read as not embedded.", where)
            del view.channel_images[key]

    metadata_users = {s["source_metadata"] for s in view.sources or () if s.get("source_metadata")}
    for key, entry in list(view.source_metadata.items()):
        where = f"core.embedded.source_metadata[{key!r}]"
        if key not in metadata_users:
            if key not in view.pointed_to_written:
                ctx.broken("embedded.unreferenced", "Source metadata no source image points to was ignored.", where)
            del view.source_metadata[key]
            continue
        problems = []
        if _hashes.sha256(_json.canonical(entry["entry"])) != key:
            problems.append("its key is not the hash of the entry")
        head, body = C.hash_parts(entry["payload_hash"])
        payload = None
        if head != "sha256":
            if not ctx.unknown_value(f"{where}.payload_hash", entry["payload_hash"]):
                problems.append("'payload_hash' uses a hash kind this version does not define")
        else:
            payload = codec2.get(body)
            if payload is None and not problems:
                _missing_body(ctx, carrier, unknown_chunks, "source metadata", where)
                del view.source_metadata[key]
                continue
            if payload is not None:
                if len(payload.payload) != entry["byte_length"]:
                    problems.append("'byte_length' does not match the payload")
                elif entry["readable"]:
                    index, why = _metadata.check_formal(entry["format"], payload.payload)
                    if why:
                        problems.append(why)
                    else:
                        entry["index"] = index
        if problems:
            ctx.broken("element.broken", "The source metadata was dropped: " + "; ".join(problems) + ".", where)
            del view.source_metadata[key]
            continue
        entry["payload"] = payload.payload if payload is not None else None

    for i, source in enumerate(view.sources or ()):
        ref = source.get("source_metadata")
        if ref and ref not in view.metadata_keys_written:
            ctx.broken("reference.missing", "The source image points to source metadata that does not exist; "
                       "it is read as having no metadata.", f"core.sources[{source['id']!r}].source_metadata")
    for body, g in codec1.items():
        if "sha256:" + body not in view.payloads_written[1]:
            ctx.broken("embedded.unused", "A channel image payload no entry uses was ignored.", f"geLs at byte {g.chunk.offset}")
    for body, g in codec2.items():
        if "sha256:" + body not in view.payloads_written[2]:
            ctx.broken("embedded.unused", "A source metadata payload no entry uses was ignored.", f"geLs at byte {g.chunk.offset}")
    return images

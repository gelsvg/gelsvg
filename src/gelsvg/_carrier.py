"""One carrier PNG: its JSON text, its `geLs` chunks, and whether its
rendered image can be used (§1.2 step 3, §1.3, §1.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
import zlib

from . import _png
from ._png import Chunk, GelsChunk, Ihdr, Walk
from .notices import Kind, Notice

GELSVG_KEYWORD = b"GelSVG"
LEGACY_KEYWORD = b"MyWB"


@dataclass
class ItxtFields:
    keyword: bytes
    compression_flag: int
    compression_method: int
    language_tag: bytes
    translated_keyword: bytes
    text: bytes
    """The text field, inflated when compressed."""


def _keyword_of(chunk: Chunk) -> bytes | None:
    if chunk.type != b"iTXt":
        return None
    end = chunk.data.find(b"\0")
    if end < 1:
        return None
    return chunk.data[:end]


def parse_itxt(data: bytes) -> tuple[ItxtFields | None, str | None]:
    """The fields of an iTXt chunk, or the reason they cannot be taken out
    (§1.3: missing NUL separators, a flag or method out of range, a text
    that cannot be inflated)."""

    end = data.find(b"\0")
    if end < 1:
        return None, "keyword not terminated"
    keyword = data[:end]
    pos = end + 1
    if pos + 2 > len(data):
        return None, "compression flag and method missing"
    flag, method = data[pos], data[pos + 1]
    pos += 2
    lang_end = data.find(b"\0", pos)
    if lang_end < 0:
        return None, "language tag not terminated"
    language = data[pos:lang_end]
    trans_end = data.find(b"\0", lang_end + 1)
    if trans_end < 0:
        return None, "translated keyword not terminated"
    translated = data[lang_end + 1 : trans_end]
    text = data[trans_end + 1 :]
    if flag not in (0, 1):
        return None, f"compression flag {flag}"
    if method != 0:
        return None, f"compression method {method}"
    if flag == 1:
        inflated = _png.inflate_complete(text)
        if inflated is None:
            return None, "compressed text cannot be inflated"
        text = inflated
    return ItxtFields(keyword, flag, method, language, translated, text), None


@dataclass
class Carrier:
    """What the container layer knows about one panel's carrier PNG."""

    png: bytes
    walk: Walk
    json_text: bytes | None = None
    """The UTF-8 bytes of the panel's JSON, when they could be taken out."""
    unreadable: str | None = None
    """Why the panel cannot be read at all (§2.6 rule 5), or None."""
    rendered_usable: bool = True
    ihdr: Ihdr | None = None
    pixel_data: bytes | None = None
    """The inflated, validated image data of the rendered image."""
    notices: list[Notice] = field(default_factory=list)

    @property
    def gels(self) -> list[GelsChunk]:
        return self.walk.gels

    def damaged(self) -> bool:
        return any(n.kind is Kind.DAMAGED for n in self.notices)


def find_keyword(walk: Walk, keyword: bytes) -> list[Chunk]:
    return [c for c in walk.chunks if _keyword_of(c) == keyword]


def assess(png: bytes, walk: Walk, json_chunks: list[Chunk]) -> Carrier:
    """Apply §1.3 to a PNG that has at least one GelSVG iTXt."""

    carrier = Carrier(png=png, walk=walk)
    notes = carrier.notices

    def note(kind: Kind, code: str, message: str, where: str | None = None) -> None:
        notes.append(Notice(kind, code, message, where=where))

    # Damage first: what §1.3 rescues.
    pixel_damage = False
    if walk.truncated:
        note(Kind.DAMAGED, "png.truncated",
             "The PNG could not be walked to its end; the rest was treated as absent.",
             where=f"from byte {walk.tail_lost_from}")
        pixel_damage = True
    present: list[Chunk] = []
    gels_by_chunk = {id(g.chunk): g for g in walk.gels}
    for chunk in walk.chunks:
        if chunk.type == _png.GELS_TYPE:
            gels = gels_by_chunk.get(id(chunk))
            if chunk.crc_ok or (gels is not None and gels.status == "rescued"):
                present.append(chunk)
            continue
        if chunk.crc_ok:
            present.append(chunk)
            continue
        if chunk.runs_past_end:
            continue  # already reported as truncation
        if _keyword_of(chunk) == GELSVG_KEYWORD:
            continue  # reported below as a damaged JSON chunk
        name = chunk.type.decode("latin-1")
        if _png.is_critical(chunk.type) and _png.is_letters(chunk.type):
            note(Kind.DAMAGED, "png.crc.critical",
                 f"The CRC of the {name} chunk does not match; the rendered image is not used.",
                 where=f"byte {chunk.offset}")
            pixel_damage = True
            carrier.rendered_usable = False
        else:
            note(Kind.DAMAGED, "png.crc.other",
                 f"The CRC of a {name!r} chunk does not match; that chunk was treated as absent.",
                 where=f"byte {chunk.offset}")
            pixel_damage = True  # its type may have been an IDAT's

    for gels in walk.gels:
        if gels.status in ("damaged", "rescued"):
            where = f"byte {gels.chunk.offset}"
            if gels.status == "rescued":
                note(Kind.DAMAGED, "gels.rescued",
                     "A damaged geLs chunk was recovered: its payload matches the SHA-256 in its header.",
                     where=where)
            else:
                note(Kind.DAMAGED, "gels.lost",
                     f"A damaged geLs chunk was treated as absent ({gels.detail}).", where=where)

    # The JSON chunk.
    if len(json_chunks) > 1:
        carrier.unreadable = "more than one GelSVG iTXt chunk"
    else:
        chunk = json_chunks[0]
        if not chunk.crc_ok:
            carrier.unreadable = "the GelSVG iTXt chunk is damaged"
            note(Kind.DAMAGED, "itxt.damaged",
                 "The CRC of the chunk holding the panel's JSON does not match.",
                 where=f"byte {chunk.offset}")
        else:
            fields, problem = parse_itxt(chunk.data)
            if fields is None:
                carrier.unreadable = f"the GelSVG iTXt chunk cannot be read: {problem}"
            else:
                carrier.json_text = fields.text

    # Writer mistakes: chunks whose CRC matches but whose content or order
    # breaks the rules.
    damage_seen = bool(notes)
    writer_errors: list[str] = []
    ordered = present
    if not ordered or ordered[0].type != b"IHDR":
        if damage_seen:
            carrier.rendered_usable = False
        else:
            writer_errors.append("the first chunk is not IHDR")
    for c in ordered:
        if not _png.is_letters(c.type):
            writer_errors.append("a chunk type is not four ASCII letters")
        elif _png.is_critical(c.type) and c.type not in _png.PERMITTED_CRITICAL:
            writer_errors.append(f"unknown critical chunk {c.type.decode('latin-1')}")
    if sum(1 for c in ordered if c.type == b"IHDR") > 1:
        writer_errors.append("more than one IHDR")
    plte = [i for i, c in enumerate(ordered) if c.type == b"PLTE"]
    idat = [i for i, c in enumerate(ordered) if c.type == b"IDAT"]
    if len(plte) > 1:
        writer_errors.append("more than one PLTE")
    if plte and idat and plte[0] > idat[0]:
        writer_errors.append("PLTE after the first IDAT")
    if idat and idat[-1] - idat[0] + 1 != len(idat):
        writer_errors.append("another chunk between IDAT chunks")
    iend = [c for c in ordered if c.type == b"IEND"]
    if walk.ended_at_iend:
        if iend and iend[-1].length != 0:
            writer_errors.append("IEND has data")
        if walk.bytes_after_iend:
            writer_errors.append("bytes after IEND")
    elif not damage_seen:
        writer_errors.append("no IEND")

    # Pixels.
    ihdr_chunk = ordered[0] if ordered and ordered[0].type == b"IHDR" else None
    if ihdr_chunk is not None:
        ihdr = Ihdr.parse(ihdr_chunk.data)
        if ihdr is None:
            writer_errors.append("IHDR is not 13 bytes")
        else:
            carrier.ihdr = ihdr
            problem = "; ".join(ihdr.problems()) or None
            raw = None
            if problem is None:
                if not idat:
                    problem = "no IDAT"
                else:
                    raw = _png.inflate_complete(b"".join(ordered[i].data for i in idat))
                    if raw is None:
                        problem = "image data is not one complete zlib stream"
                    else:
                        problem = _png.scanline_problem(ihdr, raw)
            if problem is not None:
                if pixel_damage:
                    carrier.rendered_usable = False
                    note(Kind.DAMAGED, "png.pixels.damaged",
                         f"The rendered image cannot be decoded after the damage ({problem}); it is not used.")
                else:
                    writer_errors.append(f"pixels cannot be decoded: {problem}")
            elif carrier.rendered_usable:
                carrier.pixel_data = raw

    if writer_errors and carrier.unreadable is None:
        carrier.unreadable = "the carrier PNG is not a valid PNG: " + "; ".join(writer_errors)
    return carrier

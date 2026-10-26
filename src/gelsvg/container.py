"""The container layer: from SVG bytes to the carrier PNG of each panel
(§1.1–§1.4). Nothing here reads the JSON; that is the panel layer's job."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import _carrier, _datauri, _png, _svgxml
from ._carrier import Carrier
from ._svgxml import Element, NotADocument, Tree
from .notices import Kind, Notice

_HINTS = {
    _svgxml.NOT_UTF8: "Saving it again as uncompressed UTF-8 SVG would make it readable.",
    _svgxml.COMPRESSED: "Saving it again as uncompressed UTF-8 SVG would make it readable.",
    _svgxml.DOCTYPE: "Saving it again as SVG without a document type declaration and entity references would make it readable.",
    _svgxml.REFERENCE: "Saving it again as SVG without a document type declaration and entity references would make it readable.",
}


@dataclass
class PanelSlot:
    """One carrier PNG found in the SVG, in document order."""

    index: int
    element: Element
    carrier: Carrier


@dataclass
class LegacySlot:
    """A carrier PNG of an old MyWB document (keyword `MyWB`, schema 1–6).
    Not a GelSVG panel; kept so writers can preserve it and converters can
    find it (Appendix B)."""

    element: Element
    png: bytes


@dataclass
class Container:
    tree: Tree | None
    panels: list[PanelSlot] = field(default_factory=list)
    legacy: list[LegacySlot] = field(default_factory=list)
    notices: list[Notice] = field(default_factory=list)
    not_a_document: NotADocument | None = None

    @property
    def is_document(self) -> bool:
        return self.not_a_document is None and bool(self.panels)


def read_container(svg: bytes) -> Container:
    try:
        tree = _svgxml.parse(svg)
    except NotADocument as exc:
        result = Container(tree=None, not_a_document=exc)
        message = f"This file is not a GelSVG document: {exc.detail}."
        hint = _HINTS.get(exc.reason)
        if hint:
            message += " " + hint
        result.notices.append(Notice(Kind.UNREADABLE, f"document.{exc.reason}", message))
        return result

    result = Container(tree=tree)
    for element in tree.elements:
        if element.local != "image":
            continue
        href = _datauri.image_href(element.attributes)
        if href is None:
            continue
        png = _datauri.decode_png_data_uri(href)
        if png == _datauri.NOT_PNG:
            continue
        if isinstance(png, str):
            result.notices.append(Notice(
                Kind.DAMAGED, "image.undecodable",
                "An image/png data URI could not be decoded; a panel may have been damaged.",
                where=f"element {element.order}",
            ))
            continue
        walk = _png.walk(png)
        if not walk.signature_ok:
            continue
        json_chunks = _carrier.find_keyword(walk, _carrier.GELSVG_KEYWORD)
        if json_chunks:
            carrier = _carrier.assess(png, walk, json_chunks)
            index = len(result.panels)
            carrier.notices = [
                Notice(n.kind, n.code, n.message, panel=index, where=n.where) for n in carrier.notices
            ]
            result.panels.append(PanelSlot(index, element, carrier))
            continue
        if _carrier.find_keyword(walk, _carrier.LEGACY_KEYWORD):
            result.legacy.append(LegacySlot(element, png))
            continue
        if walk.gels or walk.truncated:
            result.notices.append(Notice(
                Kind.DAMAGED, "image.panel_lost",
                "A PNG has geLs chunks or could not be walked to its end, but no GelSVG JSON was found in it; a panel may have been damaged.",
                where=f"element {element.order}",
            ))

    if not result.panels:
        if result.legacy:
            result.not_a_document = NotADocument(
                "legacy", "it holds only panels of an old MyWB document (schema 1–6)")
            message = ("This file is an old MyWB document (schema 1–6), not a GelSVG document; "
                       "it can be converted (Appendix B).")
        else:
            result.not_a_document = NotADocument("no_panel", "it holds no carrier PNG")
            message = "This file is not a GelSVG document: it holds no carrier PNG."
        result.notices.append(Notice(Kind.UNREADABLE, f"document.{result.not_a_document.reason}", message))
    return result

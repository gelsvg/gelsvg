"""Reading a whole GelSVG document: its panels and their assemblies
(§1.7)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from . import _svghash
from .container import Container, LegacySlot, read_container
from .notices import Kind, Notice
from .panel import Panel, read_panel


@dataclass
class Assembly:
    id: str
    members: int
    panels: dict[int, int]
    """member number → panel index."""

    @property
    def missing(self) -> list[int]:
        return [m for m in range(1, self.members + 1) if m not in self.panels]


@dataclass
class Document:
    container: Container
    panels: list[Panel] = field(default_factory=list)
    assemblies: list[Assembly] = field(default_factory=list)
    notices: list[Notice] = field(default_factory=list)

    @property
    def is_document(self) -> bool:
        return self.container.is_document

    @property
    def legacy(self) -> list[LegacySlot]:
        return self.container.legacy

    def all_notices(self) -> list[Notice]:
        out = list(self.notices)
        for panel in self.panels:
            out.extend(panel.notices)
        return out


def read(svg: bytes) -> Document:
    """Read a document. Nothing is raised for bad input: what could not be
    read is reported in the notices."""

    container = read_container(svg)
    doc = Document(container=container, notices=list(container.notices))
    for slot in container.panels:
        doc.panels.append(read_panel(slot.index, slot.element, slot.carrier))
    _check_svg_hash(doc, svg)
    _group_assemblies(doc)
    return doc


def _check_svg_hash(doc: Document, svg: bytes) -> None:
    """Compare each panel's svg_hash with the SVG as it is now (§6.5)."""

    wanted = [p for p in doc.panels if p.figure.get("svg_hash") and p.figure.get("svg_hash_method")]
    if not wanted:
        return
    try:
        actual = _svghash.svg_hash(svg)
    except _svghash.Unavailable:
        actual = None
    for panel in wanted:
        recorded = panel.figure["svg_hash"]
        if actual is None or not recorded.startswith("sha256:"):
            panel.checks["svg_hash"] = None
            panel.notices.append(Notice(Kind.CHECK, "check.svg_hash.unknown",
                                        "Whether the SVG changed after this panel was written could not be checked.",
                                        panel=panel.index))
            continue
        panel.checks["svg_hash"] = recorded == actual
        if recorded == actual:
            message = "Checked whether the SVG changed after this panel was written: it has not."
        else:
            message = ("Checked whether the SVG changed after this panel was written: it has (by an editor, by being "
                       "moved to another figure, or by a later writer that kept this panel as it was).")
        panel.notices.append(Notice(Kind.CHECK, "check.svg_hash." + ("matched" if recorded == actual else "differs"),
                                    message, panel=panel.index))


def _stop_element(image, member_of, number_set, counts):
    """Climb from the image's parent to the element that holds this copy
    (§1.7, the copy-splitting procedure, step 1)."""

    element = image.parent
    while element is not None:
        inside = counts.get(id(element), Counter())
        if element.parent is None:
            return element
        if sum(inside.values()) >= 2:
            if number_set <= set(inside):
                return element
            outer = counts.get(id(element.parent), Counter())
            if any(outer[m] > inside[m] for m in inside):
                return element
        element = element.parent
    return image


def _group_assemblies(doc: Document) -> None:
    by_id: dict[str, list[Panel]] = {}
    for panel in doc.panels:
        if panel.assembly is not None:
            by_id.setdefault(panel.assembly["id"], []).append(panel)

    def note(kind: Kind, code: str, message: str, ident: str) -> None:
        doc.notices.append(Notice(kind, code, message, where=f"assembly {ident!r}"))

    def drop(panels: list[Panel]) -> None:
        for p in panels:
            p.assembly = None

    for ident, panels in by_id.items():
        if len({p.assembly["members"] for p in panels}) > 1:
            note(Kind.BROKEN, "assembly.members", "The panels of this assembly disagree on how many members it has; "
                 "none of them is read as part of an assembly.", ident)
            drop(panels)
            continue
        members = panels[0].assembly["members"]
        numbers = [p.assembly["member"] for p in panels]
        if len(set(numbers)) == len(numbers):
            groups = [panels]
        elif members == 1:
            groups = [[p] for p in panels]
            note(Kind.ASSEMBLY, "assembly.copies_split", "Copies of a one-panel assembly were each read as an assembly.", ident)
        else:
            counts: dict[int, Counter] = {}
            for p in panels:
                for ancestor in p.element.ancestors():
                    counts.setdefault(id(ancestor), Counter())[p.assembly["member"]] += 1
            number_set = set(numbers)
            grouped: dict[int, list[Panel]] = {}
            for p in panels:
                stop = _stop_element(p.element, None, number_set, counts)
                grouped.setdefault(id(stop), []).append(p)
            groups = list(grouped.values())
            if any(len({p.assembly["member"] for p in g}) != len(g) for g in groups):
                note(Kind.ASSEMBLY, "assembly.copies_unsplit", "The assembly is drawn more than once and its copies "
                     "could not be told apart; none of its panels is read as part of an assembly.", ident)
                drop(panels)
                continue
            note(Kind.ASSEMBLY, "assembly.copies_split", f"The assembly is drawn {len(groups)} times; each copy was "
                 "read as its own assembly.", ident)
        for group in groups:
            assembly = Assembly(ident, members, {p.assembly["member"]: p.index for p in group})
            doc.assemblies.append(assembly)
            if assembly.missing:
                note(Kind.ASSEMBLY, "assembly.missing", "Members " + ", ".join(map(str, assembly.missing))
                     + " of the assembly were not found.", ident)

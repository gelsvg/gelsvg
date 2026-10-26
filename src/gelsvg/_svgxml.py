"""Reading the SVG as XML (§1.1): which bytes are a document at all, and a
light element tree for finding panels (§1.2) and splitting assembly copies
(§1.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from xml.parsers import expat

SVG_NS = "http://www.w3.org/2000/svg"

# Why a file is not a document; the hint is what the reader should suggest.
NOT_UTF8 = "encoding"
COMPRESSED = "compressed"
DOCTYPE = "doctype"
REFERENCE = "reference"
VERSION = "version"
MALFORMED = "malformed"
NOT_SVG = "not_svg"

_XML_DECL = re.compile(
    r"<\?xml[ \t\r\n]+version[ \t\r\n]*=[ \t\r\n]*([\"'])([^\"']*)\1"
    r"(?:[ \t\r\n]+encoding[ \t\r\n]*=[ \t\r\n]*([\"'])([^\"']*)\3)?"
    r"(?:[ \t\r\n]+standalone[ \t\r\n]*=[ \t\r\n]*([\"'])(yes|no)\5)?"
    r"[ \t\r\n]*\?>"
)
_WS = " \t\r\n"


@dataclass(eq=False)
class Element:
    namespace: str | None
    local: str
    attributes: dict[str, str]
    parent: "Element | None"
    children: list["Element"] = field(default_factory=list)
    order: int = 0
    """Position in document order, counting from 0."""
    offset: int = 0
    """Byte offset of the start tag's `<` in the SVG."""

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent


@dataclass
class Tree:
    root: Element
    elements: list[Element]
    """Every element in document order."""


class NotADocument(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _split_name(name: str) -> tuple[str | None, str]:
    if " " in name:
        namespace, local = name.split(" ", 1)
        return namespace, local
    return None, name


def _skip_misc(text: str, pos: int) -> int:
    """Skip white space, comments and processing instructions."""

    while True:
        while pos < len(text) and text[pos] in _WS:
            pos += 1
        if text.startswith("<!--", pos):
            end = text.find("-->", pos + 4)
            if end < 0:
                return len(text)
            pos = end + 3
        elif text.startswith("<?", pos):
            end = text.find("?>", pos + 2)
            if end < 0:
                return len(text)
            pos = end + 2
        else:
            return pos


def _check_doctype(text: str, pos: int) -> None:
    """Refuse an internal subset holding anything but white space and
    comments. A document type declaration that only names a DTD is read as
    if it were not there; its DTD is never fetched."""

    pos = _skip_misc(text, pos)
    if not text.startswith("<!DOCTYPE", pos):
        return
    i = pos + len("<!DOCTYPE")
    in_quote = None
    while i < len(text):
        ch = text[i]
        if in_quote:
            if ch == in_quote:
                in_quote = None
        elif ch in "\"'":
            in_quote = ch
        elif ch == ">":
            return
        elif ch == "[":
            break
        i += 1
    else:
        return  # unterminated: left to the XML parser
    i += 1
    while i < len(text):
        while i < len(text) and text[i] in _WS:
            i += 1
        if text.startswith("<!--", i):
            end = text.find("-->", i + 4)
            if end < 0:
                return
            i = end + 3
            continue
        if text.startswith("]", i):
            return
        raise NotADocument(DOCTYPE, "the document type declaration has an internal subset with declarations")


def parse(svg: bytes) -> Tree:
    """Check that `svg` is a document in the sense of §1.1 and return its
    element tree. Raises NotADocument otherwise."""

    if svg[:2] == b"\x1f\x8b":
        raise NotADocument(COMPRESSED, "the file is gzip-compressed (.svgz)")
    if svg[:2] in (b"\xfe\xff", b"\xff\xfe") or svg[:4] in (b"\x00\x00\xfe\xff",):
        raise NotADocument(NOT_UTF8, "the file starts with a UTF-16 or UTF-32 byte order mark")
    if svg[:4] in (b"\x00<\x00?", b"<\x00?\x00", b"\x00\x00\x00<", b"<\x00\x00\x00"):
        raise NotADocument(NOT_UTF8, "the file is not UTF-8 (UTF-16 or UTF-32 without a byte order mark)")
    body = svg[3:] if svg.startswith(b"\xef\xbb\xbf") else svg
    if b"\0" in body:
        # XML never holds U+0000, so NUL bytes mean UTF-16 or UTF-32.
        raise NotADocument(NOT_UTF8, "the file holds NUL bytes, so it is not UTF-8 (UTF-16 or UTF-32)")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise NotADocument(NOT_UTF8, "the file is not valid UTF-8") from None

    pos = 0
    match = _XML_DECL.match(text)
    if match:
        if match.group(2) != "1.0":
            raise NotADocument(VERSION, f"XML version {match.group(2)!r}, not '1.0'")
        encoding = match.group(4)
        if encoding is not None and encoding.lower() != "utf-8":
            raise NotADocument(NOT_UTF8, f"the XML declaration names the encoding {encoding!r}")
        pos = match.end()
    _check_doctype(text, pos)

    # The encoding is given so that expat never guesses UTF-16 from the
    # first bytes; the declaration has already been checked to be UTF-8.
    parser = expat.ParserCreate(encoding="UTF-8", namespace_separator=" ")
    parser.buffer_text = True
    elements: list[Element] = []
    stack: list[Element] = []
    refusals: list[NotADocument] = []

    def refuse(reason: str, detail: str):
        def handler(*_args):
            refusals.append(NotADocument(reason, detail))
        return handler

    def start(name, attrs):
        namespace, local = _split_name(name)
        parent = stack[-1] if stack else None
        element = Element(namespace, local, dict(attrs), parent, order=len(elements), offset=parser.CurrentByteIndex)
        if parent is not None:
            parent.children.append(element)
        elements.append(element)
        stack.append(element)

    def end(_name):
        stack.pop()

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.SkippedEntityHandler = refuse(REFERENCE, "an entity reference other than the five predefined ones")
    parser.EntityDeclHandler = refuse(DOCTYPE, "an entity declaration")
    parser.AttlistDeclHandler = refuse(DOCTYPE, "an attribute list declaration")
    parser.ElementDeclHandler = refuse(DOCTYPE, "an element declaration")
    parser.NotationDeclHandler = refuse(DOCTYPE, "a notation declaration")
    try:
        parser.Parse(svg, True)
    except expat.ExpatError as exc:
        if refusals:
            raise refusals[0] from None
        if exc.code == expat.errors.codes[expat.errors.XML_ERROR_UNDEFINED_ENTITY]:
            raise NotADocument(REFERENCE, "an undeclared entity reference") from None
        raise NotADocument(MALFORMED, f"not well-formed XML: {expat.errors.messages[exc.code]} (line {exc.lineno})") from None
    if refusals:
        raise refusals[0]
    if not elements:
        raise NotADocument(MALFORMED, "no root element")
    root = elements[0]
    if root.namespace != SVG_NS or root.local != "svg":
        raise NotADocument(NOT_SVG, "the root element is not an svg element in the SVG namespace")
    return Tree(root=root, elements=elements)

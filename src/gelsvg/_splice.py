"""Changing a few bytes and leaving the rest as read: the top-level
`assembly` in a panel's JSON text (§2.6 rule 6, the assembly-only rewrite)
and the `href` of a carrier PNG's `image` element in the SVG."""

from __future__ import annotations

import base64
import json
import re

from . import _datauri, _json
from ._svgxml import Element

_WS = b" \t\r\n"
XMLNS_NS = "http://www.w3.org/2000/xmlns/"


# --- the JSON text ---------------------------------------------------------------------------


def _skip_ws(text: bytes, pos: int) -> int:
    while pos < len(text) and text[pos] in _WS:
        pos += 1
    return pos


def _end_of_string(text: bytes, pos: int) -> int:
    """`pos` is at the opening quote; returns the index after the closing one."""

    pos += 1
    while True:
        ch = text[pos]
        if ch == 0x5C:  # backslash
            pos += 2
        elif ch == 0x22:
            return pos + 1
        else:
            pos += 1


def _end_of_value(text: bytes, pos: int) -> int:
    ch = text[pos]
    if ch == 0x22:
        return _end_of_string(text, pos)
    if ch in b"{[":
        depth = 0
        while True:
            ch = text[pos]
            if ch == 0x22:
                pos = _end_of_string(text, pos)
                continue
            if ch in b"{[":
                depth += 1
            elif ch in b"}]":
                depth -= 1
                if depth == 0:
                    return pos + 1
            pos += 1
    while pos < len(text) and text[pos] not in b",]} \t\r\n":
        pos += 1
    return pos


def top_members(text: bytes):
    """The members of the top-level object of a JSON text already known to
    be valid: (key, key_start, value_start, value_end) each, and the index
    of the opening brace."""

    pos = _skip_ws(text, 0)
    if text[pos : pos + 1] != b"{":
        raise ValueError("the JSON text is not an object")
    opening = pos
    members = []
    pos = _skip_ws(text, pos + 1)
    if text[pos : pos + 1] == b"}":
        return members, opening
    while True:
        key_start = pos
        key_end = _end_of_string(text, pos)
        key = json.loads(text[key_start:key_end].decode("utf-8"))
        pos = _skip_ws(text, key_end) + 1  # past the colon
        value_start = _skip_ws(text, pos)
        value_end = _end_of_value(text, value_start)
        members.append((key, key_start, value_start, value_end))
        pos = _skip_ws(text, value_end)
        if text[pos : pos + 1] == b"}":
            return members, opening
        pos = _skip_ws(text, pos + 1)  # past the comma


def replace_top_member(text: bytes, key: str, value) -> bytes:
    """Put `value` in place of the top-level member `key`'s value, add the
    member after the last one, or (value None) take it out with its comma.
    Every other byte stays as it was."""

    members, opening = top_members(text)
    found = [i for i, m in enumerate(members) if m[0] == key]
    if value is None:
        if not found:
            return text
        i = found[0]
        _, key_start, _, value_end = members[i]
        if i > 0:
            start = members[i - 1][3]
            comma = text.index(b",", start)
            return text[:comma] + text[value_end:]
        if len(members) > 1:
            comma = text.index(b",", value_end)
            return text[:key_start] + text[comma + 1 :]
        return text[:key_start] + text[value_end:]
    encoded = _json.dumps(value)
    if found:
        _, _, value_start, value_end = members[found[0]]
        return text[:value_start] + encoded + text[value_end:]
    member = _json.dumps(key) + b":" + encoded
    if members:
        end = members[-1][3]
        return text[:end] + b"," + member + text[end:]
    return text[: opening + 1] + member + text[opening + 1 :]


# --- the SVG ---------------------------------------------------------------------------------

_NAME = re.compile(rb"<[^ \t\r\n/>]+")
_ATTRIBUTE = re.compile(rb"[ \t\r\n]+([^ \t\r\n=/>]+)[ \t\r\n]*=[ \t\r\n]*(\"[^\"]*\"|'[^']*')")
_REFERENCE = re.compile(r"&(#x[0-9A-Fa-f]+|#[0-9]+|amp|lt|gt|quot|apos);")
_PREDEFINED = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}


def _unescape(value: str) -> str:
    def one(m):
        ref = m.group(1)
        if ref.startswith("#x"):
            return chr(int(ref[2:], 16))
        if ref.startswith("#"):
            return chr(int(ref[1:]))
        return _PREDEFINED[ref]
    return _REFERENCE.sub(one, value)


def _start_tag_attributes(svg: bytes, offset: int):
    """(raw name, value start, value end) of each attribute in the start
    tag at `offset`, the quotes left out."""

    match = _NAME.match(svg, offset)
    pos = match.end()
    out = []
    while True:
        m = _ATTRIBUTE.match(svg, pos)
        if m is None:
            return out
        out.append((m.group(1).decode("utf-8"), m.start(2) + 1, m.end(2) - 1))
        pos = m.end()


def _prefix_namespace(svg: bytes, element: Element, prefix: str) -> str | None:
    declaration = "xmlns:" + prefix
    node = element
    while node is not None:
        for name, start, end in _start_tag_attributes(svg, node.offset):
            if name == declaration:
                return _unescape(svg[start:end].decode("utf-8"))
        node = node.parent
    return None


def href_span(svg: bytes, element: Element) -> tuple[int, int]:
    """The bytes of the attribute value §1.2 step 2 reads as the image's
    PNG: the `href` without a namespace when it is there and not empty,
    else the `href` in the XLink namespace (whatever its prefix)."""

    attributes = _start_tag_attributes(svg, element.offset)
    plain = element.attributes.get("href")
    for name, start, end in attributes:
        if plain and name == "href":
            return start, end
        if not plain and name.endswith(":href"):
            if _prefix_namespace(svg, element, name.split(":", 1)[0]) == _datauri.XLINK_NS:
                return start, end
    raise ValueError("the image element's href could not be found in the SVG")


def png_data_uri(png: bytes) -> bytes:
    return b"data:image/png;base64," + base64.b64encode(png)


_TAG_END = re.compile(rb"[ \t\r\n]*(/?)>")


def element_span(svg: bytes, element: Element) -> tuple[int, int]:
    """The bytes of an element, from its start tag to the end of its end
    tag (an element whose content holds no element of the same name)."""

    match = _NAME.match(svg, element.offset)
    name = match.group(0)[1:]
    pos = match.end()
    while (m := _ATTRIBUTE.match(svg, pos)) is not None:
        pos = m.end()
    close = _TAG_END.match(svg, pos)
    if close.group(1):
        return element.offset, close.end()
    end = re.compile(rb"</" + re.escape(name) + rb"[ \t\r\n]*>").search(svg, close.end())
    return element.offset, end.end()


def apply(svg: bytes, edits: list[tuple[int, int, bytes]]) -> bytes:
    """Replace each (start, end) byte range with its bytes; every other byte
    stays as it was."""

    out, pos = [], 0
    for start, end, data in sorted(edits, key=lambda e: e[0]):
        if start < pos:
            raise ValueError("the edits overlap")
        out += [svg[pos:start], data]
        pos = end
    out.append(svg[pos:])
    return b"".join(out)


def replace_hrefs(svg: bytes, replacements: list[tuple[Element, bytes]]) -> bytes:
    """Put a new PNG in each element's href; every other byte of the SVG
    stays as it was."""

    return apply(svg, [(*href_span(svg, element), png_data_uri(png)) for element, png in replacements])

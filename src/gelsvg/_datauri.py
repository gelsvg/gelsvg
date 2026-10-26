"""PNG data URIs in SVG `image` elements (§1.2, steps 1 and 2)."""

from __future__ import annotations

import base64
import binascii

XLINK_NS = "http://www.w3.org/1999/xlink"

# RFC 4648 §4 alphabet and the ASCII white space the specification removes
# before decoding (space, tab, LF, CR, FF, VT).
_ASCII_WHITESPACE = b" \t\n\r\f\v"
_WHITESPACE_TABLE = bytes.maketrans(b"", b"")


def image_href(attributes: dict[str, str]) -> str | None:
    """The `href` the specification reads: the one without a namespace, or,
    when that is missing or empty, the one in the XLink namespace. Prefixes
    play no part, so `attributes` is keyed by `"namespace local"` or plain
    `"local"`."""

    value = attributes.get("href")
    if value:
        return value
    value = attributes.get(f"{XLINK_NS} href")
    if value:
        return value
    return None


NOT_PNG = "not_png"
"""The value is not a base64 `image/png` data URI."""
UNDECODABLE = "undecodable"
"""It is one, but its base64 cannot be decoded."""


def decode_png_data_uri(value: str) -> bytes | str:
    """The bytes of a base64 `image/png` data URI, or NOT_PNG or UNDECODABLE
    (either way it is not a panel)."""

    if len(value) < 5 or value[:5].lower() != "data:":
        return NOT_PNG
    comma = value.find(",")
    if comma < 0:
        return NOT_PNG
    header = value[5:comma]
    parts = [part.strip(" \t\n\r\f\v") for part in header.split(";")]
    if not parts or parts[0].lower() != "image/png":
        return NOT_PNG
    if not any(part.lower() == "base64" for part in parts[1:]):
        return NOT_PNG
    try:
        payload = value[comma + 1 :].encode("ascii")
    except UnicodeEncodeError:
        return UNDECODABLE
    payload = payload.translate(_WHITESPACE_TABLE, _ASCII_WHITESPACE)
    if len(payload) % 4:
        return UNDECODABLE
    try:
        # validate=True rejects anything outside the alphabet and any
        # character after the padding; unused bits that are not zero are
        # accepted, as the specification allows.
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return UNDECODABLE

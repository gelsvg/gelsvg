"""The input of `svg_hash` (§6.5, method `c14n2_without_panel_data`): the SVG
as Canonical XML 2.0, without the panels' own data."""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET

from . import _carrier, _datauri, _hashes, _png

_HREF_KEYS = ("href", "{http://www.w3.org/1999/xlink}href")


class Unavailable(Exception):
    """The input cannot be built, so no `svg_hash` can be checked."""


def png_without_panel_data(png: bytes) -> bytes:
    """The PNG rebuilt from the chunks §1.2 step 3 reaches, without `geLs`
    chunks and the GelSVG iTXt, bytes kept as they are."""

    walk = _png.walk(png)
    if not walk.signature_ok:
        raise Unavailable("a PNG data URI holds no PNG")
    out = [_png.PNG_SIGNATURE]
    for chunk in walk.chunks:
        if chunk.runs_past_end:
            break
        if chunk.type == _png.GELS_TYPE:
            continue
        if chunk.type == b"iTXt" and _carrier._keyword_of(chunk) == _carrier.GELSVG_KEYWORD:
            continue
        out.append(png[chunk.offset : chunk.offset + 12 + chunk.length])
    if walk.truncated and walk.final_iend_ok:
        out.append(_png.VALID_IEND)
    return b"".join(out)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class _Filter:
    """A parser target passing the SVG on to the canonical writer, minus the
    old settings metadata and the panels' data. Namespace declarations made
    on a dropped element go with it."""

    def __init__(self, target):
        self._target = target
        self._skip = 0
        self._pending: list[tuple[str, str]] = []
        self._dropped_ns = 0

    def start_ns(self, prefix, uri):
        if self._skip:
            self._dropped_ns += 1
        else:
            self._pending.append((prefix, uri))

    def end_ns(self, prefix):
        if self._dropped_ns:
            self._dropped_ns -= 1

    def start(self, tag, attrs):
        if self._skip:
            self._skip += 1
            return
        if _local(tag) == "metadata" and attrs.get("data-mywb-role") == "settings":
            self._dropped_ns += len(self._pending)
            self._pending = []
            self._skip = 1
            return
        for prefix, uri in self._pending:
            self._target.start_ns(prefix, uri)
        self._pending = []
        if _local(tag) == "image":
            attrs = self._image(attrs)
        self._target.start(tag, attrs)

    def _image(self, attrs):
        key = next((k for k in _HREF_KEYS if attrs.get(k)), None)
        if key is None:
            return attrs
        png = _datauri.decode_png_data_uri(attrs[key])
        if png == _datauri.NOT_PNG:
            return attrs
        if isinstance(png, str):
            raise Unavailable("a PNG data URI cannot be decoded")
        rebuilt = png_without_panel_data(png)
        if rebuilt == png:
            return attrs
        return {**attrs, key: "data:image/png;base64," + base64.b64encode(rebuilt).decode("ascii")}

    def end(self, tag):
        if self._skip:
            self._skip -= 1
            return
        self._target.end(tag)

    def data(self, text):
        if not self._skip:
            self._target.data(text)

    def comment(self, text):
        pass  # Canonical XML 2.0 with its defaults drops comments

    def pi(self, target, text=None):
        if not self._skip:
            self._target.pi(target, text)

    def close(self):
        return None


def svg_hash_input(svg: bytes) -> bytes:
    out: list[str] = []
    parser = ET.XMLParser(target=_Filter(ET.C14NWriterTarget(out.append)), encoding="utf-8")
    parser.feed(svg)
    parser.close()
    return "".join(out).encode("utf-8")


def svg_hash(svg: bytes) -> str:
    return _hashes.sha256(svg_hash_input(svg))

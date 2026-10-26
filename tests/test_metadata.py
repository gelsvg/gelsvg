import hashlib
import struct

import pytest

import builders as B
import panels as P
from gelsvg import Kind, _hashes, _json
from gelsvg._metadata import check_formal
from gelsvg.document import read


def payload(index: dict, data: bytes, canonical: bool = True) -> bytes:
    text = _json.canonical(index) if canonical else _json.canonical(index).replace(b":", b": ")
    return struct.pack(">I", len(text)) + text + data


PNG_INDEX = {"segments": [{"chunk_type": "tEXt", "after_pixels": False, "offset": 0, "length": 5},
                          {"chunk_type": "IEND", "after_pixels": True, "offset": 5, "length": 0}]}


def test_good_png_payload():
    index, problem = check_formal("png", payload(PNG_INDEX, b"hello"))
    assert problem is None and index == PNG_INDEX


@pytest.mark.parametrize("fmt, index, data, problem", [
    ("png", PNG_INDEX, b"hell", "cover"),
    ("png", PNG_INDEX, b"hello!", "cover"),
    ("png", {"segments": [{"chunk_type": "IDAT", "after_pixels": False, "offset": 0, "length": 1}]}, b"x", "chunk type"),
    ("png", {"segments": [{"chunk_type": "tEXt", "after_pixels": 0, "offset": 0, "length": 1}]}, b"x", "after_pixels"),
    ("png", {"segments": [{"chunk_type": "tEXt", "after_pixels": False, "offset": 1, "length": 1}]}, b"xx", "gap"),
    ("png", {"segments": [{"chunk_type": "tEXt", "after_pixels": False, "offset": 0, "length": 1, "x": 1}]}, b"x", "exactly"),
    ("png", {"segments": [], }, b"", "empty"),
    ("png", {"segments": [{"chunk_type": "tEXt", "after_pixels": False, "offset": 0, "length": 1}], "byte_order": "II"}, b"x", "may not have"),
    ("jpeg", {"segments": [{"marker": "SOF0", "offset": 0, "length": 1}]}, b"x", "marker"),
    ("tiff", {"segments": [{"ifd": "ifd0", "tag": 256, "type": 3, "count": 1, "offset": 0, "length": 2}]}, b"xx", "byte_order"),
    ("tiff", {"byte_order": "II", "tiff_version": 42,
              "segments": [{"ifd": "ifd0", "tag": 256, "type": 3, "count": 2, "offset": 0, "length": 2}]}, b"xx", "count"),
    ("tiff", {"byte_order": "II", "tiff_version": 42,
              "segments": [{"ifd": "ifd0", "tag": 256, "type": 16, "count": 1, "offset": 0, "length": 8}]}, b"x" * 8, "TIFF type"),
    ("tiff", {"byte_order": "MM", "tiff_version": 42,
              "segments": [{"ifd": "gps", "tag": 1, "type": 2, "count": 1, "offset": 0, "length": 1},
                           {"ifd": "exif", "tag": 2, "type": 2, "count": 1, "offset": 1, "length": 1}]}, b"xx", "order"),
    ("scn", {"segments": [{"part": "after_pixels", "offset": 0, "length": 1},
                          {"part": "before_pixels", "offset": 1, "length": 1}]}, b"xx", "out of order"),
    ("scn", {"segments": [{"part": "before_pixels", "offset": 0, "length": 0}]}, b"", "empty"),
])
def test_bad_payloads(fmt, index, data, problem):
    index_, why = check_formal(fmt, payload(index, data))
    assert index_ is None and problem in why


def test_index_must_be_canonical():
    assert "canonical" in check_formal("png", payload(PNG_INDEX, b"hello", canonical=False))[1]


def test_index_length_beyond_the_payload():
    assert "exceeds" in check_formal("png", struct.pack(">I", 999) + b"{}")[1]


def test_good_tiff_and_scn():
    tiff = {"byte_order": "II", "tiff_version": 43,
            "segments": [{"ifd": "ifd0", "tag": 271, "type": 2, "count": 3, "offset": 0, "length": 3},
                         {"ifd": "exif", "tag": 36867, "type": 16, "count": 1, "offset": 3, "length": 8}]}
    assert check_formal("tiff", payload(tiff, b"abc" + b"x" * 8))[1] is None
    scn = {"segments": [{"part": "before_pixels", "offset": 0, "length": 2}, {"part": "after_pixels", "offset": 2, "length": 1}]}
    assert check_formal("scn", payload(scn, b"abc"))[1] is None


# --- in a panel ------------------------------------------------------------------------------


def panel_with_metadata(entry_change=None, body: bytes | None = None, revision: int = 0):
    data = payload(PNG_INDEX, b"hello")
    entry = {"format": "png", "version": 1, "byte_length": len(data),
             "payload_hash": "sha256:" + hashlib.sha256(data).hexdigest()}
    if entry_change:
        entry_change(entry)
    key = _hashes.sha256(_json.canonical(entry))

    def change(p):
        p["schema_revision"] = revision
        p["core"]["sources"][0]["source_metadata"] = key
        p["core"]["embedded"]["source_metadata"] = {key: entry}
    panel, channel, rendered = P.base_panel()
    change(panel)
    panel["figure"]["drawn_from"] = _hashes.core_hash(panel["core"])
    png = P.carrier_png(panel, channel, rendered, extra_after=[B.gels(body if body is not None else data, codec=2)])
    (result,) = read(P.svg_of(png)).panels
    return result, key


def test_metadata_in_a_panel():
    panel, key = panel_with_metadata()
    entry = panel.core.source_metadata[key]
    assert entry["index"] == PNG_INDEX and entry["payload"].endswith(b"hello")
    assert not [n for n in panel.notices if n.kind in (Kind.BROKEN, Kind.DAMAGED)]


def test_bad_payload_drops_only_the_metadata():
    data = payload({"segments": [{"chunk_type": "tEXt", "after_pixels": False, "offset": 0, "length": 9}]}, b"hello")
    panel, key = panel_with_metadata(lambda e: e.update(byte_length=len(data),
                                                        payload_hash="sha256:" + hashlib.sha256(data).hexdigest()), body=data)
    assert key not in panel.core.source_metadata
    assert panel.channel_images  # the pixels stay


def test_free_form_metadata_is_kept_unread():
    panel, key = panel_with_metadata(lambda e: e.update(format="https://vendor.example/meta", version=3))
    assert panel.core.source_metadata[key]["readable"] is False
    assert "metadata.free_form" in [n.code for n in panel.notices]


@pytest.mark.parametrize("revision", [0, 1])
@pytest.mark.parametrize("change", [lambda e: e.update(version=2), lambda e: e.update(format="bmp")])
def test_unknown_format_or_version_is_kept_unread_in_any_revision(revision, change):
    panel, key = panel_with_metadata(change, revision=revision)
    assert panel.core.source_metadata[key]["readable"] is False
    assert [n.kind for n in panel.notices if n.code == "metadata.unknown"] == [Kind.UNKNOWN]
    assert not [n for n in panel.notices if n.kind is Kind.BROKEN]

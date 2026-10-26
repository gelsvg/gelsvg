import copy
import json

import numpy as np
import pytest

import builders as B
import panels as P
from gelsvg import Content, Kind, Tool, WriteError, _carrier, _encode, _hashes, _png, _svghash, save
from gelsvg._check import Context
from gelsvg._core import read_core
from gelsvg._embedded import decode_channel_png
from gelsvg.document import read

TOOL = Tool("writer-test", "2.0")
STAMP = "2026-10-26T12:00:00-04:00"


def problems(panel):
    return [n for n in panel.notices if n.kind in (Kind.BROKEN, Kind.DAMAGED, Kind.UNKNOWN, Kind.UNREADABLE,
                                                   Kind.REINTERPRETED)]


def json_of(png: bytes) -> dict:
    walk = _png.walk(png)
    (chunk,) = _carrier.find_keyword(walk, _carrier.GELSVG_KEYWORD)
    fields, _ = _carrier.parse_itxt(chunk.data)
    return json.loads(fields.text)


def png_of(svg: bytes, index: int = 0) -> bytes:
    return read(svg).container.panels[index].carrier.png


def edited(doc, index: int, change) -> Content:
    data = doc.panels[index].to_json()
    change(data)
    return Content(data)


# --- PNG encoding ---------------------------------------------------------------------------------


@pytest.mark.parametrize("shape, dtype", [((1, 1), np.uint8), ((300, 7), np.uint8), ((5, 1), np.uint16),
                                          ((513, 33), np.uint16)])
def test_channel_png_round_trip(shape, dtype):
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
    decoded, why = decode_channel_png(_encode.channel_png(pixels))
    assert why is None and decoded.dtype == dtype and np.array_equal(decoded, pixels)


@pytest.mark.parametrize("depth", [8, 16])
def test_rgb_round_trip(depth):
    pixels = B.small_pixels(270, 9, rgb=True, depth=depth, seed=3)
    header, stream = _encode.encode_image(pixels)
    png = _png.PNG_SIGNATURE + header + _encode.idat_chunks(stream) + _encode.IEND
    walk = _png.walk(png)
    ihdr = _png.Ihdr.parse(walk.chunks[0].data)
    assert np.array_equal(_png.decode_pixels(ihdr, _png.inflate_complete(walk.chunks[1].data)), pixels)


# --- the JSON a rewrite starts from ---------------------------------------------------------------


def test_to_json_of_a_good_panel_is_its_json():
    panel, _, _ = P.base_panel()
    (read_panel,) = read(P.svg_of(P.build())).panels
    expected = copy.deepcopy(panel)
    del expected["core"]["embedded"]
    assert read_panel.to_json() == expected


def test_to_json_leaves_out_what_reading_dropped():
    def change(p):
        p["core"]["bands"].append({"lane": "1", "row": "missing", "shape": "point", "x": 1, "y": 1})
        p["core"]["rows"][0]["colour"] = "red"
    (panel,) = read(P.svg_of(P.build(change))).panels
    data = panel.to_json()
    assert len(data["core"]["bands"]) == 2 and "colour" not in data["core"]["rows"][0]


def test_to_json_settles_what_a_drop_makes_invalid():
    def change(p):
        p["core"]["blot"][0]["color"] = [1, 0, 0]
        p["core"]["blot"].append({"id": "c2", "source": "s1", "source_channel": "gray", "turn": 0, "flip": False,
                                  "invert": False, "contrast": [9, 1], "color": [0, 1, 0]})
        p["core"]["rows"][0]["channel"] = "c1"
    (panel,) = read(P.svg_of(P.build(change))).panels
    core = panel.to_json()["core"]
    assert [e["id"] for e in core["blot"]] == ["c1"] and "channel" not in core["rows"][0]
    ctx = Context(panel=None, newer_revision=False)
    read_core(ctx, core)
    assert ctx.notices == []


# --- saving ---------------------------------------------------------------------------------------


def two_panel_doc(change0=None, change1=None):
    return read(P.svg_of(P.build(change0), P.build(change1)))


def test_saving_without_changes_writes_the_same_bytes():
    doc = two_panel_doc()
    assert save(doc, TOOL).svg == doc.svg


def test_a_changed_panel_is_written_again_and_the_rest_kept():
    doc = two_panel_doc()
    change = edited(doc, 0, lambda d: d["core"]["samples"]["lanes"][0].update(details="KO"))
    saved = save(doc, TOOL, rewrite={0: change}, saved_at=STAMP)
    new = read(saved.svg)
    p = new.panels[0]
    assert not problems(p), [n.message for n in problems(p)]
    assert p.core.lanes[0]["details"] == "KO"
    assert p.written_by == {"name": "writer-test", "version": "2.0", "build": None} and p.saved_at == STAMP
    assert p.checks["drawn_from"] is False  # the figure was not drawn again (§6.4)
    assert p.checks["rendered_hash"] is True and np.array_equal(p.rendered, doc.panels[0].rendered)
    assert len(p.channel_images) == 1 and p.quantifications and p.quantifications[0]["stale"] is False
    assert new.container.panels[1].carrier.png == doc.container.panels[1].carrier.png
    old_uri = B.data_uri(doc.container.panels[0].carrier.png).encode()
    new_uri = B.data_uri(new.container.panels[0].carrier.png).encode()
    assert saved.svg == doc.svg.replace(old_uri, new_uri, 1)


def test_moving_a_band_drops_the_quantification():
    doc = two_panel_doc()
    saved = save(doc, TOOL, rewrite={0: edited(doc, 0, lambda d: d["core"]["bands"][0].update(x=0))})
    assert [n.code for n in saved.notes] == ["quantification.stale"]
    p = read(saved.svg).panels[0]
    assert not problems(p) and p.quantifications == []


def test_a_new_rendered_image():
    doc = two_panel_doc()
    pixels = P.source_pixels()[1:5, 2:7].copy()

    def change(d):
        d["core"]["roi"] = {"x": 2, "y": 1, "w": 5, "h": 4}
        d["figure"]["crop"] = {"x": 2, "y": 1, "w": 5, "h": 4}
    rewrite = edited(doc, 0, change)
    rewrite.rendered = pixels
    p = read(save(doc, TOOL, rewrite={0: rewrite}).svg).panels[0]
    assert not problems(p)
    assert p.checks["drawn_from"] is True and p.checks["rendered_hash"] is True
    assert p.figure["crop"] == {"x": 2, "y": 1, "w": 5, "h": 4} and np.array_equal(p.rendered, pixels)


def test_svg_hash_of_kept_panels_still_matches():
    unsigned = P.svg_of(P.build(), P.build())
    value = _svghash.svg_hash(unsigned)

    def sign(p):
        p["figure"]["svg_hash"] = value
        p["figure"]["svg_hash_method"] = "c14n2_without_panel_data"
    doc = read(P.svg_of(P.build(sign), P.build(sign)))
    assert [p.checks["svg_hash"] for p in doc.panels] == [True, True]
    saved = save(doc, TOOL, rewrite={0: edited(doc, 0, lambda d: d["core"].update(note="checked"))})
    assert "svg_hash.removed" in [n.code for n in saved.notes]
    new = read(saved.svg)
    assert new.panels[0].figure["svg_hash"] is None and new.panels[1].checks["svg_hash"] is True


def test_a_new_channel_image_replaces_the_old_payload():
    doc = two_panel_doc()
    pixels = (P.source_pixels() // 2).astype(np.uint16)

    def change(d):
        source = d["core"]["sources"][0]
        source["dtype"] = "uint16"
        source["channels"][0]["pixel_hash"] = _hashes.pixel_hash(pixels)
    rewrite = edited(doc, 0, change)
    rewrite.channel_images = [pixels]
    saved = save(doc, TOOL, rewrite={0: rewrite})
    new = read(saved.svg)
    p = new.panels[0]
    assert not problems(p), [n.message for n in problems(p)]
    (decoded,) = p.channel_images.values()
    assert decoded.dtype == np.uint16 and np.array_equal(decoded, pixels)
    assert len(new.container.panels[0].carrier.gels) == 1


def test_a_broken_change_is_refused():
    doc = two_panel_doc()
    with pytest.raises(WriteError) as exc:
        save(doc, TOOL, rewrite={0: edited(doc, 0, lambda d: d["core"]["rows"][0].update(name=""))})
    assert exc.value.panel == 0 and [n.code for n in exc.value.notices] == ["key.empty"]


def test_a_newer_revision_needs_allow_loss():
    doc = read(P.svg_of(P.build(lambda p: p.update(schema_revision=3))))
    rewrite = edited(doc, 0, lambda d: d["core"].update(note="x"))
    with pytest.raises(WriteError, match="allow_loss"):
        save(doc, TOOL, rewrite={0: rewrite})
    p = read(save(doc, TOOL, rewrite={0: rewrite}, allow_loss=True).svg).panels[0]
    assert p.schema_revision == 0 and not problems(p)


def test_an_unusable_rendered_image_must_be_drawn_again():
    png = P.build()
    idat = next(c for c in _png.walk(png).chunks if c.type == b"IDAT")
    doc = read(P.svg_of(B.flip_bit(png, idat.offset + 8)))
    with pytest.raises(WriteError, match="rendered image"):
        save(doc, TOOL, rewrite={0: edited(doc, 0, lambda d: None)})
    rewrite = edited(doc, 0, lambda d: None)
    rewrite.rendered = P.source_pixels()
    p = read(save(doc, TOOL, rewrite={0: rewrite}).svg).panels[0]
    assert not problems(p) and p.checks["rendered_hash"] is True


def test_requests_that_cannot_be_met():
    doc = two_panel_doc()
    with pytest.raises(WriteError):
        save(doc, TOOL, rewrite={0: Content(doc.panels[0].to_json())}, assembly={0: None})
    with pytest.raises(WriteError):
        save(doc, TOOL, assembly={5: None})
    with pytest.raises(WriteError):
        save(doc, TOOL, assembly={0: {"id": "a", "member": 2, "members": 1}})


def test_an_xlink_href_under_another_prefix():
    png = P.build()
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:x="http://www.w3.org/1999/xlink">'
           f'<image href="" x:href="{B.data_uri(png)}"/></svg>').encode()
    doc = read(svg)
    p = read(save(doc, TOOL, rewrite={0: edited(doc, 0, lambda d: d["core"].update(note="x"))}).svg).panels[0]
    assert p.core.note == "x"


# --- the assembly-only rewrite --------------------------------------------------------------------


def assembled(record, compressed: bool = False) -> bytes:
    panel, channel_png, rendered = P.base_panel()
    if record is not None:
        panel["assembly"] = record
    text = json.dumps(panel).encode()
    return B.image_png(rendered, before_idat=[B.itxt(text, compressed=compressed)], after_idat=[B.gels(channel_png)])


def chunks_but_json(png: bytes) -> list[bytes]:
    walk = _png.walk(png)
    return [png[c.offset : c.offset + 12 + c.length] for c in walk.chunks if c.type != b"iTXt"]


@pytest.mark.parametrize("before, after", [
    ({"id": "a1", "member": 1, "members": 2}, {"id": "b2", "member": 1, "members": 1}),
    ({"id": "a1", "member": 1, "members": 2}, None),
    (None, {"id": "b2", "member": 1, "members": 1}),
])
@pytest.mark.parametrize("compressed", [False, True])
def test_assembly_only_changes_only_the_record(before, after, compressed):
    png = assembled(before, compressed)
    doc = read(P.svg_of(png))
    new_png = png_of(save(doc, TOOL, assembly={0: after}).svg)
    assert chunks_but_json(new_png) == chunks_but_json(png)
    old_json, new_json = json_of(png), json_of(new_png)
    old_json.pop("assembly", None)
    assert new_json.pop("assembly", None) == after and new_json == old_json
    (chunk,) = _carrier.find_keyword(_png.walk(new_png), _carrier.GELSVG_KEYWORD)
    assert _carrier.parse_itxt(chunk.data)[0].compression_flag == (1 if compressed else 0)
    p = read(P.svg_of(new_png)).panels[0]
    assert p.assembly == after and p.saved_at == "2026-10-26T10:00:00-04:00" and not problems(p)


def test_assembly_only_keeps_the_text_bytes():
    png = assembled({"id": "a1", "member": 1, "members": 2})
    text = json.dumps(json_of(png)).encode()
    new_text = json.dumps(json_of(png_of(save(read(P.svg_of(png)), TOOL, assembly={0: None}).svg))).encode()
    assert new_text == text.replace(b', "assembly": {"id": "a1", "member": 1, "members": 2}', b"")


def test_assembly_only_needs_this_version():
    doc = read(P.svg_of(P.build(lambda p: p.update(schema_version=2), rehash=False)))
    with pytest.raises(WriteError, match="version"):
        save(doc, TOOL, assembly={0: None})


def test_an_unchanged_assembly_record_keeps_the_svg():
    png = assembled({"id": "a1", "member": 1, "members": 1})
    doc = read(P.svg_of(png))
    assert save(doc, TOOL, assembly={0: {"id": "a1", "member": 1, "members": 1}}).svg == doc.svg


# --- what a rewrite carries over ------------------------------------------------------------------


def with_metadata(fmt: str = "png", version: int = 1, revision: int = 0):
    import hashlib
    from gelsvg import _json
    from test_metadata import PNG_INDEX, payload
    data = payload(PNG_INDEX, b"hello")
    entry = {"format": fmt, "version": version, "byte_length": len(data),
             "payload_hash": "sha256:" + hashlib.sha256(data).hexdigest()}
    key = _hashes.sha256(_json.canonical(entry))
    panel, channel, rendered = P.base_panel()
    panel["core"]["sources"][0]["source_metadata"] = key
    panel["core"]["embedded"]["source_metadata"] = {key: entry}
    panel["schema_revision"] = revision
    return read(P.svg_of(P.carrier_png(panel, channel, rendered, extra_after=[B.gels(data, codec=2)]))), key, data


@pytest.mark.parametrize("fmt", ["png", "https://vendor.example/meta"])
def test_source_metadata_is_carried_over(fmt):
    doc, key, data = with_metadata(fmt)
    p = read(save(doc, TOOL, rewrite={0: edited(doc, 0, lambda d: d["core"].update(note="x"))}).svg).panels[0]
    assert p.core.sources[0]["source_metadata"] == key and p.core.source_metadata[key]["payload"] == data
    assert not [n for n in problems(p) if n.code != "metadata.free_form"]


def test_metadata_of_a_newer_version_moves_into_this_revision_unread():
    doc, key, data = with_metadata("tiff", version=2, revision=1)
    saved = save(doc, TOOL, rewrite={0: edited(doc, 0, lambda d: d["core"].update(note="x"))}, allow_loss=True)
    p = read(saved.svg).panels[0]
    assert p.schema_revision == 0 and p.core.source_metadata[key]["payload"] == data
    assert [n.code for n in problems(p)] == ["metadata.unknown"]


def test_a_damaged_ancillary_chunk_is_left_out():
    panel, channel, rendered = P.base_panel()
    text = B.chunk(b"tEXt", b"Comment\0hello")
    bad = B.flip_bit(text, len(text) - 1)
    doc = read(P.svg_of(P.carrier_png(panel, channel, rendered, extra_after=[bad])))
    assert "png.crc.other" in [n.code for n in doc.panels[0].notices]
    new_png = png_of(save(doc, TOOL, rewrite={0: edited(doc, 0, lambda d: None)}).svg)
    assert b"tEXt" not in new_png and not problems(read(P.svg_of(new_png)).panels[0])


A = {"id": "a", "member": 1, "members": 1}
A_TEXT = b'{"id":"a","member":1,"members":1}'


@pytest.mark.parametrize("text, value, expected", [
    (b'{"a": 1 ,\n "assembly" : {"x": [1, "}"]} }', None, b'{"a": 1  }'),
    (b'{ "assembly":{"y":"\\"},"} , "b":2}', None, b'{  "b":2}'),
    (b'{"assembly":1}', None, b"{}"),
    (b'{"a":[1,{"assembly":2}] }', None, b'{"a":[1,{"assembly":2}] }'),
    (b"{ }", A, b"{" + b'"assembly":' + A_TEXT + b" }"),
    (b'{"a":[1,2] }', A, b'{"a":[1,2],"assembly":' + A_TEXT + b" }"),
    (b'{"assembly" : null, "b": "\\u00e9"}', A, b'{"assembly" : ' + A_TEXT + b', "b": "\\u00e9"}'),
])
def test_replacing_a_top_level_member(text, value, expected):
    from gelsvg._splice import replace_top_member
    assert replace_top_member(text, "assembly", value) == expected


def test_a_channel_image_the_panel_holds_keeps_its_bytes():
    doc = two_panel_doc()
    rewrite = edited(doc, 0, lambda d: None)
    rewrite.channel_images = [P.source_pixels()]
    new_png = png_of(save(doc, TOOL, rewrite={0: rewrite}).svg)
    (gels,) = _png.walk(new_png).gels
    assert gels.payload == B.image_png(P.source_pixels())

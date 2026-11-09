import hashlib
import uuid

import numpy as np
import pytest

import builders as B
import panels as P
from gelsvg import (Content, Kind, Tool, WriteError, _check, _hashes, _json, image_element, new_assembly_id,
                    new_panel, rewrite_panel, sign)
from gelsvg.document import read

TOOL = Tool("writer-test", "2.0", build="abc")
STAMP = "2026-10-26T12:00:00-04:00"


def problems(panel):
    return [n for n in panel.notices if n.kind in (Kind.BROKEN, Kind.DAMAGED, Kind.UNKNOWN, Kind.UNREADABLE,
                                                   Kind.REINTERPRETED)]


def content(change=None, **kwargs) -> Content:
    panel, _, rendered = P.base_panel()
    data = {"core": panel["core"], "quantifications": panel["quantifications"], "figure": {"crop": panel["figure"]["crop"]}}
    del data["core"]["embedded"]
    if change is not None:
        change(data)
    for q in data.get("quantifications", []):
        q["computed_on"] = _hashes.quant_core_hash(data["core"])
    kwargs.setdefault("rendered", rendered)
    kwargs.setdefault("source_channel_images", [P.source_pixels()])
    return Content(data, **kwargs)


def svg_around(*pngs: bytes, extra: str = "") -> bytes:
    images = "".join(image_element(png, x=10 * i, y=0, scale=2) for i, png in enumerate(pngs))
    return f'<svg xmlns="http://www.w3.org/2000/svg"><g>{images}{extra}</g></svg>'.encode()


def test_a_new_panel_reads_without_a_problem():
    png = new_panel(content(), TOOL, saved_at=STAMP)
    (p,) = read(svg_around(png)).panels
    assert not problems(p), [n.message for n in problems(p)]
    assert p.schema_revision == 0 and p.saved_at == STAMP
    assert p.written_by == {"name": "writer-test", "version": "2.0", "build": "abc"}
    assert p.checks["drawn_from"] and p.checks["rendered_hash"] and p.checks["computed_on[0]"]
    (pixels,) = p.source_channel_images.values()
    assert np.array_equal(pixels, P.source_pixels())


def test_the_image_element():
    png = new_panel(content(), TOOL)
    (p,) = read(svg_around(png)).panels
    attributes = p.element.attributes
    assert (attributes["width"], attributes["height"], attributes["preserveAspectRatio"]) == ("16", "12", "xMidYMid meet")


def test_a_source_channel_without_a_payload_is_not_embedded():
    png = new_panel(content(source_channel_images=[]), TOOL)
    (p,) = read(svg_around(png)).panels
    assert not problems(p) and p.core.source_channel_images == {}
    assert p.core.sources[0]["source_channels"][0]["pixel_hash"]


def test_source_metadata_is_embedded():
    from test_metadata import PNG_INDEX, payload
    data = payload(PNG_INDEX, b"hello")
    entry = {"format": "png", "version": 1, "byte_length": len(data),
             "payload_hash": "sha256:" + hashlib.sha256(data).hexdigest()}
    key = _hashes.sha256(_json.canonical(entry))
    png = new_panel(content(lambda d: d["core"]["sources"][0].update(source_metadata=key),
                            source_metadata=[(entry, data)]), TOOL)
    (p,) = read(svg_around(png)).panels
    assert not problems(p) and p.core.source_metadata[key]["index"] == PNG_INDEX


def test_a_coloured_drawn_channel_needs_an_rgb_rendered_image():
    colour = lambda d: d["core"]["channels"][0].update(color=[1, 0, 0])
    with pytest.raises(WriteError, match="form"):
        new_panel(content(colour), TOOL)
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 0] = P.source_pixels()
    (p,) = read(svg_around(new_panel(content(colour, rendered=rgb), TOOL))).panels
    assert not problems(p) and p.checks["rendered_hash"]


def test_what_a_new_panel_cannot_be_written_without():
    with pytest.raises(WriteError, match="rendered image"):
        new_panel(content(rendered=None), TOOL)
    with pytest.raises(WriteError):
        new_panel(content(lambda d: d["core"].pop("orientation")), TOOL)
    with pytest.raises(WriteError):
        new_panel(content(lambda d: d.pop("figure")), TOOL)
    with pytest.raises(WriteError, match="channel image"):
        new_panel(content(source_channel_images=[np.zeros((2, 2, 3), dtype=np.uint8)]), TOOL)


# --- panel channels (§5.6, §5.7, §5.12, §7.1) -------------------------------------------------------


def marker_channel(**keys):
    """A marker channel on the base panel's source channel; a key given None is left out."""

    out = {"id": "k1", "role": "marker", "drawn": False, "source": "s1", "source_channel": "gray", "turn": 0,
           "flip": False, "invert": True, "contrast": [0, 255], **keys}
    return {k: v for k, v in out.items() if v is not None}


def refused(change, code, **kwargs):
    with pytest.raises(WriteError) as caught:
        new_panel(content(change, **kwargs), TOOL)
    assert code in [n.code for n in caught.value.notices], [n.code for n in caught.value.notices]


def test_a_panel_with_a_marker_channel_and_blend_is_written():
    def change(d):
        d["core"]["channels"].append(marker_channel(name="Ladder"))
        d["core"]["blend"] = "add"
    (p,) = read(svg_around(new_panel(content(change), TOOL))).panels
    assert not problems(p), [n.message for n in problems(p)]
    assert [e["id"] for e in p.core.marker_channels()] == ["k1"] and p.core.blend == "add"
    assert p.raw["core"]["reference_channel"] == "c1" and "turn" not in p.raw["core"]["channels"][0]


def test_a_panel_whose_reference_channel_is_not_drawn_is_written():
    def change(d):
        core = d["core"]
        core["channels"][0].update(turn=0, flip=False)
        core["channels"].append(marker_channel(turn=None, flip=None))
        core["reference_channel"] = "k1"
    (p,) = read(svg_around(new_panel(content(change), TOOL))).panels
    assert not problems(p), [n.message for n in problems(p)]
    assert p.core.reference()["id"] == "k1" and [e["id"] for e in p.core.drawn()] == ["c1"]


def test_a_row_on_a_marker_channel_is_refused():
    def change(d):
        d["core"]["channels"].append(marker_channel())
        d["core"]["rows"][0]["channel"] = "k1"
    refused(change, "reference.marker")


@pytest.mark.parametrize("mode", ["lane", "region"])
def test_a_quantification_on_a_marker_channel_is_refused(mode):
    def change(d):
        d["core"]["channels"].append(marker_channel())
        q = d["quantifications"][0]
        q["channel"] = "k1"
        if mode == "region":
            q["mode"] = "region"
            for value in q["values"]:
                del value["lane"]
    refused(change, "reference.marker")


def test_a_panel_with_no_drawn_channel_is_refused():
    refused(lambda d: d["core"]["channels"][0].update(drawn=False), "channels.none_drawn")


@pytest.mark.parametrize("change", [
    lambda d: d["core"].pop("reference_channel"),
    lambda d: d["core"].update(reference_channel="k9"),
])
def test_a_panel_without_its_reference_channel_is_refused(change):
    refused(change, "core.unreadable")


def test_a_row_without_its_channel_is_refused():
    refused(lambda d: d["core"]["rows"][0].pop("channel"), "element.broken")


def test_a_lane_quantification_without_its_channel_is_refused():
    refused(lambda d: d["quantifications"][0].pop("channel"), "element.broken")


def test_two_channels_on_one_source_channel_turned_otherwise_are_written():
    # §5.6 recommends one orientation for one source channel, as a SHOULD: not refused
    png = new_panel(content(lambda d: d["core"]["channels"].append(marker_channel(turn=90))), TOOL)
    assert png.startswith(b"\x89PNG")


def test_two_drawn_channels_without_colours_are_refused():
    refused(lambda d: d["core"]["channels"].append(marker_channel(id="c2", drawn=True, role=None)), "element.broken")


@pytest.mark.parametrize("change", [
    lambda d: d["core"].update(blend="multiply"),
    lambda d: d["core"]["channels"][0].update(role="scale_bar"),
])
def test_values_no_revision_of_this_version_defines_are_refused(change):
    refused(change, "key.invalid")


def test_new_assembly_ids():
    ident = new_assembly_id()
    assert _check.is_id(ident) and uuid.UUID(ident).version == 4 and ident != new_assembly_id()


# --- signing --------------------------------------------------------------------------------------


def two_new_panels():
    return new_panel(content(), TOOL), new_panel(content(lambda d: d["core"].update(note="second")), TOOL)


def test_signed_panels_match_the_svg():
    a, b = two_new_panels()
    svg = svg_around(a, b, extra='<text x="1" y="2">p-ERK</text>')
    signed = sign(svg, [a, b])
    assert signed.notes == []
    doc = read(signed.svg)
    assert [p.checks["svg_hash"] for p in doc.panels] == [True, True]
    assert all(not problems(p) for p in doc.panels)
    new_a, new_b = (slot.carrier.png for slot in doc.container.panels)
    assert signed.svg == svg.replace(B.data_uri(a).encode(), B.data_uri(new_a).encode()).replace(
        B.data_uri(b).encode(), B.data_uri(new_b).encode())


def test_an_svg_edited_after_signing_differs():
    a, b = two_new_panels()
    signed = sign(svg_around(a, b, extra="<text>A</text>"), [a]).svg
    doc = read(signed.replace(b"<text>A</text>", b"<text>B</text>"))
    assert doc.panels[0].checks["svg_hash"] is False and doc.panels[1].figure["svg_hash"] is None


def test_signing_a_png_that_is_not_in_the_svg():
    a, b = two_new_panels()
    with pytest.raises(WriteError):
        sign(svg_around(a), [b])


def test_signing_needs_the_hash_input():
    a, _ = two_new_panels()
    svg = svg_around(a, extra='<image href="data:image/png;base64,abc"/>')
    signed = sign(svg, [a])
    assert signed.svg == svg and [n.code for n in signed.notes] == ["svg_hash.unavailable"]


def test_a_tool_redraws_the_figure_around_a_changed_and_a_kept_panel():
    doc = read(P.svg_of(P.build(), P.build()))
    data = doc.panels[0].to_json()
    data["core"]["rows"][0]["name"] = "p-AKT"
    changed = rewrite_panel(doc.panels[0], Content(data), TOOL)
    kept = doc.panels[1].carrier.png
    signed = sign(svg_around(changed, kept, extra='<text>p-AKT</text>'), [changed]).svg
    new = read(signed)
    assert new.panels[0].checks["svg_hash"] is True and new.panels[0].core.rows[0]["name"] == "p-AKT"
    assert new.container.panels[1].carrier.png == kept


def test_reading_back_does_not_decode_the_source_channel_images_again(monkeypatch):
    from gelsvg import _embedded, _encode
    pixels = P.source_pixels()
    calls = []
    decode = _embedded.decode_channel_png
    monkeypatch.setattr(_embedded, "decode_channel_png", lambda png: calls.append(png) or decode(png))
    new_panel(content(), TOOL)  # from pixels: encoded here, never decoded
    assert calls == []
    png = _encode.channel_png(pixels)
    written = new_panel(content(source_channel_images=[], source_channel_pngs=[png]), TOOL)
    assert calls == [png]  # once, to know its pixels
    (p,) = read(svg_around(written)).panels
    calls.clear()
    rewrite_panel(p, Content(p.to_json()), TOOL)
    assert calls == []  # the panel read it already

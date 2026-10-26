import copy
import json

import numpy as np
import pytest

import builders as B
import panels as P
from gelsvg import Kind
from gelsvg.document import read


def read_one(png: bytes):
    doc = read(P.svg_of(png))
    (panel,) = doc.panels
    return panel


def kinds(panel, *wanted):
    return [n for n in panel.notices if n.kind in wanted]


def codes(panel):
    return [n.code for n in panel.notices]


def problems(panel):
    return kinds(panel, Kind.BROKEN, Kind.DAMAGED, Kind.UNKNOWN, Kind.UNREADABLE, Kind.REINTERPRETED)


def test_good_panel():
    panel = read_one(P.build())
    assert panel.readable and not problems(panel), [str(n) for n in problems(panel)]
    assert panel.checks["drawn_from"] and panel.checks["rendered_hash"] and panel.checks["computed_on[0]"]
    view = panel.core
    (ph,) = view.channel_images
    assert np.array_equal(panel.channel_images[ph], P.source_pixels())
    assert [lane["id"] for lane in view.lanes] == ["1", "n1"]
    assert view.lanes[0]["values"] == {"t1": "WT"} and len(view.bands) == 2
    assert view.markers[0]["size"] == {"value": 50, "unit": "kDa"}
    assert panel.quantifications[0]["stale"] is False


# --- rule 5 and versions ---------------------------------------------------------------


@pytest.mark.parametrize("change, reason", [
    (lambda p: p.update(format="https://example.com/other"), "format"),
    (lambda p: p.update(schema_version="1"), "schema_version"),
    (lambda p: p.pop("core"), "core"),
])
def test_unreadable_panels(change, reason):
    panel = read_one(P.build(change, rehash=False))
    assert not panel.readable and reason in panel.unreadable_reason


def test_json_that_is_not_ijson_makes_the_panel_unreadable():
    panel, png, rendered = P.base_panel()
    text = json.dumps(panel).replace('"note"', '"x"').encode()[:-1] + b', "a": 1, "a": 2}'
    result = read_one(P.carrier_png(panel, png, rendered, text=text))
    assert not result.readable and "twice" in result.unreadable_reason


def test_newer_version_is_not_read():
    panel = read_one(P.build(lambda p: p.update(schema_version=2), rehash=False))
    assert panel.newer_version and panel.core is None
    assert codes(panel) == ["panel.newer_version"]


def test_missing_revision_is_read_as_newer():
    panel = read_one(P.build(lambda p: p.pop("schema_revision")))
    assert panel.newer_revision and "top.schema_revision" in codes(panel)


@pytest.mark.parametrize("revision, kind", [(0, Kind.BROKEN), (1, Kind.UNKNOWN)])
def test_unknown_keys_depend_on_the_revision(revision, kind):
    def change(p):
        p["schema_revision"] = revision
        p["core"]["future"] = 1
    panel = read_one(P.build(change))
    (notice,) = kinds(panel, Kind.BROKEN, Kind.UNKNOWN)
    assert notice.kind is kind and notice.where == "core.future"


@pytest.mark.parametrize("revision, kept", [(0, False), (1, True)])
def test_unknown_band_shape(revision, kept):
    def change(p):
        p["schema_revision"] = revision
        p["core"]["bands"][1] = {"lane": "n1", "row": "r1", "shape": "ellipse", "cx": 1}
    panel = read_one(P.build(change))
    shapes = [b["shape"] for b in panel.core.bands]
    assert shapes == (["rect", None] if kept else ["rect"])


# --- rule 3 --------------------------------------------------------------------------------


def test_null_and_empty_optional_keys_are_absent():
    def change(p):
        p["core"]["note"] = None
        p["core"]["annotation"] = ""
    panel = read_one(P.build(change))
    assert panel.core.note is None and panel.core.annotation is None
    assert codes(panel).count("key.null") == 1 and codes(panel).count("key.empty") == 1


def test_reversed_contrast_breaks_the_reference_channel():
    panel = read_one(P.build(lambda p: p["core"]["blot"][0].update(contrast=[200, 10])))
    assert panel.core.blot is None and "blot" in panel.core.unreadable


def test_second_blot_element_needs_a_colour_and_rows_need_a_channel():
    def change(p):
        p["core"]["blot"].append({"id": "c2", "source": "s1", "source_channel": "gray", "turn": 0, "flip": False,
                                  "invert": False, "contrast": [0, 255], "color": [0, 1, 0]})
    panel = read_one(P.build(change))
    # two elements: the first now needs a colour too, so the blot cannot be read
    assert panel.core.blot is None
    # and the row, whose channel cannot be checked, is kept as written
    assert [r["id"] for r in panel.core.rows] == ["r1"]


def test_duplicate_lane_id_drops_the_later_lane_and_its_references():
    def change(p):
        p["core"]["samples"]["lanes"].append({"id": "1", "number": 2, "x": 5.0, "y": 1.0})
    panel = read_one(P.build(change))
    assert [lane["id"] for lane in panel.core.lanes] == ["1", "n1"]
    assert [b["lane"] for b in panel.core.bands] == ["n1"]
    assert panel.quantifications[0]["values"] == []
    assert "id.duplicate" in codes(panel)


def test_duplicate_lane_number_drops_the_later_point():
    def change(p):
        p["core"]["samples"]["lanes"].append({"id": "2", "number": 1, "x": 5.0, "y": 1.0, "details": "KO"})
    panel = read_one(P.build(change))
    assert [lane["id"] for lane in panel.core.lanes] == ["1", "n1"]  # lane 2 had no band, so it went
    assert "lane.point" in codes(panel)


def test_half_point_keeps_a_lane_with_bands():
    def change(p):
        lane = p["core"]["samples"]["lanes"][0]
        del lane["y"]
    panel = read_one(P.build(change))
    lane = panel.core.lanes[0]
    assert lane == {"id": "1", "has_point": False}
    assert "lane.point" in codes(panel)


def test_second_band_in_a_cell_is_dropped():
    def change(p):
        p["core"]["bands"].append({"lane": "1", "row": "r1", "shape": "point", "x": 2, "y": 2})
    panel = read_one(P.build(change))
    assert len(panel.core.bands) == 2 and "band.cell" in codes(panel)


def test_cr_is_read_as_lf():
    panel = read_one(P.build(lambda p: p["core"].update(note="a\r\nb\rc")))
    assert panel.core.note == "a\nb\nc" and "text.cr" in codes(panel)


def test_bad_saved_at_and_written_by_are_only_unknown():
    def change(p):
        p["saved_at"] = "2026-10-26 10:00"
        p["written_by"] = {"name": "test"}
    panel = read_one(P.build(change))
    assert panel.readable and panel.saved_at is None and panel.written_by is None
    assert {"top.saved_at", "top.written_by"} <= set(codes(panel))


# --- embedded ---------------------------------------------------------------------------------


def test_channel_image_with_wrong_pixels_is_not_embedded():
    panel, png, rendered = P.base_panel()
    other = B.image_png(P.source_pixels() + 1)
    entry = next(iter(panel["core"]["embedded"]["channel_images"].values()))
    import hashlib
    entry["png_hash"] = "sha256:" + hashlib.sha256(other).hexdigest()
    entry["byte_length"] = len(other)
    from gelsvg import _hashes
    panel["figure"]["drawn_from"] = _hashes.core_hash(panel["core"])
    result = read_one(P.carrier_png(panel, other, rendered))
    assert not result.channel_images and not result.core.channel_images
    assert any("pixel hash" in n.message for n in result.notices)


def test_missing_body_is_a_writer_mistake():
    panel, png, rendered = P.base_panel()
    result = read_one(P.carrier_png(panel, None, rendered))
    assert "embedded.body_missing" in codes(result)


def test_missing_body_after_damage_is_damage():
    panel, png, rendered = P.base_panel()
    carrier = bytearray(P.carrier_png(panel, png, rendered))
    at = carrier.index(b"geLs") + 4 + 44 + 10
    carrier[at] ^= 1  # inside the payload: the chunk is lost
    result = read_one(bytes(carrier))
    assert "embedded.body_damaged" in codes(result)


def test_unreferenced_payload_is_reported():
    panel, png, rendered = P.base_panel()
    result = read_one(P.carrier_png(panel, png, rendered, extra_after=[B.gels(b"stray", codec=2)]))
    assert "embedded.unused" in codes(result)


# --- figure and quantification checks ---------------------------------------------------------


def test_text_change_leaves_the_quantification_current():
    panel = read_one(P.build(lambda p: p["core"].update(note="changed"), rehash=False))
    assert panel.checks["drawn_from"] is False and panel.checks["computed_on[0]"] is True


def test_band_change_makes_both_out_of_date():
    panel = read_one(P.build(lambda p: p["core"]["bands"][0].update(x=1.5), rehash=False))
    assert panel.checks["drawn_from"] is False and panel.checks["computed_on[0]"] is False
    assert panel.quantifications[0]["stale"] is True


def test_changed_rendered_image_is_detected():
    panel, png, rendered = P.base_panel()
    rendered = rendered.copy()
    rendered[0, 0] ^= 1
    result = read_one(P.carrier_png(panel, png, rendered))
    assert result.checks["rendered_hash"] is False


def test_crop_must_match_the_rendered_size():
    panel = read_one(P.build(lambda p: p["figure"]["crop"].update(w=7)))
    assert panel.figure["crop"] is None and "figure.crop" in codes(panel)


def test_crop_outside_a_rotated_aligned_image():
    # 6x8 turned by 30 degrees: W' = 8 cos 30 + 6 sin 30 = 9.93 -> 10, H' = 8 sin 30 + 6 cos 30 = 9.20 -> 9
    def change(p):
        p["core"]["orientation"]["rotation"] = 30.0
        p["figure"]["crop"] = {"x": 3, "y": 3, "w": 8, "h": 6}
    panel = read_one(P.build(change))
    assert panel.figure["crop"] is None and "outside" in " ".join(n.message for n in panel.notices)


# --- assemblies --------------------------------------------------------------------------------


def _member(ident, member, members):
    return P.build(lambda p: p.update(assembly={"id": ident, "member": member, "members": members}))


def _svg(groups):
    """groups: list of lists of carrier PNGs; each inner list goes in its own <g>."""
    body = "".join("<g>" + "".join(f'<image href="{B.data_uri(png)}"/>' for png in group) + "</g>" for group in groups)
    return f'<svg xmlns="http://www.w3.org/2000/svg">{body}</svg>'.encode()


def test_assembly_with_all_members():
    doc = read(_svg([[_member("a", 1, 2), _member("a", 2, 2)]]))
    (assembly,) = doc.assemblies
    assert assembly.panels == {1: 0, 2: 1} and not assembly.missing


def test_copies_in_their_own_groups_are_split():
    doc = read(_svg([[_member("a", 1, 2), _member("a", 2, 2)], [_member("a", 1, 2)]]))
    assert sorted(len(a.panels) for a in doc.assemblies) == [1, 2]
    assert "assembly.copies_split" in [n.code for n in doc.notices]


def test_copies_without_groups_cannot_be_split():
    body = "".join(f'<image href="{B.data_uri(p)}"/>' for p in [_member("a", 1, 2), _member("a", 2, 2), _member("a", 1, 2)])
    doc = read(f'<svg xmlns="http://www.w3.org/2000/svg">{body}</svg>'.encode())
    assert not doc.assemblies and all(p.assembly is None for p in doc.panels)


def test_members_disagree():
    doc = read(_svg([[_member("a", 1, 2), _member("a", 2, 3)]]))
    assert not doc.assemblies and "assembly.members" in [n.code for n in doc.notices]


def test_one_panel_assembly_copies():
    doc = read(_svg([[_member("a", 1, 1)], [_member("a", 1, 1)]]))
    assert len(doc.assemblies) == 2


def test_member_out_of_range_is_absent():
    doc = read(_svg([[_member("a", 3, 2)]]))
    assert not doc.assemblies and "assembly.broken" in [n.code for n in doc.panels[0].notices]

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


# --- hashes of a kind this version does not define (§2.2) ----------------------------------

FOREIGN = "sha512:" + "ab" * 64
BY_REVISION = [(0, Kind.BROKEN), (1, Kind.UNKNOWN)]


def reported(panel):
    return kinds(panel, Kind.BROKEN, Kind.UNKNOWN)


@pytest.mark.parametrize("revision, kind", BY_REVISION)
def test_a_quantification_whose_computed_on_is_of_a_foreign_kind(revision, kind):
    def change(p):
        p["schema_revision"] = revision
        p["quantifications"][0]["computed_on"] = FOREIGN
    panel = read_one(P.build(change, rehash=False))
    (notice,) = reported(panel)
    assert notice.kind is kind
    if revision:  # a newer addition: kept, whether it is stale unknown
        assert panel.quantifications[0]["stale"] is None
    else:  # an error in a required key: the quantification is dropped
        assert panel.quantifications == []


@pytest.mark.parametrize("key", ["rendered_hash", "drawn_from", "svg_hash"])
@pytest.mark.parametrize("revision, kind", BY_REVISION)
def test_a_figure_hash_of_a_foreign_kind_cannot_be_checked(key, revision, kind):
    def change(p):
        p["schema_revision"] = revision
        p["figure"][key] = FOREIGN
        if key == "svg_hash":
            p["figure"]["svg_hash_method"] = "c14n2_without_panel_data"
    panel = read_one(P.build(change, rehash=False))
    (notice,) = reported(panel)
    assert notice.kind is kind and notice.where == f"figure.{key}"
    assert panel.checks[key] is None


@pytest.mark.parametrize("revision, kind", BY_REVISION)
def test_a_file_hash_of_a_foreign_kind(revision, kind):
    def change(p):
        p["schema_revision"] = revision
        p["core"]["sources"][0]["file_hash"] = FOREIGN
    panel = read_one(P.build(change))
    (notice,) = reported(panel)
    assert notice.kind is kind and notice.where.endswith("file_hash")
    assert panel.core.sources[0]["file_hash"] == (FOREIGN if revision else None)


@pytest.mark.parametrize("revision, kind", BY_REVISION)
def test_a_channel_image_keyed_by_a_foreign_hash_is_not_used(revision, kind):
    def change(p):
        p["schema_revision"] = revision
        images = p["core"]["embedded"]["channel_images"]
        (key,) = images
        images[FOREIGN] = images.pop(key)
        p["core"]["sources"][0]["channels"][0]["pixel_hash"] = FOREIGN
    panel = read_one(P.build(change))
    entry = [n for n in panel.notices if n.where == f"core.embedded.channel_images[{FOREIGN!r}]"]
    assert [n.kind for n in entry] == [kind]
    assert panel.channel_images == {}


def test_a_source_metadata_pointer_of_a_foreign_kind_is_an_error():
    panel = read_one(P.build(lambda p: p["core"]["sources"][0].update(source_metadata=FOREIGN)))
    (notice,) = reported(panel)
    assert notice.kind is Kind.BROKEN and notice.code == "key.invalid"
    assert panel.core.sources[0]["source_metadata"] is None


@pytest.mark.parametrize("revision, kind", BY_REVISION)
def test_the_pixel_hash_of_an_average_of_a_foreign_kind(revision, kind):
    from gelsvg._check import Context
    from gelsvg._core import read_core

    core = P.base_panel()[0]["core"]
    del core["embedded"]
    core["sources"][0]["channels"] = [{"name": c, "pixel_hash": "sha256:" + d * 64}
                                      for c, d in (("red", "1"), ("green", "2"), ("blue", "3"))]
    core["blot"][0].update(source_channel="average", pixel_hash=FOREIGN)
    ctx = Context(panel=0, newer_revision=bool(revision))
    view = read_core(ctx, core)
    (notice,) = [n for n in ctx.notices if n.kind in (Kind.BROKEN, Kind.UNKNOWN)]
    assert notice.kind is kind and notice.where == "core.blot[0].pixel_hash"
    assert view.blot[0]["pixel_hash"] == (FOREIGN if revision else None)


@pytest.mark.parametrize("revision, kind", BY_REVISION)
def test_based_on_of_a_foreign_kind(revision, kind):
    def change(p):
        p["schema_revision"] = revision
        p["extensions"] = {"https://example.com/tool": {"based_on": FOREIGN}}
    panel = read_one(P.build(change, rehash=False))
    (notice,) = reported(panel)
    assert notice.kind is kind
    assert panel.extensions["https://example.com/tool"]["based_on"] is None


def test_based_on_that_is_not_a_hash_is_reported():
    panel = read_one(P.build(lambda p: p.update(extensions={"https://example.com/tool": {"based_on": 5}}),
                             rehash=False))
    (notice,) = reported(panel)
    assert notice.kind is Kind.BROKEN


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


def test_a_body_whose_chunk_type_was_damaged_is_damage():
    panel, png, rendered = P.base_panel()
    carrier = bytearray(P.carrier_png(panel, png, rendered))
    carrier[carrier.index(b"geLs")] ^= 0x20  # geLs -> GeLs: it looks critical, and its CRC no longer matches
    result = read_one(bytes(carrier))
    assert "embedded.body_damaged" in codes(result) and "embedded.body_missing" not in codes(result)


def test_unreferenced_payload_is_reported():
    panel, png, rendered = P.base_panel()
    result = read_one(P.carrier_png(panel, png, rendered, extra_after=[B.gels(b"stray", codec=2)]))
    assert "embedded.unused" in codes(result)


# Whether something refers to an embedding is judged on what is written (§5.3, §5.4).

def test_the_payload_of_a_dropped_entry_is_not_reported_again():
    def change(p):
        (entry,) = p["core"]["embedded"]["channel_images"].values()
        entry["compression"] = "none"  # an unknown key in a closed object
    panel = read_one(P.build(change))
    assert [n.code for n in reported(panel)] == ["element.broken"]
    assert panel.channel_images == {}


def test_the_payload_of_an_entry_of_a_newer_kind_is_no_writers_error():
    def change(p):
        p["schema_revision"] = 1
        (entry,) = p["core"]["embedded"]["channel_images"].values()
        entry["kind"] = "future_plane"
    panel = read_one(P.build(change))
    assert [n.kind for n in reported(panel)] == [Kind.UNKNOWN]
    assert panel.channel_images == {}


def test_an_entry_pointed_to_only_by_a_dropped_source_is_not_reported_again():
    def change(p):
        p["core"]["sources"].append(dict(copy.deepcopy(p["core"]["sources"][0]), id="s2", shape=[6, 9]))
        p["core"]["sources"][0]["channels"][0]["pixel_hash"] = "sha256:" + "cd" * 32
        p["core"]["sources"][1]["dtype"] = "int8"  # the only source left pointing to it is broken
    panel = read_one(P.build(change))
    assert [n.where for n in reported(panel)] == ["core.sources[1]"]


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


def _form_notices(panel):
    return [n for n in panel.notices if n.code.endswith("rendered_form")]


def test_a_form_the_writer_got_wrong_is_broken():
    # One element with a colour draws an RGB picture (§6.2); the grey one was written so.
    panel = read_one(P.build(lambda p: p["core"]["blot"][0].update(color=[1, 1, 1])))
    assert panel.checks["rendered_hash"] is True and panel.checks["drawn_from"] is True
    assert [n.kind for n in _form_notices(panel)] == [Kind.BROKEN]


def test_a_form_changed_after_drawing_is_a_check_result():
    panel, png, rendered = P.base_panel()
    result = read_one(P.carrier_png(panel, png, np.stack([rendered] * 3, axis=-1)))
    assert result.checks["rendered_hash"] is False
    assert [n.kind for n in _form_notices(result)] == [Kind.CHECK]


def test_a_size_that_cannot_be_checked_against_rendered_hash_is_broken():
    def change(p):
        p["figure"]["crop"]["w"] = 7
        p["figure"]["rendered_hash"] = "sha512:" + "ab" * 64
    panel = read_one(P.build(change))
    assert panel.checks["rendered_hash"] is None
    assert [n.kind for n in panel.notices if n.where == "figure.crop"] == [Kind.BROKEN]


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

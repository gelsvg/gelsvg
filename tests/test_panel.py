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
    (ph,) = view.source_channel_images
    assert np.array_equal(panel.source_channel_images[ph], P.source_pixels())
    assert [lane["id"] for lane in view.lanes] == ["1", "n1"]
    assert view.lanes[0]["values"] == {"t1": "WT"} and len(view.bands) == 2
    assert view.markers[0]["label"] == "50 kDa" and "size" not in view.markers[0]
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


def test_an_svg_hash_method_without_an_svg_hash_is_ignored():
    # §6.1: there is no hash to take; rule 3.
    panel = read_one(P.build(lambda p: p["figure"].update(svg_hash_method="c14n2_without_panel_data")))
    (notice,) = problems(panel)
    assert notice.kind is Kind.BROKEN and notice.where == "figure.svg_hash_method"
    assert panel.figure["svg_hash_method"] is None and panel.checks.get("svg_hash") is None


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
        images = p["core"]["embedded"]["source_channel_images"]
        (key,) = images
        images[FOREIGN] = images.pop(key)
        p["core"]["sources"][0]["source_channels"][0]["pixel_hash"] = FOREIGN
    panel = read_one(P.build(change))
    entry = [n for n in panel.notices if n.where == f"core.embedded.source_channel_images[{FOREIGN!r}]"]
    assert [n.kind for n in entry] == [kind]
    assert panel.source_channel_images == {}


def test_a_source_metadata_pointer_of_a_foreign_kind_is_an_error():
    panel = read_one(P.build(lambda p: p["core"]["sources"][0].update(source_metadata=FOREIGN)))
    (notice,) = reported(panel)
    assert notice.kind is Kind.BROKEN and notice.code == "key.invalid"
    assert panel.core.sources[0]["source_metadata"] is None


@pytest.mark.parametrize("revision, kind", BY_REVISION)
def test_an_average_element_has_no_pixel_hash(revision, kind):
    # §4.4: the pixels of `average` follow from red, green and blue; an
    # element's `pixel_hash` is a key no revision 0 defines (rule 1).
    from gelsvg._check import Context
    from gelsvg._core import read_core

    core = P.base_panel()[0]["core"]
    del core["embedded"]
    core["sources"][0]["source_channels"] = [{"name": c, "pixel_hash": "sha256:" + d * 64}
                                      for c, d in (("red", "1"), ("green", "2"), ("blue", "3"))]
    core["channels"][0].update(source_channel="average", pixel_hash="sha256:" + "4" * 64)
    ctx = Context(panel=0, newer_revision=bool(revision))
    view = read_core(ctx, core)
    (notice,) = [n for n in ctx.notices if n.kind in (Kind.BROKEN, Kind.UNKNOWN)]
    assert notice.kind is kind and notice.where == "core.channels[0].pixel_hash"
    assert "pixel_hash" not in view.channels[0] and view.channels[0]["source_channel"] == "average"


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
        p["core"]["annotations"] = {}
    panel = read_one(P.build(change))
    assert panel.core.note is None and panel.core.annotations == {}
    assert codes(panel).count("key.null") == 1 and codes(panel).count("key.empty") == 1


def test_annotations_are_read_side_by_side():
    def change(p):
        p["core"]["annotations"] = {"top": "IP: FLAG", "left": "", "right": 3, "center": "x", "bottom": "a\r\nb"}
    panel = read_one(P.build(change))
    assert panel.core.annotations == {"top": "IP: FLAG", "bottom": "a\nb"}
    assert list(panel.core.annotations) == ["top", "bottom"]
    assert codes(panel).count("key.empty") == 1 and codes(panel).count("key.invalid") == 1
    assert [n.where for n in panel.notices if n.code == "key.undefined"] == ["core.annotations.center"]
    assert panel.to_json()["core"]["annotations"] == {"top": "IP: FLAG", "bottom": "a\nb"}


def test_the_unit_of_the_marker_labels_is_read_and_written_back():
    panel = read_one(P.build(lambda p: p["core"].update(marker_unit="kDa")))
    assert panel.core.marker_unit == "kDa" and not problems(panel)
    assert panel.to_json()["core"]["marker_unit"] == "kDa"


@pytest.mark.parametrize("value, code", [("", "key.empty"), (["kDa"], "key.invalid"), (None, "key.null")])
def test_a_marker_unit_that_is_not_text_is_absent(value, code):
    panel = read_one(P.build(lambda p: p["core"].update(marker_unit=value)))
    assert panel.core.marker_unit is None and "marker_unit" not in panel.to_json()["core"]
    assert codes(panel).count(code) == 1


def test_annotations_that_are_not_an_object_are_absent():
    panel = read_one(P.build(lambda p: p["core"].update(annotations="anti-p-ERK")))
    assert panel.core.annotations == {} and "annotations" not in panel.to_json()["core"]
    assert codes(panel).count("key.invalid") == 1


# --- panel channels (§5.6, §5.7) ---------------------------------------------------------------


def channel(ident, **keys):
    """A panel channel on the base panel's source channel, undrawn and
    turned as the reference channel is, with `keys` changed (None leaves a
    key out)."""

    out = {"id": ident, "drawn": False, "source": "s1", "source_channel": "gray", "turn": 0, "flip": False,
           "invert": False, "contrast": [0, 255]}
    out.update(keys)
    return {k: v for k, v in out.items() if v is not None}


def with_channels(*extra, change=None, revision=0):
    """The base panel with panel channels added after `c1`, the reference
    channel, and then `change(panel)`."""

    def edit(p):
        p["schema_revision"] = revision
        p["core"]["channels"].extend(copy.deepcopy(extra))
        if change is not None:
            change(p)
    return read_one(P.build(edit))


def where_codes(panel):
    return [(n.code, n.where) for n in problems(panel)]


def core_problems(panel):
    return [n for n in problems(panel) if (n.where or "").startswith(("core", "quantifications"))]


def test_reversed_contrast_breaks_the_only_channel_and_the_reference_channel():
    panel = read_one(P.build(lambda p: p["core"]["channels"][0].update(contrast=[200, 10])))
    view = panel.core
    assert view.channels is None and view.reference_channel is None and view.reference() is None
    assert view.unreadable == {"channels", "reference_channel"}
    assert [n.where for n in kinds(panel, Kind.UNREADABLE)] == ["core.channels", "core.reference_channel"]
    # rule 4: what points to them is kept as written, unchecked
    assert [r["channel"] for r in view.rows] == ["c1"] and len(view.bands) == 2
    assert panel.quantifications[0]["channel"] == "c1"


def test_a_broken_reference_channel_leaves_the_other_channels_readable():
    panel = with_channels(channel("c2"), change=lambda p: p["core"]["channels"][0].update(contrast=[200, 10]))
    view = panel.core
    assert [e["id"] for e in view.channels] == ["c2"] and view.reference_channel is None
    assert view.unreadable == {"reference_channel"}
    # `channels` can be read, so the row and the quantification on the dropped channel go with it (rule 3)
    assert view.rows == [] and view.bands == [] and panel.quantifications == []


@pytest.mark.parametrize("change", [
    lambda core: core.pop("reference_channel"),
    lambda core: core.update(reference_channel=None),
    lambda core: core.update(reference_channel=1),
    lambda core: core.update(reference_channel=""),
    lambda core: core.update(reference_channel="c 1"),
    lambda core: core.update(reference_channel="c9"),
    lambda core: core["channels"][0].update(source="s9"),
], ids=["missing", "null", "number", "empty", "not-an-id", "no-such-channel", "its-element-dropped"])
def test_reference_channel_that_cannot_be_read(change):
    panel = with_channels(channel("c2"), change=lambda p: change(p["core"]))
    view = panel.core
    assert view.reference_channel is None and view.reference() is None
    assert "reference_channel" in view.unreadable and "channels" not in view.unreadable
    assert "core.reference_channel" in [n.where for n in kinds(panel, Kind.UNREADABLE)]
    assert "c2" in view.channel_ids()
    assert "reference_channel" not in panel.to_json()["core"]


def test_a_reference_channel_whose_id_is_written_twice_cannot_be_read():
    panel = with_channels(channel("c1"))
    view = panel.core
    assert view.reference_channel is None and "reference_channel" in view.unreadable
    assert [e["id"] for e in view.channels] == ["c1"] and "id.duplicate" in codes(panel)
    # the ID cannot say which panel channel the row meant (rule 3)
    assert view.ambiguous["channels"] == {"c1"} and view.rows == []


def test_without_a_reference_channel_turn_and_flip_are_read_as_written():
    def change(p):
        del p["core"]["reference_channel"]
        p["core"]["channels"][0].update(turn=90, flip=True)
    panel = with_channels(channel("c2", turn=None, flip=None), change=change)
    assert [(e["id"], e["turn"], e["flip"]) for e in panel.core.channels] == [("c1", 90, True), ("c2", None, None)]
    assert not [n for n in core_problems(panel) if n.where != "core.reference_channel"]
    assert [c.get("turn") for c in panel.to_json()["core"]["channels"]] == [90, None]


def test_the_reference_channel_has_no_turn_and_the_others_need_one():
    panel = with_channels(channel("c2", flip=None), change=lambda p: p["core"]["channels"][0].update(turn=0, flip=False))
    assert [e["id"] for e in panel.core.channels] == ["c1"]
    assert sorted(where_codes(panel)) == [("element.broken", "core.channels[1]"),
                                          ("key.not_allowed", "core.channels[0].flip"),
                                          ("key.not_allowed", "core.channels[0].turn")]
    assert "turn" not in panel.to_json()["core"]["channels"][0]


def test_the_reference_channel_is_found_by_its_id_not_its_place():
    # §5.7: it may be undrawn; the drawn c1 is turned to it.
    def change(p):
        p["core"]["channels"][0].update(turn=0, flip=False)
        p["core"]["reference_channel"] = "c2"
    panel = with_channels(channel("c2", turn=None, flip=None), change=change)
    assert not problems(panel), [str(n) for n in problems(panel)]
    view = panel.core
    assert view.reference()["id"] == "c2" and [e["id"] for e in view.drawn()] == ["c1"]
    out = panel.to_json()["core"]
    assert out["reference_channel"] == "c2" and [("turn" in c) for c in out["channels"]] == [True, False]


def test_one_drawn_channel_needs_no_colour_and_undrawn_ones_may_keep_one():
    panel = with_channels(channel("c2"), channel("c3", color=[0.25, 0, 0]))
    assert not problems(panel), [str(n) for n in problems(panel)]
    assert [e["color"] for e in panel.core.channels] == [None, None, [0.25, 0, 0]]
    assert panel.to_json()["core"]["channels"][2]["color"] == [0.25, 0, 0]


def test_two_drawn_channels_need_a_colour_each():
    panel = with_channels(channel("c2", drawn=True, color=[0, 1, 0]))
    # c1, drawn and without a colour, is dropped; it was the reference channel
    assert [e["id"] for e in panel.core.channels] == ["c2"] and "reference_channel" in panel.core.unreadable
    assert ("element.broken", "core.channels[0]") in where_codes(panel)


def test_two_drawn_channels_with_colours():
    panel = with_channels(channel("c2", drawn=True, color=[0, 1, 0]),
                          change=lambda p: p["core"]["channels"][0].update(color=[1, 0, 1]))
    assert not core_problems(panel), [str(n) for n in core_problems(panel)]
    assert [e["id"] for e in panel.core.drawn()] == ["c1", "c2"]


def test_whether_a_colour_is_needed_counts_the_drawn_channels_as_written():
    # c2 is dropped for its contrast, but it was written drawn: c1 needs a colour still (rule 3)
    panel = with_channels(channel("c2", drawn=True, color=[0, 1, 0], contrast=[9, 1]))
    assert panel.core.drawn_count_as_written == 2
    assert panel.core.channels is None
    assert [w for c, w in where_codes(panel) if c == "element.broken"] == ["core.channels[0]", "core.channels[1]"]


@pytest.mark.parametrize("color", [[1, 0], [0, 2, 0], [0, -1, 0], "red", [0, None, 1], [True, 0, 0]])
def test_a_bad_colour_where_it_is_optional_is_read_as_absent(color):
    panel = with_channels(channel("c2", color=color))
    assert panel.core.channel("c2")["color"] is None
    assert where_codes(panel) == [("key.invalid", "core.channels[1].color")]


def test_a_bad_colour_where_it_is_required_breaks_the_channel():
    panel = with_channels(channel("c2", drawn=True, color=[0, 2, 0]),
                          change=lambda p: p["core"]["channels"][0].update(color=[1, 0, 0]))
    assert [e["id"] for e in panel.core.channels] == ["c1"]
    assert ("element.broken", "core.channels[1]") in where_codes(panel)


def test_no_drawn_channel_is_told_and_the_rest_is_read():
    panel = read_one(P.build(lambda p: p["core"]["channels"][0].update(drawn=False)))
    (notice,) = problems(panel)
    assert notice.code == "channels.none_drawn" and notice.kind is Kind.BROKEN and notice.where == "core.channels"
    view = panel.core
    assert view.reference()["id"] == "c1" and view.drawn() == [] and view.unreadable == set()
    assert [r["channel"] for r in view.rows] == ["c1"] and len(view.bands) == 2
    assert panel.quantifications[0]["stale"] is False and panel.checks["rendered_hash"] is True


@pytest.mark.parametrize("drawn", ["true", 1, None])
def test_drawn_is_required_and_true_or_false(drawn):
    panel = with_channels(channel("c2", drawn=drawn))
    assert [e["id"] for e in panel.core.channels] == ["c1"]
    assert where_codes(panel) == [("element.broken", "core.channels[1]")]


def test_blend_add_is_read_and_written_back():
    panel = read_one(P.build(lambda p: p["core"].update(blend="add")))
    assert panel.core.blend == "add" and not panel.core.blend_unknown and not problems(panel)
    assert panel.to_json()["core"]["blend"] == "add"


def test_no_blend_is_add():
    panel = read_one(P.build())
    assert panel.core.blend is None and not panel.core.blend_unknown
    assert "blend" not in panel.to_json()["core"]


def test_a_blend_of_a_newer_revision_cannot_draw_again_but_the_picture_stays():
    def change(p):
        p["schema_revision"] = 1
        p["core"]["blend"] = "multiply"
    panel = read_one(P.build(change))
    (notice,) = problems(panel)
    assert notice.kind is Kind.UNKNOWN and notice.code == "value.unknown" and notice.where == "core.blend"
    assert panel.core.blend_unknown and panel.core.blend is None
    assert panel.checks["rendered_hash"] is True and panel.rendered is not None
    assert "blend" not in panel.to_json()["core"]  # rule 6: not written back


@pytest.mark.parametrize("revision, blend, code", [
    (0, "multiply", "key.invalid"), (0, 5, "key.invalid"), (0, "", "key.empty"), (0, None, "key.null"),
    (1, 5, "key.invalid"),
])
def test_a_blend_that_is_not_a_value_of_this_revision_is_ignored(revision, blend, code):
    def change(p):
        p["schema_revision"] = revision
        p["core"]["blend"] = blend
    panel = read_one(P.build(change))
    assert where_codes(panel) == [(code, "core.blend")] and problems(panel)[0].kind is Kind.BROKEN
    assert panel.core.blend is None and not panel.core.blend_unknown  # read as absent: `add`


def test_a_marker_channel():
    panel = with_channels(channel("k1", role="marker", invert=True, name="Ladder"))
    assert not problems(panel), [str(n) for n in problems(panel)]
    assert [e["id"] for e in panel.core.marker_channels()] == ["k1"]
    assert panel.to_json()["core"]["channels"][1] == {
        "id": "k1", "name": "Ladder", "role": "marker", "drawn": False, "source": "s1", "source_channel": "gray",
        "turn": 0, "flip": False, "invert": True, "contrast": [0, 255]}


def test_a_drawn_marker_channel_is_drawn_like_any_other():
    panel = with_channels(channel("k1", role="marker", drawn=True, color=[0, 0, 1]),
                          change=lambda p: p["core"]["channels"][0].update(color=[1, 1, 0]))
    assert not core_problems(panel), [str(n) for n in core_problems(panel)]
    assert [e["id"] for e in panel.core.drawn()] == ["c1", "k1"]


def test_a_role_of_a_newer_revision_is_kept_unused():
    panel = with_channels(channel("k1", role="scale_bar"), revision=1,
                          change=lambda p: p["core"]["rows"].append({"id": "r2", "channel": "k1"}))
    (notice,) = problems(panel)
    assert notice.kind is Kind.UNKNOWN and notice.code == "value.unknown" and notice.where == "core.channels[1].role"
    k1 = panel.core.channel("k1")
    assert k1["role"] == "scale_bar" and panel.core.marker_channels() == []
    # neither an ordinary nor a marker channel: rows may point to it, as written
    assert [r["id"] for r in panel.core.rows] == ["r1", "r2"]
    out = panel.to_json()["core"]
    assert "role" not in out["channels"][1] and out["rows"][1] == {"id": "r2", "channel": "k1"}


@pytest.mark.parametrize("role, code", [("scale_bar", "key.invalid"), (1, "key.invalid"), ("", "key.empty"),
                                        (None, "key.null")])
def test_a_role_that_is_not_a_value_of_this_revision_is_ignored(role, code):
    panel = with_channels(dict(channel("k1"), role=role))
    assert where_codes(panel) == [(code, "core.channels[1].role")]
    assert panel.core.channel("k1")["role"] is None and panel.core.marker_channels() == []


def test_a_row_always_names_its_panel_channel():
    # §5.12: even when the panel has one panel channel
    panel = read_one(P.build(lambda p: p["core"]["rows"][0].pop("channel")))
    assert panel.core.rows == [] and panel.core.bands == []
    assert ("element.broken", "core.rows[0]") in where_codes(panel)


@pytest.mark.parametrize("value, code", [("c9", "reference.missing"), (3, "element.broken")])
def test_a_row_on_a_panel_channel_that_cannot_be_used_is_dropped(value, code):
    panel = read_one(P.build(lambda p: p["core"]["rows"][0].update(channel=value)))
    assert panel.core.rows == [] and (code, "core.rows[0]") in where_codes(panel)


def test_rows_on_different_panel_channels_and_a_measured_undrawn_one():
    def change(p):
        p["core"]["rows"].append({"id": "r2", "channel": "c2"})
        p["quantifications"][0]["channel"] = "c2"
    panel = with_channels(channel("c2"), change=change)
    assert not problems(panel), [str(n) for n in problems(panel)]
    assert [r["channel"] for r in panel.core.rows] == ["c1", "c2"]
    assert panel.quantifications[0]["channel"] == "c2"


@pytest.mark.parametrize("mode", ["lane", "region"])
def test_a_lane_or_region_quantification_always_names_its_panel_channel(mode):
    def change(p):
        if mode == "region":
            region_mode(p)
        del p["quantifications"][0]["channel"]
    panel = read_one(P.build(change))
    assert panel.quantifications == [] and where_codes(panel) == [("element.broken", "quantifications[0]")]


def test_a_quantification_on_a_panel_channel_that_does_not_exist_is_dropped():
    panel = read_one(P.build(lambda p: p["quantifications"][0].update(channel="c9")))
    assert panel.quantifications == [] and where_codes(panel) == [("element.broken", "quantifications[0]")]


def band_mode(p):
    q = p["quantifications"][0]
    q["mode"] = "band"
    del q["channel"]
    q["values"] = [{"lane": "1", "row": "r1", "region": {"shape": "rect", "x": 1, "y": 2, "w": 2, "h": 1},
                    "raw": 10, "signal": 9}]


def test_a_band_quantification_measures_each_rows_channel():
    panel = read_one(P.build(band_mode))
    assert not problems(panel) and panel.quantifications[0]["channel"] is None


def test_a_band_quantification_names_no_channel():
    def change(p):
        band_mode(p)
        p["quantifications"][0]["channel"] = "c1"
    panel = read_one(P.build(change))
    assert where_codes(panel) == [("key.not_allowed", "quantifications[0].channel")]
    (q,) = panel.quantifications
    assert q["channel"] is None and "channel" not in panel.to_json()["quantifications"][0]


@pytest.mark.parametrize("revision, told", [(0, (Kind.BROKEN, "quantifications[0]")),
                                            (1, (Kind.UNKNOWN, "quantifications[0].mode"))])
def test_the_channel_of_a_quantification_of_a_mode_not_known_is_not_checked(revision, told):
    # rule 2: a newer mode decides the form of the quantification, `channel` included; it is not used.
    # In this revision the mode is a value error (rule 1): the quantification is dropped, told once.
    def change(p):
        p["schema_revision"] = revision
        p["quantifications"][0]["mode"] = "ratio"
    panel = read_one(P.build(change))
    assert [(n.kind, n.where) for n in problems(panel)] == [told]
    assert panel.quantifications == []


def test_a_row_on_a_marker_channel_is_read_as_written_and_told():
    panel = with_channels(channel("k1", role="marker"), change=lambda p: p["core"]["rows"][0].update(channel="k1"))
    (notice,) = problems(panel)
    assert notice.kind is Kind.BROKEN and notice.code == "reference.marker" and notice.where == "core.rows[0]"
    assert panel.core.rows[0]["channel"] == "k1" and len(panel.core.bands) == 2


@pytest.mark.parametrize("mode", ["lane", "region"])
def test_a_quantification_on_a_marker_channel_is_read_as_written_and_told(mode):
    def change(p):
        if mode == "region":
            region_mode(p)
        p["quantifications"][0]["channel"] = "k1"
    panel = with_channels(channel("k1", role="marker"), change=change)
    (notice,) = problems(panel)
    assert notice.kind is Kind.BROKEN and notice.code == "reference.marker"
    assert notice.where == "quantifications[0].channel"
    assert panel.quantifications[0]["channel"] == "k1" and panel.quantifications[0]["stale"] is False


@pytest.mark.parametrize("keys", [{"turn": 90}, {"flip": True}])
def test_two_channels_on_one_source_channel_turned_otherwise_are_read_as_written(keys):
    # §5.6: a writer should turn them alike; each still says how its image lies, so nothing is unclear
    panel = with_channels(channel("c2", **keys), channel("k1", role="marker", turn=180))
    assert not problems(panel), [str(n) for n in problems(panel)]
    c2 = panel.core.channel("c2")
    assert (c2["turn"], c2["flip"]) == (keys.get("turn", 0), keys.get("flip", False))
    assert panel.core.channel("k1")["turn"] == 180


def test_channels_on_one_source_channel_turned_alike_are_not_told():
    panel = with_channels(channel("c2"), channel("k1", role="marker", invert=True))
    assert not problems(panel), [str(n) for n in problems(panel)]


def test_a_source_lists_its_source_channels_as_source_channels():
    def change(p):
        source = p["core"]["sources"][0]
        source["channels"] = source.pop("source_channels")
    panel = read_one(P.build(change))
    assert panel.core.sources is None and "sources" in panel.core.unreadable


def test_the_catalog_of_embedded_pixels_is_source_channel_images():
    def change(p):
        embedded = p["core"]["embedded"]
        embedded["channel_images"] = embedded.pop("source_channel_images")
    panel = read_one(P.build(change))
    assert panel.source_channel_images == {} and panel.core.source_channel_images == {}
    assert ("key.undefined", "core.embedded.channel_images") in where_codes(panel)


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
    entry = next(iter(panel["core"]["embedded"]["source_channel_images"].values()))
    import hashlib
    entry["png_hash"] = "sha256:" + hashlib.sha256(other).hexdigest()
    entry["byte_length"] = len(other)
    from gelsvg import _hashes
    panel["figure"]["drawn_from"] = _hashes.core_hash(panel["core"])
    result = read_one(P.carrier_png(panel, other, rendered))
    assert not result.source_channel_images and not result.core.source_channel_images
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
        (entry,) = p["core"]["embedded"]["source_channel_images"].values()
        entry["compression"] = "none"  # an unknown key in a closed object
    panel = read_one(P.build(change))
    assert [n.code for n in reported(panel)] == ["element.broken"]
    assert panel.source_channel_images == {}


def test_the_payload_of_an_entry_of_a_newer_kind_is_no_writers_error():
    def change(p):
        p["schema_revision"] = 1
        (entry,) = p["core"]["embedded"]["source_channel_images"].values()
        entry["kind"] = "future_plane"
    panel = read_one(P.build(change))
    assert [n.kind for n in reported(panel)] == [Kind.UNKNOWN]
    assert panel.source_channel_images == {}


def test_an_entry_pointed_to_only_by_a_dropped_source_is_not_reported_again():
    def change(p):
        p["core"]["sources"].append(dict(copy.deepcopy(p["core"]["sources"][0]), id="s2", shape=[6, 9]))
        p["core"]["sources"][0]["source_channels"][0]["pixel_hash"] = "sha256:" + "cd" * 32
        p["core"]["sources"][1]["dtype"] = "int8"  # the only source left pointing to it is broken
    panel = read_one(P.build(change))
    assert [n.where for n in reported(panel)] == ["core.sources[1]"]


# --- figure and quantification checks ---------------------------------------------------------


def test_text_change_leaves_the_quantification_current():
    panel = read_one(P.build(lambda p: p["core"].update(note="changed"), rehash=False))
    assert panel.checks["drawn_from"] is False and panel.checks["computed_on[0]"] is True


def test_a_new_marker_unit_leaves_the_quantification_current():
    # §2.7: text, like the labels it is the unit of.
    panel = read_one(P.build(lambda p: p["core"].update(marker_unit="kDa"), rehash=False))
    assert panel.checks["drawn_from"] is False and panel.checks["computed_on[0]"] is True


def test_drawing_otherwise_leaves_the_quantification_current():
    # §2.7: `drawn`, `blend` and a panel channel's name, invert, contrast and colour only draw
    from gelsvg import _hashes

    def change(p):
        core = p["core"]
        core["channels"].append(channel("c2"))
        p["quantifications"][0]["computed_on"] = _hashes.quant_core_hash(core)
        p["figure"]["drawn_from"] = _hashes.core_hash(core)
        core["blend"] = "add"
        core["channels"][0].update(drawn=False, name="p-ERK", invert=True, contrast=[3, 250])
        core["channels"][1].update(drawn=True, name="total", color=[0, 1, 0])
    panel = read_one(P.build(change, rehash=False))
    assert not problems(panel), [str(n) for n in problems(panel)]
    assert panel.checks["computed_on[0]"] is True and panel.checks["drawn_from"] is False


@pytest.mark.parametrize("change", [
    lambda core: core.update(blend="add"),
    lambda core: core["channels"][1].update(drawn=True),
    lambda core: core["channels"][1].update(name="Ladder", invert=True, contrast=[0, 9], color=[1, 0, 0]),
    lambda core: core["channels"][1].pop("drawn"),
])
def test_what_only_draws_is_not_in_the_hash_for_quantification(change):
    from gelsvg import _hashes

    core = P.base_panel()[0]["core"]
    core["channels"].append(channel("c2"))
    before, whole = _hashes.quant_core_hash(core), _hashes.core_hash(core)
    change(core)
    assert _hashes.quant_core_hash(core) == before and _hashes.core_hash(core) != whole


@pytest.mark.parametrize("change", [
    lambda core: core["channels"][1].update(turn=90),
    lambda core: core["channels"][1].update(flip=True),
    lambda core: core["channels"][1].update(role="marker"),
    lambda core: core["channels"][1].update(source_channel="red"),
    lambda core: core["channels"][1].update(id="c3"),
    lambda core: core.update(reference_channel="c2"),
    lambda core: core["channels"].reverse(),
])
def test_what_a_quantification_may_depend_on_is_in_its_hash(change):
    from gelsvg import _hashes

    core = P.base_panel()[0]["core"]
    core["channels"].append(channel("c2"))
    before = _hashes.quant_core_hash(core)
    change(core)
    assert _hashes.quant_core_hash(core) != before


def region_mode(p):
    q = p["quantifications"][0]
    q["mode"] = "region"
    q["values"] = [{"region": {"shape": "rect", "x": x, "y": 0, "w": 4, "h": 6}, "raw": 100, "signal": 90}
                   for x in (0, 4, 4)]


def test_a_region_quantification_names_no_lane_and_may_repeat_a_region():
    panel = read_one(P.build(region_mode))
    assert not problems(panel), [str(n) for n in problems(panel)]
    (q,) = panel.quantifications
    assert q["mode"] == "region" and q["stale"] is False
    assert [v["region"]["x"] for v in q["values"]] == [0, 4, 4]
    assert all("lane" not in v for v in q["values"])
    assert [v.keys() >= {"region", "raw", "signal"} for v in panel.to_json()["quantifications"][0]["values"]] == [True] * 3
    assert all("lane" not in v for v in panel.to_json()["quantifications"][0]["values"])


def test_a_lane_written_on_a_region_value_is_ignored():
    def change(p):
        region_mode(p)
        p["quantifications"][0]["values"][0].update(lane="1", row="r1")
    panel = read_one(P.build(change))
    assert codes(panel).count("key.not_allowed") == 2
    (q,) = panel.quantifications
    assert len(q["values"]) == 3 and "lane" not in q["values"][0]


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


def test_a_panel_channel_named_by_the_user_keeps_its_quantifications():
    # §5.6: a name the user gave the channel; §2.7: text, which no number of
    # a quantification depends on. The hashes are left as they were before.
    panel = read_one(P.build(lambda p: p["core"]["channels"][0].update(name="p-ERK"), rehash=False))
    assert not problems(panel), [str(n) for n in problems(panel)]
    assert panel.to_json()["core"]["channels"][0]["name"] == "p-ERK"
    assert panel.checks["computed_on[0]"] is True
    # The whole core's hash has it, as it has every row's name.
    assert panel.checks["drawn_from"] is False


def _form_notices(panel):
    return [n for n in panel.notices if n.code.endswith("rendered_form")]


def test_a_form_the_writer_got_wrong_is_broken():
    # One element with a colour draws an RGB picture (§6.2); the grey one was written so.
    panel = read_one(P.build(lambda p: p["core"]["channels"][0].update(color=[1, 1, 1])))
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


# --- figure.style (§6.7) ----------------------------------------------------------------------------

STYLE = {
    "font_family": "Helvetica", "image_scale": 2, "image_border_width": 0,
    "sample_label": {"form": "table", "side": "split", "side_automatic": True, "bottom_from": "t2",
                     "merge_same_values": True, "angle": 45, "font_size": 20},
    "details_list": {"visible": False, "font_size": 12},
    "marker_label": {"side": "mixed", "side_automatic": True, "font_size": 20},
    "row_name": {"side": "right", "side_automatic": False, "font_size": 20, "hidden_rows": ["r2"]},
    "annotation": {"font_size": 20},
}


def styled(style, revision=0, rehash=True, change=None):
    def edit(p):
        p["schema_revision"] = revision
        p["core"]["samples"]["conditions"].append({"id": "t2", "name": "Time"})
        p["core"]["rows"].append({"id": "r2", "name": "total", "channel": "c1"})
        p["figure"]["style"] = copy.deepcopy(style)
        if change is not None:
            change(p)
    return read_one(P.build(edit, rehash=rehash))


def test_a_style_is_read_as_written():
    panel = styled(STYLE)
    assert panel.figure["style"] == STYLE and not problems(panel), [str(n) for n in problems(panel)]
    assert panel.to_json()["figure"]["style"] == STYLE


def test_no_style_is_not_reported():
    panel = read_one(P.build())
    assert panel.figure["style"] is None and not problems(panel)


@pytest.mark.parametrize("key, value", [
    ("font_family", ""), ("image_scale", 0), ("image_border_width", -1), ("image_scale", "2"),
    ("sample_label", "table"), ("details_list", {"visible": "no"}), ("marker_label", {"font_size": -3}),
    ("sample_label", {"form": "table", "merge_same_values": "yes"}),
    ("annotation", {"font_size": None}),
])
def test_a_bad_key_is_read_as_absent_alone(key, value):
    style = {**copy.deepcopy(STYLE), key: value}
    panel = styled(style)
    got = panel.figure["style"]
    assert key not in got or got[key] != value
    assert {k: v for k, v in got.items() if k != key} == {k: v for k, v in STYLE.items() if k != key}
    assert [n.kind for n in problems(panel)] == [Kind.BROKEN]


def test_annotation_has_no_side():
    style = {"annotation": {"side": "left", "font_size": 20}}
    panel = styled(style)
    assert panel.figure["style"] == {"annotation": {"font_size": 20}} and "key.undefined" in codes(panel)


@pytest.mark.parametrize("revision, kind", [(0, Kind.BROKEN), (1, Kind.UNKNOWN)])
def test_an_unknown_form_leaves_what_depends_on_it_unused(revision, kind):
    style = {"sample_label": {"form": "wheel", "side": "split", "bottom_from": "t2", "angle": 30, "font_size": 9}}
    panel = styled(style, revision)
    expected = {"font_size": 9} if revision else {"bottom_from": "t2", "angle": 30, "font_size": 9}
    assert panel.figure["style"] == {"sample_label": expected}
    assert kind in [n.kind for n in problems(panel)]


@pytest.mark.parametrize("sample_label", [
    {"form": "numbers", "side": "split", "bottom_from": "t2"},
    {"form": "table", "side": "split"},
    {"side": "split", "bottom_from": "t2"},
])
def test_split_needs_a_table_and_bottom_from(sample_label):
    panel = styled({"sample_label": sample_label})
    assert "side" not in panel.figure["style"]["sample_label"] and "style.split" in codes(panel)


@pytest.mark.parametrize("sample_label, expected", [
    ({"form": "table", "side": "split", "bottom_from": "t9"}, {"form": "table"}),
    ({"form": "table", "side": "split", "bottom_from": "t1"}, {"form": "table"}),
    ({"form": "table", "side": "top", "bottom_from": "t9"}, {"form": "table", "side": "top"}),
    ({"form": "table", "side": "top", "bottom_from": "t1"}, {"form": "table", "side": "top", "bottom_from": "t1"}),
])
def test_ids_are_checked_when_the_figure_matches_the_core(sample_label, expected):
    panel = styled({"sample_label": sample_label, "row_name": {"hidden_rows": ["r2", "r9"]}})
    assert panel.figure["style"] == {"sample_label": expected, "row_name": {"hidden_rows": ["r2"]}}
    assert "style.hidden_rows" in codes(panel)


def test_ids_are_kept_unchecked_when_the_core_changed_since():
    style = {"sample_label": {"form": "table", "side": "split", "bottom_from": "t9"},
             "row_name": {"hidden_rows": ["r9"]}}
    panel = styled(style, rehash=False)
    assert panel.checks["drawn_from"] is False
    assert panel.figure["style"] == style and not problems(panel)


@pytest.mark.parametrize("marker_label, expected, code", [
    ({"side_automatic": True, "font_size": 9}, {"font_size": 9}, "style.side_automatic"),
    ({"side": "mixed", "font_size": 9}, {"font_size": 9}, "style.mixed"),
    ({"side": "mixed", "side_automatic": False}, None, "style.mixed"),
    ({"side": "left", "side_automatic": "yes"}, {"side": "left"}, "key.invalid"),
])
def test_side_automatic_goes_with_a_side(marker_label, expected, code):
    panel = styled({"marker_label": marker_label})
    assert panel.figure["style"] == ({"marker_label": expected} if expected else None)
    assert code in codes(panel) and len(problems(panel)) == 1, [str(n) for n in problems(panel)]


@pytest.mark.parametrize("revision", [0, 1])
def test_side_automatic_beside_a_side_not_known_is_not_used(revision):
    panel = styled({"row_name": {"side": "upward", "side_automatic": True, "font_size": 9}}, revision)
    assert panel.figure["style"] == {"row_name": {"font_size": 9}}
    assert len(problems(panel)) == 1  # the side alone: unknown (newer revision) or broken


@pytest.mark.parametrize("revision, kind", [(0, Kind.BROKEN), (1, Kind.UNKNOWN)])
def test_row_names_have_no_visible(revision, kind):
    # §6.7: none drawn is every row in `hidden_rows`; `visible` is a key no
    # revision 0 defines (rule 1).
    panel = styled({"row_name": {"visible": False, "hidden_rows": ["r1"]}}, revision=revision)
    assert panel.figure["style"]["row_name"] == {"hidden_rows": ["r1"]}
    assert [n.kind for n in problems(panel)] == [kind]


def test_a_row_hidden_twice():
    panel = styled({"row_name": {"hidden_rows": ["r1", "r2", "r1"]}})
    assert panel.figure["style"]["row_name"]["hidden_rows"] == ["r1", "r2"]
    assert "style.hidden_rows.repeated" in codes(panel)


@pytest.mark.parametrize("revision, kind", [(0, Kind.BROKEN), (1, Kind.UNKNOWN)])
def test_unknown_style_keys(revision, kind):
    panel = styled({"glow": 1, "marker_label": {"tilt": 3, "font_size": 8}}, revision)
    assert panel.figure["style"] == {"marker_label": {"font_size": 8}}
    assert [n.kind for n in problems(panel)] == [kind, kind]


def test_an_empty_style_is_read_as_absent():
    panel = styled({})
    assert panel.figure["style"] is None and "key.empty" in codes(panel)


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


def test_a_panel_keeps_its_name_in_the_set():
    named = P.build(lambda p: p.update(assembly={"id": "a", "member": 1, "members": 2, "name": "HA 0412"}))
    doc = read(_svg([[named, _member("a", 2, 2)]]))
    assert [p.assembly for p in doc.panels] == [{"id": "a", "member": 1, "members": 2, "name": "HA 0412"},
                                                 {"id": "a", "member": 2, "members": 2}]
    assert doc.assemblies and not problems(doc.panels[0])


@pytest.mark.parametrize("name, code", [(3, "key.invalid"), ("", "key.empty"), (None, "key.null")])
def test_a_bad_name_alone_is_ignored(name, code):
    # §1.7: the assembly is not broken by its optional name (rule 3).
    panel = read(_svg([[P.build(lambda p: p.update(assembly={"id": "a", "member": 1, "members": 1, "name": name}))]])).panels[0]
    assert panel.assembly == {"id": "a", "member": 1, "members": 1}
    assert [n.code for n in problems(panel)] == [code]


def _omitted(value, revision=None):
    def edit(p):
        p.update(assembly={"id": "a", "member": 1, "members": 1, "omitted": value})
        if revision is not None:
            p["schema_revision"] = revision
    return read(_svg([[P.build(edit)]])).panels[0]


def test_the_parts_left_out_in_the_set():
    panel = _omitted(["sample_label", "details_list"])
    assert panel.assembly["omitted"] == ["sample_label", "details_list"] and not problems(panel)


def test_the_unit_of_the_marker_labels_may_be_left_out():
    panel = _omitted(["marker_label", "marker_unit"])
    assert panel.assembly["omitted"] == ["marker_label", "marker_unit"] and not problems(panel)


@pytest.mark.parametrize("value, code", [
    ("sample_label", "key.invalid"), (["sample_label", "sample_label"], "key.invalid"), ([3], "key.invalid"),
    (["scale_bar"], "key.invalid"), ([], "key.empty"),
])
def test_bad_parts_left_out_alone_are_ignored(value, code):
    # §1.7: `omitted` alone is read as absent (rule 3); an unknown value of
    # this revision is a value error (rule 1).
    panel = _omitted(value)
    assert panel.assembly == {"id": "a", "member": 1, "members": 1}
    assert [n.code for n in problems(panel)] == [code]


def test_a_part_of_a_newer_revision_is_not_used():
    # §2.6 rule 2: that value only; the rest is read.
    panel = _omitted(["scale_bar", "marker_label"], revision=99)
    assert panel.assembly["omitted"] == ["marker_label"]
    assert "value.unknown" in [n.code for n in panel.notices]


def test_a_name_with_cr_reads_as_lf():
    panel = read(_svg([[P.build(lambda p: p.update(assembly={"id": "a", "member": 1, "members": 1, "name": "a\r\nb"}))]])).panels[0]
    assert panel.assembly["name"] == "a\nb"


@pytest.mark.parametrize("svg", [
    lambda: _svg([[_member("a", 1, 2), _member("a", 2, 2)]]),
    lambda: _svg([[_member("a", 1, 2), _member("a", 2, 2)], [_member("a", 1, 2)]]),
    lambda: _svg([[_member("a", 1, 2), _member("a", 2, 3)]]),
    lambda: _svg([[_member("a", 1, 1)], [_member("a", 1, 1)]]),
    lambda: _svg([[_member("a", 3, 2), P.build()], [_member("b", 2, 2)]]),
    lambda: _svg([[P.build(lambda p: p.update(schema_version=2, assembly={"id": "a", "member": 1, "members": 1}))]]),
])
def test_read_sets_groups_as_read_does(svg):
    from gelsvg import read_sets
    data = svg()
    full, light = read(data), read_sets(data)
    assert [(a.id, a.members, a.panels) for a in light.assemblies] == [(a.id, a.members, a.panels) for a in full.assemblies]
    assert [p.assembly for p in light.panels] == [p.assembly for p in full.panels]
    assert [n.code for n in light.notices] == [n.code for n in full.notices]


@pytest.mark.parametrize("revision, kind", [(0, Kind.BROKEN), (1, Kind.UNKNOWN), (None, Kind.UNKNOWN)])
def test_read_sets_reads_the_revision_as_read_does(revision, kind):
    from gelsvg import read_sets

    def change(p):
        p["assembly"] = {"id": "a", "member": 1, "members": 1, "colour": "red"}
        if revision is None:
            p.pop("schema_revision")
        else:
            p["schema_revision"] = revision
    data = _svg([[P.build(change)]])
    light, full = read_sets(data).panels[0], read(data).panels[0]
    assert [n.kind for n in light.notices] == [kind]
    assert [n.kind for n in full.notices if n.where == "assembly.colour"] == [kind]


# --- overlaps and §5.10, judged as written (rule 3) ---------------------------------------------


def test_a_row_written_twice_drops_both_when_the_first_is_broken():
    # The later r1 is a later element with that ID as written; the first is broken itself.
    def change(p):
        p["core"]["rows"].insert(0, {"id": "r1", "name": "first"})  # no channel: broken
    panel = read_one(P.build(change))
    assert panel.core.rows == [] and panel.core.bands == []
    assert ("id.duplicate", "core.rows[1]") in where_codes(panel)


def test_a_row_written_twice_keeps_the_first_when_both_are_good():
    def change(p):
        p["core"]["rows"].append({"id": "r1", "name": "second", "channel": "c1"})
    panel = read_one(P.build(change))
    assert [r["name"] for r in panel.core.rows] == ["p-ERK"]
    assert panel.core.bands == []  # pointing to an ID written twice: which row is meant cannot be told
    assert ("id.duplicate", "core.rows[1]") in where_codes(panel)


def test_a_later_band_in_a_cell_is_dropped_when_the_first_is_broken():
    def change(p):
        p["core"]["bands"].insert(0, {"lane": "1", "row": "r1", "shape": "rect", "x": 1, "y": 2})  # no w, h
    panel = read_one(P.build(change))
    assert [(b["lane"], b["row"]) for b in panel.core.bands] == [("n1", "r1")]
    assert ("band.cell", "core.bands[1]") in where_codes(panel)


def test_a_later_value_for_a_lane_is_dropped_when_the_first_is_broken():
    def change(p):
        values = p["quantifications"][0]["values"]
        values.insert(0, {"lane": "1", "region": values[0]["region"], "signal": 1})  # no raw: broken
    panel = read_one(P.build(change))
    assert panel.quantifications[0]["values"] == []
    assert ("value.duplicate", "quantifications[0].values[1]") in where_codes(panel)


def test_the_bit_depth_is_judged_on_the_drawn_channels_as_written():
    # §5.10: c2 draws a 16-bit source but is broken; as written, 16 is allowed.
    def change(p):
        p["core"]["sources"].append({"id": "s2", "dtype": "uint16", "shape": [6, 8],
                                     "source_channels": [{"name": "gray"}]})
        p["core"]["channels"][0]["color"] = [1, 0, 0]
        p["core"]["channels"].append(channel("c2", source="s2", drawn=True, color=[0, 1, 0], contrast=[9, 1]))
        p["core"]["output_bit_depth"] = 16
    panel = read_one(P.build(change))
    assert panel.core.output_bit_depth == 16 and "output_bit_depth" not in panel.core.unreadable
    assert "core.output_bit_depth" not in codes(panel)
    assert [e["id"] for e in panel.core.channels] == ["c1"]  # c2 itself is dropped


def test_a_16_bit_depth_over_8_bit_drawn_sources_is_still_wrong():
    def change(p):
        p["core"]["output_bit_depth"] = 16
    panel = read_one(P.build(change))
    assert "output_bit_depth" in panel.core.unreadable and "core.output_bit_depth" in codes(panel)

import copy
import struct

import numpy as np
import pytest

import builders as B
import oldmywb as O
import panels as P
from gelsvg import Kind, Tool, aligned_image, read, render
from gelsvg._metadata import check_formal
from gelsvg.legacy import MYWB_EXTENSION, LegacyError, convert

TOOL = Tool("converter-test", "1.0")


def converted(svg: bytes):
    result = convert(svg, TOOL, saved_at="2026-10-26T12:00:00-04:00")
    doc = read(result.svg)
    for panel in doc.panels:
        problems = [n for n in panel.notices if n.kind in (Kind.BROKEN, Kind.DAMAGED, Kind.UNREADABLE, Kind.UNKNOWN)]
        assert not problems, [n.message for n in problems]
    assert not doc.legacy
    return result, doc


def schema6(change=None, extra=""):
    doc, payloads = O.base()
    if change is not None:
        change(doc, payloads)
    return O.signed(doc, payloads, O.picture(doc), extra)


def test_a_schema6_document():
    result, doc = converted(schema6(extra='<text x="1">p-ERK</text>'))
    (panel,) = doc.panels
    core = panel.to_json()["core"]
    (source,) = core["sources"]
    assert (source["file_name"], source["file_hash"], source["dtype"], source["shape"]) == ("blot.tif", "sha256:" + "ab" * 32,
                                                                                             "uint8", [6, 8])
    assert np.array_equal(next(iter(panel.source_channel_images.values())), P.source_pixels())
    # Appendix B: the blot is the one drawn panel channel, and the reference channel; no marker image, no marker channel
    assert core["channels"] == [{"id": "blot", "drawn": True, "source": "s1", "source_channel": "gray", "invert": False,
                                 "contrast": [0, 255]}]
    assert core["reference_channel"] == "blot" and "blend" not in core
    assert core["orientation"] == {"turn": 0, "flip": False, "rotation": 0}
    assert core["roi"] == {"x": 1, "y": 1, "w": 6, "h": 4} and core["output_bit_depth"] == 8
    assert core["samples"]["lanes"] == [{"id": "1", "number": 1, "x": 2.5, "y": 0.5, "details": "WT\nrep 1"},
                                        {"id": "2", "number": 2, "x": 5.5, "y": 0.5}]
    assert core["rows"] == [{"id": "r1", "name": "p-ERK", "channel": "blot"}]
    assert core["bands"] == [{"lane": "1", "row": "r1", "shape": "rect", "x": 2, "y": 2, "w": 1, "h": 1},
                             {"lane": "2", "row": "r1", "shape": "point", "x": 5, "y": 2}]
    assert all("size" not in label for label in core["markers"])  # the label holds it (§5.14)
    assert (core["annotations"], core["note"]) == ({"right": "line 1\nline 2"}, "a note")
    assert panel.figure["crop"] == {"x": 1, "y": 1, "w": 6, "h": 4}
    assert panel.checks["drawn_from"] and panel.checks["rendered_hash"] and panel.checks["svg_hash"]
    assert np.array_equal(render(core, panel.source_channel_images, panel.figure["crop"]), panel.rendered)
    (meta,) = panel.core.source_metadata.values()
    assert meta["index"]["segments"][0] == {"ifd": "ifd0", "tag": 271, "type": 2, "count": 4, "offset": 0, "length": 4}
    mywb = panel.extensions[MYWB_EXTENSION]
    assert mywb["based_on"] is True
    assert mywb["value"]["converted_from"]["written_by"] == {"name": "MyWB", "version": "0.6.0", "build": "build-1"}
    assert panel.written_by["name"] == "converter-test" and result.panels[0].schema_version == 6


def test_the_marker_image_turned_and_flipped_on_its_own():
    marker = np.arange(48, dtype=np.uint8).reshape(6, 8) * 3

    def change(doc, payloads):
        asset_id, asset, png = O.plane(marker)
        roots = doc["embedded_png_info"]["source_assets"]
        roots["assets"][asset_id] = asset
        roots["bundles"]["sha256:" + "dd" * 32] = {"color_format": "grayscale", "dtype": "uint8", "shape": [6, 8],
                                                   "planes": {"gray": asset_id}}
        roots["role_views"]["marker"] = {"bundle_id": "sha256:" + "dd" * 32, "selected_channel": "gray",
                                         "selected_pixel_sha256": asset["pixel_sha256"]}
        doc["source_image_info"].update(marker_file_name="marker.png", marker_sha256="ef" * 32)
        doc["filter_data_m"].update(rotation_offset=90, flip=1)
        doc["filter_data_b"].update(rotation_offset=90, flip=0)
        doc["roi_info"] = {"roi_x": None, "roi_y": None, "roi_w": None, "roi_h": None}
        payloads[1].append(png)

    doc, payloads = O.base()
    change(doc, payloads)
    picture = np.rot90(P.source_pixels(), k=-1).copy()
    _, read_doc = converted(O.signed(doc, payloads, picture))
    (panel,) = read_doc.panels
    core = panel.to_json()["core"]
    old_display = np.rot90(marker, k=-1)[:, ::-1]  # old: rotation_offset, then flip
    assert np.array_equal(aligned_image(core, panel.source_channel_images, "marker"), old_display)
    (blot, marker_channel) = core["channels"]
    assert blot["id"] == core["reference_channel"] == "blot" and "turn" not in blot
    assert marker_channel["role"] == "marker" and marker_channel["drawn"] is False and marker_channel["turn"] in (0, 180)
    assert core["rows"][0]["channel"] == "blot"  # rows never point to the marker channel
    assert [e["id"] for e in panel.core.marker_channels()] == ["marker"]


def test_the_marker_image_on_the_blots_own_pixels():
    # Appendix B: the same pixels are one source image; the marker channel is turned as the reference channel is
    def change(doc, payloads):
        roots = doc["embedded_png_info"]["source_assets"]
        roots["role_views"]["marker"] = dict(roots["role_views"]["blot"])
        doc["filter_data_m"].update(invert=1, contrast=[0, 100])
    _, read_doc = converted(schema6(change))
    core = read_doc.panels[0].to_json()["core"]
    assert len(core["sources"]) == 1
    assert core["channels"][1] == {"id": "marker", "role": "marker", "drawn": False, "source": "s1",
                                   "source_channel": "gray", "turn": 0, "flip": False, "invert": True,
                                   "contrast": [0, 100]}


@pytest.mark.parametrize("filter_m, turned", [({"rotation_offset": 180}, (180, False)), ({"flip": 1}, (0, True))])
def test_the_marker_image_on_the_blots_own_pixels_turned_otherwise_keeps_its_turn(filter_m, turned):
    # §5.6 recommends one orientation for one source channel (SHOULD); the old document's is kept as it was
    def change(doc, payloads):
        roots = doc["embedded_png_info"]["source_assets"]
        roots["role_views"]["marker"] = dict(roots["role_views"]["blot"])
        doc["filter_data_m"].update(filter_m)
    _, read_doc = converted(schema6(change))
    core = read_doc.panels[0].to_json()["core"]
    assert len(core["sources"]) == 1
    marker = core["channels"][1]
    assert (marker["source"], marker["role"], (marker["turn"], marker["flip"])) == ("s1", "marker", turned)


def table(label, rows, **settings):
    import json
    return json.dumps({"version": 1, "label": label, "rows": rows, "cell_render_mode": "per_lane", **settings})


def test_table_labels():
    def change(doc, payloads):
        doc["misc_info"]["sample_label_type"] = "table"
        doc["sample_label_info"] = {
            "1": {"x_spinbox": 2.5, "y_spinbox": 0.5, "label": table("WT", [["Genotype", "WT"], ["Time", "10"]])},
            "2": {"x_spinbox": 5.5, "y_spinbox": 0.5, "label": table("", [["Strain", "KO"], ["Time", ""], ["", "x"]])},
            "3": {"x_spinbox": 7.0, "y_spinbox": 0.5, "label": "not json"},
        }
        doc["band_info"] = []
    result, doc = converted(schema6(change))
    samples = doc.panels[0].to_json()["core"]["samples"]
    assert samples["conditions"] == [{"id": "t1", "name": "Genotype"}, {"id": "t2", "name": "Time"}, {"id": "t3"}]
    lanes = {lane["id"]: lane for lane in samples["lanes"]}
    assert lanes["1"]["values"] == {"t1": "WT", "t2": "10"} and lanes["1"]["details"] == "WT"
    assert lanes["2"]["values"] == {"t1": "KO", "t3": "x"} and "details" not in lanes["2"]
    assert lanes["3"]["details"] == "not json" and "values" not in lanes["3"]
    notes = " ".join(result.panels[0].notes)
    assert "Strain" in notes and "could not be read" in notes


def test_rotated_labels():
    import json

    def change(doc, payloads):
        doc["misc_info"]["sample_label_type"] = "rotated"
        doc["sample_label_info"] = {
            "1": {"x_spinbox": 2.5, "y_spinbox": 0.5, "label": json.dumps({"version": 1, "label": "WT", "angle": 45})},
            "2": {"x_spinbox": 5.5, "y_spinbox": 0.5,
                  "label": json.dumps({"version": 1, "label": "KO", "angle": 30, "rotated_label": "KO (b)"})},
        }
    _, doc = converted(schema6(change))
    samples = doc.panels[0].to_json()["core"]["samples"]
    assert samples["conditions"] == [{"id": "r1"}]
    assert [(l["details"], l["values"]) for l in samples["lanes"][:2]] == [("WT", {"r1": "WT"}), ("KO", {"r1": "KO (b)"})]


def test_bands_of_lanes_without_labels():
    def change(doc, payloads):
        doc["band_info"] = [{"x": 1, "y": 1, "w": 1, "h": 1, "lane": 10007, "row": 2},
                            {"x": 3, "y": 1, "w": 1, "h": 1, "lane": 10003, "row": 0},
                            {"x": 4, "y": 1, "w": 1, "h": 1, "lane": 3, "row": 0},
                            {"x": 4, "y": 2, "w": 1, "h": 1, "lane": 3, "row": 0},
                            {"x": 4, "y": 2, "w": -1, "h": 1, "lane": 3, "row": 1}]
        doc["band_row_names"] = ["A", "", "C"]
    result, doc = converted(schema6(change))
    core = doc.panels[0].to_json()["core"]
    assert [l["id"] for l in core["samples"]["lanes"]] == ["1", "2", "n2", "n1", "3"]
    assert core["rows"] == [{"id": "r1", "name": "A", "channel": "blot"}, {"id": "r2", "name": "C", "channel": "blot"}]
    assert [(b["lane"], b["row"]) for b in core["bands"]] == [("n2", "r2"), ("n1", "r1"), ("3", "r1")]
    notes = " ".join(result.panels[0].notes)
    assert "second band" in notes and "skipped" in notes


def test_a_schema4_document():
    doc, _ = O.base(4)
    del doc["embedded_png_info"]
    svg = O.schema4(doc, O.picture(doc))
    result, read_doc = converted(svg)
    assert b"data-mywb-role" not in result.svg and b"<text>A</text>" in result.svg
    (panel,) = read_doc.panels
    source = panel.to_json()["core"]["sources"][0]
    assert "pixel_hash" not in source["source_channels"][0] and "dtype" not in source and source["shape"] == [6, 8]
    assert panel.checks["drawn_from"] and panel.checks["rendered_hash"]


def test_a_full_rendered_fallback():
    doc, _ = O.base(4)
    del doc["embedded_png_info"]
    doc["roi_info"] = {"roi_x": -5, "roi_y": 0, "roi_w": 3, "roi_h": 2}
    _, read_doc = converted(O.schema4(doc, P.source_pixels(), fallback="full-rendered"))
    assert read_doc.panels[0].figure["crop"] == {"x": 0, "y": 0, "w": 8, "h": 6}


def test_an_svg_edited_after_the_old_mywb_keeps_no_hash():
    svg = schema6(extra="<text>A</text>").replace(b"<text>A</text>", b"<text>B</text>")
    result, doc = converted(svg)
    assert doc.panels[0].figure["svg_hash"] is None
    assert any("changed after" in n for n in result.panels[0].notes)


def test_an_old_hash_that_cannot_be_checked_is_not_carried():
    doc, payloads = O.base(5)
    doc["svg_integrity"] = {"algorithm": "sha256", "target": "svg_without_mywb_metadata", "hash": "aa" * 32}
    result, read_doc = converted(O.svg_with(O.carrier(doc, payloads, O.picture(doc))))
    assert read_doc.panels[0].figure["svg_hash"] is None and result.panels[0].notes


def test_gelsvg_panels_beside_it_stay_as_they_were():
    gelsvg_png = P.build()
    svg = schema6().replace(b"</g></svg>", b'<image href="' + B.data_uri(gelsvg_png).encode() + b'"/></g></svg>')
    result, doc = converted(svg)
    assert [slot.carrier.png == gelsvg_png for slot in doc.container.panels] == [False, True]


def test_what_cannot_be_converted():
    from gelsvg.legacy import NoOldDocument
    with pytest.raises(NoOldDocument):
        convert(P.svg_of(P.build()), TOOL)
    doc, payloads = O.base()
    doc["schema_version"] = 7
    with pytest.raises(LegacyError, match="schema"):
        convert(O.svg_with(O.carrier(doc, payloads, O.picture(doc))), TOOL)
    doc, payloads = O.base()
    doc["filter_data_b"]["flip"] = 2
    with pytest.raises(LegacyError, match="flip"):
        convert(O.svg_with(O.carrier(doc, payloads, O.picture(doc))), TOOL)


def test_damaged_pixels_and_metadata_are_noted():
    doc, payloads = O.base()
    payloads = {1: [payloads[1][0][:-1] + b"\0"], 2: payloads[2]}
    doc["embedded_png_info"]["source_assets"]["file_metadata"][next(iter(
        doc["embedded_png_info"]["source_assets"]["file_metadata"]))]["units"][1]["offset"] = 9
    result, read_doc = converted(O.svg_with(O.carrier(doc, payloads, O.picture(doc))))
    assert read_doc.panels[0].source_channel_images == {} and read_doc.panels[0].core.source_metadata == {}
    notes = " ".join(result.panels[0].notes)
    assert "not embedded" in notes and "metadata could not be carried over" in notes


def test_the_picture_decides_the_bit_depth():
    doc, payloads = O.base()
    doc["misc_info"]["output_bit_depth"] = "16"
    result, read_doc = converted(O.svg_with(O.carrier(doc, payloads, O.picture(doc))))
    assert read_doc.panels[0].to_json()["core"]["output_bit_depth"] == 8


def test_a_reversed_contrast_and_an_roi_past_the_edge():
    doc, payloads = O.base()
    doc["filter_data_b"]["contrast"] = [255, 0]
    doc["roi_info"] = {"roi_x": -2, "roi_y": 1, "roi_w": 6, "roi_h": 4}
    picture = P.source_pixels()[1:5, 0:4].copy()
    _, read_doc = converted(O.svg_with(O.carrier(doc, payloads, picture)))
    (panel,) = read_doc.panels
    assert panel.to_json()["core"]["channels"][0]["contrast"] == [0, 255]
    assert panel.figure["crop"] == {"x": 0, "y": 1, "w": 4, "h": 4}
    assert np.array_equal(render(panel.to_json()["core"], panel.source_channel_images, panel.figure["crop"]), picture)


def payloads_of(payloads: dict) -> dict:
    return {(codec, O.sha(p)): p for codec, items in payloads.items() for p in items}


def test_the_mapping_alone_for_a_document_held_in_memory():
    from gelsvg.legacy import from_schema6
    doc, payloads = O.base()
    mapped = from_schema6(doc, payloads_of(payloads), converted=False)
    assert "converted_from" not in mapped.extension
    # Appendix B: what §6.7 covers is figure.style, not the extension (§8).
    assert "svg_style" not in mapped.extension and "sample_label_type" not in mapped.extension
    assert mapped.style == {"font_family": "Helvetica", "image_scale": 1.0, "sample_label": {"form": "numbers"}}
    assert mapped.extension["highlight_saturated"] == {"blot": False, "marker": False}
    assert mapped.core["output_bit_depth"] == 8 and len(mapped.source_channel_pngs) == 1 and len(mapped.source_metadata) == 1
    assert [e["id"] for e in mapped.core["channels"]] == ["blot"] and mapped.core["reference_channel"] == "blot"
    assert {row["channel"] for row in mapped.core["rows"]} == {"blot"}
    converted = from_schema6(doc, payloads_of(payloads))
    assert converted.core == mapped.core and converted.extension["converted_from"] == {"schema_version": 6}


def test_a_document_held_in_memory_keeps_its_sides():
    from gelsvg.legacy import from_schema6
    doc, payloads = O.base()
    del doc["misc_info"]["annotation_text"]
    doc["misc_info"]["annotations"] = {"left": "IP: FLAG", "top": "HeLa\\n293T", "bottom": "", "middle": "x"}
    doc["svg_style"]["annotation_pos"] = "right"
    mapped = from_schema6(doc, payloads_of(payloads), converted=False)
    assert mapped.core["annotations"] == {"left": "IP: FLAG", "top": "HeLa\n293T"}
    assert "annotation_pos" not in mapped.extension.get("svg_style", {}) and "annotation" not in mapped.style


@pytest.mark.parametrize("unit, expected", [("kDa", "kDa"), (" (kDa) ", " (kDa) "), ("  ", None), ("", None)])
def test_a_document_held_in_memory_keeps_its_marker_unit(unit, expected):
    from gelsvg.legacy import from_schema6
    doc, payloads = O.base()
    doc["misc_info"]["marker_unit"] = unit
    mapped = from_schema6(doc, payloads_of(payloads), converted=False)
    assert mapped.core.get("marker_unit") == expected


def _side_of(change, picture_size=None):
    from gelsvg.legacy import annotation_side
    doc, _payloads = O.base()
    change(doc)
    return annotation_side(doc, picture_size)


def test_an_old_annotation_goes_where_the_old_mywb_drew_it():
    # The base document: ROI x 1-7, y 1-5; both marker labels left of its middle.
    assert _side_of(lambda d: None) == "right"
    assert _side_of(lambda d: d["svg_style"].update(annotation_pos="left")) == "left"
    assert _side_of(lambda d: d["svg_style"].update(annotation_pos="auto", marker_pos="left")) == "right"
    assert _side_of(lambda d: d["svg_style"].update(marker_pos="right")) == "left"
    for item in O.base()[0]["marker_label_info"].values():
        assert item["x_spinbox"] < 4
    right_of_middle = lambda d: [m.update(x_spinbox=6) for m in d["marker_label_info"].values()]
    assert _side_of(right_of_middle) == "left"
    # Counted against the picture, when it is given, not the ROI.
    assert _side_of(right_of_middle, picture_size=(20, 4)) == "right"


def test_old_marker_labels_off_the_picture_are_not_counted():
    def one_each(doc):
        doc["marker_label_info"]["1"].update(x_spinbox=6)
        doc["marker_label_info"]["2"].update(x_spinbox=2)
    assert _side_of(one_each) == "right"  # a tie goes right
    def above_the_roi(doc):
        one_each(doc)
        doc["marker_label_info"]["2"].update(y_spinbox=0)
    assert _side_of(above_the_roi) == "left"
    def schema3(doc):
        doc["schema_version"] = 3
        doc["misc_info"]["marker_pos"] = "right"
    assert _side_of(schema3) == "left"
    assert _side_of(lambda d: d.update(roi_info={"roi_x": None, "roi_y": None, "roi_w": None, "roi_h": None})) == "right"


@pytest.mark.parametrize("stated, expected", [("8", 8), ("16", 8), (None, 8), ("32", 8)])
def test_output_bit_depth_without_a_picture(stated, expected):
    from gelsvg.legacy import from_schema6
    doc, payloads = O.base()
    if stated is None:
        del doc["misc_info"]["output_bit_depth"]
    else:
        doc["misc_info"]["output_bit_depth"] = stated
    assert from_schema6(doc, payloads_of(payloads), converted=False).core["output_bit_depth"] == expected


def test_a_new_panel_keeps_source_channel_pngs_as_given():
    from gelsvg import Content, default_crop, new_panel, render
    from gelsvg.legacy import from_schema6
    doc, payloads = O.base()
    mapped = from_schema6(doc, payloads_of(payloads), converted=False)
    pixels = {}
    from gelsvg._embedded import decode_channel_png
    from gelsvg import pixel_hash
    for png in mapped.source_channel_pngs:
        image = decode_channel_png(png)[0]
        pixels[pixel_hash(image)] = image
    crop = default_crop(mapped.core)
    png = new_panel(Content({"core": mapped.core, "figure": {"crop": crop}}, rendered=render(mapped.core, pixels, crop),
                            source_channel_pngs=mapped.source_channel_pngs, source_metadata=mapped.source_metadata), TOOL)
    (panel,) = read(B.svg(B.data_uri(png))).panels
    assert not [n for n in panel.notices if n.kind in (Kind.BROKEN, Kind.DAMAGED, Kind.UNREADABLE)]
    (gels,) = [g for g in read(B.svg(B.data_uri(png))).container.panels[0].carrier.gels if g.codec == 1]
    assert gels.payload == payloads[1][0]


# --- Appendix B: the style the old MyWB drew with goes to figure.style ----------------------------------


def mapped_style(change):
    from gelsvg.legacy import from_schema6
    doc, payloads = O.base()
    change(doc)
    return from_schema6(doc, payloads_of(payloads), converted=False)


def test_old_style_settings_move_as_the_old_mywb_read_them():
    def change(doc):
        doc["svg_style"] = {
            "font_family": " 'Arial' ", "image_scale_percent": "150", "image_border_stroke_width": -2,
            "sample_label_font_size": 0.1, "sample_pos": "auto", "sample_details_font_size": "9",
            "hide_sample_details": "yes", "marker_label_font_size": "big", "marker_pos": "left",
            "band_label_font_size": 12, "band_label_pos": "middle", "band_label_visible": 0,
            "annotation_font_size": None, "annotation_pos": "left", "svg_margin_left": 4, "band_tick_style": "dashed",
        }
    mapped = mapped_style(change)
    assert mapped.style == {
        "font_family": "Arial", "image_scale": 1.5, "image_border_width": 0,
        # "auto": where it was drawn is not known here, so no side (Appendix B).
        "sample_label": {"form": "numbers", "font_size": 0.5},
        "details_list": {"font_size": 9.0, "visible": False},
        "marker_label": {"side": "left", "side_automatic": False},  # "big" was drawn with the default: not moved
        # "middle" was drawn automatic: not moved. Names switched off: every row hidden (§6.7 has no `visible`).
        "row_name": {"font_size": 12.0, "hidden_rows": ["r1"]},
    }
    assert mapped.extension["svg_style"] == {"svg_margin_left": 4, "band_tick_style": "dashed"}
    assert mapped.core["annotations"] == {"left": "line 1\nline 2"}


def test_only_the_keys_written_move():
    mapped = mapped_style(lambda doc: (doc.pop("svg_style"), doc["misc_info"].pop("sample_label_type")))
    assert mapped.style == {} and "svg_style" not in mapped.extension
    mapped = mapped_style(lambda doc: doc["misc_info"].update(sample_label_type="spiral"))
    assert "sample_label" not in mapped.style


def test_schema3_positions_come_from_misc_info():
    def change(doc):
        doc["schema_version"] = 3
        doc["misc_info"].update(marker_pos="right", sample_pos="bottom")
    style = mapped_style(change).style
    assert style["marker_label"] == {"side": "right", "side_automatic": False}
    assert style["sample_label"]["side"] == "bottom" and style["sample_label"]["side_automatic"] is False


def _table_doc(doc, divider=None, sample_pos="automatic", readable=True):
    rows = [["Genotype", "WT"], ["Time", "10"], ["Dose", "1"]]
    settings = {} if divider is None else {"row_separator_index": divider}
    doc["misc_info"]["sample_label_type"] = "table"
    doc["svg_style"]["sample_pos"] = sample_pos
    doc["sample_label_info"] = {
        "1": {"x_spinbox": 2.5, "y_spinbox": 0.5, "label": table("WT", rows, **settings) if readable else "x"},
        "2": {"x_spinbox": 5.5, "y_spinbox": 0.5, "label": table("KO", rows, **settings) if readable else "y"},
    }
    doc["band_info"] = []


@pytest.mark.parametrize("divider, sample_pos, expected", [
    (1, "automatic", {"form": "table", "bottom_from": "t2", "side": "split", "side_automatic": True}),
    (0, "auto", {"form": "table", "side": "bottom", "side_automatic": True}),
    (3, "automatic", {"form": "table", "side": "top", "side_automatic": True}),
    (9, "automatic", {"form": "table", "side": "top", "side_automatic": True}),
    (None, "automatic", {"form": "table"}),  # where it was drawn is not known
    (2, "top", {"form": "table", "bottom_from": "t3", "side": "top", "side_automatic": False}),  # kept, not used
    (3, "bottom", {"form": "table", "side": "bottom", "side_automatic": False}),
])
def test_the_table_divider_becomes_the_side(divider, sample_pos, expected):
    mapped = mapped_style(lambda doc: _table_doc(doc, divider, sample_pos))
    assert mapped.style["sample_label"] == {**expected, "merge_same_values": False}  # "per_lane"
    assert all("row_separator_index" not in lane for lane in mapped.extension.get("sample_labels", {}).values())


@pytest.mark.parametrize("modes, merged, shapes", [
    # The leftmost lane within the ROI decides (lane "0" is left of it).
    (("per_lane", "bracket_span", "per_lane"), True, [None, "bracket", None]),
    (("bracket_span", "underline_span", "bracket_span"), True, ["bracket", None, "bracket"]),
    (("underline_span", "per_lane", "bracket_span"), False, [None, None, "bracket"]),
    (("per_lane", None, "per_lane"), None, [None, None, None]),  # unsaid: drawn by the default, not moved
])
def test_merging_same_values_comes_from_the_lane_the_table_was_drawn_by(modes, merged, shapes):
    # Appendix B: merge_same_values in figure.style, a bracket in MyWB's extension.
    import json

    def change(doc):
        _table_doc(doc)
        doc["roi_info"] = {"roi_x": 2, "roi_y": 0, "roi_w": 6, "roi_h": 4}
        doc["sample_label_info"] = {}
        for key, x, mode in zip(("0", "1", "2"), (0.5, 2.5, 5.5), modes):
            label = json.loads(table(key, [["Genotype", "WT"]]))
            label.pop("cell_render_mode")
            if mode is not None:
                label["cell_render_mode"] = mode
            doc["sample_label_info"][key] = {"x_spinbox": x, "y_spinbox": 0.5, "label": json.dumps(label)}
    mapped = mapped_style(change)
    assert mapped.style["sample_label"].get("merge_same_values") is merged
    lanes = mapped.extension.get("sample_labels", {})
    assert [lanes.get(key, {}).get("span_shape") for key in ("0", "1", "2")] == shapes
    assert all("cell_render_mode" not in lane for lane in lanes.values())


def test_a_table_the_old_mywb_could_not_read_was_drawn_as_numbers():
    mapped = mapped_style(lambda doc: _table_doc(doc, 1, readable=False))
    assert mapped.style["sample_label"] == {"form": "numbers"}


@pytest.mark.parametrize("angles, style_angle", [((30, 30), 30.0), ((30, 45), None), ((120, "x"), None),
                                                  (("x", 45), 45.0)])
def test_rotated_angles(angles, style_angle):
    import json

    def change(doc):
        doc["misc_info"]["sample_label_type"] = "rotated"
        doc["sample_label_info"] = {
            str(i + 1): {"x_spinbox": 2.5 + 3 * i, "y_spinbox": 0.5,
                         "label": json.dumps({"version": 1, "label": "L", "angle": a, "anchor": "left"})}
            for i, a in enumerate(angles)}
    mapped = mapped_style(change)
    assert mapped.style["sample_label"].get("angle") == style_angle and mapped.style["sample_label"]["form"] == "rotated"
    lanes = mapped.extension["sample_labels"]
    assert all(lane["anchor"] == "left" for lane in lanes.values())
    assert all(("angle" in lane) == (style_angle is None) for lane in lanes.values())


def test_hidden_rows_become_row_ids():
    def change(doc):
        doc["band_info"] = [{"x": 2, "y": 2, "w": 1, "h": 1, "lane": 1, "row": 0},
                            {"x": 2, "y": 3, "w": 1, "h": 1, "lane": 1, "row": 2}]
        doc["misc_info"]["band_label_hidden_rows"] = [2, 2, 5, "0", 0]
    mapped = mapped_style(change)
    assert mapped.style["row_name"] == {"hidden_rows": ["r2", "r1"]}
    assert "band_label_hidden_rows" not in mapped.extension


def test_a_converted_panel_carries_its_style():
    def change(doc, payloads):
        _table_doc(doc, 1)
        doc["svg_style"]["band_label_visible"] = True
        doc["misc_info"]["band_label_hidden_rows"] = [0]
    _, doc = converted(schema6(change))
    (panel,) = doc.panels
    assert panel.figure["style"]["sample_label"] == {"form": "table", "bottom_from": "t2", "side": "split",
                                                     "side_automatic": True, "merge_same_values": False}
    assert panel.to_json()["figure"]["style"]["font_family"] == "Helvetica"
    mywb = panel.extensions[MYWB_EXTENSION]
    assert mywb["based_on"] is True and "svg_style" not in mywb["value"] and "sample_label_type" not in mywb["value"]

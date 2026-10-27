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
    assert np.array_equal(next(iter(panel.channel_images.values())), P.source_pixels())
    assert core["blot"] == [{"id": "blot", "source": "s1", "source_channel": "gray", "invert": False, "contrast": [0, 255]}]
    assert core["orientation"] == {"turn": 0, "flip": False, "rotation": 0}
    assert core["roi"] == {"x": 1, "y": 1, "w": 6, "h": 4} and core["output_bit_depth"] == 8
    assert core["samples"]["lanes"] == [{"id": "1", "number": 1, "x": 2.5, "y": 0.5, "details": "WT\nrep 1"},
                                        {"id": "2", "number": 2, "x": 5.5, "y": 0.5}]
    assert core["rows"] == [{"id": "r1", "name": "p-ERK"}]
    assert core["bands"] == [{"lane": "1", "row": "r1", "shape": "rect", "x": 2, "y": 2, "w": 1, "h": 1},
                             {"lane": "2", "row": "r1", "shape": "point", "x": 5, "y": 2}]
    assert core["markers"][0]["size"] == {"value": 50, "unit": "kDa"} and "size" not in core["markers"][1]
    assert (core["annotation"], core["note"]) == ("line 1\nline 2", "a note")
    assert panel.figure["crop"] == {"x": 1, "y": 1, "w": 6, "h": 4}
    assert panel.checks["drawn_from"] and panel.checks["rendered_hash"] and panel.checks["svg_hash"]
    assert np.array_equal(render(core, panel.channel_images, panel.figure["crop"]), panel.rendered)
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
    assert np.array_equal(aligned_image(core, panel.channel_images, "marker"), old_display)
    assert core["marker"]["turn"] in (0, 180)


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
    assert core["rows"] == [{"id": "r1", "name": "A"}, {"id": "r2", "name": "C"}]
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
    assert "pixel_hash" not in source["channels"][0] and "dtype" not in source and source["shape"] == [6, 8]
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
    assert read_doc.panels[0].channel_images == {} and read_doc.panels[0].core.source_metadata == {}
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
    assert panel.to_json()["core"]["blot"][0]["contrast"] == [0, 255]
    assert panel.figure["crop"] == {"x": 0, "y": 1, "w": 4, "h": 4}
    assert np.array_equal(render(panel.to_json()["core"], panel.channel_images, panel.figure["crop"]), picture)


def payloads_of(payloads: dict) -> dict:
    return {(codec, O.sha(p)): p for codec, items in payloads.items() for p in items}


def test_the_mapping_alone_for_a_document_held_in_memory():
    from gelsvg.legacy import from_schema6
    doc, payloads = O.base()
    mapped = from_schema6(doc, payloads_of(payloads), converted=False)
    assert "converted_from" not in mapped.extension
    assert mapped.extension["svg_style"] == doc["svg_style"] and mapped.extension["sample_label_type"] == "basic"
    assert mapped.extension["highlight_saturated"] == {"blot": False, "marker": False}
    assert mapped.core["output_bit_depth"] == 8 and len(mapped.channel_pngs) == 1 and len(mapped.source_metadata) == 1
    converted = from_schema6(doc, payloads_of(payloads))
    assert converted.core == mapped.core and converted.extension["converted_from"] == {"schema_version": 6}


@pytest.mark.parametrize("stated, expected", [("8", 8), ("16", 8), (None, 8), ("32", 8)])
def test_output_bit_depth_without_a_picture(stated, expected):
    from gelsvg.legacy import from_schema6
    doc, payloads = O.base()
    if stated is None:
        del doc["misc_info"]["output_bit_depth"]
    else:
        doc["misc_info"]["output_bit_depth"] = stated
    assert from_schema6(doc, payloads_of(payloads), converted=False).core["output_bit_depth"] == expected


def test_a_new_panel_keeps_channel_pngs_as_given():
    from gelsvg import Content, default_crop, new_panel, render
    from gelsvg.legacy import from_schema6
    doc, payloads = O.base()
    mapped = from_schema6(doc, payloads_of(payloads), converted=False)
    pixels = {}
    from gelsvg._embedded import decode_channel_png
    from gelsvg import pixel_hash
    for png in mapped.channel_pngs:
        image = decode_channel_png(png)[0]
        pixels[pixel_hash(image)] = image
    crop = default_crop(mapped.core)
    png = new_panel(Content({"core": mapped.core, "figure": {"crop": crop}}, rendered=render(mapped.core, pixels, crop),
                            channel_pngs=mapped.channel_pngs, source_metadata=mapped.source_metadata), TOOL)
    (panel,) = read(B.svg(B.data_uri(png))).panels
    assert not [n for n in panel.notices if n.kind in (Kind.BROKEN, Kind.DAMAGED, Kind.UNREADABLE)]
    (gels,) = [g for g in read(B.svg(B.data_uri(png))).container.panels[0].carrier.gels if g.codec == 1]
    assert gels.payload == payloads[1][0]

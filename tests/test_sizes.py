"""Reading marker labels as sizes (§5.14 "Reading a size")."""

import pytest

from gelsvg import marker_sizes
from gelsvg.sizes import read_label, read_unit


@pytest.mark.parametrize("label, expected", [
    ("50", (50.0, None)),
    ("50 kDa", (50.0, "kDa")),
    ("  50kd ", (50.0, "kDa")),
    ("1.5 KB", (1.5, "kb")),
    ("1,000 bp", (1000.0, "bp")),
    ("10 kbp", (10.0, "kb")),
    ("250   nt", (250.0, "nt")),
    ("kDa", None), ("(kDa)", None),  # a unit alone is no size: the unit is `marker_unit`
    ("(kDa", None), ("~50", None), ("M", None), ("", None), ("0", None), ("1,00 bp", None),
    ("1,5 kb", None), ("50 kDa*", None), ("50 Da", None),
])
def test_one_label(label, expected):
    assert read_label(label) == expected


@pytest.mark.parametrize("marker_unit, expected", [
    ("kDa", "kDa"), (" (kd) ", "kDa"), ("BP", "bp"), ("kbp", "kb"), ("nt", "nt"),
    ("MW", None), ("kDa.", None), ("( kDa )", None), ("50 kDa", None), (None, None),
])
def test_the_unit_marker_unit_names(marker_unit, expected):
    assert read_unit(marker_unit) == expected


def labels(*texts):
    return [{"id": str(i), "x": 0, "y": i, "label": t} for i, t in enumerate(texts)]


@pytest.mark.parametrize("texts, marker_unit, expected", [
    (("75", "50"), "kDa", [{"value": 75, "unit": "kDa"}, {"value": 50, "unit": "kDa"}]),
    (("75", "50"), None, [{"value": 75, "unit": None}, {"value": 50, "unit": None}]),
    (("75", "50"), "MW", [{"value": 75, "unit": None}, {"value": 50, "unit": None}]),  # names no unit
    (("10 kb", "500 bp"), None, [{"value": 10, "unit": "kb"}, {"value": 500, "unit": "bp"}]),
    (("1.5 kb", "50"), "kDa", [{"value": 1.5, "unit": "kb"}, {"value": 50, "unit": "kDa"}]),  # its own unit kept
    (("kDa", "37"), None, [None, {"value": 37, "unit": None}]),  # a unit among the labels is only text
    (("M", "", "50"), "(kDa)", [None, None, {"value": 50, "unit": "kDa"}]),
])
def test_a_panels_labels(texts, marker_unit, expected):
    assert marker_sizes(labels(*texts), marker_unit) == expected

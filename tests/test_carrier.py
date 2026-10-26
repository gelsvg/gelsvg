import struct
import zlib

import numpy as np
import pytest

import builders as B
from gelsvg import _carrier, _png
from gelsvg.notices import Kind


def assess(png: bytes):
    walk = _png.walk(png)
    return _carrier.assess(png, walk, _carrier.find_keyword(walk, _carrier.GELSVG_KEYWORD))


def codes(carrier):
    return [n.code for n in carrier.notices]


@pytest.mark.parametrize("compressed", [False, True])
def test_good_carrier(compressed):
    c = assess(B.carrier(b'{"a": 1}', compressed=compressed))
    assert c.unreadable is None and c.json_text == b'{"a": 1}'
    assert c.rendered_usable and c.pixel_data is not None and not c.notices


def test_two_json_chunks_make_the_panel_unreadable():
    png = B.image_png(B.small_pixels(), before_idat=[B.itxt(b"{}"), B.itxt(b"{}")])
    assert "more than one" in assess(png).unreadable


def test_damaged_json_chunk_makes_the_panel_unreadable():
    png = B.carrier(b'{"a": 1}')
    off = png.index(b"iTXt")
    c = assess(B.flip_bit(png, off + 12))
    assert c.unreadable and "itxt.damaged" in codes(c)


@pytest.mark.parametrize("field, problem", [
    (b"GelSVG\0\x02\x00\0\0{}", "compression flag"),
    (b"GelSVG\0\x00\x01\0\0{}", "compression method"),
    (b"GelSVG\0\x00\x00\0{}", "translated keyword"),
    (b"GelSVG\0\x01\x00\0\0not zlib", "inflated"),
])
def test_bad_itxt_fields(field, problem):
    png = B.image_png(B.small_pixels(), before_idat=[B.chunk(b"iTXt", field)])
    assert problem in assess(png).unreadable


def test_damaged_idat_keeps_the_json_and_drops_the_rendered_image():
    png = B.carrier(b"{}")
    off = png.index(b"IDAT")
    c = assess(B.flip_bit(png, off + 6))
    assert c.unreadable is None and c.json_text == b"{}"
    assert not c.rendered_usable and "png.crc.critical" in codes(c)


def test_damaged_ancillary_chunk_is_only_that_chunk():
    png = B.carrier(b"{}", after=[B.chunk(b"tEXt", b"k\0v")])
    off = png.index(b"tEXt")
    c = assess(B.flip_bit(png, off + 5))
    assert c.unreadable is None and c.rendered_usable
    assert codes(c) == ["png.crc.other"]


def test_file_cut_inside_idat_reads_the_json():
    png = B.carrier(b"{}")
    cut = png[: png.index(b"IDAT") + 8]
    c = assess(cut)
    assert c.unreadable is None and c.json_text == b"{}"
    assert not c.rendered_usable
    assert "png.truncated" in codes(c)


def test_file_cut_at_a_chunk_boundary_before_iend():
    png = B.carrier(b"{}")
    c = assess(png[: -12])
    assert c.unreadable is None and c.rendered_usable
    assert codes(c) == ["png.truncated"]


@pytest.mark.parametrize("make, problem", [
    (lambda: B.carrier(b"{}") + b"x", "bytes after IEND"),
    (lambda: B.image_png(B.small_pixels(), before_idat=[B.itxt(b"{}")],
                         raw=zlib.compress(B.scanlines(B.small_pixels())) + b"\0"), "zlib"),
    (lambda: B.image_png(B.small_pixels(), before_idat=[B.itxt(b"{}"), B.chunk(b"ABCD", b"")]), "unknown critical"),
    (lambda: B.image_png(B.small_pixels(), before_idat=[B.itxt(b"{}")], after_idat=[B.chunk(b"PLTE", b"\0\0\0")]), "PLTE after"),
    (lambda: B.image_png(B.small_pixels(), before_idat=[B.itxt(b"{}")], idat_parts=2), None),
    (lambda: B.image_png(B.small_pixels(), before_idat=[B.itxt(b"{}"), B.chunk(b"a1cd", b"")]), "four ASCII letters"),
])
def test_writer_mistakes_make_the_panel_unreadable(make, problem):
    c = assess(make())
    if problem is None:
        assert c.unreadable is None
    else:
        assert c.unreadable and problem in c.unreadable


def test_chunk_between_idats_is_a_writer_mistake():
    data = zlib.compress(B.scanlines(B.small_pixels()))
    png = (B.SIG + B.ihdr(4, 3) + B.itxt(b"{}") + B.chunk(b"IDAT", data[:5])
           + B.chunk(b"tEXt", b"k\0v") + B.chunk(b"IDAT", data[5:]) + B.IEND)
    assert "between IDAT" in assess(png).unreadable


def test_bad_adler_with_good_crc_is_a_writer_mistake():
    data = bytearray(zlib.compress(B.scanlines(B.small_pixels())))
    data[-1] ^= 1
    png = B.image_png(B.small_pixels(), before_idat=[B.itxt(b"{}")], raw=bytes(data))
    assert "zlib" in assess(png).unreadable

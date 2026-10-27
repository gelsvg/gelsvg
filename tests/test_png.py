import hashlib
import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

import builders as B
from gelsvg import _png


def gels_offset(png: bytes) -> int:
    return png.index(b"geLs") - 4


# --- walking -----------------------------------------------------------------


def test_walk_plain_png_reaches_iend():
    png = B.image_png(B.small_pixels())
    walk = _png.walk(png)
    assert walk.signature_ok and walk.ended_at_iend and not walk.truncated
    assert [c.type for c in walk.chunks] == [b"IHDR", b"IDAT", b"IEND"]
    assert all(c.crc_ok for c in walk.chunks)


def test_walk_rejects_other_signature():
    assert not _png.walk(b"GIF89a....").signature_ok


def test_bytes_after_iend_are_seen():
    walk = _png.walk(B.image_png(B.small_pixels()) + b"junk")
    assert walk.ended_at_iend and walk.bytes_after_iend


def test_cut_inside_idat_loses_the_tail():
    png = B.image_png(B.small_pixels())
    cut = png[: png.index(b"IDAT") + 10]
    walk = _png.walk(cut)
    assert walk.truncated and not walk.ended_at_iend
    assert walk.chunks[-1].type == b"IDAT" and walk.chunks[-1].runs_past_end


# --- geLs ----------------------------------------------------------------------


def test_good_gels_chunk():
    payload = b"payload bytes"
    walk = _png.walk(B.image_png(B.small_pixels(), after_idat=[B.gels(payload, codec=2)]))
    (g,) = walk.gels
    assert g.status == "ok" and g.codec == 2 and g.payload == payload
    assert walk.ended_at_iend


@pytest.mark.parametrize("kwargs, detail", [
    (dict(magic=b"MWAS"), "marker"),
    (dict(payload_length=3), "disagree"),
    (dict(sha=b"\0" * 32), "SHA-256"),
])
def test_gels_writer_errors(kwargs, detail):
    walk = _png.walk(B.image_png(B.small_pixels(), after_idat=[B.gels(b"abcdef", **kwargs)]))
    (g,) = walk.gels
    assert g.status == "writer_error" and detail in g.detail
    assert walk.ended_at_iend


@pytest.mark.parametrize("kwargs", [dict(version=2), dict(codec=9), dict(flags=1)])
def test_gels_unknown_header(kwargs):
    walk = _png.walk(B.image_png(B.small_pixels(), after_idat=[B.gels(b"abc", **kwargs)]))
    assert walk.gels[0].status == "unknown_header"


def test_gels_duplicates_after_the_first_are_ignored():
    walk = _png.walk(B.image_png(B.small_pixels(), after_idat=[B.gels(b"same"), B.gels(b"same"), B.gels(b"same", codec=2)]))
    assert [g.status for g in walk.gels] == ["ok", "duplicate", "ok"]


def test_gels_crc_only_damage_is_rescued():
    png = B.image_png(B.small_pixels(), after_idat=[B.gels(b"abcdef")])
    crc_at = gels_offset(png) + 8 + 44 + 6
    walk = _png.walk(B.flip_bit(png, crc_at))
    (g,) = walk.gels
    assert g.status == "rescued" and g.rescued_by == "i" and g.payload == b"abcdef"
    assert walk.ended_at_iend


def test_gels_length_field_too_big_is_rescued_by_payload_length():
    png = B.image_png(B.small_pixels(), after_idat=[B.gels(b"abcdef")])
    walk = _png.walk(B.flip_bit(png, gels_offset(png) + 1))  # i grows past the end
    (g,) = walk.gels
    assert g.status == "rescued" and g.rescued_by == "ii"
    assert walk.ended_at_iend and not walk.truncated


def test_gels_length_field_too_small_is_rescued_and_the_walk_goes_on():
    payload = bytes(range(200))
    png = B.image_png(B.small_pixels(), after_idat=[B.gels(payload), B.chunk(b"tEXt", b"k\0v")])
    off = gels_offset(png)
    length = struct.unpack_from(">I", png, off)[0]
    damaged = png[:off] + struct.pack(">I", length - 64) + png[off + 4 :]
    walk = _png.walk(damaged)
    assert walk.gels[0].status == "rescued" and walk.gels[0].rescued_by == "ii"
    assert [c.type for c in walk.chunks][-2:] == [b"tEXt", b"IEND"]


def test_gels_length_below_44_still_reads_the_header():
    png = B.image_png(B.small_pixels(), after_idat=[B.gels(b"x" * 30)])
    off = gels_offset(png)
    damaged = png[:off] + struct.pack(">I", 10) + png[off + 4 :]
    walk = _png.walk(damaged)
    assert walk.gels[0].status == "rescued" and walk.gels[0].rescued_by == "ii"
    assert walk.ended_at_iend


def test_gels_damage_inside_the_payload_loses_only_that_chunk():
    png = B.image_png(B.small_pixels(), after_idat=[B.gels(b"abcdef"), B.gels(b"second", codec=2)])
    walk = _png.walk(B.flip_bit(png, gels_offset(png) + 8 + 44 + 2))
    assert [g.status for g in walk.gels] == ["damaged", "ok"]
    assert walk.ended_at_iend


def test_gels_unrecoverable_with_unreadable_next_chunk_loses_the_tail():
    png = B.image_png(B.small_pixels(), after_idat=[B.gels(b"abcdef")])
    off = gels_offset(png)
    damaged = B.flip_bit(B.flip_bit(png, off + 3, 3), off + 8 + 44 + 1)  # length and payload
    walk = _png.walk(damaged)
    assert walk.gels[0].status == "damaged"
    assert walk.truncated and walk.final_iend_ok


# --- pixels --------------------------------------------------------------------


def _reference(ihdr: _png.Ihdr, raw: bytes) -> np.ndarray:
    return _png.decode_pixels(ihdr, raw)


@pytest.mark.parametrize("depth", [8, 16])
@pytest.mark.parametrize("rgb", [False, True])
@pytest.mark.parametrize("interlace", [0, 1])
def test_decode_matches_the_pixels_for_every_filter(depth, rgb, interlace):
    pixels = B.small_pixels(height=9, width=11, rgb=rgb, depth=depth, seed=depth + rgb + interlace)
    filters = [r % 5 for r in range(9)] if interlace == 0 else "cycle"
    raw = B.scanlines(pixels, depth, filters, interlace)
    ihdr = _png.Ihdr(11, 9, depth, 2 if rgb else 0, 0, 0, interlace)
    assert _png.scanline_problem(ihdr, raw) is None
    assert np.array_equal(_png.decode_pixels(ihdr, raw), pixels)


@pytest.mark.parametrize("depth", [8, 16])
@pytest.mark.parametrize("rgb", [False, True])
@pytest.mark.parametrize("shape", [(9, 11), (1, 1), (1, 7), (6, 1), (40, 33)])
def test_both_ways_of_undoing_the_filters_agree(depth, rgb, shape):
    height, width = shape
    pixels = B.small_pixels(height=height, width=width, rgb=rgb, depth=depth, seed=height * width + depth + rgb)
    raw = B.scanlines(pixels, depth, [(r * 3 + 1) % 5 for r in range(height)], 0)
    ihdr = _png.Ihdr(width, height, depth, 2 if rgb else 0, 0, 0, 0)
    channels = 3 if rgb else 1
    rebuilt = _png._rebuilt_samples(ihdr, raw)
    assert np.array_equal(rebuilt.reshape(pixels.shape), pixels)
    by_pillow = _png._pillow_samples(ihdr, raw)
    if depth == 16 and rgb:
        assert by_pillow is None  # Pillow keeps only 8 bits of 16-bit RGB
    else:
        assert by_pillow.dtype == rebuilt.dtype and by_pillow.shape == (height, width, channels)
        assert np.array_equal(by_pillow, rebuilt)


def test_interlaced_scanlines_are_undone_here():
    pixels = B.small_pixels(height=9, width=11, seed=5)
    ihdr = _png.Ihdr(11, 9, 8, 0, 0, 0, 1)
    assert _png._pillow_samples(ihdr, B.scanlines(pixels, 8, "cycle", 1)) is None


def test_decode_agrees_with_pillow_for_grey_16():
    pixels = B.small_pixels(height=6, width=7, depth=16, seed=3)
    png = B.image_png(pixels, depth=16, filters=[4, 3, 1, 2, 0, 4])
    walk = _png.walk(png)
    ihdr = _png.Ihdr.parse(walk.chunks[0].data)
    raw = _png.inflate_complete(walk.chunks[1].data)
    with Image.open(io.BytesIO(png)) as image:
        assert np.array_equal(np.asarray(image), _png.decode_pixels(ihdr, raw))


def test_scanline_problems():
    pixels = B.small_pixels()
    ihdr = _png.Ihdr(4, 3, 8, 0, 0, 0, 0)
    raw = B.scanlines(pixels)
    assert _png.scanline_problem(ihdr, raw[:-1]) is not None
    bad = bytearray(raw)
    bad[0] = 5
    assert "filter" in _png.scanline_problem(ihdr, bytes(bad))


def test_inflate_complete_rejects_trailing_bytes_and_bad_adler():
    stream = zlib.compress(b"hello")
    assert _png.inflate_complete(stream) == b"hello"
    assert _png.inflate_complete(stream + b"\0") is None
    assert _png.inflate_complete(stream[:-1]) is None
    assert _png.inflate_complete(stream[:-1] + bytes([stream[-1] ^ 1])) is None

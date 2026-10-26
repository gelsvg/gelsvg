import io
import json
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

import builders as B
import sourcefiles as F
from gelsvg import (Content, Kind, SourceError, Tool, default_crop, new_panel, pixel_hash, read, read_source,
                    render)
from gelsvg._metadata import check_formal
from gelsvg.sources import DAMAGED, NOT_A_SOURCE_IMAGE, NOT_SUPPORTED


def grey(h=5, w=7, dtype=np.uint8, seed=0):
    return np.random.default_rng(seed).integers(0, np.iinfo(dtype).max + 1, size=(h, w), dtype=dtype)


def rgb(h=5, w=7, dtype=np.uint8, seed=1):
    return np.random.default_rng(seed).integers(0, np.iinfo(dtype).max + 1, size=(h, w, 3), dtype=dtype)


def index_of(source):
    entry, payload = source.metadata
    index, problem = check_formal(entry["format"], payload)
    assert problem is None
    return index, payload[4 + struct.unpack(">I", payload[:4])[0] :]


def refused(data, reason):
    with pytest.raises(SourceError) as exc:
        read_source(data)
    assert exc.value.reason == reason, str(exc.value)
    return str(exc.value)


def pillow_bytes(image, fmt, **kwargs):
    buf = io.BytesIO()
    image.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


# --- the result -----------------------------------------------------------------------------------


def test_a_source_image_as_json():
    pixels = rgb()
    source = read_source(F.tiff(pixels), file_name="blot.tif")
    data = source.to_json("s1")
    assert [c["name"] for c in data["channels"]] == ["red", "green", "blue"]
    assert data["channels"][1]["pixel_hash"] == pixel_hash(np.ascontiguousarray(pixels[..., 1]))
    assert (data["dtype"], data["shape"], data["file_name"]) == ("uint8", [5, 7], "blot.tif")
    assert data["source_metadata"].startswith("sha256:") and data["file_hash"].startswith("sha256:")


def test_from_a_file_path(tmp_path):
    path = tmp_path / "gel.png"
    path.write_bytes(F.png(grey(), 0))
    source = read_source(path)
    assert source.file_name == "gel.png" and source.format == "png"


def test_not_a_known_format():
    refused(b"GIF89a....", NOT_SUPPORTED)


# --- PNG ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_png_low_bit_depths_are_widened(depth):
    top = 2**depth - 1
    stored = np.random.default_rng(depth).integers(0, top + 1, size=(4, 11), dtype=np.uint8)
    source = read_source(F.png(stored, 0, depth))
    assert np.array_equal(source.pixels, stored * (255 // top))


def test_png_opaque_alpha_is_dropped_and_transparency_refused():
    pixels = rgb(dtype=np.uint16)
    alpha = np.full(pixels.shape[:2] + (1,), 65535, dtype=np.uint16)
    assert np.array_equal(read_source(F.png(np.concatenate([pixels, alpha], axis=2), 6, 16)).pixels, pixels)
    alpha[0, 0, 0] = 65534
    refused(F.png(np.concatenate([pixels, alpha], axis=2), 6, 16), NOT_A_SOURCE_IMAGE)
    g = grey()
    assert np.array_equal(read_source(F.png(np.stack([g, np.full_like(g, 255)], axis=2), 4)).pixels, g)


def test_png_transparent_colour():
    g = grey()
    present = B.chunk(b"tRNS", struct.pack(">H", int(g[2, 3])))
    refused(F.png(g, 0, before=[present]), NOT_A_SOURCE_IMAGE)
    absent_value = next(v for v in range(256) if v not in g)
    assert read_source(F.png(g, 0, before=[B.chunk(b"tRNS", struct.pack(">H", absent_value))]))


def test_png_that_cannot_be_a_source_image():
    g = grey()
    refused(F.png(g, 3, before=[B.chunk(b"PLTE", bytes(range(256)) * 3)]), NOT_A_SOURCE_IMAGE)
    refused(F.png(g, 0, before=[B.chunk(b"acTL", struct.pack(">II", 2, 0))]), NOT_A_SOURCE_IMAGE)
    damaged = F.png(g, 0)
    refused(damaged[:-20], DAMAGED)


def test_interlaced_rgb16_png():
    pixels = rgb(9, 13, np.uint16)
    assert np.array_equal(read_source(F.png(pixels, 2, 16, interlace=1)).pixels, pixels)


def turned(pixels, k):
    return np.rot90(pixels, k=k)


@pytest.mark.parametrize("chunk", [
    lambda: B.chunk(b"eXIf", F.exif_block(6)),
    lambda: B.chunk(b"eXIf", b"Exif\0\0" + F.exif_block(6, ">")),
    lambda: B.chunk(b"zTXt", b"Raw profile type exif\0\0" + zlib.compress(
        b"\nexif\n   100\n" + ("457869660000" + F.exif_block(6).hex()).encode() + b"\n")),
    lambda: B.chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + F.xmp('tiff:Orientation="6"')),
])
def test_png_orientation(chunk):
    g = grey()
    assert np.array_equal(read_source(F.png(g, 0, before=[chunk()])).pixels, turned(g, -1))


def test_png_damaged_exif_turns_nothing_even_with_xmp():
    g = grey()
    bad = B.chunk(b"eXIf", b"XX" + F.exif_block(6)[2:])
    x = B.chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + F.xmp('tiff:Orientation="6"'))
    assert np.array_equal(read_source(F.png(g, 0, before=[bad, x])).pixels, g)
    odd = B.chunk(b"tEXt", b"Raw profile type exif\0\nexif\n 3\nabc\n")
    assert np.array_equal(read_source(F.png(g, 0, before=[odd, x])).pixels, g)


def test_png_xmp_in_other_forms_is_not_read():
    g = grey()
    x = B.chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + F.xmp("tiff:Orientation='6'"))
    assert np.array_equal(read_source(F.png(g, 0, before=[x])).pixels, g)


def test_png_metadata_segments():
    g = grey()
    text = B.chunk(b"tEXt", b"Comment\0hi")
    source = read_source(F.png(g, 0, before=[text], after=[B.gels(b"x"), B.chunk(b"tIME", b"\x07\xea\x0a\x1a\0\0\0")]))
    index, data = index_of(source)
    assert [(s["chunk_type"], s["after_pixels"]) for s in index["segments"]] == [
        ("IHDR", False), ("tEXt", False), ("tIME", True), ("IEND", True)]
    assert b"Comment\0hi" in data and b"geLs" not in data


def test_png_with_a_damaged_ancillary_chunk_keeps_pixels_not_metadata():
    g = grey()
    text = B.chunk(b"tEXt", b"Comment\0hi")
    source = read_source(F.png(g, 0, before=[B.flip_bit(text, len(text) - 1)]))
    assert np.array_equal(source.pixels, g) and source.metadata is None and source.notes


# --- JPEG -----------------------------------------------------------------------------------------


def smooth(h=16, w=24):
    y, x = np.mgrid[0:h, 0:w]
    return ((x * 7 + y * 5) % 256).astype(np.uint8)


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_jpeg_pixels_are_pillows(mode):
    image = Image.fromarray(smooth()).convert(mode)
    data = pillow_bytes(image, "JPEG", quality=90)
    source = read_source(data)
    assert np.array_equal(source.pixels, np.asarray(Image.open(io.BytesIO(data))))
    index, _ = index_of(source)
    assert [s["marker"] for s in index["segments"]] == ["APP0"]


def test_jpeg_orientation_and_segments():
    image = Image.fromarray(smooth())
    exif = Image.Exif()
    exif[274] = 8
    data = pillow_bytes(image, "JPEG", exif=exif.tobytes(), quality=95)
    data = data[:2] + b"\xff\xfe\x00\x05hi!" + data[2:] + b"trailing bytes"
    source = read_source(data)
    plain = np.asarray(Image.open(io.BytesIO(data)))
    assert np.array_equal(source.pixels, turned(plain, 1))
    index, payload = index_of(source)
    assert [s["marker"] for s in index["segments"]] == ["COM", "APP0", "APP1"]
    assert payload.startswith(b"hi!") and b"trailing" not in payload


def test_jpeg_that_cannot_be_a_source_image():
    refused(pillow_bytes(Image.fromarray(smooth()).convert("CMYK"), "JPEG"), NOT_A_SOURCE_IMAGE)
    first, second = Image.fromarray(smooth()).convert("RGB"), Image.fromarray(255 - smooth()).convert("RGB")
    refused(pillow_bytes(first, "MPO", save_all=True, append_images=[second]), NOT_A_SOURCE_IMAGE)


# --- TIFF -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("compression", ["raw", "tiff_lzw", "tiff_adobe_deflate", "packbits"])
@pytest.mark.parametrize("pixels", [grey(), grey(dtype=np.uint16), rgb()], ids=["L", "I16", "RGB"])
def test_pillow_written_tiffs(pixels, compression):
    if compression == "packbits" and pixels.dtype == np.uint16:
        pytest.skip("Pillow writes PackBits only for 8-bit images")
    source = read_source(pillow_bytes(Image.fromarray(pixels), "TIFF", compression=compression))
    assert np.array_equal(source.pixels, pixels) and source.pixels.dtype == pixels.dtype


@pytest.mark.parametrize("compression", [1, 5, 8, 32773])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_rgb16_tiff(compression, endian):
    pixels = rgb(9, 6, np.uint16)
    assert np.array_equal(read_source(F.tiff(pixels, compression=compression, endian=endian, rows_per_strip=4)).pixels,
                          pixels)


@pytest.mark.parametrize("bits", [8, 16])
def test_tiff_predictor_planes_and_tiles(bits):
    pixels = rgb(23, 37, np.uint16 if bits == 16 else np.uint8)
    for kwargs in [dict(predictor=2, compression=8), dict(planar=2, rows_per_strip=5),
                   dict(tile=(16, 16), compression=5), dict(tile=(16, 16), planar=2, predictor=2, compression=8)]:
        assert np.array_equal(read_source(F.tiff(pixels, **kwargs)).pixels, pixels), kwargs


def test_bigtiff():
    pixels = grey(dtype=np.uint16)
    assert np.array_equal(read_source(F.tiff(pixels, big=True, endian=">")).pixels, pixels)


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
def test_tiff_white_is_zero(bits):
    top = 2**bits - 1
    stored = np.random.default_rng(bits).integers(0, top + 1, size=(6, 13)).astype(np.uint16 if bits == 16 else np.uint8)
    source = read_source(F.tiff(stored, photometric=0, bits=bits))
    widened = stored.astype(np.int64) * (255 // top if bits < 8 else 1)
    maximum = 65535 if bits == 16 else 255
    assert np.array_equal(source.pixels, maximum - widened)


def test_tiff_alpha():
    pixels = rgb()
    alpha = np.full(pixels.shape[:2] + (1,), 255, dtype=np.uint8)
    assert np.array_equal(read_source(F.tiff(np.concatenate([pixels, alpha], axis=2), extra_samples=[2])).pixels, pixels)
    alpha[1, 1, 0] = 0
    refused(F.tiff(np.concatenate([pixels, alpha], axis=2), extra_samples=[1]), NOT_A_SOURCE_IMAGE)
    alpha[1, 1, 0] = 255
    refused(F.tiff(np.concatenate([pixels, alpha], axis=2), extra_samples=[0]), NOT_A_SOURCE_IMAGE)


@pytest.mark.parametrize("kwargs, reason", [
    (dict(next_ifd=True), NOT_A_SOURCE_IMAGE),
    (dict(tags=[(330, 4, [8])]), NOT_A_SOURCE_IMAGE),
    (dict(photometric=5), NOT_A_SOURCE_IMAGE),
    (dict(photometric=3), NOT_A_SOURCE_IMAGE),
    (dict(tags=[(339, 3, [2])]), NOT_A_SOURCE_IMAGE),
    (dict(bits=12), NOT_A_SOURCE_IMAGE),
    (dict(omit=(259,), tags=[(259, 3, [7])]), NOT_SUPPORTED),
    (dict(omit=(262,)), DAMAGED),
])
def test_tiffs_that_cannot_be_source_images(kwargs, reason):
    refused(F.tiff(grey(), **kwargs), reason)


@pytest.mark.parametrize("tags, k", [
    ([(274, 3, [8])], 1),
    ([(274, 3, [6])], -1),
    ([(274, 4, [6])], 0),  # not a SHORT: read as 1
    ([(700, 1, F.xmp("tiff:Orientation>3<"))], 2),
    ([(274, 3, [1]), (700, 1, F.xmp('tiff:Orientation="6"'))], 0),
])
def test_tiff_orientation(tags, k):
    g = grey()
    assert np.array_equal(read_source(F.tiff(g, tags=tags)).pixels, turned(g, k))


def test_tiff_metadata_segments():
    g = grey()
    data = F.tiff(g, tags=[(271, 2, b"ChemiDoc\0"), (305, 2, b"Image Lab\0")],
                  exif=[(36867, 2, b"2026:10:26 10:00:00\0")], gps=[(0, 1, [2, 3, 0, 0])],
                  interop=[(1, 2, b"R98\0")], rows_per_strip=2)
    index, payload = index_of(read_source(data))
    listed = [(s["ifd"], s["tag"]) for s in index["segments"]]
    assert listed == [("ifd0", 256), ("ifd0", 257), ("ifd0", 258), ("ifd0", 259), ("ifd0", 262), ("ifd0", 271),
                      ("ifd0", 277), ("ifd0", 278), ("ifd0", 284), ("ifd0", 305),
                      ("exif", 36867), ("interop", 1), ("gps", 0)]
    assert (index["byte_order"], index["tiff_version"]) == ("II", 42)
    assert b"ChemiDoc\0" in payload and b"R98\0" in payload


def test_tiff_metadata_needs_known_types():
    g = grey()
    source = read_source(F.tiff(g, tags=[(50000, 99, 1, b"\0\0\0\0")]))
    assert np.array_equal(source.pixels, g) and source.metadata is None and source.notes
    big = read_source(F.tiff(g, big=True, tags=[(50000, 16, [7])]))
    assert index_of(big)[0]["tiff_version"] == 43


# --- .scn -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("endian", ["little", "big"])
def test_scn(endian):
    pixels = grey(4, 6, np.uint16)
    data = F.scn(pixels, endian=endian, extra_pixel_bytes=b"\x01\x02")
    source = read_source(data)
    assert np.array_equal(source.pixels, pixels) and source.format == "scn"
    index, payload = index_of(source)
    assert [s["part"] for s in index["segments"]] == ["before_pixels", "after_pixels"]
    assert payload.endswith(b"\x01\x02\r\n--BOUND--\r\n")


def test_scn_bytes_per_pixel_and_channels():
    pixels = grey(3, 5)
    assert np.array_equal(read_source(F.scn(pixels, max_value=256)).pixels, pixels)
    assert read_source(F.scn(grey(3, 5, np.uint16), max_value=None)).pixels.dtype == np.uint16
    refused(F.scn(grey(3, 5, np.uint16), channel_count=2), NOT_A_SOURCE_IMAGE)
    refused(F.scn(grey(3, 5, np.uint16))[:-60], DAMAGED)


# --- from a file to a panel -----------------------------------------------------------------------


def test_from_a_source_file_to_a_panel():
    pixels = grey(30, 40, np.uint16)
    source = read_source(F.tiff(pixels, tags=[(271, 2, b"ChemiDoc\0")]), file_name="blot.tif")
    core = {"sources": [source.to_json("s1")],
            "blot": [{"id": "c1", "source": "s1", "source_channel": "gray", "invert": True, "contrast": [0, 65535]}],
            "orientation": {"turn": 0, "flip": False, "rotation": 1.5}, "output_bit_depth": 16}
    crop = default_crop(core)
    png = new_panel(Content({"core": core, "figure": {"crop": crop}}, rendered=render(core, source.channels and
                    {pixel_hash(i): i for i in source.channel_images}, crop),
                    channel_images=source.channel_images, source_metadata=source.source_metadata), Tool("t", "1"))
    (panel,) = read(B.svg(B.data_uri(png))).panels
    assert not [n for n in panel.notices if n.kind in (Kind.BROKEN, Kind.DAMAGED, Kind.UNREADABLE)]
    key = core["sources"][0]["source_metadata"]
    assert panel.core.source_metadata[key]["index"]["byte_order"] == "II"
    assert np.array_equal(next(iter(panel.channel_images.values())), pixels)

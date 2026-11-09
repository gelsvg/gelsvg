"""Drawing, checked bit for bit against a plain reading of §4.3 and §6.2:
one pixel at a time, in Python floats (binary64, one rounding per
operation, no FMA)."""

import copy
import math

import numpy as np
import pytest

import builders as B
import panels as P
from gelsvg import (Content, Kind, Tool, _render, aligned_image, aligned_size, default_crop, new_panel, pixel_hash,
                    render, to_channel, to_panel)
from gelsvg._geometry import cos_sin
from gelsvg.document import read


def k_ref(t):
    t = abs(t)
    if t <= 1:
        return (((1.5 * t) - 2.5) * t) * t + 1
    if t < 2:
        return ((((-0.5 * t) + 2.5) * t) - 4) * t + 2
    return 0.0


def to_type(rows, dtype):
    top = 255 if dtype == np.uint8 else 65535
    return np.array([[min(top, max(0, round(v))) for v in row] for row in rows], dtype=dtype)


def rotate_ref(image, degrees):
    return to_type(rotation_values_ref(image, degrees), image.dtype)


def rotation_values_ref(image, degrees):
    c, s = cos_sin(degrees)
    h, w = image.shape
    W = (w * abs(c)) + (h * abs(s))
    H = (w * abs(s)) + (h * abs(c))
    rows = []
    for i in range(round(H)):
        row = []
        for j in range(round(W)):
            xp, yp = j + 0.5, i + 0.5
            x = ((c * (xp - W / 2)) + (s * (yp - H / 2))) + w / 2
            y = ((-s * (xp - W / 2)) + (c * (yp - H / 2))) + h / 2
            u, v = x - 0.5, y - 0.5
            total = 0.0
            for m in range(math.floor(v) - 1, math.floor(v) + 3):
                for n in range(math.floor(u) - 1, math.floor(u) + 3):
                    p = int(image[m, n]) if 0 <= m < h and 0 <= n < w else 0
                    total = total + ((k_ref(u - n) * k_ref(v - m)) * p)
            row.append(total)
        rows.append(row)
    return rows


def resize_line_ref(line, size):
    w = len(line)
    s = w / size
    f = max(s, 1)
    out = []
    for j in range(size):
        c = (j + 0.5) * s
        k_sum, p_sum = 0.0, 0.0
        for n in range(w):
            if abs((n + 0.5) - c) < 2 * f:
                k = k_ref(((n + 0.5) - c) / f)
                k_sum = k_sum + k
                p_sum = p_sum + (k * line[n])
        out.append(p_sum / k_sum)
    return out


def resize_ref(image, height, width):
    return to_type(resize_values_ref(image, height, width), image.dtype)


def resize_values_ref(image, height, width):
    across = [resize_line_ref([int(v) for v in row], width) for row in image]
    columns = [resize_line_ref([row[j] for row in across], height) for j in range(width)]
    return [[columns[j][i] for j in range(width)] for i in range(height)]


def random_image(h, w, dtype=np.uint8, seed=0):
    return np.random.default_rng(seed).integers(0, np.iinfo(dtype).max + 1, size=(h, w), dtype=dtype)


# --- the reference interpolation ------------------------------------------------------------------


def test_the_kernel():
    t = np.array([0, 0.5, 1, 1.5, 2, 2.5, -0.5, -1.5])
    assert K_values(t) == [1.0, 0.5625, 0.0, -0.0625, 0.0, 0.0, 0.5625, -0.0625]


def K_values(t):
    return [float(v) for v in _render.K(t)]


@pytest.mark.parametrize("degrees", [7.3, -33.0, 0.5, -179.9, 45.0, 90.0, 180.0, 270.0, -90.0, 12.25])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_rotation_matches_the_reference(degrees, dtype):
    image = random_image(7, 9, dtype, seed=int(degrees * 10) % 97)
    assert np.array_equal(_render.rotate(image, degrees), rotate_ref(image, degrees))


@pytest.mark.parametrize("degrees", [7.3, -33.0, 0.5, -179.9, 45.0, 12.25])
def test_rotated_values_are_the_reference_bit_for_bit(degrees):
    image = random_image(19, 23, np.uint16, seed=11)
    assert _render.rotation_values(image, degrees).tolist() == rotation_values_ref(image, degrees)


@pytest.mark.parametrize("shape, size", [((5, 7), (9, 4)), ((13, 11), (4, 6)), ((3, 10), (7, 25)), ((17, 9), (11, 23))])
def test_resized_values_are_the_reference_bit_for_bit(shape, size):
    image = random_image(*shape, np.uint16, seed=13)
    assert _render.resize_values(image, *size).tolist() == resize_values_ref(image, *size)


def test_a_window_of_a_rotation():
    image = random_image(11, 8, seed=5)
    whole = _render.rotate(image, 21.5)
    assert np.array_equal(_render.rotate(image, 21.5, (2, 9, 3, 7)), whole[2:9, 3:7])


@pytest.mark.parametrize("shape, size", [((5, 7), (9, 4)), ((13, 11), (4, 6)), ((3, 10), (7, 25)),
                                         ((6, 6), (6, 6)), ((1, 5), (3, 2)), ((20, 3), (5, 3))])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_resizing_matches_the_reference(shape, size, dtype):
    image = random_image(*shape, dtype, seed=shape[0] * 31 + size[1])
    assert np.array_equal(_render.resize(image, *size), resize_ref(image, *size))


def test_average():
    red, green, blue = (np.array([[255, 0, 1]], dtype=np.uint8), np.array([[255, 0, 1]], dtype=np.uint8),
                        np.array([[255, 1, 0]], dtype=np.uint8))
    assert _render.average(red, green, blue).tolist() == [[255, 0, 1]]


@pytest.mark.parametrize("contrast, values, expected", [
    ([10, 20], [5, 10, 15, 20, 30], [0, 0, 0.5, 1, 1]),
    ([10, 10], [9, 10, 11], [0, 0, 1]),
    ([0, 510], [255], [0.5]),
])
def test_contrast(contrast, values, expected):
    assert _render.level(np.array(values), contrast).tolist() == expected


# --- drawing a panel ------------------------------------------------------------------------------


def base_core():
    panel, _, _ = P.base_panel()
    core = panel["core"]
    del core["embedded"]
    return core


def pixels_of(*images):
    return {pixel_hash(image): image for image in images}


def test_a_grey_panel_at_full_contrast_is_its_pixels():
    source = P.source_pixels()
    assert np.array_equal(render(base_core(), pixels_of(source)), source)


def test_rotation_crop_and_invert():
    source = random_image(30, 40, seed=2)
    core = base_core()
    core["sources"][0].update(shape=[30, 40])
    core["sources"][0]["source_channels"][0]["pixel_hash"] = pixel_hash(source)
    core["orientation"] = {"turn": 90, "flip": True, "rotation": -4.5}
    core["channels"][0].update(invert=True, contrast=[20, 200])
    core["roi"] = {"x": 3, "y": 5, "w": 20, "h": 25}
    px = pixels_of(source)
    assert default_crop(core) == core["roi"]
    height, width = aligned_size(core)
    expected = rotate_ref(_render.flip(np.rot90(source, k=-1)), -4.5)
    assert expected.shape == (height, width)
    expected = 255 - expected[5:30, 3:23]
    levels = np.clip((expected.astype(float) - 20) / (200 - 20), 0, 1)
    assert np.array_equal(render(core, px), np.floor(levels * 255).astype(np.uint8))
    whole = render(core, px, {"x": 0, "y": 0, "w": width, "h": height})
    assert np.array_equal(render(core, px), whole[5:30, 3:23])


def two_channel_core(dtype=np.uint8):
    a, b = random_image(6, 8, dtype, seed=3), random_image(4, 3, dtype, seed=4)
    name = "uint8" if dtype == np.uint8 else "uint16"
    core = base_core()
    core["sources"] = [
        {"id": "s1", "dtype": name, "shape": [6, 8], "source_channels": [{"name": "gray", "pixel_hash": pixel_hash(a)}]},
        {"id": "s2", "dtype": name, "shape": [4, 3], "source_channels": [{"name": "gray", "pixel_hash": pixel_hash(b)}]},
    ]
    top = int(np.iinfo(dtype).max)
    core["channels"] = [
        {"id": "c1", "drawn": True, "source": "s1", "source_channel": "gray", "invert": False, "contrast": [0, top],
         "color": [1, 0, 0.5]},
        {"id": "c2", "drawn": True, "source": "s2", "source_channel": "gray", "turn": 90, "flip": False,
         "invert": False, "contrast": [0, top // 2], "color": [0.25, 1, 0.5]},
    ]
    core["reference_channel"] = "c1"
    return core, a, b


@pytest.mark.parametrize("dtype, depth", [(np.uint8, 8), (np.uint16, 16), (np.uint16, 8)])
def test_two_coloured_channels(dtype, depth):
    core, a, b = two_channel_core(dtype)
    core["output_bit_depth"] = depth
    top = float(np.iinfo(dtype).max)
    second = resize_ref(np.rot90(b, k=-1), 6, 8)
    assert np.array_equal(aligned_image(core, pixels_of(a, b), 1), second)
    n1 = np.clip(a.astype(float) / top, 0, 1)
    n2 = np.clip(second.astype(float) / float(int(top) // 2), 0, 1)
    colours = [[1, 0, 0.5], [0.25, 1, 0.5]]
    c = np.stack([np.minimum(1, (0.0 + n1 * colours[0][k]) + n2 * colours[1][k]) for k in range(3)], axis=-1)
    q = np.floor(c * top)
    if depth == 8 and dtype == np.uint16:
        q = np.rint((q * 255) / 65535)
    assert np.array_equal(render(core, pixels_of(a, b)), q.astype(np.uint8 if depth == 8 else np.uint16))


def test_a_panel_channel_is_named_by_its_id_or_its_place():
    core, a, b = two_channel_core()
    px = pixels_of(a, b)
    assert np.array_equal(aligned_image(core, px, "c2"), aligned_image(core, px, 1))
    assert np.array_equal(aligned_image(core, px), aligned_image(core, px, "c1"))
    assert np.array_equal(aligned_image(core, px), a)
    with pytest.raises(ValueError, match="no panel channel"):
        aligned_image(core, px, "marker")


def test_the_reference_channel_is_named_by_reference_channel_not_its_place():
    core, a, b = two_channel_core()
    # c2 becomes the reference channel: 4 × 3, unturned; c1 is turned to it and resized
    core["channels"][0].update(turn=270, flip=False)
    del core["channels"][1]["turn"], core["channels"][1]["flip"]
    core["reference_channel"] = "c2"
    px = pixels_of(a, b)
    assert aligned_size(core) == (4, 3)
    assert np.array_equal(aligned_image(core, px), b)
    assert np.array_equal(aligned_image(core, px, "c1"), resize_ref(np.rot90(a, k=1), 4, 3))


def test_an_undrawn_reference_channel_sets_the_size_alone():
    # §5.7: the reference channel may be undrawn; only drawn panel channels are drawn (§6.2),
    # each aligned to it. Its colour draws nothing: one drawn channel without one is grey.
    core, a, b = two_channel_core()
    core["channels"] = [
        {"id": "c1", "drawn": True, "source": "s1", "source_channel": "gray", "turn": 0, "flip": False,
         "invert": True, "contrast": [0, 255]},
        {"id": "ref", "drawn": False, "source": "s2", "source_channel": "gray", "invert": False,
         "contrast": [0, 255], "color": [1, 0, 0]},
    ]
    core["reference_channel"] = "ref"
    px = pixels_of(a, b)
    assert aligned_size(core) == (4, 3) and default_crop(core) == {"x": 0, "y": 0, "w": 3, "h": 4}
    assert np.array_equal(aligned_image(core, px), b)
    drawn = resize_ref(a, 4, 3)
    assert np.array_equal(aligned_image(core, px, "c1"), drawn)
    expected = np.floor(np.clip((255 - drawn).astype(float) / 255, 0, 1) * 255).astype(np.uint8)
    assert np.array_equal(render(core, px), expected)


def test_an_undrawn_channel_takes_no_part_in_the_picture():
    core, a, b = two_channel_core()
    px = pixels_of(a, b)
    core["channels"][1]["drawn"] = False
    alone = render(core, px)
    assert alone.shape == (6, 8, 3)  # c1 still has a colour: RGB
    n1 = np.clip(a.astype(float) / 255, 0, 1)
    expected = np.stack([np.floor(np.minimum(1, 0.0 + n1 * k) * 255) for k in (1, 0, 0.5)], axis=-1)
    assert np.array_equal(alone, expected.astype(np.uint8))
    del core["channels"][0]["color"]
    assert np.array_equal(render(core, px), a)  # one drawn, no colour: grey


def test_blend_add_draws_as_no_blend():
    core, a, b = two_channel_core()
    px = pixels_of(a, b)
    added = copy.deepcopy(core)
    added["blend"] = "add"
    assert np.array_equal(render(added, px), render(core, px))


def test_a_blend_this_version_does_not_know_is_not_drawn():
    core, a, b = two_channel_core()
    core["blend"] = "multiply"
    with pytest.raises(ValueError, match="cannot be drawn"):
        render(core, pixels_of(a, b))


def test_with_no_drawn_channel_nothing_is_drawn_but_the_channels_align():
    core, a, b = two_channel_core()
    for element in core["channels"]:
        element["drawn"] = False
    px = pixels_of(a, b)
    with pytest.raises(ValueError, match="no panel channel is drawn"):
        render(core, px)
    assert np.array_equal(aligned_image(core, px, "c2"), resize_ref(np.rot90(b, k=-1), 6, 8))
    assert aligned_size(core) == (6, 8)


@pytest.mark.parametrize("change", [
    lambda core: core.pop("reference_channel"),
    lambda core: core.update(reference_channel="c9"),
    lambda core: core.pop("channels"),
])
def test_without_a_reference_channel_nothing_aligns(change):
    core, a, b = two_channel_core()
    change(core)
    with pytest.raises(ValueError, match="cannot be drawn"):
        aligned_size(core)
    with pytest.raises(ValueError, match="cannot be drawn"):
        render(core, pixels_of(a, b))


def test_channels_on_one_source_channel_turned_otherwise_are_drawn_as_written():
    # §5.6: read as written; each is aligned by its own turn and flip
    core, a, b = two_channel_core()
    core["channels"].append({"id": "k1", "role": "marker", "drawn": False, "source": "s1", "source_channel": "gray",
                             "turn": 180, "flip": False, "invert": False, "contrast": [0, 255]})
    px = pixels_of(a, b)
    assert np.array_equal(aligned_image(core, px, "k1"), np.rot90(a, k=2))
    before = two_channel_core()[0]
    assert np.array_equal(render(core, px), render(before, px))


def test_drawing_needs_the_channel_images_and_a_valid_core():
    core, a, b = two_channel_core()
    with pytest.raises(ValueError, match="not given"):
        render(core, pixels_of(a))
    with pytest.raises(ValueError, match="pixel hash"):
        render(core, {pixel_hash(a): a, pixel_hash(b): a})
    broken = copy.deepcopy(core)
    broken["channels"][1]["contrast"] = [5, 1]
    with pytest.raises(ValueError, match="cannot be drawn"):
        render(broken, pixels_of(a, b))
    uncoloured = copy.deepcopy(core)
    del uncoloured["channels"][1]["color"]  # two drawn: each needs a colour
    with pytest.raises(ValueError, match="cannot be drawn"):
        render(uncoloured, pixels_of(a, b))
    with pytest.raises(ValueError, match="inside"):
        render(core, pixels_of(a, b), {"x": 1, "y": 0, "w": 8, "h": 6})
    unsized = copy.deepcopy(core)
    for source in unsized["sources"]:
        source.pop("shape"), source.pop("dtype")
        for channel in source["source_channels"]:
            channel.pop("pixel_hash")
    with pytest.raises(ValueError, match="shape"):
        render(unsized, {})


def test_a_drawn_panel_reads_back_and_draws_again_the_same():
    source = random_image(30, 40, seed=7)
    core = base_core()
    core["sources"][0].update(shape=[30, 40])
    core["sources"][0]["source_channels"][0]["pixel_hash"] = pixel_hash(source)
    core["orientation"] = {"turn": 0, "flip": False, "rotation": 3.25}
    core["roi"] = {"x": 2, "y": 2, "w": 30, "h": 20}
    crop = default_crop(core)
    rendered = render(core, pixels_of(source), crop)
    png = new_panel(Content({"core": core, "figure": {"crop": crop}}, rendered=rendered,
                            source_channel_images=[source]), Tool("t", "1"))
    (p,) = read(P.svg_of(png)).panels
    assert not [n for n in p.notices if n.kind in (Kind.BROKEN, Kind.UNREADABLE)]
    assert p.checks["drawn_from"] and p.checks["rendered_hash"]
    again = render(p.to_json()["core"], p.source_channel_images, p.figure["crop"])
    assert np.array_equal(again, p.rendered)


# --- points ---------------------------------------------------------------------------------------


def test_pixel_centres_follow_the_pixels_through_turns_and_flips():
    core, a, b = two_channel_core()
    core["channels"][1]["flip"] = True
    core["orientation"] = {"turn": 270, "flip": True, "rotation": 0}
    b_big = random_image(8, 6, seed=9)  # turned by 90°, it is already the reference size: no resize
    core["sources"][1].update(shape=[8, 6])
    core["sources"][1]["source_channels"][0]["pixel_hash"] = pixel_hash(b_big)
    image = aligned_image(core, pixels_of(a, b_big), 1)
    for i, j in [(0, 0), (7, 5), (3, 2), (5, 0)]:
        x, y = to_panel(core, j + 0.5, i + 0.5, 1)
        assert (x % 1, y % 1) == (0.5, 0.5)
        assert image[int(y), int(x)] == b_big[i, j]
        assert to_channel(core, x, y, 1) == (j + 0.5, i + 0.5)


@pytest.mark.parametrize("element", [None, 0, 1, "c2", "k1"])
def test_points_go_there_and_back(element):
    core, a, b = two_channel_core()
    core["orientation"] = {"turn": 90, "flip": True, "rotation": 7.75}
    marker = random_image(5, 7, seed=6)
    core["sources"].append({"id": "s3", "dtype": "uint8", "shape": [5, 7],
                            "source_channels": [{"name": "gray", "pixel_hash": pixel_hash(marker)}]})
    core["channels"].append({"id": "k1", "role": "marker", "drawn": False, "source": "s3", "source_channel": "gray",
                             "turn": 180, "flip": True, "invert": False, "contrast": [0, 255]})
    for point in [(0.0, 0.0), (3.25, 1.5), (5.9, 7.1)]:
        back = to_channel(core, *to_panel(core, *point, element), element)
        assert back == pytest.approx(point, abs=1e-12)


def test_a_point_through_a_resize():
    core, _, _ = two_channel_core()  # the second channel, 4 × 3, turned by 90° to 3 × 4, is resized to 6 × 8
    assert to_panel(core, 0.0, 0.0, 1) == (8.0, 0.0)
    assert to_panel(core, 1.5, 2.0, 1) == (4.0, 3.0)
    assert to_panel(core, 1.5, 2.0, "c2") == (4.0, 3.0)


def test_a_point_on_an_undrawn_reference_channel_is_not_moved():
    core, _, _ = two_channel_core()
    core["channels"][0]["drawn"] = False
    assert to_panel(core, 1.5, 2.0) == (1.5, 2.0) == to_channel(core, 1.5, 2.0, "c1")

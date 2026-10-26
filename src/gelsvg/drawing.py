"""Drawing the rendered image (§6.2) and aligned images (§4.2) from a
panel's core and its channel images, with the reference interpolation, so
every implementation gets the same pixel values."""

from __future__ import annotations

import copy
from typing import Mapping

import numpy as np

from . import _geometry, _hashes, _render
from ._check import Context
from ._core import CoreView, read_core
from .notices import Kind

_NEEDED = ("core.sources", "core.blot", "core.orientation", "core.output_bit_depth")


def _view(core: dict, marker: bool = False) -> CoreView:
    """The core read strictly, refusing one whose drawing parts are not all
    there and valid."""

    ctx = Context(panel=None, newer_revision=False)
    view = read_core(ctx, copy.deepcopy(core))
    needed = _NEEDED + (("core.marker",) if marker else ())
    bad = [n.message for n in ctx.notices
           if n.kind is not Kind.REINTERPRETED and (n.where or "").startswith(needed)]
    if bad or view.blot is None or view.orientation is None or view.sources is None:
        raise ValueError("The core cannot be drawn: " + ("; ".join(bad) or "a key it needs is missing") + ".")
    if marker and view.marker is None:
        raise ValueError("The core has no marker image.")
    return view


def _channel(view: CoreView, element: dict, pixels: Mapping[str, np.ndarray]) -> np.ndarray:
    source = view.source(element["source"])
    if source.get("dtype") is None or source.get("shape") is None:
        raise ValueError(f"The source image {source['id']!r} has no 'dtype' and 'shape'; its pixels were never read.")
    dtype = np.uint8 if source["dtype"] == "uint8" else np.uint16

    def one(name: str) -> np.ndarray:
        channel = next(c for c in source["channels"] if c["name"] == name)
        key = channel.get("pixel_hash") if channel.get("pixel_hash_known") else None
        image = pixels.get(key) if key is not None else None
        if image is None:
            raise ValueError(f"The channel image of {source['id']!r} / {name!r} is not given.")
        if image.dtype != dtype or list(image.shape) != source["shape"] or _hashes.pixel_hash(image) != key:
            raise ValueError(f"The channel image given for {source['id']!r} / {name!r} does not match its pixel hash.")
        return image

    if element["source_channel"] == "average":
        return _render.average(one("red"), one("green"), one("blue"))
    return one(element["source_channel"])


def _reference_shape(view: CoreView) -> list[int]:
    return view.source(view.blot[0]["source"])["shape"]


def aligned_size(core: dict) -> tuple[int, int]:
    """(height, width) of the panel's aligned image."""

    view = _view(core)
    return _geometry.aligned_size(_reference_shape(view), view.orientation)


def default_crop(core: dict) -> dict:
    """The rectangle §6.2 recommends drawing: the ROI, cut to the aligned
    image, or the whole aligned image when there is no ROI or it lies
    outside."""

    view = _view(core)
    height, width = _geometry.aligned_size(_reference_shape(view), view.orientation)
    roi = view.roi
    if roi is not None:
        x0, y0 = max(0, roi["x"]), max(0, roi["y"])
        x1, y1 = min(width, roi["x"] + roi["w"]), min(height, roi["y"] + roi["h"])
        if x1 > x0 and y1 > y0:
            return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
    return {"x": 0, "y": 0, "w": width, "h": height}


def _window(view: CoreView, crop: dict | None) -> tuple[int, int, int, int] | None:
    if crop is None:
        return None
    height, width = _geometry.aligned_size(_reference_shape(view), view.orientation)
    x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
    if not (x >= 0 and y >= 0 and w >= 1 and h >= 1 and x + w <= width and y + h <= height):
        raise ValueError(f"The rectangle {crop} is not inside the aligned image ({width} × {height}).")
    return y, y + h, x, x + w


def aligned_image(core: dict, pixels: Mapping[str, np.ndarray], element: int | str = 0,
                  crop: dict | None = None) -> np.ndarray:
    """The aligned image (§4.2) of the blot element at `element`, or of the
    marker image (`element="marker"`), before `invert`; only the `crop`
    rectangle of it when given. `pixels` maps pixel hashes to channel
    images (as `Panel.channel_images` does)."""

    marker = element == "marker"
    view = _view(core, marker=marker)
    item = view.marker if marker else view.blot[element]
    channel = _channel(view, item, pixels)
    first = not marker and element == 0
    return _render.aligned(channel, _reference_shape(view), None if first else item, view.orientation,
                           _window(view, crop))


def render(core: dict, pixels: Mapping[str, np.ndarray], crop: dict | None = None) -> np.ndarray:
    """The rendered image of the `crop` rectangle (default_crop() when
    None), drawn by §6.2: grey (h, w) or RGB (h, w, 3), of the type
    `output_bit_depth` says."""

    view = _view(core)
    if view.output_bit_depth is None:
        raise ValueError("The core cannot be drawn: 'output_bit_depth' cannot be read.")
    window = _window(view, crop if crop is not None else default_crop(core))
    reference = _reference_shape(view)
    levels, colors, high_bit = [], [], False
    for i, element in enumerate(view.blot):
        image = _render.aligned(_channel(view, element, pixels), reference, None if i == 0 else element,
                                view.orientation, window)
        if element["invert"]:
            image = np.iinfo(image.dtype).max - image
        levels.append(_render.level(image, element["contrast"]))
        colors.append(element.get("color"))
        high_bit = high_bit or image.dtype == np.uint16
    grey = len(view.blot) == 1 and colors[0] is None
    return _render.compose(levels, None if grey else colors, high_bit, view.output_bit_depth)


# --- points -------------------------------------------------------------------------------------


def _stages(view: CoreView, element: int | str) -> list[tuple]:
    """The transforms from the element's channel image to the aligned
    image (§4.2), each with the size of the image it starts from."""

    marker = element == "marker"
    item = view.marker if marker else view.blot[element]
    height, width = view.source(item["source"])["shape"]
    stages = []

    def turn(degrees):
        nonlocal height, width
        stages.append(("rotate", degrees, width, height))
        frame_w, frame_h = _geometry.rotated_frame(width, height, degrees)
        width, height = round(frame_w), round(frame_h)

    def flip():
        stages.append(("flip", width))

    if marker or element != 0:
        if item["turn"]:
            turn(item["turn"])
        if item["flip"]:
            flip()
        ref_h, ref_w = _reference_shape(view)
        if (height, width) != (ref_h, ref_w):
            stages.append(("resize", width, height, ref_w, ref_h))
            height, width = ref_h, ref_w
    orientation = view.orientation
    if orientation["turn"]:
        turn(orientation["turn"])
    if orientation["flip"]:
        flip()
    if orientation["rotation"]:
        turn(orientation["rotation"])
    return stages


def _forward(stage: tuple, x: float, y: float) -> tuple[float, float]:
    kind = stage[0]
    if kind == "flip":
        return stage[1] - x, y
    if kind == "resize":
        _, w, h, new_w, new_h = stage
        return (x * new_w) / w, (y * new_h) / h
    _, degrees, w, h = stage
    c, s = _geometry.cos_sin(degrees)
    frame_w, frame_h = _geometry.rotated_frame(w, h, degrees)
    return (((c * (x - w / 2)) - (s * (y - h / 2))) + frame_w / 2,
            ((s * (x - w / 2)) + (c * (y - h / 2))) + frame_h / 2)


def _backward(stage: tuple, x: float, y: float) -> tuple[float, float]:
    kind = stage[0]
    if kind == "flip":
        return stage[1] - x, y
    if kind == "resize":
        _, w, h, new_w, new_h = stage
        return (x * w) / new_w, (y * h) / new_h
    _, degrees, w, h = stage
    c, s = _geometry.cos_sin(degrees)
    frame_w, frame_h = _geometry.rotated_frame(w, h, degrees)
    return (((c * (x - frame_w / 2)) + (s * (y - frame_h / 2))) + w / 2,
            (((-s) * (x - frame_w / 2)) + (c * (y - frame_h / 2))) + h / 2)


def to_channel(core: dict, x: float, y: float, element: int | str = 0) -> tuple[float, float]:
    """Where the panel's point (x, y) lies on the channel image of the blot
    element at `element`, or of the marker image (`"marker"`): §4.2
    followed backwards (§4.3). Points are not rounded."""

    view = _view(core, marker=element == "marker")
    for stage in reversed(_stages(view, element)):
        x, y = _backward(stage, x, y)
    return x, y


def to_panel(core: dict, x: float, y: float, element: int | str = 0) -> tuple[float, float]:
    """Where a point of the element's channel image lies in the panel's
    coordinates (§4.2, §4.3)."""

    view = _view(core, marker=element == "marker")
    for stage in _stages(view, element):
        x, y = _forward(stage, x, y)
    return x, y

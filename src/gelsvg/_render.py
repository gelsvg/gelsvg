"""Drawing: channel images → aligned images (§4.2–§4.4) → the rendered
image (§4.5, §6.2), with the reference interpolation of §4.3.

numpy does each elementwise operation in binary64 with one rounding, so
the arrays below follow §4.3's arithmetic as long as every expression keeps
the specification's brackets and the terms are added in its order.
"""

from __future__ import annotations

import numpy as np

from . import _geometry

_BLOCK = 1 << 20  # output values computed at once


def K(t: np.ndarray) -> np.ndarray:
    """Keys' cubic convolution kernel with a = −0.5, in §4.3's order."""

    t = np.abs(t)
    near = ((((1.5 * t) - 2.5) * t) * t) + 1
    far = ((((((-0.5) * t) + 2.5) * t) - 4) * t) + 2
    return np.where(t <= 1, near, np.where(t < 2, far, 0.0))


def _to_type(values: np.ndarray, dtype) -> np.ndarray:
    top = 255 if dtype == np.uint8 else 65535
    return np.clip(np.rint(values), 0, top).astype(dtype)


def flip(image: np.ndarray) -> np.ndarray:
    return image[:, ::-1].copy()


def rotate(image: np.ndarray, degrees: float, window: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Rotate clockwise by `degrees` (y points down). `window` = (top,
    bottom, left, right) limits the result to those rows and columns of the
    rotated frame."""

    c, s = _geometry.cos_sin(degrees)
    if c in (0.0, 1.0, -1.0) and s in (0.0, 1.0, -1.0):
        # u and v are whole numbers: K picks one pixel, so this is a
        # reordering of the pixels (§4.3).
        turned = np.rot90(image, k={(1.0, 0.0): 0, (0.0, 1.0): -1, (-1.0, 0.0): 2, (0.0, -1.0): 1}[(c, s)])
        if window is not None:
            top, bottom, left, right = window
            turned = turned[top:bottom, left:right]
        return turned.copy()
    return _to_type(rotation_values(image, degrees, window), image.dtype)


def rotation_values(image: np.ndarray, degrees: float, window: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """The rotated values before they are rounded to the type."""

    c, s = _geometry.cos_sin(degrees)
    h, w = image.shape
    frame_w, frame_h = _geometry.rotated_frame(w, h, degrees)
    out_h, out_w = round(frame_h), round(frame_w)
    top, bottom, left, right = window or (0, out_h, 0, out_w)
    source = image.astype(np.float64)
    out = np.empty((bottom - top, right - left), dtype=np.float64)
    cols = np.arange(left, right, dtype=np.float64) + 0.5
    rows_per_block = max(1, _BLOCK // max(1, right - left))
    half_w, half_h, w2, h2 = frame_w / 2, frame_h / 2, w / 2, h / 2
    for r0 in range(top, bottom, rows_per_block):
        r1 = min(bottom, r0 + rows_per_block)
        y_ = (np.arange(r0, r1, dtype=np.float64) + 0.5)[:, None]
        dx = cols[None, :] - half_w
        dy = y_ - half_h
        x = ((c * dx) + (s * dy)) + w2
        y = (((-s) * dx) + (c * dy)) + h2
        u = x - 0.5
        v = y - 0.5
        fu, fv = np.floor(u), np.floor(v)
        acc = np.zeros(np.broadcast(u, v).shape, dtype=np.float64)
        across = []
        for j in range(4):
            n = fu + (j - 1)
            ni = n.astype(np.int64)
            across.append((K(u - n), np.clip(ni, 0, w - 1), (ni >= 0) & (ni < w)))
        for i in range(4):
            m = fv + (i - 1)
            kv = K(v - m)
            mi = m.astype(np.int64)
            m_ok = (mi >= 0) & (mi < h)
            mc = np.clip(mi, 0, h - 1)
            for ku, nc, n_ok in across:
                p = np.where(m_ok & n_ok, source[mc, nc], 0.0)
                acc = acc + ((ku * kv) * p)
        out[r0 - top : r1 - top] = acc
    return out


def _resize_axis(values: np.ndarray, size: int) -> np.ndarray:
    """Resize the last axis to `size` (§4.3, one direction)."""

    w = values.shape[-1]
    s = w / size
    f = max(s, 1.0)
    reach = 2 * f
    c = (np.arange(size, dtype=np.float64) + 0.5) * s
    start = np.floor((c - 0.5) - reach).astype(np.int64) - 1
    k_sum = np.zeros(size, dtype=np.float64)
    p_sum = np.zeros(values.shape[:-1] + (size,), dtype=np.float64)
    for d in range(int(np.ceil(4 * f)) + 4):
        n = start + d
        offset = (n.astype(np.float64) + 0.5) - c
        ok = (n >= 0) & (n < w) & (np.abs(offset) < reach)
        k = K(offset / f)
        k_sum = np.where(ok, k_sum + k, k_sum)
        p_sum = np.where(ok, p_sum + (k * values[..., np.clip(n, 0, w - 1)]), p_sum)
    return p_sum / k_sum


def resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape == (height, width):
        return image.copy()
    return _to_type(resize_values(image, height, width), image.dtype)


def resize_values(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Across first, then down; the values across are not rounded. The
    values before they are rounded to the type."""

    across = _resize_axis(image.astype(np.float64), width)
    return _resize_axis(across.T, height).T


def average(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """§4.4."""

    total = red.astype(np.int64) + green + blue + 1
    return (total // 3).astype(red.dtype)


def aligned(channel: np.ndarray, reference_shape, element: dict | None, orientation: dict,
            window: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """The aligned image of one blot element or the marker image (§4.2).
    `element` is None for the reference channel."""

    image = channel
    if element is not None:
        if element["turn"]:
            image = rotate(image, element["turn"])
        if element["flip"]:
            image = flip(image)
        image = resize(image, *reference_shape)
    if orientation["turn"]:
        image = rotate(image, orientation["turn"])
    if orientation["flip"]:
        image = flip(image)
    if orientation["rotation"]:
        return rotate(image, orientation["rotation"], window)
    if window is not None:
        top, bottom, left, right = window
        image = image[top:bottom, left:right]
    return image.copy() if np.shares_memory(image, channel) else image


def level(values: np.ndarray, contrast) -> np.ndarray:
    """n of §4.5: the values (after `invert`) on 0 to 1."""

    low, high = float(contrast[0]), float(contrast[1])
    v = values.astype(np.float64)
    if low == high:
        return np.where(v > low, 1.0, 0.0)
    return np.clip((v - low) / (high - low), 0.0, 1.0)


def compose(levels: list[np.ndarray], colors: list | None, high_bit: bool, output_bit_depth: int) -> np.ndarray:
    """Steps 3 and 4 of §6.2. `colors` is None for a grey rendered image."""

    if colors is None:
        (c,) = levels
    else:
        channels = []
        for k in range(3):
            total = np.zeros_like(levels[0])
            for n, color in zip(levels, colors):
                total = total + (n * float(color[k]))
            channels.append(np.minimum(1.0, total))
        c = np.stack(channels, axis=-1)
    top = 65535.0 if high_bit else 255.0
    q = np.floor(c * top)
    if high_bit and output_bit_depth == 8:
        return np.rint((q * 255) / 65535).astype(np.uint8)
    return q.astype(np.uint16 if output_bit_depth == 16 else np.uint8)

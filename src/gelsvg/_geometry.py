"""The exact numbers §4.3 asks for: cos θ and sin θ correctly rounded, and
the size of the aligned image."""

from __future__ import annotations

from decimal import Decimal, localcontext

_PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798"
)


def _cos_sin_decimal(degrees: float) -> tuple[Decimal, Decimal]:
    with localcontext() as ctx:
        ctx.prec = 80
        x = Decimal(degrees) * _PI / 180  # Decimal(float) is the exact value of the double
        # Reduce to [-π, π] for a short series.
        two_pi = 2 * _PI
        x = x - two_pi * (x / two_pi).to_integral_value()
        term_c, term_s = Decimal(1), x
        cos, sin = Decimal(0), Decimal(0)
        n = 0
        while True:
            cos += term_c
            sin += term_s
            n += 2
            term_c = -term_c * x * x / (n * (n - 1))
            term_s = -term_s * x * x / ((n + 1) * n)
            if abs(term_c) < Decimal("1e-70") and abs(term_s) < Decimal("1e-70"):
                break
        return cos, sin


def cos_sin(degrees: float) -> tuple[float, float]:
    """cos and sin of degrees × π ÷ 180 taken exactly, each rounded to the
    nearest double. Multiples of 90° give exactly 0, 1 and −1."""

    if float(degrees) % 90 == 0:
        quarter = int(round(float(degrees) / 90)) % 4
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[quarter]
    cos, sin = _cos_sin_decimal(float(degrees))
    # float(str) rounds correctly to the nearest double.
    return float(str(cos)), float(str(sin))


def rotated_frame(width: float, height: float, degrees: float) -> tuple[float, float]:
    """W' and H' before rounding (§4.3)."""

    c, s = cos_sin(degrees)
    return (width * abs(c)) + (height * abs(s)), (width * abs(s)) + (height * abs(c))


def aligned_size(reference_shape: list[int], orientation: dict) -> tuple[int, int]:
    """(height, width) of the aligned image, from the reference channel's
    channel image size and the panel's orientation (§4.2, §4.3)."""

    height, width = reference_shape
    if orientation["turn"] in (90, 270):
        height, width = width, height
    rotation = orientation["rotation"]
    if rotation != 0:
        w, h = rotated_frame(width, height, rotation)
        width, height = round(w), round(h)  # round() is round-half-even
    return height, width

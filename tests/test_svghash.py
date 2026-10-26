import builders as B
import panels as P
from gelsvg import _hashes, _svghash
from gelsvg.document import read


def svg_with(png: bytes, extra: str = "") -> bytes:
    return (f'<svg xmlns="http://www.w3.org/2000/svg"><image href="{B.data_uri(png)}"/>{extra}</svg>').encode()


def signed(extra: str = "", after_signing: str = "") -> bytes:
    """An SVG whose panel records the svg_hash of the SVG as first written."""

    unsigned = svg_with(P.build(), extra)
    value = _svghash.svg_hash(unsigned)

    def change(p):
        p["figure"]["svg_hash"] = value
        p["figure"]["svg_hash_method"] = "c14n2_without_panel_data"
    return svg_with(P.build(change), extra + after_signing)


def test_the_panel_data_is_left_out_of_the_input():
    png = P.build()
    assert _svghash.svg_hash(svg_with(png)) == _svghash.svg_hash(svg_with(P.build(lambda p: p.update(note="x"))))


def test_unchanged_svg_matches():
    (panel,) = read(signed('<text x="1">A</text>')).panels
    assert panel.checks["svg_hash"] is True


def test_comments_do_not_count():
    (panel,) = read(signed('<text x="1">A</text>', "<!-- note -->")).panels
    assert panel.checks["svg_hash"] is True


def test_edited_svg_differs():
    (panel,) = read(signed('<text x="1">A</text>', '<text x="2">B</text>')).panels
    assert panel.checks["svg_hash"] is False


def test_undecodable_png_data_uri_makes_it_uncheckable():
    (panel,) = read(signed("", '<image href="data:image/png;base64,abc"/>')).panels
    assert panel.checks["svg_hash"] is None


def test_png_rebuilt_without_geLs_and_json():
    png = P.build()
    rebuilt = _svghash.png_without_panel_data(png + b"trailing")
    assert b"geLs" not in rebuilt and b"GelSVG" not in rebuilt
    assert rebuilt.endswith(B.IEND) and rebuilt.startswith(B.SIG)
    plain = B.image_png(B.small_pixels())
    assert _svghash.png_without_panel_data(plain) == plain

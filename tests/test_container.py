import base64

import builders as B
from gelsvg import Kind, read_container


def test_panels_in_document_order_and_other_images_skipped():
    plain = B.data_uri(B.image_png(B.small_pixels()))
    first = B.data_uri(B.carrier(b'{"n": 1}'))
    second = B.data_uri(B.carrier(b'{"n": 2}'))
    result = read_container(B.svg(first, plain, "pic.png", second))
    assert result.is_document
    assert [p.carrier.json_text for p in result.panels] == [b'{"n": 1}', b'{"n": 2}']
    assert [p.index for p in result.panels] == [0, 1]


def test_xlink_href_is_used_when_href_is_missing():
    result = read_container(B.svg(B.data_uri(B.carrier()), xlink=True))
    assert len(result.panels) == 1


def test_plain_href_wins_over_xlink():
    carrier = B.data_uri(B.carrier())
    data = ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
            f'<image href="pic.png" xlink:href="{carrier}"/></svg>').encode()
    assert not read_container(data).panels


def test_data_uri_variants():
    payload = base64.b64encode(B.carrier()).decode()
    spaced = " ".join(payload[i:i + 60] for i in range(0, len(payload), 60))
    for href in (f"DATA:Image/PNG ; charset=x ; BASE64,{payload}", f"data:image/png;base64,{spaced}"):
        assert len(read_container(B.svg(href)).panels) == 1


def test_undecodable_png_data_uri_is_reported():
    result = read_container(B.svg("data:image/png;base64,abc", B.data_uri(B.carrier())))
    assert len(result.panels) == 1
    assert [n.code for n in result.notices] == ["image.undecodable"]


def test_old_mywb_document_is_not_a_gelsvg_document():
    old = B.image_png(B.small_pixels(), before_idat=[B.itxt(b"{}", keyword=b"MyWB")])
    result = read_container(B.svg(B.data_uri(old)))
    assert not result.is_document and len(result.legacy) == 1
    assert result.notices[0].kind is Kind.UNREADABLE and result.notices[0].code == "document.legacy"


def test_svg_without_panels_is_not_a_document():
    result = read_container(B.svg(B.data_uri(B.image_png(B.small_pixels()))))
    assert not result.is_document and result.notices[0].code == "document.no_panel"


def test_png_that_lost_its_json_is_reported():
    png = B.image_png(B.small_pixels(), after_idat=[B.gels(b"x")])
    result = read_container(B.svg(B.data_uri(png), B.data_uri(B.carrier())))
    assert "image.panel_lost" in [n.code for n in result.notices]


def test_not_a_document_carries_a_hint():
    result = read_container('<?xml version="1.0" encoding="Shift_JIS"?><svg/>'.encode())
    assert result.notices[0].code == "document.encoding"
    assert "UTF-8" in result.notices[0].message


def test_panel_notices_carry_the_panel_index():
    damaged = bytearray(B.carrier(b"{}", after=[B.chunk(b"tEXt", b"k\0v")]))
    damaged[damaged.index(b"tEXt") + 5] ^= 1
    result = read_container(B.svg(B.data_uri(B.carrier()), B.data_uri(bytes(damaged))))
    (notice,) = result.panels[1].carrier.notices
    assert notice.panel == 1 and notice.kind is Kind.DAMAGED

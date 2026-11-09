"""Finding the element that draws some panels, and replacing it or the
root's attributes with every other byte kept (a writer redrawing one
assembly, §1.7)."""

import pytest

import builders as B
import panels as P
from gelsvg import _splice, _svgxml, drawing_of, read, replace_element, set_attributes

SVG_OPEN = '<svg xmlns="http://www.w3.org/2000/svg" width="10" viewBox="0 0 10 10">'


def doc_of(body: str, *pngs: bytes):
    """A document whose body is `body` with `{0}`, `{1}`… standing for the
    carrier PNGs' `image` elements."""

    images = [f'<image href="{B.data_uri(p)}"/>' for p in pngs]
    return read((SVG_OPEN + body.format(*images) + "</svg>").encode())


@pytest.fixture(scope="module")
def two():
    return P.build(), P.build(lambda p: p["core"].update(note="second"))


def test_end_tag_offsets_tell_nested_groups_apart():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><g id="a"><g id="b"><rect/></g><circle/></g></svg>'
    tree = _svgxml.parse(svg)
    outer, inner, rect = tree.elements[1], tree.elements[2], tree.elements[3]
    assert svg[slice(*_splice.element_span(svg, outer))] == b'<g id="a"><g id="b"><rect/></g><circle/></g>'
    assert svg[slice(*_splice.element_span(svg, inner))] == b'<g id="b"><rect/></g>'
    assert svg[slice(*_splice.element_span(svg, rect))] == b"<rect/>"


def test_end_tag_with_white_space_before_its_close():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><g><g/></g ></svg>'
    outer = _svgxml.parse(svg).elements[1]
    assert svg[slice(*_splice.element_span(svg, outer))] == b"<g><g/></g >"


def test_drawing_of_a_set_is_its_group(two):
    doc = doc_of('<g id="set"><g>{0}<text>a</text></g><g>{1}</g></g><path d="M0 0"/>', *two)
    element = drawing_of(doc, [0, 1])
    assert element.attributes["id"] == "set"


def test_drawing_of_one_panel_is_the_element_holding_its_image(two):
    doc = doc_of('<g id="one">{0}<text>a</text></g><g id="other">{1}</g>', *two)
    assert drawing_of(doc, [0]).attributes["id"] == "one"
    assert drawing_of(doc, [1]).attributes["id"] == "other"


def test_no_drawing_when_drawn_straight_in_the_root(two):
    doc = doc_of("{0}{1}", *two)
    assert drawing_of(doc, [0, 1]) is None
    assert drawing_of(doc, [0]) is None


def test_no_drawing_when_another_panel_is_drawn_with_them(two):
    doc = doc_of('<g id="both">{0}{1}</g>', *two)
    assert drawing_of(doc, [0]) is None
    assert drawing_of(doc, [0, 1]).attributes["id"] == "both"


def test_replace_element_keeps_every_other_byte(two):
    doc = doc_of('<g id="set"><g>{0}</g></g><!-- kept --><path d="M0 0"/>', *two)
    element = drawing_of(doc, [0])
    out = replace_element(doc.svg, element, b'<g id="new"/>')
    start, end = _splice.element_span(doc.svg, element)
    assert out == doc.svg[:start] + b'<g id="new"/>' + doc.svg[end:]
    assert b"<!-- kept -->" in out and read(out).panels == []


def test_set_attributes_replaces_and_adds():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" viewBox="0 0 10 10"><g/></svg>'
    root = _svgxml.parse(svg).root
    out = set_attributes(svg, root, {"width": "20px", "height": 'a"b'})
    assert out == (b'<svg xmlns="http://www.w3.org/2000/svg" width="20px" viewBox="0 0 10 10" height="a&quot;b">'
                   b"<g/></svg>")
    assert _svgxml.parse(out).root.attributes["height"] == 'a"b'

import gzip

import pytest

from gelsvg import _svgxml
from gelsvg._svgxml import NotADocument

SVG = '<svg xmlns="http://www.w3.org/2000/svg"><g><image href="x"/></g></svg>'


def reason(data: bytes) -> str:
    with pytest.raises(NotADocument) as info:
        _svgxml.parse(data)
    return info.value.reason


def test_plain_svg_tree():
    tree = _svgxml.parse(SVG.encode())
    assert [e.local for e in tree.elements] == ["svg", "g", "image"]
    image = tree.elements[2]
    assert [a.local for a in image.ancestors()] == ["g", "svg"]


def test_utf8_bom_and_declaration_are_fine():
    _svgxml.parse(b"\xef\xbb\xbf<?xml version='1.0' encoding='utf-8'?>" + SVG.encode())


@pytest.mark.parametrize("data, expected", [
    (gzip.compress(SVG.encode()), _svgxml.COMPRESSED),
    (SVG.encode("utf-16"), _svgxml.NOT_UTF8),
    (SVG.encode("utf-16-le"), _svgxml.NOT_UTF8),
    (b'<?xml version="1.0" encoding="ISO-8859-1"?>' + SVG.encode(), _svgxml.NOT_UTF8),
    (b'<svg xmlns="http://www.w3.org/2000/svg">\xff</svg>', _svgxml.NOT_UTF8),
    (b'<?xml version="1.1"?>' + SVG.encode(), _svgxml.VERSION),
    (b"<svg><image/></svg>", _svgxml.NOT_SVG),
    (b'<svg xmlns="http://www.w3.org/2000/svg"><a:b/></svg>', _svgxml.MALFORMED),
    (b'<svg xmlns="http://www.w3.org/2000/svg"><g></svg>', _svgxml.MALFORMED),
])
def test_files_that_are_not_documents(data, expected):
    assert reason(data) == expected


def test_doctype_naming_a_dtd_is_read_as_absent():
    doctype = '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">'
    _svgxml.parse((doctype + SVG).encode())


def test_internal_subset_with_only_comments_is_fine():
    _svgxml.parse(("<!DOCTYPE svg [ <!-- nothing --> ]>" + SVG).encode())


@pytest.mark.parametrize("subset", [
    '<!ENTITY ns_svg "http://www.w3.org/2000/svg">',
    "<!ATTLIST svg version CDATA '1.1'>",
    "%pe;",
    "<?pi x?>",
])
def test_internal_subset_with_anything_else_is_refused(subset):
    assert reason(f"<!DOCTYPE svg [ {subset} ]>{SVG}".encode()) == _svgxml.DOCTYPE


def test_entity_reference_with_an_external_dtd_is_refused():
    doctype = '<!DOCTYPE svg SYSTEM "svg11.dtd">'
    data = doctype + '<svg xmlns="http://www.w3.org/2000/svg"><text>&nbsp;</text></svg>'
    assert reason(data.encode()) == _svgxml.REFERENCE


def test_undeclared_entity_reference_is_refused():
    data = '<svg xmlns="http://www.w3.org/2000/svg"><text>&nbsp;</text></svg>'
    assert reason(data.encode()) == _svgxml.REFERENCE


def test_predefined_and_character_references_are_fine():
    _svgxml.parse('<svg xmlns="http://www.w3.org/2000/svg"><text>&amp;&lt;&#x41;</text></svg>'.encode())

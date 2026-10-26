import struct

import pytest

from gelsvg import _json
from gelsvg._json import JsonError


def test_reads_an_object_keeping_number_notation():
    value = _json.parse(b'{"a": 90, "b": 90.0, "c": 1e1, "d": -0, "e": [1, "x"]}')
    assert type(value["a"]) is int and type(value["b"]) is float and type(value["c"]) is float
    assert value["d"] == 0 and value["e"] == [1, "x"]


@pytest.mark.parametrize("text, problem", [
    (b"\xef\xbb\xbf{}", "byte order mark"),
    (b'{"a": "\xff"}', "UTF-8"),
    (b"[]", "not an object"),
    (b'"x"', "not an object"),
    (b'{"a": 1, "a": 2}', "twice"),
    (b'{"x": {"a": 1, "a": 1}}', "twice"),
    (b'{"a": NaN}', "NaN"),
    (b'{"a": -Infinity}', "Infinity"),
    (b'{"a": 9007199254740992}', "2^53"),
    (b'{"a": -9007199254740992}', "2^53"),
    (b'{"a": 1e400}', "too large"),
    (b'{"a": -1e400}', "too large"),
    (b'{"a": 1e-400}', "too small"),
    (b'{"a": "\\ud800"}', "surrogate"),
    (b'{"\\udc00": 1}', "surrogate"),
    (b'{"a": ["\\ud83d"]}', "surrogate"),
    (b'{"a": "\\uffff"}', "noncharacter"),
    (b'{"a": "\\ufdd0"}', "noncharacter"),
    (b'{"a": "\xf0\x9f\xbf\xbe"}', "noncharacter"),
    (b'{"a": 01}', "not JSON"),
    (b'{"a": 1,}', "not JSON"),
    (b'{"a": "\x01"}', "not JSON"),
])
def test_rejects(text, problem):
    with pytest.raises(JsonError) as info:
        _json.parse(text)
    assert problem in str(info.value)


def test_accepts_edges():
    value = _json.parse(b'{"a": 9007199254740991, "b": 0e5, "c": 5e-324, "d": "\\ud83d\\ude00"}')
    assert value["a"] == 2**53 - 1 and value["b"] == 0.0 and value["d"] == "\U0001F600"


# RFC 8785 Appendix B, checked against JavaScript's JSON.stringify.
NUMBER_SAMPLES = [
    ("0000000000000000", "0"),
    ("8000000000000000", "0"),
    ("0000000000000001", "5e-324"),
    ("8000000000000001", "-5e-324"),
    ("7fefffffffffffff", "1.7976931348623157e+308"),
    ("ffefffffffffffff", "-1.7976931348623157e+308"),
    ("4340000000000000", "9007199254740992"),
    ("c340000000000000", "-9007199254740992"),
    ("4430000000000000", "295147905179352830000"),
    ("44b52d02c7e14af5", "9.999999999999997e+22"),
    ("44b52d02c7e14af6", "1e+23"),
    ("44b52d02c7e14af7", "1.0000000000000001e+23"),
    ("444b1ae4d6e2ef4e", "999999999999999700000"),
    ("444b1ae4d6e2ef4f", "999999999999999900000"),
    ("444b1ae4d6e2ef50", "1e+21"),
    ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
    ("3eb0c6f7a0b5ed8d", "0.000001"),
    ("41b3de4355555553", "333333333.3333332"),
    ("41b3de4355555554", "333333333.33333325"),
    ("41b3de4355555555", "333333333.3333333"),
    ("41b3de4355555556", "333333333.3333334"),
    ("41b3de4355555557", "333333333.33333343"),
    ("becbf647612f3696", "-0.0000033333333333333333"),
    ("43143ff3c1cb0959", "1424953923781206.2"),
]


@pytest.mark.parametrize("bits, expected", NUMBER_SAMPLES)
def test_number_serialisation(bits, expected):
    (x,) = struct.unpack(">d", bytes.fromhex(bits))
    assert _json.number(x) == expected


def test_integers_and_small_values():
    assert _json.number(10) == "10" and _json.number(-3) == "-3"
    assert _json.number(0.001) == "0.001" and _json.number(1e-7) == "1e-7"
    assert _json.number(1e16) == "10000000000000000" and _json.number(123.0) == "123"


def test_key_order_is_by_utf16_code_units():
    value = {"€": "Euro Sign", "\r": "Carriage Return", "דּ": "Hebrew Letter Dalet With Dagesh",
             "1": "One", "\U0001f600": "Emoji: Grinning Face", "\u0080": "Control", "ö": "Latin Small Letter O With Diaeresis"}
    keys = list(_json.canonical(value).decode("utf-8")[1:-1].split(","))
    assert [k.split(":")[0] for k in keys] == ['"\\r"', '"1"', '"\u0080"', '"ö"', '"€"', '"\U0001f600"', '"דּ"']


def test_canonical_strings_and_literals():
    value = {"b": [None, True, False, 1.5], "a": "q\"\\\b\f\n\r\t\x01 é"}
    assert _json.canonical(value) == ('{"a":"q\\"\\\\\\b\\f\\n\\r\\t\\u0001 é","b":[null,true,false,1.5]}').encode("utf-8")

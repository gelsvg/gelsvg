"""The panel's JSON: strict I-JSON reading (§2.1, §2.2) and canonical JSON
(RFC 8785, §2.7).

Numbers keep how they were written: an integer literal (no fraction, no
exponent) is read as `int`, anything else as `float`. §2.2 decides integer
keys by notation, so `90.0` stays a float and is the wrong type for an
integer key.
"""

from __future__ import annotations

import json
import math

MAX_SAFE_INTEGER = 2**53 - 1


class JsonError(ValueError):
    """The text is not JSON this format reads (§2.1, rule 5)."""


class _Reject(Exception):
    pass


def _is_noncharacter(cp: int) -> bool:
    return 0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFE) == 0xFFFE


def _check_string(s: str) -> None:
    for ch in s:
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:
            raise _Reject("a lone surrogate")
        if _is_noncharacter(cp):
            raise _Reject(f"the Unicode noncharacter U+{cp:04X}")


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise _Reject(f"the key {key!r} twice in one object")
        _check_string(key)
        if isinstance(value, str):
            _check_string(value)
        out[key] = value
    return out


def _parse_int(literal: str) -> int:
    value = int(literal)
    if abs(value) > MAX_SAFE_INTEGER:
        raise _Reject(f"the integer {literal} beyond ±(2^53 − 1)")
    return value


def _parse_float(literal: str) -> float:
    value = float(literal)
    if math.isinf(value):
        raise _Reject(f"the number {literal} too large for a double")
    if value == 0.0:
        mantissa = literal.lstrip("-").split("e")[0].split("E")[0]
        if any(c in "123456789" for c in mantissa):
            raise _Reject(f"the number {literal} too small for a double")
    return value


def _parse_constant(name: str):
    raise _Reject(f"{name}, which is not JSON")


def _check_tree(value) -> None:
    # Strings inside arrays are not seen by the object hook.
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            for element in item:
                if isinstance(element, str):
                    _check_string(element)
                elif isinstance(element, (list, dict)):
                    stack.append(element)
        elif isinstance(item, dict):
            stack.extend(v for v in item.values() if isinstance(v, (list, dict)))


def parse(text: bytes) -> dict:
    """Read the iTXt text as the panel's JSON object, or raise JsonError."""

    if text.startswith(b"\xef\xbb\xbf"):
        raise JsonError("the text starts with a byte order mark")
    try:
        decoded = text.decode("utf-8")
    except UnicodeDecodeError:
        raise JsonError("the text is not valid UTF-8") from None
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_pairs,
            parse_int=_parse_int,
            parse_float=_parse_float,
            parse_constant=_parse_constant,
        )
        if isinstance(value, str):
            _check_string(value)
        _check_tree(value)
    except _Reject as exc:
        raise JsonError(f"the text holds {exc}") from None
    except json.JSONDecodeError as exc:
        raise JsonError(f"not JSON: {exc.msg} at character {exc.pos}") from None
    except RecursionError:
        raise JsonError("the JSON is nested too deeply to read") from None
    if not isinstance(value, dict):
        raise JsonError("the JSON is not an object")
    return value


# --- canonical JSON (RFC 8785) -------------------------------------------------

_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _string(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _shortest_digits(x: float) -> tuple[str, int]:
    """The shortest digit string d and exponent n with x = 0.d × 10^n
    (ECMAScript's s, k and n: d has k digits, the point sits after n)."""

    text = repr(abs(x))
    if "e" in text:
        mantissa, exponent = text.split("e")
        exponent = int(exponent)
    else:
        mantissa, exponent = text, 0
    if "." in mantissa:
        whole, fraction = mantissa.split(".")
    else:
        whole, fraction = mantissa, ""
    digits = whole + fraction
    point = len(whole) + exponent
    stripped = digits.lstrip("0")
    point -= len(digits) - len(stripped)
    stripped = stripped.rstrip("0")
    return stripped, point


def number(x: int | float) -> str:
    """ECMAScript Number.prototype.toString, as RFC 8785 §3.2.2.3 requires."""

    if isinstance(x, bool):
        raise TypeError("booleans are not numbers")
    if isinstance(x, int):
        x = float(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError("NaN and infinities have no JSON form")
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    digits, n = _shortest_digits(x)
    k = len(digits)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    exponent = n - 1
    exp_text = ("+" if exponent >= 0 else "-") + str(abs(exponent))
    if k == 1:
        return sign + digits + "e" + exp_text
    return sign + digits[0] + "." + digits[1:] + "e" + exp_text


def _serialize(value, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, (int, float)):
        out.append(number(value))
    elif isinstance(value, str):
        out.append(_string(value))
    elif isinstance(value, list):
        out.append("[")
        for i, item in enumerate(value):
            if i:
                out.append(",")
            _serialize(item, out)
        out.append("]")
    elif isinstance(value, dict):
        out.append("{")
        for i, key in enumerate(sorted(value, key=lambda k: k.encode("utf-16-be"))):
            if i:
                out.append(",")
            out.append(_string(key))
            out.append(":")
            _serialize(value[key], out)
        out.append("}")
    else:
        raise TypeError(f"cannot canonicalize {type(value).__name__}")


def canonical(value) -> bytes:
    """The RFC 8785 canonical form of a JSON value, as UTF-8 bytes."""

    out: list[str] = []
    _serialize(value, out)
    return "".join(out).encode("utf-8")

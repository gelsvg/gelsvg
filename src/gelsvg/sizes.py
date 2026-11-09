"""Reading marker labels as sizes (§5.14 "Reading a size").

A marker label holds only the text drawn. It reads as a size when it is a
number, optionally followed by a unit; as nothing otherwise. A size written
without a unit takes the unit `core.marker_unit` names, when it names one: a
unit alone, optionally in parentheses (the `kDa` drawn beside a ladder's
numbers).
"""

from __future__ import annotations

import re

UNITS = {"kda": "kDa", "kd": "kDa", "bp": "bp", "kb": "kb", "kbp": "kb", "nt": "nt"}
"""The units a label may name, without regard to case, and the unit each is."""

_UNIT = "kda|kd|kbp|kb|bp|nt"
_SIZE = re.compile(rf"([0-9]{{1,3}}(?:,[0-9]{{3}})+|[0-9]+)(\.[0-9]+)? *({_UNIT})?", re.IGNORECASE)
_ALONE = re.compile(rf"({_UNIT})|\(({_UNIT})\)", re.IGNORECASE)


def read_label(label: str) -> tuple | None:
    """What one label reads as: (number, unit or None), or None for any other
    text."""

    match = _SIZE.fullmatch(label.strip())
    if not match:
        return None
    number = float(match.group(1).replace(",", "") + (match.group(2) or ""))
    if number <= 0:
        return None
    return number, UNITS[match.group(3).lower()] if match.group(3) else None


def read_unit(marker_unit: str | None) -> str | None:
    """The unit `core.marker_unit` names, or None: absent, or other text
    (`MW`), which is drawn but names no unit."""

    if not isinstance(marker_unit, str):
        return None
    match = _ALONE.fullmatch(marker_unit.strip())
    return UNITS[(match.group(1) or match.group(2)).lower()] if match else None


def marker_sizes(markers, marker_unit: str | None = None) -> list[dict | None]:
    """The size each of a panel's marker labels reads as, in their order:
    `{"value", "unit"}` (`unit` None when it cannot be determined), or None
    for a label that is no size. `markers` and `marker_unit` as
    `core.markers` and `core.marker_unit` hold them."""

    shared = read_unit(marker_unit)
    out = []
    for m in markers:
        read = read_label(m["label"]) if isinstance(m.get("label"), str) else None
        if read is None:
            out.append(None)
            continue
        number, unit = read
        out.append({"value": int(number) if number.is_integer() else number,
                    "unit": unit if unit is not None else shared})
    return out

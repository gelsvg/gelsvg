"""What a reader tells its caller about what it dropped, ignored or changed.

The specification (§2.6) requires a reader to report everything it discarded,
ignored or read differently, grouped by kind. Each notice carries one of the
kinds below, a stable code that programs can match on, and an English message
for people.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum


class Kind(enum.Enum):
    """The notice kinds of §2.6."""

    BROKEN = "broken"
    """A writer's mistake: a broken element dropped, an optional key ignored,
    a reference with nothing to point to, a chunk written wrongly."""

    DAMAGED = "damaged"
    """Bytes changed after writing (a CRC mismatch, a file cut short)."""

    UNKNOWN = "unknown"
    """Something added by a newer revision or version, not used."""

    REINTERPRETED = "reinterpreted"
    """Read differently from how it is written (CR LF and CR read as LF)."""

    UNREADABLE = "unreadable"
    """A core key, a panel or the whole document could not be read."""

    UNHANDLED = "unhandled"
    """Refused because of its size (§1.5)."""

    ASSEMBLY = "assembly"
    """How the assemblies were put together (copies split or not, members
    missing)."""

    CHECK = "check"
    """The result of comparing a hash or another recorded value."""


@dataclass(frozen=True)
class Notice:
    kind: Kind
    code: str
    message: str
    panel: int | None = None
    """The panel's position in document order, counting from 0, when the
    notice is about one panel."""
    where: str | None = None
    """Where in the panel or the PNG, in a form people can read."""

    def __str__(self) -> str:
        place = []
        if self.panel is not None:
            place.append(f"panel {self.panel}")
        if self.where:
            place.append(self.where)
        prefix = f"[{self.kind.value}] "
        if place:
            prefix += "(" + ", ".join(place) + ") "
        return prefix + self.message

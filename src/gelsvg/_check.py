"""Shared pieces of the panel layer: value types (§2.2, §2.3), the
reinterpretations every reader makes (§2.4, §2.5), and the bookkeeping of
notices by kind (§2.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .notices import Kind, Notice

KNOWN_VERSION = 1
KNOWN_REVISION = 0

_ID = re.compile(r"[A-Za-z0-9._-]+")
_HASH = re.compile(r"([a-z0-9-]+):(.+)", re.DOTALL)
_SHA256_BODY = re.compile(r"[0-9a-f]{64}")


def is_int(value) -> bool:
    return type(value) is int


def is_number(value) -> bool:
    return type(value) in (int, float)


def is_bool(value) -> bool:
    return type(value) is bool


def is_str(value) -> bool:
    return type(value) is str


def is_id(value) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def is_shape(value) -> bool:
    return (type(value) is list and len(value) == 2
            and all(type(v) is int and v > 0 for v in value))


def hash_parts(value) -> tuple[str, str] | None:
    """(head, body) of a value of the hash type, or None when the value is
    not one (§2.2). A head this reader does not know still has the type."""

    if type(value) is not str:
        return None
    match = _HASH.fullmatch(value)
    if match is None:
        return None
    head, body = match.groups()
    if head == "sha256" and _SHA256_BODY.fullmatch(body) is None:
        return None
    return head, body


def is_hash(value) -> bool:
    return hash_parts(value) is not None


def sha256_hash(digest_hex: str) -> str:
    return "sha256:" + digest_hex


def is_empty(value) -> bool:
    return value == "" or value == [] or value == {}


def lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


@dataclass
class Context:
    """Per-panel state: the panel's revision decides whether something
    unknown is a newer addition or a writer's mistake (rule 1)."""

    panel: int | None
    newer_revision: bool = True
    notices: list[Notice] = field(default_factory=list)

    def note(self, kind: Kind, code: str, message: str, where: str | None = None) -> None:
        self.notices.append(Notice(kind, code, message, panel=self.panel, where=where))

    def broken(self, code: str, message: str, where: str) -> None:
        self.note(Kind.BROKEN, code, message, where)

    def unknown_key(self, where: str, key: str) -> None:
        if self.newer_revision:
            self.note(Kind.UNKNOWN, "key.unknown",
                      f"The key {key!r} is not known to this reader and was not used; "
                      "rewriting this panel would drop it.", f"{where}.{key}")
        else:
            self.broken("key.undefined",
                        f"The key {key!r} is not defined by this version of the format and was ignored.",
                        f"{where}.{key}")

    def unknown_value(self, where: str, value) -> bool:
        """Report an enumerated key's value this reader does not know.
        Returns True when it is a newer addition (rule 2: the dependent thing
        is not used, the element is not broken) and False when it is a
        writer's mistake (rule 1: a value error)."""

        if self.newer_revision:
            self.note(Kind.UNKNOWN, "value.unknown",
                      f"The value {value!r} is not known to this reader; what depends on it was not used. "
                      "Rewriting this panel would drop it.", where)
            return True
        return False

    def hash_error(self, where: str, value: str) -> bool:
        """§2.2 for a value of the hash type: whether its kind is a writer's
        error. A kind this reader does not know only cannot be checked in a
        panel of a newer revision (reported as unknown, rule 2); in a panel
        of the same or an older revision it is an error in the value (rule
        1), which the caller handles."""

        return hash_parts(value)[0] != "sha256" and not self.unknown_value(where, value)

    def reinterpreted(self, where: str) -> None:
        self.note(Kind.REINTERPRETED, "text.cr",
                  "CR LF and CR were read as LF.", where)


def strip_text(ctx: Context, value: str, where: str) -> str:
    """Apply §2.5 to a string value of the format-defined parts."""

    if "\r" in value:
        ctx.reinterpreted(where)
        return lf(value)
    return value


class Obj:
    """Read one JSON object key by key, recording what was ignored. The
    caller decides what makes the element broken; this class handles the
    common cases of rules 1 and 3 and §2.4."""

    def __init__(self, ctx: Context, value: dict, where: str, closed: bool = False):
        self.ctx = ctx
        self.value = value
        self.where = where
        self.closed = closed
        self.seen: set[str] = set()
        self.unknown_in_closed = False

    def has(self, key: str) -> bool:
        return key in self.value

    def raw(self, key: str):
        self.seen.add(key)
        return self.value.get(key)

    def optional(self, key: str, check, describe: str):
        """An optional key: absent, null, empty or of the wrong type or range
        means absent; all but the first are reported (rule 3, §2.4)."""

        self.seen.add(key)
        if key not in self.value:
            return None
        v = self.value[key]
        if v is None:
            self.ctx.broken("key.null", f"{key!r} is null and was read as absent.", f"{self.where}.{key}")
            return None
        if is_empty(v):
            self.ctx.broken("key.empty", f"{key!r} is empty and was read as absent.", f"{self.where}.{key}")
            return None
        if not check(v):
            self.ctx.broken("key.invalid", f"{key!r} is not {describe} and was ignored.", f"{self.where}.{key}")
            return None
        return self.text(key, v)

    def required(self, key: str, check, describe: str, problems: list[str]):
        """A required key: anything wrong is a reason the element is broken."""

        self.seen.add(key)
        if key not in self.value or self.value[key] is None:
            problems.append(f"{key!r} is missing")
            return None
        v = self.value[key]
        if not check(v):
            problems.append(f"{key!r} is not {describe}")
            return None
        return self.text(key, v)

    def forbidden(self, key: str, reason: str) -> None:
        """A key that must not be written in this case (rule 3: ignored)."""

        self.seen.add(key)
        if key in self.value:
            self.ctx.broken("key.not_allowed", f"{key!r} must not be written here ({reason}) and was ignored.",
                            f"{self.where}.{key}")

    def text(self, key: str, v):
        if type(v) is str:
            return strip_text(self.ctx, v, f"{self.where}.{key}")
        return v

    def finish(self) -> None:
        """Report the keys nobody asked for (rule 1)."""

        for key in self.value:
            if key in self.seen:
                continue
            if self.closed:
                self.unknown_in_closed = True
            else:
                self.ctx.unknown_key(self.where, key)

"""TIFF structure: the header and IFDs of a TIFF file, or of a TIFF-shaped
block inside another file (EXIF, MPF)."""

from __future__ import annotations

from dataclasses import dataclass
import struct

# §5.4: bytes per value of each field type (BigTIFF's 16–18 included).
TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8}
_UNSIGNED = {1: "B", 3: "H", 4: "I", 13: "I", 16: "Q", 18: "Q"}


class TiffError(ValueError):
    """The structure cannot be read."""


@dataclass(frozen=True)
class Entry:
    tag: int
    type: int
    count: int
    value: bytes | None
    """The value's bytes in the file's byte order; None when the type is not
    known or the value lies outside the data."""
    known_type: bool
    inside: bool


class Tiff:
    def __init__(self, data: bytes, bigtiff: bool = True):
        self.data = data
        if len(data) < 8 or data[:2] not in (b"II", b"MM"):
            raise TiffError("not a TIFF header")
        self.byte_order = data[:2].decode("ascii")
        self.endian = "<" if self.byte_order == "II" else ">"
        self.version = self._unpack("H", 2)
        if self.version == 42:
            self.offset_size = 4
            self.first_ifd = self._unpack("I", 4)
        elif self.version == 43 and bigtiff:
            if len(data) < 16 or self._unpack("H", 4) != 8 or self._unpack("H", 6) != 0:
                raise TiffError("the BigTIFF header is damaged")
            self.offset_size = 8
            self.first_ifd = self._unpack("Q", 8)
        else:
            raise TiffError("not a TIFF header")
        self.offset_format = "I" if self.offset_size == 4 else "Q"

    def _unpack(self, fmt: str, offset: int):
        size = struct.calcsize(fmt)
        if offset < 0 or offset + size > len(self.data):
            raise TiffError("a structure runs past the end of the data")
        return struct.unpack_from(self.endian + fmt, self.data, offset)[0]

    def ifd(self, offset: int) -> tuple[list[Entry], int]:
        """The entries of the IFD at `offset`, in directory order, and the
        offset of the next IFD. Raises TiffError when the directory itself
        lies outside the data."""

        count_format = "H" if self.offset_size == 4 else "Q"
        entry_size = 12 if self.offset_size == 4 else 20
        count = self._unpack(count_format, offset)
        first = offset + struct.calcsize(count_format)
        if first + count * entry_size + self.offset_size > len(self.data):
            raise TiffError("an IFD runs past the end of the data")
        entries = []
        for i in range(count):
            at = first + i * entry_size
            tag, field_type = self._unpack("H", at), self._unpack("H", at + 2)
            if self.offset_size == 4:
                n, value_field = self._unpack("I", at + 4), at + 8
            else:
                n, value_field = self._unpack("Q", at + 4), at + 12
            size = TYPE_SIZES.get(field_type)
            if size is None:
                entries.append(Entry(tag, field_type, n, None, False, True))
                continue
            length = n * size
            start = value_field if length <= self.offset_size else self._unpack(self.offset_format, value_field)
            if start + length > len(self.data):
                entries.append(Entry(tag, field_type, n, None, True, False))
                continue
            entries.append(Entry(tag, field_type, n, self.data[start : start + length], True, True))
        return entries, self._unpack(self.offset_format, first + count * entry_size)

    def ints(self, entry: Entry) -> list[int] | None:
        """The values of an entry of an unsigned integer type, or None."""

        fmt = _UNSIGNED.get(entry.type)
        if fmt is None or entry.value is None:
            return None
        return list(struct.unpack(f"{self.endian}{entry.count}{fmt}", entry.value))


def first(entries: list[Entry], tag: int) -> Entry | None:
    return next((e for e in entries if e.tag == tag), None)

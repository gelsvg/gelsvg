"""Reading `figure.style` (§6.7): the style the figure was drawn with.

Every key is optional and has no default: an absent `style`, or an absent
key, is not reported, and means "left to the tool". Each key that cannot be
read is read as absent on its own (rule 3); an enumerated key's value this
reader does not know leaves what depends on it unused (rule 2). The result
keeps, in the order written, everything that was read, keys a figure did not
use included (they are the tool's settings, written back as read, §6.7); a
reader that draws the figure must not take those as the figure's style.

With `row_name` written, an absent `hidden_rows` means that no row's name
was left out; none drawn is every row in it (there is no `visible`).

`side` is the side drawn on; `side_automatic`, whether the tool decided it.
A `side_automatic` without a usable `side` is not used (reported when no
`side` was written), and `mixed` only goes with `side_automatic` true.

`bottom_from` and `hidden_rows` point into the core the figure was drawn
from. They are checked against the core only when `drawn_from` matches it;
otherwise IDs the core lacks are kept for writing back, and a reader does
not use them (nor `side: split` without a usable `bottom_from`).
"""

from __future__ import annotations

from . import _check as C
from ._check import Context, Obj

SAMPLE_FORMS = ("numbers", "table", "rotated")
SAMPLE_SIDES = ("top", "bottom", "split", "mixed")
SIDES = ("left", "right", "mixed")


def _positive(value) -> bool:
    return C.is_number(value) and value > 0


def _non_negative(value) -> bool:
    return C.is_number(value) and value >= 0


def _is_object(value) -> bool:
    return type(value) is dict


def _in_order(raw: dict, values: dict) -> dict | None:
    out = {key: values[key] for key in raw if values.get(key) is not None}
    return out or None


def _enum(ctx: Context, o: Obj, key: str, allowed: tuple[str, ...]) -> tuple[str | None, bool]:
    """(value, unknown): the value when it is one of `allowed`; unknown is
    True for a value of a newer revision (rule 2)."""

    value = o.optional(key, C.is_str, "a string")
    if value is None or value in allowed:
        return value, False
    if ctx.unknown_value(f"{o.where}.{key}", value):
        return None, True
    ctx.broken("key.invalid", f"{key!r} is not one of {', '.join(allowed)} and was ignored.", f"{o.where}.{key}")
    return None, False


def _side(ctx: Context, o: Obj, allowed: tuple[str, ...]) -> tuple[str | None, bool | None]:
    """`side` and `side_automatic` as read (§6.7 "Sides"): `mixed` without
    `side_automatic` true is a writer's error, read as no side."""

    side, _ = _enum(ctx, o, "side", allowed)
    automatic = o.optional("side_automatic", C.is_bool, "true or false")
    if side == "mixed" and automatic is not True:
        ctx.broken("style.mixed", "'side' is 'mixed', but 'side_automatic' is not true; 'side' was read as absent.",
                   f"{o.where}.side")
        side = None
    return side, automatic


def _automatic(ctx: Context, o: Obj, raw: dict, side: str | None, automatic: bool | None) -> bool | None:
    """`side_automatic`, which goes with a `side`: without one written it is
    a writer's error; with one that could not be used, it is unused too."""

    if automatic is None or side is not None:
        return automatic
    if "side" not in raw:
        ctx.broken("style.side_automatic", "'side_automatic' is written without 'side' and was ignored.",
                   f"{o.where}.side_automatic")
    return None


def _sample_label(ctx: Context, raw: dict, condition_ids: list[str], drawn: bool) -> dict | None:
    o = Obj(ctx, raw, "figure.style.sample_label")
    form, form_unknown = _enum(ctx, o, "form", SAMPLE_FORMS)
    side, automatic = _side(ctx, o, SAMPLE_SIDES)
    bottom_from = o.optional("bottom_from", C.is_id, "an ID")
    angle = o.optional("angle", lambda v: C.is_number(v) and -90 <= v <= 90, "a number from −90 to 90")
    merge = o.optional("merge_same_values", C.is_bool, "true or false")
    font_size = o.optional("font_size", _positive, "a number greater than 0")
    o.finish()
    if form_unknown:
        # Rule 2: an unknown form leaves the form, angle, bottom_from, merging and a split unused.
        angle = bottom_from = merge = None
        if side == "split":
            side = None
    elif side == "split" and (form != "table" or bottom_from is None):
        ctx.broken("style.split", "'side' is 'split', but 'form' is not 'table' or 'bottom_from' is missing; "
                   "'side' was read as absent.", f"{o.where}.side")
        side = None
    if bottom_from is not None and drawn:
        first = condition_ids[0] if condition_ids else None
        if bottom_from not in condition_ids or (side == "split" and bottom_from == first):
            why = "is not a condition of the core" if bottom_from not in condition_ids else "is the first condition"
            ctx.broken("style.bottom_from", f"'bottom_from' {why}; it was read as absent"
                       + (", and so was 'side'." if side == "split" else "."), f"{o.where}.bottom_from")
            bottom_from = None
            if side == "split":
                side = None
    automatic = _automatic(ctx, o, raw, side, automatic)
    return _in_order(raw, {"form": form, "side": side, "side_automatic": automatic, "bottom_from": bottom_from,
                           "merge_same_values": merge, "angle": angle, "font_size": font_size})


def _details_list(ctx: Context, raw: dict) -> dict | None:
    o = Obj(ctx, raw, "figure.style.details_list")
    values = {"visible": o.optional("visible", C.is_bool, "true or false"),
              "font_size": o.optional("font_size", _positive, "a number greater than 0")}
    o.finish()
    return _in_order(raw, values)


def _labels(ctx: Context, raw: dict, where: str, side: bool) -> tuple[Obj, dict]:
    o = Obj(ctx, raw, where)
    values = {"font_size": o.optional("font_size", _positive, "a number greater than 0")}
    if side:
        found, automatic = _side(ctx, o, SIDES)
        values["side"] = found
        values["side_automatic"] = _automatic(ctx, o, raw, found, automatic)
    return o, values


def _row_name(ctx: Context, raw: dict, row_ids: set[str], drawn: bool) -> dict | None:
    o, values = _labels(ctx, raw, "figure.style.row_name", side=True)
    hidden = o.optional("hidden_rows", lambda v: type(v) is list and all(C.is_id(x) for x in v), "an array of row IDs")
    o.finish()
    if hidden is not None:
        once = list(dict.fromkeys(hidden))
        if len(once) != len(hidden):
            ctx.broken("style.hidden_rows.repeated", "'hidden_rows' names a row twice; the repeats were dropped.",
                       f"{o.where}.hidden_rows")
        if drawn:
            missing = [row for row in once if row not in row_ids]
            if missing:
                ctx.broken("style.hidden_rows", f"'hidden_rows' names rows the core does not have "
                           f"({', '.join(missing)}); they were dropped.", f"{o.where}.hidden_rows")
                once = [row for row in once if row in row_ids]
        hidden = once or None
    values["hidden_rows"] = hidden
    return _in_order(raw, values)


def read_style(ctx: Context, raw: dict | None, view, drawn: bool) -> dict | None:
    """The style as read, or None. `view` is the core as read (CoreView);
    `drawn` whether `drawn_from` matches it."""

    if raw is None:
        return None
    condition_ids = [c["id"] for c in view.conditions] if view is not None else []
    row_ids = ({r["id"] for r in view.rows} - view.ambiguous.get("rows", set())) if view is not None else set()
    o = Obj(ctx, raw, "figure.style")
    values: dict = {
        "font_family": o.optional("font_family", C.is_str, "a string"),
        "image_scale": o.optional("image_scale", _positive, "a number greater than 0"),
        "image_border_width": o.optional("image_border_width", _non_negative, "a number of 0 or more"),
    }
    for key in ("sample_label", "details_list", "marker_label", "row_name", "annotation"):
        sub = o.optional(key, _is_object, "an object")
        if sub is None:
            continue
        if key == "sample_label":
            values[key] = _sample_label(ctx, sub, condition_ids, drawn)
        elif key == "details_list":
            values[key] = _details_list(ctx, sub)
        elif key == "row_name":
            values[key] = _row_name(ctx, sub, row_ids, drawn)
        else:
            inner, got = _labels(ctx, sub, f"figure.style.{key}", side=key == "marker_label")
            inner.finish()
            values[key] = _in_order(sub, got)
    o.finish()
    return _in_order(raw, values)

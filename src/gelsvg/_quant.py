"""Reading `quantifications` (§7)."""

from __future__ import annotations

from . import _check as C
from ._check import Context, Obj
from ._core import _firsts, CoreView

MODES = ("lane", "band", "region")
PIXELS = ("aligned",)
REGION_SHAPES = ("rect",)


def _enum(ctx: Context, o: Obj, key: str, allowed, problems: list[str], where: str):
    v = o.raw(key)
    if v in allowed:
        return v
    if key not in o.value or v is None:
        problems.append(f"{key!r} is missing")
        return None
    if C.is_str(v) and ctx.unknown_value(f"{where}.{key}", v):
        return "unknown"
    problems.append(f"{key!r} has the value {v!r}")
    return None


def _read_value(ctx: Context, value, where: str, mode: str, view: CoreView, lane_ids, row_ids, cells):
    if type(value) is not dict:
        return None, ["not an object"]
    o = Obj(ctx, value, where)
    problems: list[str] = []
    out = {}
    if mode == "region":
        o.forbidden("lane", "a region-mode value names no lane")
        o.forbidden("row", "a region-mode value names no lane")
    else:
        out["lane"] = o.required("lane", C.is_id, "an ID", problems)
    if mode == "band":
        out["row"] = o.required("row", C.is_id, "an ID", problems)
    elif mode == "lane":
        o.forbidden("row", "a lane-mode value measures the whole lane")
    region = o.raw("region")
    if type(region) is not dict:
        problems.append("'region' is missing or not an object")
    else:
        r = Obj(ctx, region, f"{where}.region")
        shape = _enum(ctx, r, "shape", REGION_SHAPES, problems, f"{where}.region")
        if shape == "rect":
            rect = {k: r.required(k, C.is_number, "a number", problems) for k in ("x", "y")}
            rect.update({k: r.required(k, lambda v: C.is_number(v) and v > 0, "a number above 0", problems) for k in ("w", "h")})
            out["region"] = {"shape": "rect", **rect}
        else:
            for key in ("x", "y", "w", "h"):
                r.seen.add(key)
            out["region"] = None  # unknown shape: the region cannot be used
        r.finish()
    out["raw"] = o.required("raw", C.is_number, "a number", problems)
    out["signal"] = o.required("signal", C.is_number, "a number", problems)
    out["background"] = o.optional("background", C.is_number, "a number")
    out["peak"] = o.optional("peak", C.is_number, "a number")
    o.finish()
    if problems:
        return None, problems
    if mode == "region":
        return out, []
    if out["lane"] not in lane_ids:
        return None, [f"its lane {out['lane']!r} cannot be used"]
    if mode == "band":
        if out["row"] not in row_ids:
            return None, [f"its row {out['row']!r} cannot be used"]
        if (out["lane"], out["row"]) not in cells:
            return None, ["its cell has no band"]
    return out, []


def read_quantifications(ctx: Context, items: list, view: CoreView) -> list[dict]:
    lane_ids = view.lane_ids() - view.ambiguous.get("lanes", set())
    row_ids = view.row_ids() - view.ambiguous.get("rows", set())
    cells = view.band_cells()
    channels_readable = view.channels is not None
    channel_ids = view.channel_ids() - view.ambiguous.get("channels", set()) if channels_readable else set()
    out = []
    for i, item in enumerate(items):
        where = f"quantifications[{i}]"
        if type(item) is not dict:
            ctx.broken("element.broken", "The quantification was dropped: not an object.", where)
            continue
        o = Obj(ctx, item, where)
        problems: list[str] = []
        mode = _enum(ctx, o, "mode", MODES, problems, where)
        pixels = _enum(ctx, o, "pixels", PIXELS, problems, where)
        q: dict = {"mode": mode, "pixels": None if pixels == "unknown" else pixels}
        method = o.raw("method")
        if type(method) is not dict:
            problems.append("'method' is missing or not an object")
        else:
            m = Obj(ctx, method, f"{where}.method")
            q["method"] = {"name": m.required("name", C.is_str, "a string", problems),
                           "by": m.required("by", C.is_str, "a string", problems)}
            parameters = m.optional("parameters", lambda v: type(v) is dict, "an object")
            if parameters is not None:
                q["method"]["parameters"] = parameters  # the method's own; read as written (§7.1)
            m.finish()
        if mode in ("lane", "region"):
            q["channel"] = o.required("channel", C.is_id, "an ID", problems)
            if channels_readable and q.get("channel") is not None:
                if q["channel"] not in channel_ids:
                    problems.append(f"its panel channel {q['channel']!r} cannot be used")
                elif view.channel(q["channel"])["role"] == "marker":
                    ctx.broken("reference.marker", f"It measures the marker channel {q['channel']!r}, which a "
                               "quantification may not do (§5.6); it was read as written.", f"{where}.channel")
        elif mode == "band":
            o.forbidden("channel", "only a lane- or region-mode quantification names one; a band-mode one measures "
                        "each row's panel channel")
            q["channel"] = None
        else:
            # A mode not known (rule 2) decides the quantification's form, so its `channel` is not checked; a
            # missing or wrong mode drops the quantification anyway.
            o.seen.add("channel")
            q["channel"] = None
        q["inverted"] = o.required("inverted", C.is_bool, "true or false", problems)
        q["computed_on"] = o.required("computed_on", C.is_hash, "a hash", problems)
        if q["computed_on"] is not None and ctx.hash_error(f"{where}.computed_on", q["computed_on"]):
            problems.append("'computed_on' uses a hash kind this version does not define")
        values = o.raw("values")
        if type(values) is not list or not values:
            problems.append("'values' is missing or empty")
            values = []
        o.finish()
        if problems:
            ctx.broken("element.broken", "The quantification was dropped: " + "; ".join(problems) + ".", where)
            continue
        if mode == "unknown":
            continue  # a newer kind of quantification: not used at all (rule 2)
        # The lanes and cells as written: a later value is dropped even when the earlier one is
        # broken (rule 3). Regions name no lane, and may repeat (§7.3).
        def written_key(value):
            if type(value) is not dict or not C.is_id(value.get("lane")):
                return None
            if mode == "lane":
                return value["lane"]
            return (value["lane"], value.get("row")) if C.is_id(value.get("row")) else None
        first, _ = _firsts([written_key(value) for value in values]) if mode != "region" else ({}, set())
        kept = []
        for j, value in enumerate(values):
            w = f"{where}.values[{j}]"
            v, why = _read_value(ctx, value, w, mode, view, lane_ids, row_ids, cells)
            if v is None:
                ctx.broken("element.broken", "The value was dropped: " + "; ".join(why) + ".", w)
                continue
            if mode != "region":
                key = v["lane"] if mode == "lane" else (v["lane"], v["row"])
                if first.get(key, j) != j:
                    ctx.broken("value.duplicate", "A later value for the same lane or cell was dropped.", w)
                    continue
            kept.append(v)
        q["values"] = kept
        q["index"] = i
        out.append(q)
    return out

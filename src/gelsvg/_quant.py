"""Reading `quantifications` (§7)."""

from __future__ import annotations

from . import _check as C
from ._check import Context, Obj
from ._core import CoreView

MODES = ("lane", "band")
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
    out = {"lane": o.required("lane", C.is_id, "an ID", problems)}
    if mode == "band":
        out["row"] = o.required("row", C.is_id, "an ID", problems)
    else:
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
    blot_readable = view.blot is not None
    blot_ids = view.blot_ids() - view.ambiguous.get("blot", set()) if blot_readable else set()
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
        if mode == "lane" and (not blot_readable or view.blot_count_as_written >= 2):
            if blot_readable:
                q["channel"] = o.required("channel", C.is_id, "an ID", problems)
                if q.get("channel") is not None and q["channel"] not in blot_ids:
                    problems.append(f"its channel {q['channel']!r} is not a blot element that can be used")
            else:
                q["channel"] = item.get("channel") if C.is_id(item.get("channel")) else None
                o.seen.add("channel")
        else:
            o.forbidden("channel", "only a lane-mode quantification of a blot with two or more elements names one")
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
        kept, seen = [], set()
        for j, value in enumerate(values):
            w = f"{where}.values[{j}]"
            v, why = _read_value(ctx, value, w, mode, view, lane_ids, row_ids, cells)
            if v is None:
                ctx.broken("element.broken", "The value was dropped: " + "; ".join(why) + ".", w)
                continue
            key = v["lane"] if mode == "lane" else (v["lane"], v["row"])
            if key in seen:
                ctx.broken("value.duplicate", "A later value for the same lane or cell was dropped.", w)
                continue
            seen.add(key)
            kept.append(v)
        q["values"] = kept
        q["index"] = i
        out.append(q)
    return out

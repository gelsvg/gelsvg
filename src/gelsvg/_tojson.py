"""The panel as read, as JSON a writer may write again (§2.6 rule 6): what
the reader ignored or dropped is left out, everything else keeps its value,
its order and its IDs. `embedded` is left out too; the writer rebuilds it
from the payloads it has."""

from __future__ import annotations

import copy

from . import _json
from ._check import Context
from ._core import CoreView, read_core
from ._quant import read_quantifications


def _put(out: dict, record: dict, *keys: str) -> None:
    for key in keys:
        if record.get(key) is not None:
            out[key] = record[key]


def _source(source: dict, metadata_keys: set[str]) -> dict:
    out = {"id": source["id"]}
    _put(out, source, "file_name", "file_hash", "dtype", "shape")
    channels = []
    for channel in source["channels"]:
        item = {"name": channel["name"]}
        if channel.get("pixel_hash_known"):
            item["pixel_hash"] = channel["pixel_hash"]
        channels.append(item)
    out["channels"] = channels
    if source.get("source_metadata") in metadata_keys:
        out["source_metadata"] = source["source_metadata"]
    return out


def _view_element(element: dict, first: bool, marker: bool) -> dict:
    out = {} if marker else {"id": element["id"]}
    out["source"] = element["source"]
    out["source_channel"] = element["source_channel"]
    if not first:
        out["turn"], out["flip"] = element["turn"], element["flip"]
    out["invert"], out["contrast"] = element["invert"], element["contrast"]
    _put(out, element, "color", "pixel_hash")
    return out


def _lane(lane: dict) -> dict:
    out = {"id": lane["id"]}
    if lane["has_point"]:
        _put(out, lane, "number", "x", "y", "details", "values")
    return out


def _band(band: dict) -> dict:
    out = {"lane": band["lane"], "row": band["row"], "shape": band["shape"], "x": band["x"], "y": band["y"]}
    _put(out, band, "w", "h", "detection")
    return out


def _core(view: CoreView, metadata_keys: set[str]) -> dict:
    core: dict = {}
    if view.sources is not None:
        core["sources"] = [_source(s, metadata_keys) for s in view.sources]
    if view.blot is not None:
        core["blot"] = [_view_element(e, i == 0, marker=False) for i, e in enumerate(view.blot)]
    if view.marker is not None:
        core["marker"] = _view_element(view.marker, False, marker=True)
    if view.orientation is not None:
        core["orientation"] = view.orientation
    if view.roi is not None:
        core["roi"] = view.roi
    if view.output_bit_depth is not None:
        core["output_bit_depth"] = view.output_bit_depth
    samples: dict = {}
    if view.conditions:
        samples["conditions"] = [{"id": c["id"], **({"name": c["name"]} if c["name"] is not None else {})}
                                 for c in view.conditions]
    if view.lanes:
        samples["lanes"] = [_lane(lane) for lane in view.lanes]
    if samples:
        core["samples"] = samples
    if view.rows:
        rows = []
        for row in view.rows:
            item = {"id": row["id"]}
            _put(item, row, "name", "channel")
            rows.append(item)
        core["rows"] = rows
    bands = [_band(b) for b in view.bands if b["shape"] is not None]
    if bands:
        core["bands"] = bands
    if view.markers:
        markers = []
        for label in view.markers:
            item = {"id": label["id"], "x": label["x"], "y": label["y"], "label": label["label"]}
            _put(item, label, "size")
            markers.append(item)
        core["markers"] = markers
    if view.annotation is not None:
        core["annotation"] = view.annotation
    if view.note is not None:
        core["note"] = view.note
    return core


def _quantifications(items: list[dict]) -> list[dict]:
    out = []
    for q in items:
        if q["pixels"] is None:
            continue  # an unknown value (rule 2): not written back
        values = []
        for v in q["values"]:
            if v["region"] is None:
                continue
            item = {"lane": v["lane"]}
            _put(item, v, "row")
            item.update(region=v["region"], raw=v["raw"], signal=v["signal"])
            _put(item, v, "background", "peak")
            values.append(item)
        if not values:
            continue
        item = {"mode": q["mode"], "pixels": q["pixels"], "method": q["method"]}
        _put(item, q, "channel")
        item.update(inverted=q["inverted"], computed_on=q["computed_on"], values=values)
        out.append(item)
    return out


def _settle(core: dict, quantifications: list[dict], metadata_keys: set[str]):
    """Read the JSON again until reading it drops nothing more. Leaving
    something out can make another thing invalid (a row's `channel` once
    the blot has one element, a lane whose only band was dropped), and
    rule 6 forbids writing what is invalid."""

    for _ in range(16):
        ctx = Context(panel=None, newer_revision=False)
        view = read_core(ctx, copy.deepcopy(core))
        quants = read_quantifications(ctx, copy.deepcopy(quantifications), view)
        new_core, new_quants = _core(view, metadata_keys), _quantifications(quants)
        if _json.dumps([new_core, new_quants]) == _json.dumps([core, quantifications]):
            return core, quantifications
        core, quantifications = new_core, new_quants
    return core, quantifications


def panel_json(panel) -> dict:
    """The JSON of a readable panel of a known version, cleaned as read."""

    if not panel.readable or panel.newer_version:
        raise ValueError("only a panel that was read has JSON to write again")
    view: CoreView = panel.core
    metadata_keys = {k for k, rec in view.source_metadata.items() if rec.get("payload") is not None}
    core, quants = _settle(_core(view, metadata_keys), _quantifications(panel.quantifications), metadata_keys)

    out: dict = {"format": panel.raw["format"], "schema_version": panel.schema_version}
    if panel.schema_revision is not None:
        out["schema_revision"] = panel.schema_revision
    if panel.saved_at is not None:
        out["saved_at"] = panel.saved_at
    if panel.written_by is not None:
        out["written_by"] = {k: v for k, v in panel.written_by.items() if v is not None}
    out["core"] = core
    if quants:
        out["quantifications"] = quants
    out["figure"] = {k: v for k, v in panel.figure.items() if v is not None}
    if panel.assembly is not None:
        out["assembly"] = panel.assembly
    if panel.extensions:
        out["extensions"] = {name: record["value"] for name, record in panel.extensions.items()}
    return copy.deepcopy(out)

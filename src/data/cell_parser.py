"""Extract PathCellObject geometries from QuPath .qpdata files.

QuPath stores each cell as `PathCellObject` with two `PolygonROI` children:
the nucleus and the cell membrane. Each PolygonROI is serialised as
`PolygonROI$SerializationProxy` with two `float[]` arrays (X and Y coords).

This module sits next to `qpdata_parser.py` (which only handles
PathAnnotationObject). It reuses the same `walk_stream` walker but extracts
the cell-shaped events instead.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Polygon

from src.data.java_oois import StreamEvent, walk_stream

# QuPath serialises with `setrecursionlimit`-hostile depth on files that have
# tens of thousands of cells. The walker is recursive; lift the cap at
# import time so callers don't have to.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 1_000_000))

CELL_OBJECT = "qupath.lib.objects.PathCellObject"
DETECTION_OBJECT = "qupath.lib.objects.PathDetectionObject"
POLYGON_PROXY = "qupath.lib.roi.PolygonROI$SerializationProxy"
FLOAT_ARRAY = "[F"


@dataclass(frozen=True)
class ParsedCell:
    """One cell with its nucleus + cell-membrane polygons (in level-0 px)."""

    nucleus: Polygon
    cell: Polygon | None  # None for PathDetectionObject (single ROI)
    class_name: str | None


def _decode_float_array(payload: object) -> list[float]:
    """Decode a `[F` array event payload (cls_name, size, raw_bytes)."""
    if not (isinstance(payload, tuple) and len(payload) == 3):
        return []
    cls_name, size, raw = payload
    if cls_name != FLOAT_ARRAY or not isinstance(raw, (bytes, bytearray)):
        return []
    if len(raw) != size * 4:
        return []
    return list(struct.unpack(f">{size}f", raw))


def _read_polygon_after(events: list[StreamEvent], start: int) -> tuple[Polygon | None, int]:
    """Read one PolygonROI$SerializationProxy starting at index `start`
    (which must be `object PolygonROI$SerializationProxy`). Returns
    (polygon_or_None, next_index_to_continue_from).

    The two `[F` arrays following the object are X and Y. We do NOT inspect
    intermediate ClosedShapeStatistics objects — we just grab the first two
    `[F` arrays we encounter before the next non-numeric object.
    """
    n = len(events)
    if start >= n or events[start].kind != "object" or events[start].payload != POLYGON_PROXY:
        return None, start + 1

    xs: list[float] | None = None
    ys: list[float] | None = None
    i = start + 1
    while i < n:
        ev = events[i]
        # Stop scanning if we run into the next ROI proxy or the next
        # PathCellObject — we won't find more arrays for *this* polygon.
        if ev.kind == "object" and ev.payload in (POLYGON_PROXY, CELL_OBJECT, DETECTION_OBJECT):
            break
        if ev.kind == "array" and isinstance(ev.payload, tuple) and ev.payload[0] == FLOAT_ARRAY:
            decoded = _decode_float_array(ev.payload)
            if xs is None:
                xs = decoded
            else:
                ys = decoded
                i += 1
                break
        i += 1

    if xs is None or ys is None or len(xs) != len(ys) or len(xs) < 3:
        return None, i

    coords = list(zip(xs, ys))
    try:
        poly = Polygon(coords)
        if poly.is_empty:
            return None, i
        return poly, i
    except Exception:
        return None, i


def _resolve_pathclass_handles(events: list[StreamEvent]) -> dict[int, str]:
    """Same logic as qpdata_parser, but tolerant of cell objects between PathClass and its name."""
    PATHCLASS = "qupath.lib.objects.classes.PathClass"
    mapping: dict[int, str] = {}
    for i, ev in enumerate(events):
        if ev.kind == "object" and ev.payload == PATHCLASS and ev.handle is not None:
            for j in range(i + 1, min(i + 30, len(events))):
                if events[j].kind == "string":
                    mapping[ev.handle] = events[j].payload
                    break
                if events[j].kind == "object" and events[j].payload in (
                    CELL_OBJECT,
                    DETECTION_OBJECT,
                    "qupath.lib.objects.PathAnnotationObject",
                ):
                    break
    return mapping


def extract_cell_objects(data: bytes) -> list[ParsedCell]:
    """Walk a `.qpdata` byte stream and return every PathCellObject as
    a ParsedCell with both polygons decoded."""
    events = walk_stream(data)
    handle_to_class = _resolve_pathclass_handles(events)

    cells: list[ParsedCell] = []
    n = len(events)
    i = 0
    while i < n:
        ev = events[i]
        if not (ev.kind == "object" and ev.payload in (CELL_OBJECT, DETECTION_OBJECT)):
            i += 1
            continue

        is_cell = ev.payload == CELL_OBJECT

        # Walk forward inside this PathCellObject body to find:
        #   1. its PathClass (object or reference)
        #   2. the first PolygonROI$SerializationProxy (nucleus)
        #   3. the second one (cell) — only for PathCellObject
        current_class: str | None = None
        nucleus: Polygon | None = None
        cell_poly: Polygon | None = None

        j = i + 1
        while j < n:
            ev2 = events[j]
            # Hit the next cell — stop, this one is done.
            if ev2.kind == "object" and ev2.payload in (CELL_OBJECT, DETECTION_OBJECT):
                break

            if current_class is None:
                if ev2.kind == "object" and ev2.payload == "qupath.lib.objects.classes.PathClass":
                    current_class = handle_to_class.get(ev2.handle or -1)
                elif ev2.kind == "reference" and ev2.payload in handle_to_class:
                    current_class = handle_to_class[ev2.payload]

            if ev2.kind == "object" and ev2.payload == POLYGON_PROXY:
                poly, j_next = _read_polygon_after(events, j)
                # QuPath serialises PathCellObject as (cellROI, nucleusROI):
                # the larger cell-membrane polygon comes first, the nucleus
                # second. PathDetectionObject only has one ROI (treat as
                # nucleus to keep downstream consumers simple).
                if cell_poly is None and is_cell:
                    cell_poly = poly
                elif nucleus is None:
                    nucleus = poly
                    if not is_cell:
                        # Detection object only has one ROI; we're done.
                        j = j_next
                        break
                j = j_next
                continue

            j += 1

        if nucleus is not None:
            cells.append(ParsedCell(nucleus=nucleus, cell=cell_poly, class_name=current_class))

        i = max(j, i + 1)

    return cells


def parse_cores_qpdata(filepath: str | Path) -> list[ParsedCell]:
    """Convenience wrapper: read a cores .qpdata file and return all cells."""
    return extract_cell_objects(Path(filepath).read_bytes())

"""Parser for QuPath .qpdata files.

Extracts annotation geometries (as Shapely objects) and their class names
from QuPath's binary serialization format (Java serialized objects with
embedded WKB geometry data).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from shapely import wkb
from shapely.geometry.base import BaseGeometry

from configs.classes import normalize_class_name, class_name_to_idx
from src.data.java_oois import extract_annotations


@dataclass(frozen=True)
class WSIMetadata:
    """Whole-slide image metadata extracted from .qpdata header."""
    name: str
    width: int
    height: int
    magnification: float
    pixel_size_um: float  # µm per pixel


@dataclass(frozen=True)
class AnnotationEntry:
    """Single annotation: class name + Shapely geometry."""
    class_name: str
    class_idx: int
    geometry: BaseGeometry
    area_px: float


@dataclass(frozen=True)
class ParsedQPData:
    """Full parse result for one .qpdata file."""
    metadata: WSIMetadata
    annotations: list[AnnotationEntry]

    @property
    def class_names(self) -> list[str]:
        return [a.class_name for a in self.annotations]


def _locate_header(data: bytes) -> tuple[int, int]:
    """Return (start, end) byte offsets of the JSON header block."""
    start = data.find(b'{"dataVersion"')
    if start < 0:
        raise ValueError("No header JSON found in .qpdata file")

    depth = 0
    end = start
    for i in range(start, min(start + 10000, len(data))):
        if data[i : i + 1] == b"{":
            depth += 1
        elif data[i : i + 1] == b"}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    return start, end


def _extract_header(data: bytes) -> dict:
    """Extract the JSON header from the binary .qpdata file."""
    start, end = _locate_header(data)
    return json.loads(data[start:end].decode("utf-8", errors="replace"))


def _extract_metadata(header: dict) -> WSIMetadata:
    """Parse WSI metadata from the header JSON."""
    meta = header.get("server", {}).get("metadata", {})
    cal = meta.get("pixelCalibration", {})
    pw = cal.get("pixelWidth", {})

    return WSIMetadata(
        name=meta.get("name", "unknown"),
        width=meta.get("width", 0),
        height=meta.get("height", 0),
        magnification=meta.get("magnification", 0.0),
        pixel_size_um=pw.get("value", 0.0),
    )


def parse_qpdata(filepath: str | Path) -> ParsedQPData:
    """Parse a .qpdata file and return metadata + geojson.

    Uses a Java OIS walker (see `src/data/java_oois.py`) to recover every
    (PathClass, WKB) pair, including geojson that share a class via a
    Java TC_REFERENCE handle. Unknown class names are silently skipped;
    they are surfaced separately by `scripts/validate_dataset.py`.
    """
    filepath = Path(filepath)
    data = filepath.read_bytes()

    header = _extract_header(data)
    metadata = _extract_metadata(header)

    annotations: list[AnnotationEntry] = []
    for raw in extract_annotations(data):
        if raw.class_name is None:
            continue
        idx = class_name_to_idx(raw.class_name)
        if idx is None:
            continue
        try:
            geom = wkb.loads(raw.wkb)
        except Exception:
            continue
        normalized = normalize_class_name(raw.class_name)
        annotations.append(
            AnnotationEntry(
                class_name=normalized,
                class_idx=idx,
                geometry=geom,
                area_px=geom.area,
            )
        )

    return ParsedQPData(metadata=metadata, annotations=annotations)


def extract_class_colors(filepath: str | Path) -> dict[str, tuple[int, int, int]]:
    """Extract every (class_name, RGB) pair from a .qpdata file's PathClass table.

    QuPath stores each PathClass as `(colorRGB: Integer, name: String, parent)`.
    Our walker now retains Integer values, so we can read the colour that the
    annotator actually used in QuPath instead of guessing.

    Returns a dict {normalized_class_name: (R, G, B)} for every class that
    appeared in the file. Unknown / null PathClass entries are skipped.
    """
    from src.data.java_oois import walk_stream

    data = Path(filepath).read_bytes()
    events = walk_stream(data)

    pathclass_name = "qupath.lib.objects.classes.PathClass"
    integer_name = "java.lang.Integer"
    annotation_name = "qupath.lib.objects.PathAnnotationObject"

    out: dict[str, tuple[int, int, int]] = {}
    n = len(events)
    for i, ev in enumerate(events):
        if not (ev.kind == "object" and ev.payload == pathclass_name):
            continue
        # Look ahead for the next Integer (colorRGB) and String (name).
        color_int: int | None = None
        cls_name: str | None = None
        for j in range(i + 1, min(i + 30, n)):
            ev2 = events[j]
            if (
                color_int is None
                and ev2.kind == "object"
                and isinstance(ev2.payload, tuple)
                and ev2.payload[0] == integer_name
            ):
                color_int = int(ev2.payload[1])
                continue
            if cls_name is None and ev2.kind == "string":
                cls_name = ev2.payload
                break
            if ev2.kind == "object" and ev2.payload == annotation_name:
                break
        if cls_name is None or color_int is None:
            continue
        # QuPath packs colorRGB as 0xAARRGGBB (alpha in top byte). Mask to RGB.
        r = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        b = color_int & 0xFF
        normalized = normalize_class_name(cls_name)
        out.setdefault(normalized, (r, g, b))
    return out


def find_qpdata_for_svs(svs_path: str | Path) -> Path | None:
    """Find the matching .qpdata file for an SVS file.

    Handles both naming conventions: name.svs.qpdata and name.qpdata
    """
    svs_path = Path(svs_path)
    parent = svs_path.parent
    stem = svs_path.stem  # e.g., "101"

    for suffix in [f"{stem}.svs.qpdata", f"{stem}.qpdata"]:
        candidate = parent / suffix
        if candidate.exists():
            return candidate

    return None

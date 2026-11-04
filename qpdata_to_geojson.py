import json
import sys
from pathlib import Path

import shapely
from shapely.geometry import mapping

from configs.classes import CLASS_COLORS
from src.data.qpdata_parser import find_qpdata_for_svs, parse_qpdata

sys.setrecursionlimit(1_000_000)

ROOT = Path(__file__).parent
SLIDES_DIR = ROOT / "data" / "slides"
GEOJSON_DIR = ROOT / "data" / "geojson"
GEOJSON_DIR.mkdir(exist_ok=True)


for svs_path in sorted(SLIDES_DIR.glob("*.svs")):
    qpdata_path = find_qpdata_for_svs(svs_path)

    if qpdata_path is None:
        continue


    parsed = parse_qpdata(qpdata_path)
    features = []
    for annotation in parsed.annotations:
        geometry = annotation.geometry

        if not geometry.is_valid:
            geometry = shapely.make_valid(geometry, method="structure", keep_collapsed=False)

        for polygon in shapely.get_parts(geometry):
            if polygon.area < 100:
                continue

            features.append({
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": annotation.class_name,
                        "color": list(CLASS_COLORS[annotation.class_idx]),
                    },
                },
            })

    geojson_path = GEOJSON_DIR / (svs_path.stem + ".geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)

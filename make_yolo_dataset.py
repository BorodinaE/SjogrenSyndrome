import json
import random
from pathlib import Path

import numpy as np
import openslide
import shapely
from shapely.geometry import box, shape

from configs.classes import CLASS_NAME_TO_IDX

ROOT = Path(__file__).parent
SLIDES_DIR = ROOT / "data" / "slides"
GEOJSON_DIR = ROOT / "data" / "geojson"
DATASET_DIR = ROOT / "data" / "yolo_data"

LEVEL = 1
TILE_SIZE = 640
STEP = 512         # шаг окна
MIN_PART_AREA = 50


YOLO_NAMES = ["parenchyma", "ducts", "infiltrate", "foci", "sclerosis", "vessels", "adipose"]


names = sorted(p.stem for p in GEOJSON_DIR.glob("*.geojson"))
random.seed(42)
random.shuffle(names)
n_train = round(len(names) * 0.70)
n_val = round(len(names) * 0.15)
split = {
    "train": names[:n_train],
    "val": names[n_train:n_train + n_val],
    "test": names[n_train + n_val:],
}


DATASET_DIR.mkdir(parents=True, exist_ok=True)
with open(DATASET_DIR / "split.json", "w", encoding="utf-8") as f:
    json.dump(split, f, ensure_ascii=False, indent=2)

for part, slide_names in split.items():
    (DATASET_DIR / "images" / part).mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / part).mkdir(parents=True, exist_ok=True)

    for name in slide_names:
        slide = openslide.OpenSlide(SLIDES_DIR / (name + ".svs"))
        scale = slide.level_downsamples[LEVEL]
        level_width, level_height = slide.level_dimensions[LEVEL]

        with open(GEOJSON_DIR / (name + ".geojson"), encoding="utf-8") as f:
            features = json.load(f)["features"]
        polygons = [shape(feat["geometry"]) for feat in features]   # словарь -> объект Shapely
        classes = [CLASS_NAME_TO_IDX[feat["properties"]["classification"]["name"]] - 1 for feat in features]

        tree = shapely.STRtree(polygons)
        n_saved = 0

        for y in range(0, level_height - TILE_SIZE + 1, STEP):
            for x in range(0, level_width - TILE_SIZE + 1, STEP):
                x0, y0 = int(x * scale), int(y * scale)
                tile_box = box(x0, y0, x0 + TILE_SIZE * scale, y0 + TILE_SIZE * scale)
                hits = tree.query(tile_box, predicate="intersects")
                if len(hits) == 0:
                    continue

                image = slide.read_region((x0, y0), LEVEL, (TILE_SIZE, TILE_SIZE)).convert("RGB")

                lines = []
                for i in hits:
                    piece = polygons[i].intersection(tile_box)

                    for p in shapely.get_parts(piece):
                        if p.geom_type != "Polygon" or p.area / scale ** 2 < MIN_PART_AREA:
                            continue

                        p = p.simplify(scale)
                        coords = []
                        for px, py in p.exterior.coords[:-1]:
                            coords.append((px - x0) / scale / TILE_SIZE) #нормализация (требование YOLO)
                            coords.append((py - y0) / scale / TILE_SIZE)

                        if len(coords) >= 6:
                            numbers = [classes[i]] + [round(c, 5) for c in coords]
                            line = " ".join(str(n) for n in numbers)
                            lines.append(line)

                if not lines:
                    continue

                tile_name = f"{name}_{x0}_{y0}"
                image.save(DATASET_DIR / "images" / part / (tile_name + ".jpg"), quality=95)
                with open(DATASET_DIR / "labels" / part / (tile_name + ".txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")
                n_saved += 1


with open(DATASET_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
    f.write(f"path: {DATASET_DIR.as_posix()}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("names:\n")
    for i, class_name in enumerate(YOLO_NAMES):
        f.write(f"  {i}: {class_name}\n")


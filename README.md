# SjogrenSyndrome — сегментация гистологических срезов слюнных желёз

Подготовка данных и обучение модели YOLO для сегментации тканей на гистологических
сканах малых слюнных желёз (7 классов), с последующей детекцией ядер клеток.

## Что куда класть

Сами данные в git не хранятся (слишком большие) — в репозитории лежит только
структура папок. Файлы нужно положить вручную:

| Папка | Что класть | Откуда берётся |
|-------|-----------|----------------|
| `data/slides/` | сканы `.svs` + разметка тканей `.qpdata` | исходные данные |
| `data/nuclei/` | разметка ядер (`* ядра.qpdata` и т.п.) | исходные данные |
| `data/geojson/` | — (создаётся автоматически) | `qpdata_to_geojson.py` |
| `data/yolo_data/` | — (создаётся автоматически) | `make_yolo_dataset.py` |
| `checkpoints/` | — (создаётся при обучении) | веса обученной модели |

## Пайплайн

```
data/slides/*.qpdata  ──[qpdata_to_geojson.py]──►  data/geojson/*.geojson
                                                          │
data/slides/*.svs  ──────[make_yolo_dataset.py]───────────┘
                                   │
                                   ▼
                         data/yolo_data/  (тайлы 640×640 + разметка YOLO)
                                   │
                                   ▼
                          обучение YOLO-сегментации
```

### 1. Разметка `.qpdata` → GeoJSON

```bash
python qpdata_to_geojson.py
```
Читает бинарные файлы разметки QuPath из `data/slides/` и сохраняет полигоны в
`data/geojson/` (удобный текстовый формат, открывается обратно в QuPath).

### 2. Нарезка на тайлы для YOLO

```bash
python make_yolo_dataset.py
```
По `.svs` + GeoJSON режет сканы на квадраты 640×640 (с уменьшением в 4 раза),
переводит разметку в формат YOLO и делит сканы на train/val/test (70/15/15).
Результат — в `data/yolo_data/`.


## Структура кода

```
configs/classes.py        # 7 классов тканей, их цвета и нормализация имён
src/data/
  java_oois.py            # разбор бинарной Java-сериализации QuPath
  qpdata_parser.py        # .qpdata (разметка тканей) → полигоны
qpdata_to_geojson.py      # шаг 1: .qpdata → GeoJSON
make_yolo_dataset.py      # шаг 2: GeoJSON + .svs → датасет YOLO
```

## Установка

```bash
pip install -r requirements.txt
```


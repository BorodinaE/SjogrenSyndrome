"""Annotation class definitions for salivary gland biopsy segmentation."""

# Canonical class names (lowercase) → class index (0 = background)
CLASS_NAME_TO_IDX: dict[str, int] = {
    "паренхима": 1,
    "выводные протоки": 2,
    "воспалительный инфильтрат": 3,
    "фокусы воспаления": 4,
    "склероз": 5,
    "расширенные сосуды": 6,
    "клетчатка": 7,
}

# Aliases for non-standard naming (e.g., file 429)
CLASS_ALIASES: dict[str, str] = {
    "сосуды": "расширенные сосуды",
}

NUM_CLASSES = len(CLASS_NAME_TO_IDX) + 1  # +1 for background (index 0)

# Pack A: at training time we collapse "фокусы воспаления" (4) into
# "воспалительный инфильтрат" (3). Both are morphologically identical
# inflammatory tissue — focus is just a denser cluster within infiltrate
# — and pathologists split them with subjective criteria, which gave the
# model conflicting supervision. Postprocessing splits them back by
# nuclei count: ≥FOCUS_CELL_THRESHOLD → focus, otherwise → infiltrate.
INFLAMMATION_BASE_IDX = 3   # "воспалительный инфильтрат"
INFLAMMATION_FOCUS_IDX = 4  # "фокусы воспаления"
FOCUS_CELL_THRESHOLD = 50   # ≥N cells in a region promotes infiltrate → focus

# Sentinel value used in masks to mark pixels that must be excluded from
# both the loss and the metrics. CrossEntropy/Focal use this via
# `ignore_index`; Dice masks them out before reduction.
IGNORE_INDEX = 255

# Reverse mapping
IDX_TO_CLASS_NAME: dict[int, str] = {0: "фон"}
IDX_TO_CLASS_NAME.update({v: k for k, v in CLASS_NAME_TO_IDX.items()})

# Colors for visualization (RGB).
# Mirror exactly what pathologists used in QuPath — these are the colorRGB
# values stored inside every PathClass in `initial_dataset/*.qpdata` and are
# consistent across all WSIs
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),           # фон — чёрный
    1: (255, 255, 0),       # паренхима — жёлтый  (#FFFF00)
    2: (255, 179, 102),     # выводные протоки — светло-оранжевый  (#FFB366)
    3: (77, 0, 26),         # воспалительный инфильтрат — тёмно-бордовый  (#4D001A)
    4: (255, 0, 0),         # фокусы воспаления — красный  (#FF0000)
    5: (0, 255, 0),         # склероз — зелёный  (#00FF00)
    6: (255, 0, 255),       # расширенные сосуды — магента  (#FF00FF)
    7: (0, 0, 255),         # клетчатка — синий  (#0000FF)
}


def normalize_class_name(name: str) -> str:
    """Normalize class name: lowercase + resolve aliases."""
    lower = name.lower().strip()
    return CLASS_ALIASES.get(lower, lower)


def class_name_to_idx(name: str) -> int | None:
    """Get class index for a (possibly non-standard) class name."""
    normalized = normalize_class_name(name)
    return CLASS_NAME_TO_IDX.get(normalized)

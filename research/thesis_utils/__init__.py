"""Common utilities shared across thesis EDA notebooks."""
from .alignment import vector_alignment
from .io import (
    load_json,
    load_state,
    save_figure,
    save_json,
    save_state,
)
from .paths import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    DATA_DIR,
    EMBED_CACHE_DIR,
    FIGURES_DIR,
    METADATA_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RAW_IMAGES_DIR,
    SHARED_STATE_DIR,
    THESIS_ROOT,
    ensure_dirs,
)
from .seeding import set_seed

__all__ = [
    "vector_alignment",
    "THESIS_ROOT",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_IMAGES_DIR",
    "METADATA_DIR",
    "CACHE_DIR",
    "EMBED_CACHE_DIR",
    "OUTPUTS_DIR",
    "ARTIFACTS_DIR",
    "FIGURES_DIR",
    "SHARED_STATE_DIR",
    "ensure_dirs",
    "set_seed",
    "save_json",
    "load_json",
    "save_figure",
    "save_state",
    "load_state",
]

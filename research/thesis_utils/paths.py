"""Standard paths shared across the thesis EDA pipeline."""
from __future__ import annotations

from pathlib import Path

# thesis_utils/paths.py 기준으로 thesis/ 루트 추론
THESIS_ROOT: Path = Path(__file__).resolve().parents[1]
PROJECT_ROOT: Path = THESIS_ROOT.parent

# 입력 데이터 (집컴에서 채워질 자리)
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_IMAGES_DIR: Path = DATA_DIR / "raw_images"
METADATA_DIR: Path = DATA_DIR / "metadata"

# 임베딩 등 캐시
CACHE_DIR: Path = PROJECT_ROOT / "cache"
EMBED_CACHE_DIR: Path = CACHE_DIR / "embeddings"

# 산출물
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR: Path = OUTPUTS_DIR / "artifacts"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
SHARED_STATE_DIR: Path = OUTPUTS_DIR / "shared_state"


def ensure_dirs() -> None:
    """모든 표준 디렉터리를 멱등적으로 생성한다."""
    for directory in (
        DATA_DIR,
        RAW_IMAGES_DIR,
        METADATA_DIR,
        CACHE_DIR,
        EMBED_CACHE_DIR,
        OUTPUTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        SHARED_STATE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
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
]


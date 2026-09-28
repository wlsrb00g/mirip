"""Seed helpers shared across notebooks."""
from __future__ import annotations

import os
import random


def set_seed(seed: int = 42) -> None:
    """random / numpy / torch / PYTHONHASHSEED를 동시에 고정한다.

    numpy / torch가 설치되지 않은 환경에서도 안전하게 동작한다.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


__all__ = ["set_seed"]

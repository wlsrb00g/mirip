"""Vector alignment utilities for WSAV v4 Prototype/Direct contrasts."""
from __future__ import annotations

from typing import Any

import numpy as np


def vector_alignment(proto_vec: Any, direct_vec: Any, eps: float = 1e-12) -> dict[str, float | int]:
    """Compute pair-level alignment between Prototype and Direct contrast vectors.

    Parameters
    ----------
    proto_vec:
        Prototype contrast vector, Δ^P.
    direct_vec:
        Direct contrast vector, Δ^D.
    eps:
        Numerical lower bound for nonzero vector norms.

    Returns
    -------
    dict
        dot product, vector norms, cosine similarity, angle in degrees,
        Sign Concordance (SC), and magnitude ratio R.
    """
    proto_vec = np.asarray(proto_vec, dtype=float)
    direct_vec = np.asarray(direct_vec, dtype=float)

    dot_pd = float(np.dot(proto_vec, direct_vec))
    norm_proto = float(np.linalg.norm(proto_vec))
    norm_direct = float(np.linalg.norm(direct_vec))

    if norm_proto < eps or norm_direct < eps:
        return {
            "dot_pd": dot_pd,
            "norm_proto": norm_proto,
            "norm_direct": norm_direct,
            "cos_pd": float("nan"),
            "angle_deg": float("nan"),
            "SC": float("nan"),
            "R": float("nan"),
        }

    cos_pd = dot_pd / (norm_proto * norm_direct)
    cos_pd = float(np.clip(cos_pd, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_pd)))

    return {
        "dot_pd": dot_pd,
        "norm_proto": norm_proto,
        "norm_direct": norm_direct,
        "cos_pd": cos_pd,
        "angle_deg": angle_deg,
        "SC": int(dot_pd > 0),
        "R": float(norm_direct / norm_proto),
    }


__all__ = ["vector_alignment"]

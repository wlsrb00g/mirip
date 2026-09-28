"""11_sorting_gap_decomposition — decompose Prototype/Direct gap by department.

Target pair:
    craft / visual_design

This script checks whether the low Prototype-Direct cosine is driven by:
1. craft full-sample mean vs paired-subset mean shift
2. visual_design full-sample mean vs paired-subset mean shift
3. both

No applicant names, schools, or raw post_metadata text are printed.
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

THESIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_ROOT))
os.chdir(THESIS_ROOT / "notebooks")

from thesis_utils import METADATA_DIR, SHARED_STATE_DIR, load_state, vector_alignment


PAIR_A = "craft"
PAIR_B = "visual_design"
EXCLUDED_DEPTS = {"other", "exclude"}

OUT_DIR = SHARED_STATE_DIR / "alignment_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_images(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            return json.loads(value)
        except Exception:
            return []
    return []


def cos_safe(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def norm(x):
    return float(np.linalg.norm(x))


def main():
    print("=" * 76, flush=True)
    print(" 11_sorting_gap_decomposition — full vs paired subset decomposition", flush=True)
    print("=" * 76, flush=True)

    works = pd.read_csv(METADATA_DIR / "works.csv", dtype=str)
    works["images"] = works["images"].apply(parse_images)

    embed_state = load_state("03_embeddings_state")
    cache_path = Path(embed_state["cache_path"])
    edf = pd.read_parquet(cache_path)

    emb = {}
    for _, r in edf.iterrows():
        emb[str(r["image_path"])] = np.asarray(r["embedding"], dtype=float)

    print(f"  embeddings: {len(emb):,}", flush=True)

    # Build cell-level accepted-work vectors, matching run_04 logic.
    cell_vectors_raw = defaultdict(list)

    for _, w in works.iterrows():
        if w.get("work_type") != "재현작":
            continue

        sid = w.get("student_id")
        dept = w.get("main_dept_normalized")

        if pd.isna(sid) or pd.isna(dept) or dept in EXCLUDED_DEPTS:
            continue

        imgs = w.get("images") or []
        vecs = [emb[str(p)] for p in imgs if str(p) in emb]
        if not vecs:
            continue

        key = (str(sid), str(dept))
        cell_vectors_raw[key].extend(vecs)

    cell_vec = {
        key: np.mean(np.stack(vecs), axis=0)
        for key, vecs in cell_vectors_raw.items()
    }

    # Global centering, matching run_04.
    x_bar = np.mean(np.stack(list(cell_vec.values())), axis=0)
    centered = {key: v - x_bar for key, v in cell_vec.items()}

    # Full-sample dept sets.
    full_a = [v for (sid, dept), v in centered.items() if dept == PAIR_A]
    full_b = [v for (sid, dept), v in centered.items() if dept == PAIR_B]

    full_mean_a = np.mean(np.stack(full_a), axis=0)
    full_mean_b = np.mean(np.stack(full_b), axis=0)

    # Paired subset.
    pair_sids = sorted(
        sid for (sid, dept) in centered.keys()
        if dept == PAIR_A and (sid, PAIR_B) in centered
    )

    pair_a = [centered[(sid, PAIR_A)] for sid in pair_sids]
    pair_b = [centered[(sid, PAIR_B)] for sid in pair_sids]

    pair_mean_a = np.mean(np.stack(pair_a), axis=0)
    pair_mean_b = np.mean(np.stack(pair_b), axis=0)

    # Prototype, Direct, Sorting gap.
    P = full_mean_a - full_mean_b
    D = pair_mean_a - pair_mean_b
    G = P - D

    # Department-specific shifts.
    shift_a = full_mean_a - pair_mean_a
    shift_b = full_mean_b - pair_mean_b

    # Check: G should equal shift_a - shift_b.
    G_reconstructed = shift_a - shift_b
    reconstruction_error = norm(G - G_reconstructed)

    align = vector_alignment(P, D)

    rows = []

    def add_vec_row(name, vec):
        rows.append({
            "pair_key": f"{PAIR_A}__{PAIR_B}",
            "component": name,
            "norm": norm(vec),
            "cos_to_P": cos_safe(vec, P),
            "cos_to_D": cos_safe(vec, D),
            "cos_to_G": cos_safe(vec, G),
            "dot_to_P": float(np.dot(vec, P)),
            "dot_to_D": float(np.dot(vec, D)),
            "dot_to_G": float(np.dot(vec, G)),
        })

    add_vec_row("P_proto_full_contrast", P)
    add_vec_row("D_direct_pair_contrast", D)
    add_vec_row("G_sorting_gap_P_minus_D", G)
    add_vec_row(f"{PAIR_A}_shift_full_minus_pair", shift_a)
    add_vec_row(f"{PAIR_B}_shift_full_minus_pair", shift_b)
    add_vec_row("G_reconstructed_shift_a_minus_shift_b", G_reconstructed)

    decomp_csv = OUT_DIR / "11_sorting_gap_decomposition_craft_visual_design.csv"
    pd.DataFrame(rows).to_csv(decomp_csv, index=False, encoding="utf-8-sig")

    summary = {
        "pair_key": f"{PAIR_A}__{PAIR_B}",
        "n_full_a": len(full_a),
        "n_full_b": len(full_b),
        "n_pair": len(pair_sids),

        "norm_P": norm(P),
        "norm_D": norm(D),
        "norm_G": norm(G),
        "R": align["R"],
        "SC": align["SC"],
        "cos_pd": align["cos_pd"],
        "angle_deg": align["angle_deg"],

        f"norm_{PAIR_A}_shift": norm(shift_a),
        f"norm_{PAIR_B}_shift": norm(shift_b),
        f"cos_{PAIR_A}_shift_to_G": cos_safe(shift_a, G),
        f"cos_{PAIR_B}_shift_to_G": cos_safe(shift_b, G),
        f"cos_{PAIR_A}_shift_to_P": cos_safe(shift_a, P),
        f"cos_{PAIR_B}_shift_to_P": cos_safe(shift_b, P),
        f"cos_{PAIR_A}_shift_to_D": cos_safe(shift_a, D),
        f"cos_{PAIR_B}_shift_to_D": cos_safe(shift_b, D),

        "reconstruction_error_norm": reconstruction_error,
    }

    summary_csv = OUT_DIR / "11_sorting_gap_decomposition_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n  Pair summary:", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}", flush=True)
        else:
            print(f"    {k}: {v}", flush=True)

    print("\n  Component table:", flush=True)
    print(
        pd.DataFrame(rows)[
            ["component", "norm", "cos_to_P", "cos_to_D", "cos_to_G"]
        ].to_string(index=False),
        flush=True,
    )

    print(f"\n  Saved: {decomp_csv}", flush=True)
    print(f"  Saved: {summary_csv}", flush=True)
    print("===== 11 sorting-gap decomposition done =====", flush=True)


if __name__ == "__main__":
    main()

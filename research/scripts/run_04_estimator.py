"""04_estimator 전체 실행 — WSAV v4 dual-estimator framework.

For all viable department pairs, compute Direct and Prototype contrasts
from accepted-work embeddings.
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

THESIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_ROOT))
os.chdir(THESIS_ROOT / 'notebooks')

from thesis_utils import (
    METADATA_DIR, EMBED_CACHE_DIR, SHARED_STATE_DIR,
    save_state, load_state, vector_alignment,
)

import numpy as np
import pandas as pd

print('=' * 70, flush=True)
print(' 04_estimator - Prototype/Direct estimator construction (WSAV v4 sec3-4)', flush=True)
print('=' * 70, flush=True)


# ── 1) Load metadata
print('\n[A] Load metadata + 03 embeddings parquet ...', flush=True)
works = pd.read_csv(METADATA_DIR / 'works.csv')
students = pd.read_csv(METADATA_DIR / 'students.csv')

# Parse JSON columns
works['images'] = works['images'].apply(
    lambda s: json.loads(s) if isinstance(s, str) and s.strip().startswith('[') else (s if isinstance(s, list) else [])
)
print(f'  works  : {len(works):,d}', flush=True)
print(f'  students: {len(students):,d}', flush=True)


# ── 2) 02 state (split audit)
prep_state = load_state('02_preprocessing_state')
student_to_split = prep_state['split']['student_to_split']
train_sids = {sid for sid, sp in student_to_split.items() if sp == 'train'}
multi_dept_sids = {sid for sid, sp in student_to_split.items() if sp == 'multi_dept_holdout'}
print(f'  train students       : {len(train_sids):,d}', flush=True)
print(f'  multi_dept students  : {len(multi_dept_sids):,d}', flush=True)


# ── 3) 03 embeddings (parquet → dict)
embed_state = load_state('03_embeddings_state')
cache_path = Path(embed_state['cache_path'])
if not cache_path.exists():
    cache_path = EMBED_CACHE_DIR / cache_path.name
print(f'  embeddings cache: {cache_path}', flush=True)

edf = pd.read_parquet(cache_path)
embeddings: dict = {}
for _, r in edf.iterrows():
    embeddings[r['image_path']] = np.asarray(r['embedding'], dtype=np.float32)
print(f'  embeddings dict: {len(embeddings):,d} images, dim={next(iter(embeddings.values())).shape[0]}', flush=True)


# ── 4) Cell-level embedding (WSAV v4 §4.1)
print('\n[B] Cell-level accepted-work embedding aggregation (sec4.1) ...', flush=True)


# 분석 제외 라벨: 'other' (정규화 실패) + 'exclude' (한국교원대/학교명누락, chan v1.2.1 결정)
EXCLUDED_DEPTS = {'other', 'exclude'}


def compute_cell_embeddings(works_df, embeddings_dict):
    cell_imgs = defaultdict(list)
    for _, w in works_df.iterrows():
        if w['work_type'] != '재현작':
            continue
        sid = w['student_id']
        d = w['main_dept_normalized']
        if d in EXCLUDED_DEPTS or pd.isna(sid):
            continue
        for img_path in (w['images'] or []):
            if img_path in embeddings_dict:
                cell_imgs[(sid, d)].append(embeddings_dict[img_path])
    cells_ = {k: np.mean(np.stack(v), axis=0) for k, v in cell_imgs.items() if len(v) > 0}
    n_per_cell_ = {k: len(v) for k, v in cell_imgs.items() if len(v) > 0}
    return cells_, n_per_cell_


cells, n_per_cell = compute_cell_embeddings(works, embeddings)
print(f'  cells: {len(cells):,d}', flush=True)
print(f'  n_per_cell distribution:', flush=True)
n_dist = pd.Series(list(n_per_cell.values())).value_counts().sort_index()
for n, c in n_dist.items():
    pct = 100 * c / len(n_per_cell)
    print(f'    n_{{i,K}}={n}: {c:>5d} cells ({pct:>5.1f}%)', flush=True)


# -- 5) Global centering for Prototype/Direct contrasts (WSAV v4 sec4.1)
print('\n[C] Global centering for Prototype/Direct contrasts (sec4.1) ...', flush=True)


def center_cells(cells_dict):
    X = np.stack(list(cells_dict.values()))
    x_bar = X.mean(axis=0)
    return {k: v - x_bar for k, v in cells_dict.items()}, x_bar


centered, x_bar = center_cells(cells)
print(f'  |cells| = {len(centered):,d}', flush=True)
print(f'  x_bar shape = {x_bar.shape}, mean = {x_bar.mean():.6f}, std = {x_bar.std():.6f}', flush=True)


# ── 6) Per-dept membership (S_K)
print('\n[D] Per-dept membership |S_K| ...', flush=True)
S_K = defaultdict(set)
for (sid, d) in centered.keys():
    S_K[d].add(sid)
all_depts = sorted(S_K.keys())
print(f'  depts ({len(all_depts)}): {all_depts}', flush=True)
NK_table = {K: len(S_K[K]) for K in all_depts}
median_NK = float(np.median(list(NK_table.values())))
print(f'  |S_K| table:', flush=True)
for K in sorted(all_depts, key=lambda k: -NK_table[k]):
    print(f'    {K:25s}  {NK_table[K]:>5d}', flush=True)
print(f'  median |S_K| = {median_NK:.1f}', flush=True)


# ── 7) Direct estimator (WSAV v4 §3.3)
print('\n[E] Direct estimator — multi-acceptance contrast (§3.3) ...', flush=True)


def direct_estimator(centered_dict, K, Kp):
    diffs = []
    for (sid, d), v in centered_dict.items():
        if d != K:
            continue
        v_kp = centered_dict.get((sid, Kp))
        if v_kp is None:
            continue
        diffs.append(v - v_kp)
    if not diffs:
        return None, 0
    return np.mean(np.stack(diffs), axis=0), len(diffs)


# ?? 8) Prototype estimator (WSAV v4 ?3.2)
def prototype_estimator(centered_dict, K):
    """Full admitted-sample accepted-work mean for department K."""
    K_cells = [v for (sid, d), v in centered_dict.items() if d == K]
    if not K_cells:
        return None
    return np.mean(np.stack(K_cells), axis=0)


def prototype_contrast(centered_dict, K, Kp):
    proto_K = prototype_estimator(centered_dict, K)
    proto_Kp = prototype_estimator(centered_dict, Kp)
    if proto_K is None or proto_Kp is None:
        return None
    return proto_K - proto_Kp


def summarize_alignment(rows, label, pair_filter):
    """Summarize cosine alignment for a named pair subset."""
    subset = [r for r in rows if pair_filter(r)]
    cos_vals = [r['cos_pd'] for r in subset if pd.notna(r['cos_pd'])]
    angle_vals = [r['angle_deg'] for r in subset if pd.notna(r['angle_deg'])]
    return {
        'group': label,
        'n_pairs': int(len(subset)),
        'SC_count': int(sum(r['SC'] for r in subset if pd.notna(r['SC']))),
        'SC_total': int(len(subset)),
        'mean_cos': float(np.mean(cos_vals)) if cos_vals else np.nan,
        'median_cos': float(np.median(cos_vals)) if cos_vals else np.nan,
        'min_cos': float(np.min(cos_vals)) if cos_vals else np.nan,
        'max_angle': float(np.max(angle_vals)) if angle_vals else np.nan,
        'median_R': float(np.median([r['R'] for r in subset if pd.notna(r['R'])])) if subset else np.nan,
    }


# ── 9) 모든 pair 순회 (K < K' lexicographic)
print('  Computing Δ̂_{K,K\'} for all pairs ...', flush=True)
Delta_direct = {}
Delta_proto = {}
n_pair_table = {}
viable_pairs = []
empty_pairs = []

for K, Kp in combinations(all_depts, 2):
    delta_d, n_pair = direct_estimator(centered, K, Kp)
    delta_p = prototype_contrast(centered, K, Kp)
    n_pair_table[(K, Kp)] = n_pair
    if delta_d is not None and delta_p is not None:
        Delta_direct[(K, Kp)] = delta_d
        Delta_proto[(K, Kp)] = delta_p
        viable_pairs.append((K, Kp))
    else:
        empty_pairs.append((K, Kp))

print(f'  total pairs   : {len(list(combinations(all_depts, 2)))}', flush=True)
print(f'  viable (n≥1) : {len(viable_pairs)}', flush=True)
print(f'  empty (n=0)   : {len(empty_pairs)}', flush=True)

# Core pairs (WSAV v4 — Design 4 core {design_general, industrial_design, craft, visual_design})
core_set = {'design_general', 'industrial_design', 'craft', 'visual_design'}
core_pairs = [(K, Kp) for (K, Kp) in viable_pairs if K in core_set and Kp in core_set]
print(f'  core (design 4): {len(core_pairs)} pairs', flush=True)

# Pair-level alignment table: SC/R preserved, cos_pd and angle_deg added.
pair_alignment_rows = []
for K, Kp in viable_pairs:
    align = vector_alignment(Delta_proto[(K, Kp)], Delta_direct[(K, Kp)])
    pair_alignment_rows.append({
        'dept_a': K,
        'dept_b': Kp,
        'pair_key': f'{K}__{Kp}',
        'n_pair': int(n_pair_table[(K, Kp)]),
        **align,
    })

DESIGN_6 = {'craft', 'design_general', 'fashion', 'industrial_design', 'interior', 'visual_design'}
summary_rows = [
    summarize_alignment(pair_alignment_rows, 'all_viable', lambda r: True),
    summarize_alignment(pair_alignment_rows, 'design6_viable', lambda r: r['dept_a'] in DESIGN_6 and r['dept_b'] in DESIGN_6),
    summarize_alignment(pair_alignment_rows, 'design4_core', lambda r: r['dept_a'] in core_set and r['dept_b'] in core_set),
    summarize_alignment(pair_alignment_rows, 'design6_periphery', lambda r: (r['dept_a'] in DESIGN_6 and r['dept_b'] in DESIGN_6) and not (r['dept_a'] in core_set and r['dept_b'] in core_set)),
]

pair_alignment_df = pd.DataFrame(pair_alignment_rows)
summary_df = pd.DataFrame(summary_rows)

pair_alignment_csv = SHARED_STATE_DIR / '04_pair_alignment.csv'
summary_csv = SHARED_STATE_DIR / '04_pair_alignment_summary.csv'
pair_alignment_df.to_csv(pair_alignment_csv, index=False, encoding='utf-8-sig')
summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

print('\n  Per-pair n & alignment metrics (top 12 by n):', flush=True)
print(f'    {"K":<22s} {"K\'":<22s} {"n":>4s} {"||P||":>10s} {"||D||":>10s} {"R":>7s} {"SC":>3s} {"cos":>7s} {"angle":>8s}', flush=True)
for row in sorted(pair_alignment_rows, key=lambda r: -r['n_pair'])[:12]:
    print(f"    {row['dept_a']:<22s} {row['dept_b']:<22s} {row['n_pair']:>4d} "
          f"{row['norm_proto']:>10.4f} {row['norm_direct']:>10.4f} {row['R']:>7.3f} "
          f"{int(row['SC']):>3d} {row['cos_pd']:>7.3f} {row['angle_deg']:>8.2f}", flush=True)

print('\n  Alignment summary:', flush=True)
for row in summary_rows:
    print(f"    {row['group']:<18s} n={row['n_pairs']:>2d}, SC={row['SC_count']}/{row['SC_total']}, "
          f"mean cos={row['mean_cos']:.3f}, median cos={row['median_cos']:.3f}, "
          f"min cos={row['min_cos']:.3f}, max angle={row['max_angle']:.2f}, "
          f"median R={row['median_R']:.3f}", flush=True)

print(f'  Saved pair CSV   : {pair_alignment_csv.relative_to(THESIS_ROOT.parent)}', flush=True)
print(f'  Saved summary CSV: {summary_csv.relative_to(THESIS_ROOT.parent)}', flush=True)


# ── 10) Handoff state
# ── 12) Handoff state
print('\n[H] Handoff — save state ...', flush=True)
estimator_state = {
    'framework': 'dual_estimator (WSAV v4)',
    'embedding_dim': int(x_bar.shape[0]),
    'n_cells_total': int(len(centered)),
    'NK_table': NK_table,
    'median_NK': median_NK,
    'cell_centered_grand_mean': x_bar.tolist(),
    'all_depts': all_depts,
    'core_set': sorted(core_set),

    'Delta_direct': {f'{K}__{Kp}': v.tolist() for (K, Kp), v in Delta_direct.items()},
    'Delta_proto':  {f'{K}__{Kp}': v.tolist() for (K, Kp), v in Delta_proto.items()},
    'n_pair_table': {f'{K}__{Kp}': v for (K, Kp), v in n_pair_table.items()},
    'viable_pairs': [f'{K}__{Kp}' for (K, Kp) in viable_pairs],
    'core_pairs':   [f'{K}__{Kp}' for (K, Kp) in core_pairs],
    'empty_pairs':  [f'{K}__{Kp}' for (K, Kp) in empty_pairs],
    'pair_alignment_results': pair_alignment_rows,
    'alignment_summary': {row['group']: row for row in summary_rows},
    'pair_alignment_csv': str(pair_alignment_csv.relative_to(THESIS_ROOT.parent)),
    'pair_alignment_summary_csv': str(summary_csv.relative_to(THESIS_ROOT.parent)),

}
save_state('04_estimator_state', estimator_state)
print(f'  Saved: outputs/shared_state/04_estimator_state.json', flush=True)
print(f'  Saved: {pair_alignment_csv.relative_to(THESIS_ROOT.parent)}', flush=True)
print(f'  Saved: {summary_csv.relative_to(THESIS_ROOT.parent)}', flush=True)


# ── 13) Summary
print('\n' + '=' * 70, flush=True)
print(' 04 완료 요약', flush=True)
print('=' * 70, flush=True)
print(f'  embedding dim     : {x_bar.shape[0]}', flush=True)
print(f'  cells total       : {len(centered):,d}', flush=True)
print(f'  median |S_K|      : {median_NK:.1f}', flush=True)
print(f'  viable pairs      : {len(viable_pairs)}', flush=True)
print(f'  core pairs (4 코어): {len(core_pairs)}', flush=True)
print(f'  empty pairs       : {len(empty_pairs)}', flush=True)
print('===== 04 완료 =====', flush=True)







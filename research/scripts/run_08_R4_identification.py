"""08_R4_identification — Bootstrap inference for WSAV v4 dual estimator.

Hierarchical cluster bootstrap (B=1000, student-level resample):
- Per-pair R = ||Δ̂^D|| / ||Δ̂^P|| with 95% CI
- Per-pair cos(Δ̂^D, Δ̂^P) bootstrap distribution
- Sign Concordance bootstrap probability
- Aggregate median R across core 6 / periphery 8 with CI

Per-student sign-flip permutation test (B=1000) for H0: Δ̂^D = 0.

Output: outputs/shared_state/08_R4_state.json
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from time import time

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
print(' 08_R4 — Bootstrap inference (WSAV v4)', flush=True)
print('=' * 70, flush=True)

SEED = 42
B = 1000
rng_boot = np.random.default_rng(SEED)
rng_perm = np.random.default_rng(SEED + 1)


# ── 1) Load metadata + embeddings (04 패턴)
print('\n[A] Load metadata + embeddings ...', flush=True)
works = pd.read_csv(METADATA_DIR / 'works.csv')
works['images'] = works['images'].apply(
    lambda s: json.loads(s) if isinstance(s, str) and s.strip().startswith('[')
    else (s if isinstance(s, list) else [])
)
embed_state = load_state('03_embeddings_state')
cache_path = Path(embed_state['cache_path'])
if not cache_path.exists():
    cache_path = EMBED_CACHE_DIR / cache_path.name
edf = pd.read_parquet(cache_path)
embeddings = {
    r['image_path']: np.asarray(r['embedding'], dtype=np.float32)
    for _, r in edf.iterrows()
}
print(f'  works: {len(works):,d}, embeddings: {len(embeddings):,d}', flush=True)


# ── 2) Cell-level embedding (04 동일)
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
    return {k: np.mean(np.stack(v), axis=0)
            for k, v in cell_imgs.items() if len(v) > 0}


cells = compute_cell_embeddings(works, embeddings)
print(f'  cells: {len(cells):,d}', flush=True)


# ── 3) Centering (04 동일)
X = np.stack(list(cells.values()))
x_bar = X.mean(axis=0)
centered = {k: v - x_bar for k, v in cells.items()}
print(f'  |cells| = {len(centered)}, dim = {x_bar.shape[0]}', flush=True)


# ── 4) Indexing structures
all_sids = sorted({sid for (sid, _) in centered.keys()})
all_depts = sorted({d for (_, d) in centered.keys()})
N_students = len(all_sids)
sid_to_idx = {sid: i for i, sid in enumerate(all_sids)}
print(f'  N students: {N_students}, depts: {len(all_depts)}', flush=True)


# Per-pair multi-dept diffs (학생 단위)
pair_diffs = {}
for K, Kp in combinations(all_depts, 2):
    K_dict = {sid: v for (sid, d), v in centered.items() if d == K}
    Kp_dict = {sid: v for (sid, d), v in centered.items() if d == Kp}
    common_sids = set(K_dict) & set(Kp_dict)
    if not common_sids:
        continue
    diffs = [(sid, K_dict[sid] - Kp_dict[sid]) for sid in common_sids]
    pair_diffs[(K, Kp)] = diffs


# Design 6 / core 4 scoping
DESIGN_6 = {'craft', 'design_general', 'fashion',
            'industrial_design', 'interior', 'visual_design'}
CORE_4 = {'craft', 'design_general', 'industrial_design', 'visual_design'}

viable_pairs = sorted(pair_diffs.keys())
design6_pairs = [p for p in viable_pairs if p[0] in DESIGN_6 and p[1] in DESIGN_6]
core_pairs = [p for p in viable_pairs if p[0] in CORE_4 and p[1] in CORE_4]
periphery_pairs = [p for p in design6_pairs if p not in core_pairs]
print(f'  viable pairs: {len(viable_pairs)} | Design 6: {len(design6_pairs)} '
      f'(core 6: {len(core_pairs)} + periphery: {len(periphery_pairs)})', flush=True)


# ── 5) Pre-compute arrays for vectorized bootstrap
print('\n[B] Pre-compute arrays for bootstrap ...', flush=True)

dept_to_sid_vec = defaultdict(list)
for (sid, d), v in centered.items():
    dept_to_sid_vec[d].append((sid, v))

dept_sids = {}      # K -> array of sid indices (length nK)
dept_vecs = {}      # K -> array (nK, dim)
for K in all_depts:
    sid_vec_list = dept_to_sid_vec[K]
    dept_sids[K] = np.array([sid_to_idx[sid] for sid, _ in sid_vec_list])
    dept_vecs[K] = np.stack([v for _, v in sid_vec_list])

pair_sids = {}      # (K, Kp) -> array of sid indices
pair_diff_vecs = {}  # (K, Kp) -> array (n_pair, dim)
for (K, Kp), diffs in pair_diffs.items():
    pair_sids[(K, Kp)] = np.array([sid_to_idx[sid] for sid, _ in diffs])
    pair_diff_vecs[(K, Kp)] = np.stack([d for _, d in diffs])

print(f'  Pre-computed: {len(dept_vecs)} dept arrays, '
      f'{len(pair_diff_vecs)} pair arrays', flush=True)


# ── 6) Bootstrap loop (B=1000, hierarchical cluster, student-level resample)
print(f'\n[C] Bootstrap (B={B}) ...', flush=True)

t0 = time()

boot_R = {p: [] for p in viable_pairs}
boot_cos = {p: [] for p in viable_pairs}
boot_norm_D = {p: [] for p in viable_pairs}
boot_norm_P = {p: [] for p in viable_pairs}

for b in range(B):
    sampled_idx = rng_boot.choice(N_students, size=N_students, replace=True)
    weights = np.bincount(sampled_idx, minlength=N_students).astype(np.float32)

    # Per-dept proto means (weighted average)
    dept_proto = {}
    for K in all_depts:
        w = weights[dept_sids[K]]
        total = w.sum()
        if total < 2:
            continue
        dept_proto[K] = (w[:, None] * dept_vecs[K]).sum(axis=0) / total

    # Per-pair direct + proto
    for K, Kp in viable_pairs:
        w = weights[pair_sids[(K, Kp)]]
        total = w.sum()
        if total < 2:
            continue
        delta_d = (w[:, None] * pair_diff_vecs[(K, Kp)]).sum(axis=0) / total
        if K not in dept_proto or Kp not in dept_proto:
            continue
        delta_p = dept_proto[K] - dept_proto[Kp]

        align = vector_alignment(delta_p, delta_d, eps=1e-9)
        if pd.isna(align['R']) or pd.isna(align['cos_pd']):
            continue
        boot_R[(K, Kp)].append(align['R'])
        boot_cos[(K, Kp)].append(align['cos_pd'])
        boot_norm_D[(K, Kp)].append(align['norm_direct'])
        boot_norm_P[(K, Kp)].append(align['norm_proto'])

    if (b + 1) % 100 == 0:
        elapsed = time() - t0
        print(f'  b={b+1}/{B}  elapsed={elapsed:.1f}s', flush=True)

print(f'  Bootstrap loop done in {time() - t0:.1f}s', flush=True)


# ── 7) Per-pair statistics
print('\n[D] Per-pair statistics (point + 95% CI) ...', flush=True)


def compute_point(K, Kp):
    direct_vec = pair_diff_vecs[(K, Kp)].mean(axis=0)
    proto_K = dept_vecs[K].mean(axis=0)
    proto_Kp = dept_vecs[Kp].mean(axis=0)
    proto_vec = proto_K - proto_Kp
    align = vector_alignment(proto_vec, direct_vec, eps=1e-9)

    # Backward-compatible aliases for existing paper/state readers.
    align['norm_D'] = align['norm_direct']
    align['norm_P'] = align['norm_proto']
    align['cos'] = align['cos_pd']
    return align


per_pair_results = {}
for (K, Kp) in viable_pairs:
    point = compute_point(K, Kp)
    Rs = np.array(boot_R[(K, Kp)])
    coss = np.array(boot_cos[(K, Kp)])
    n_eff = len(Rs)

    if n_eff < 100:
        ci_R = ci_cos = sc_prob = None
        R_med = cos_med = None
    else:
        ci_R = [float(np.quantile(Rs, 0.025)), float(np.quantile(Rs, 0.975))]
        ci_cos = [float(np.quantile(coss, 0.025)), float(np.quantile(coss, 0.975))]
        sc_prob = float((coss > 0).mean())
        R_med = float(np.median(Rs))
        cos_med = float(np.median(coss))

    per_pair_results[f'{K}__{Kp}'] = {
        'n_pair': int(len(pair_diff_vecs[(K, Kp)])),
        'point': point,
        'boot': {
            'B_effective': n_eff,
            'R_median': R_med,
            'R_CI95': ci_R,
            'cos_median': cos_med,
            'cos_CI95': ci_cos,
            'SC_prob': sc_prob,
        },
    }

# Print tables
print(f"\n  {'Pair':<45s} {'n':>4s} {'R_pt':>7s} {'R_med':>7s} "
      f"{'SC':>3s} {'cos':>7s} {'angle':>8s} {'SCp':>5s}", flush=True)
print(f"  Core 6 (Design 4):", flush=True)
for (K, Kp) in core_pairs:
    r = per_pair_results[f'{K}__{Kp}']
    pt, bt = r['point'], r['boot']
    sc_p = f"{bt['SC_prob']:.3f}" if bt['SC_prob'] is not None else '  -  '
    print(f"  {K+'__'+Kp:<45s} {r['n_pair']:>4d} {pt['R']:>7.3f} "
          f"{(bt['R_median'] or 0):>7.3f} {int(pt['SC']):>3d} "
          f"{pt['cos_pd']:>7.3f} {pt['angle_deg']:>8.2f} {sc_p:>5s}",
          flush=True)
print(f"\n  Periphery 8 (Design 6 minus Design 4):", flush=True)
for (K, Kp) in periphery_pairs:
    r = per_pair_results[f'{K}__{Kp}']
    pt, bt = r['point'], r['boot']
    sc_p = f"{bt['SC_prob']:.3f}" if bt['SC_prob'] is not None else '  -  '
    print(f"  {K+'__'+Kp:<45s} {r['n_pair']:>4d} {pt['R']:>7.3f} "
          f"{(bt['R_median'] or 0):>7.3f} {int(pt['SC']):>3d} "
          f"{pt['cos_pd']:>7.3f} {pt['angle_deg']:>8.2f} {sc_p:>5s}",
          flush=True)


# ── 8) Aggregate median R across pair sets
print('\n[E] Aggregate median R (core 6 + periphery 8) ...', flush=True)


def aggregate_median_R(pair_set, boot_R_dict, B):
    medians = []
    for b in range(B):
        rs = [boot_R_dict[p][b] for p in pair_set
              if b < len(boot_R_dict[p])]
        if rs:
            medians.append(float(np.median(rs)))
    return np.array(medians)


core_med = aggregate_median_R(core_pairs, boot_R, B)
periph_med = aggregate_median_R(periphery_pairs, boot_R, B)
design6_med = aggregate_median_R(design6_pairs, boot_R, B)


def agg_block(pair_set, point_results, boot_med):
    cos_vals = [point_results[f'{K}__{Kp}']['point']['cos_pd'] for K, Kp in pair_set]
    angle_vals = [point_results[f'{K}__{Kp}']['point']['angle_deg'] for K, Kp in pair_set]
    return {
        'point_median_R': float(np.median(
            [point_results[f'{K}__{Kp}']['point']['R'] for K, Kp in pair_set])),
        'boot_median_R': float(np.median(boot_med)) if len(boot_med) else None,
        'boot_median_R_CI95': (
            [float(np.quantile(boot_med, 0.025)),
             float(np.quantile(boot_med, 0.975))]
            if len(boot_med) else None),
        'SC_count': int(sum(point_results[f'{K}__{Kp}']['point']['SC']
                            for K, Kp in pair_set)),
        'SC_total': len(pair_set),
        'mean_cos': float(np.mean(cos_vals)) if cos_vals else None,
        'median_cos': float(np.median(cos_vals)) if cos_vals else None,
        'min_cos': float(np.min(cos_vals)) if cos_vals else None,
        'max_angle': float(np.max(angle_vals)) if angle_vals else None,
    }


agg_results = {
    'core_6': agg_block(core_pairs, per_pair_results, core_med),
    'periphery_8': agg_block(periphery_pairs, per_pair_results, periph_med),
    'design6_14': agg_block(design6_pairs, per_pair_results, design6_med),
}

for label, r in agg_results.items():
    print(f"  {label}: pt={r['point_median_R']:.3f}, "
          f"boot={r['boot_median_R']:.3f} "
          f"CI95={r['boot_median_R_CI95']}, "
          f"SC={r['SC_count']}/{r['SC_total']}, "
          f"mean cos={r['mean_cos']:.3f}, median cos={r['median_cos']:.3f}, "
          f"min cos={r['min_cos']:.3f}, max angle={r['max_angle']:.2f}", flush=True)


# ── 8.5) Export point alignment CSVs
pair_alignment_rows = []
for K, Kp in viable_pairs:
    pt = per_pair_results[f'{K}__{Kp}']['point']
    bt = per_pair_results[f'{K}__{Kp}']['boot']
    pair_alignment_rows.append({
        'dept_a': K,
        'dept_b': Kp,
        'pair_key': f'{K}__{Kp}',
        'n_pair': per_pair_results[f'{K}__{Kp}']['n_pair'],
        'norm_proto': pt['norm_proto'],
        'norm_direct': pt['norm_direct'],
        'R': pt['R'],
        'dot_pd': pt['dot_pd'],
        'SC': pt['SC'],
        'cos_pd': pt['cos_pd'],
        'angle_deg': pt['angle_deg'],
        'boot_R_median': bt['R_median'],
        'boot_R_CI95_low': bt['R_CI95'][0] if bt['R_CI95'] else np.nan,
        'boot_R_CI95_high': bt['R_CI95'][1] if bt['R_CI95'] else np.nan,
        'boot_cos_median': bt['cos_median'],
        'boot_cos_CI95_low': bt['cos_CI95'][0] if bt['cos_CI95'] else np.nan,
        'boot_cos_CI95_high': bt['cos_CI95'][1] if bt['cos_CI95'] else np.nan,
        'boot_SC_prob': bt['SC_prob'],
    })

summary_rows = []
for label, r in agg_results.items():
    summary_rows.append({'group': label, **r})

pair_alignment_csv = SHARED_STATE_DIR / '08_pair_alignment.csv'
summary_csv = SHARED_STATE_DIR / '08_pair_alignment_summary.csv'
pd.DataFrame(pair_alignment_rows).to_csv(pair_alignment_csv, index=False, encoding='utf-8-sig')
pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding='utf-8-sig')
print(f'  Saved pair CSV   : {pair_alignment_csv.relative_to(THESIS_ROOT.parent)}', flush=True)
print(f'  Saved summary CSV: {summary_csv.relative_to(THESIS_ROOT.parent)}', flush=True)


# ── 9) Sign-flip permutation test (per-pair, H0: Δ^D = 0)
print(f'\n[F] Sign-flip permutation test (B={B}) ...', flush=True)

permutation_results = {}
for (K, Kp) in viable_pairs:
    diffs = pair_diff_vecs[(K, Kp)]
    n_pair = diffs.shape[0]
    if n_pair < 2:
        permutation_results[f'{K}__{Kp}'] = {'p_value': None, 'note': 'n<2'}
        continue
    T_obs = float(np.linalg.norm(diffs.mean(axis=0)))
    T_perm = []
    for _ in range(B):
        signs = rng_perm.choice([-1, 1], size=n_pair).astype(np.float32)
        delta = (signs[:, None] * diffs).mean(axis=0)
        T_perm.append(float(np.linalg.norm(delta)))
    T_perm = np.array(T_perm)
    permutation_results[f'{K}__{Kp}'] = {
        'T_obs': T_obs,
        'T_perm_median': float(np.median(T_perm)),
        'p_value': float((T_perm >= T_obs).mean()),
        'n_pair': int(n_pair),
        'note': 'one-sided sign-flip permutation under H0: Δ^D = 0',
    }

print(f"  Sign-flip p-values (one-sided, H0: Δ^D = 0):", flush=True)
print(f"  Core 6 (Bonferroni α/6 = 0.0083):", flush=True)
for (K, Kp) in core_pairs:
    p = permutation_results[f'{K}__{Kp}']
    sig = '***' if p['p_value'] < 0.001 else ('**' if p['p_value'] < 0.0083 else '')
    print(f"    {K+'__'+Kp:<45s} p = {p['p_value']:.4f} {sig}", flush=True)
print(f"  Periphery 8:", flush=True)
for (K, Kp) in periphery_pairs:
    p = permutation_results[f'{K}__{Kp}']
    sig = '***' if p['p_value'] < 0.001 else ('**' if p['p_value'] < 0.05 else '')
    print(f"    {K+'__'+Kp:<45s} p = {p['p_value']:.4f} {sig}", flush=True)


# ── 10) Save state
print('\n[G] Save state ...', flush=True)
r4_state = {
    'framework': 'bootstrap_inference (WSAV v4)',
    'B': B,
    'seed': SEED,
    'embedding_dim': int(x_bar.shape[0]),
    'N_students': int(N_students),
    'design6_pairs': [f'{K}__{Kp}' for K, Kp in design6_pairs],
    'core_pairs': [f'{K}__{Kp}' for K, Kp in core_pairs],
    'periphery_pairs': [f'{K}__{Kp}' for K, Kp in periphery_pairs],
    'per_pair_results': per_pair_results,
    'aggregate_results': agg_results,
    'pair_alignment_csv': str(pair_alignment_csv.relative_to(THESIS_ROOT.parent)),
    'pair_alignment_summary_csv': str(summary_csv.relative_to(THESIS_ROOT.parent)),
    'permutation_results': permutation_results,
}
save_state('08_R4_state', r4_state)
print(f'  Saved: outputs/shared_state/08_R4_state.json', flush=True)
print(f'  Saved: {pair_alignment_csv.relative_to(THESIS_ROOT.parent)}', flush=True)
print(f'  Saved: {summary_csv.relative_to(THESIS_ROOT.parent)}', flush=True)


# ── 11) Summary
print('\n' + '=' * 70, flush=True)
print(' 08 완료 요약', flush=True)
print('=' * 70, flush=True)
print(f"  Core 6 (Design 4):", flush=True)
print(f"    point median R = {agg_results['core_6']['point_median_R']:.3f}", flush=True)
print(f"    boot  median R = {agg_results['core_6']['boot_median_R']:.3f}  "
      f"CI95 = {agg_results['core_6']['boot_median_R_CI95']}", flush=True)
print(f"    SC = {agg_results['core_6']['SC_count']}/{agg_results['core_6']['SC_total']}", flush=True)
print(f"    mean cos = {agg_results['core_6']['mean_cos']:.3f}, "
      f"median cos = {agg_results['core_6']['median_cos']:.3f}, "
      f"min cos = {agg_results['core_6']['min_cos']:.3f}, "
      f"max angle = {agg_results['core_6']['max_angle']:.2f}", flush=True)
print(f"  Design 6 viable pairs:", flush=True)
print(f"    point median R = {agg_results['design6_14']['point_median_R']:.3f}", flush=True)
print(f"    SC = {agg_results['design6_14']['SC_count']}/{agg_results['design6_14']['SC_total']}", flush=True)
print(f"    mean cos = {agg_results['design6_14']['mean_cos']:.3f}, "
      f"median cos = {agg_results['design6_14']['median_cos']:.3f}, "
      f"min cos = {agg_results['design6_14']['min_cos']:.3f}, "
      f"max angle = {agg_results['design6_14']['max_angle']:.2f}", flush=True)
print(f"  Periphery 8 (Design 6 minus Design 4):", flush=True)
print(f"    point median R = {agg_results['periphery_8']['point_median_R']:.3f}", flush=True)
print(f"    boot  median R = {agg_results['periphery_8']['boot_median_R']:.3f}  "
      f"CI95 = {agg_results['periphery_8']['boot_median_R_CI95']}", flush=True)
print(f"    SC = {agg_results['periphery_8']['SC_count']}/{agg_results['periphery_8']['SC_total']}", flush=True)
print(f"    mean cos = {agg_results['periphery_8']['mean_cos']:.3f}, "
      f"median cos = {agg_results['periphery_8']['median_cos']:.3f}, "
      f"min cos = {agg_results['periphery_8']['min_cos']:.3f}, "
      f"max angle = {agg_results['periphery_8']['max_angle']:.2f}", flush=True)
print('===== 08 완료 =====', flush=True)

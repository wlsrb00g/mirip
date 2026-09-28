"""부서 라벨 재정규화 — chan님 결정 (2026-05-09) 반영.

원본 dept_mapping.py 의 한계 (회화/조소 mix in fine_art) 를 thesis 단에서 보정.

핵심 변경:
  1. 조소 family (조소/조각/환경조각/입체미술/입체조형) 모두 'sculpture' 로 분리
     (단 경기대 입체조형 13건은 chan 결정으로 'visual_design')
  2. 회화 family (회화/동양화/서양화/한국화/판화) → 'fine_art'
  3. Mixed family (조형예술/현대미술/응용미술/미술교육/예술학과 등):
     - 학교별 chan 명시 결정 우선
     - 명시 없는 곳: 일반 규칙 ('조형예술학과' → sculpture, 그 외 → fine_art)
  4. 학교 정보 없는 경우 → 'exclude'

입력: (department_raw, university)
출력: 새 normalized label
"""
from __future__ import annotations


# ═══════════════════════════════════════════════════════════════
# 학교별 명시 결정 (chan 2026-05-09)
# ═══════════════════════════════════════════════════════════════
SCHOOL_MIXED_OVERRIDE = {
    # 학교 → mixed family raw 부서명에 적용할 새 라벨
    '서울과기대':   'fine_art',     # painting (사실화)
    '서울여대':     'fine_art',     # painting (사실화)
    '경기대':       'fine_art',     # painting (mixed family — 파인아트 등)
    '한예종':       'sculpture',    # 전부 조소
    '수원대':       'sculpture',    # 조형예술학부
    '상명대':       'animation',    # 아트앤컬쳐 → 만화/애니
    '한국교원대':   'exclude',      # 사람 사진 섞임
}


# ═══════════════════════════════════════════════════════════════
# Keyword sets
# ═══════════════════════════════════════════════════════════════
# 조소 family (3D 입체)
SCULPTURE_KEYS = ['조소', '조각', '환경조각', '입체미술', '입체조형', '3D조형']

# 회화 family (2D painting)
PAINTING_KEYS = ['회화', '동양화', '서양화', '한국화', '판화']

# Mixed family (매체 모호 — 학교별 결정 또는 일반 규칙)
MIXED_KEYS = [
    '조형예술', '현대미술', '응용미술', '미술교육', '예술학과',
    '불교미술', '트랜스아트', '아트앤컬쳐',
    '파인아트', 'Fine Art', 'Fine Arts',
    '미술학부', '미술과', '미술',
]

# 다른 카테고리에 이미 정확히 매핑된 라벨 — mixed 처리 skip
# (chan 2026-05-09 review: '미술' / '예술학과' 키워드가 craft/industrial/animation/interior 등 침범 방지)
SAFE_LABELS = {
    'craft', 'industrial_design', 'visual_design',
    'animation', 'interior', 'fashion', 'design_general',
}


def has_keyword(text: str, keys: list) -> bool:
    if not text:
        return False
    return any(k in text for k in keys)


def remap_dept(department_raw: str, university: str = None,
               existing_normalized: str = None) -> str:
    """새 정규화 함수.

    Parameters
    ----------
    department_raw  : 원본 한국어 학과명 (또는 영문 'fine_art', 'craft' 등 사전변환된 값)
    university      : 학교명 (학교별 결정에 필요)
    existing_normalized : 기존 dept_mapping.py 결과 (fallback 용)

    Returns
    -------
    새 라벨: 'sculpture', 'fine_art', 'visual_design', 'animation',
            'craft', 'industrial_design', 'fashion', 'interior',
            'design_general', 'exclude', 'other'
    """
    if not department_raw:
        # raw 가 없으면 기존 라벨 fallback
        return existing_normalized or 'other'

    raw = department_raw.strip()

    # ──────────────────────────────────────────────────────────
    # [1] 조소 family — 모든 학교 sculpture (단 경기대 입체조형 예외)
    # ──────────────────────────────────────────────────────────
    if has_keyword(raw, SCULPTURE_KEYS):
        # chan 결정: 경기대 입체조형 → visual_design (시디 입시)
        if university == '경기대' and ('입체조형' in raw):
            return 'visual_design'
        return 'sculpture'

    # ──────────────────────────────────────────────────────────
    # [2] 회화 family — fine_art
    # ──────────────────────────────────────────────────────────
    if has_keyword(raw, PAINTING_KEYS):
        return 'fine_art'

    # ──────────────────────────────────────────────────────────
    # [3] Mixed family — 학교별 명시 우선, 없으면 일반 규칙
    # 단 existing_normalized 가 SAFE_LABELS 인 경우 skip (다른 카테고리 침범 방지)
    # ──────────────────────────────────────────────────────────
    if has_keyword(raw, MIXED_KEYS) and existing_normalized not in SAFE_LABELS:
        # 학교 정보 없으면 exclude (학교명 누락 케이스)
        if not university or university in ('', '?'):
            return 'exclude'

        # 학교별 명시 결정
        if university in SCHOOL_MIXED_OVERRIDE:
            return SCHOOL_MIXED_OVERRIDE[university]

        # 일반 규칙: 조형예술학과 / ( 글로컬) 조형예술학과 → sculpture
        if '조형예술학과' in raw:
            return 'sculpture'

        # 일반 규칙: 조형예술과 (단순 형태, 위 학교 외) → painting / fine_art
        # (실제로 조형예술과 분포는 위 명시 학교에 거의 다 들어 있음)
        if raw.endswith('조형예술과') or raw == '조형예술과':
            return 'fine_art'

        # 나머지 mixed (현대미술/응용미술/미술교육/예술학과/불교미술/트랜스아트/조형예술 단순) → fine_art
        return 'fine_art'

    # ──────────────────────────────────────────────────────────
    # [4] Sculpture / fine_art / mixed 키워드 모두 안 맞음
    #     → 기존 normalized label 그대로 (craft, visual_design, design_general,
    #     industrial_design, fashion, interior, animation 등)
    # ──────────────────────────────────────────────────────────
    return existing_normalized or 'other'


def get_remap_summary() -> dict:
    """디버그/문서화용 요약 반환."""
    return {
        'scope': 'thesis-side relabeling (parser.py 안 건드림)',
        'school_explicit': dict(SCHOOL_MIXED_OVERRIDE),
        'sculpture_keys': list(SCULPTURE_KEYS),
        'painting_keys': list(PAINTING_KEYS),
        'mixed_keys': list(MIXED_KEYS),
        'general_rule': "조형예술학과 → sculpture, 조형예술과 → fine_art",
        'special_case': "경기대 입체조형 → visual_design (chan 결정)",
        'reference': 'chan 2026-05-09 (axis_inspection 시각 검증 후)',
    }

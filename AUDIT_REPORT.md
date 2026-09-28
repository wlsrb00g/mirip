# 공개 저장소 선별 감사 — MIRIP / WSAV

- 기준 소스: `Desktop/mirip/mirip-thesis-main`; 제품 프런트엔드 저장소 `mirip-master`는 별도 버전이며 본 연구 저장소에 합치지 않음.
- 포함: thesis_utils 일부, estimator / bootstrap / decomposition 코드 일부, 공개 가능한 aggregate department-pair heatmap, 기존 requirements와 MIT LICENSE.
- 제외: `.git` history, manuscript PDF/TeX 원문(저자 이메일 포함), raw artworks, applicant metadata, embedding cache, per-applicant records, internal experiment outputs, unrelated MIRIP web app build artifacts.
- 결과 버전: 최종 원고의 Design 6=14 viable pair / Design 4=6 pairs 우선. root README의 17-pair summary는 stale version으로 README에서 차이를 명시함.
- 실행성: 데이터·공유상태 경로 미포함. 코드 entrypoints는 데이터가 승인된 환경에 존재해야 실행 가능하며 여기서는 import/run을 실행하지 않음.
- 개인 기여: Git author metadata를 복사하지 않았고 코드만으로 저자별 기여를 확정하지 않음.

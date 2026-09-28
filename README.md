# MIRIP — WSAV dual-estimator study

Selection-only admissions data에는 합격 기록만 있고 탈락 기록이나 직접적인 선호 관측이 없습니다. 이 연구는 accepted-work embedding으로 department-level prototype contrast와 동일 지원자 내 direct contrast를 나란히 추정하고, 두 결과의 차이를 Sorting Gap으로 분석합니다.

## 방법

- **Accepted-work representation:** 최종 원고는 DINOv2 ViT-L/14 frozen backbone의 1,024-D CLS embedding을 사용하고, 같은 지원자·학과의 accepted works를 cell 단위 평균으로 표현합니다.
- **Prototype estimator:** 학과별 accepted-work cell 평균의 pairwise 차이.
- **Direct estimator:** 두 학과 모두 합격한 지원자의 within-person embedding 차이를 평균.
- **Sorting Gap:** 두 추정 벡터의 불일치. 최종 원고는 방향의 Sign Concordance (SC)와 크기의 (R=\|\widehat{\Delta}^{D}\|/\|\widehat{\Delta}^{P}\|)를 구분합니다.

```text
accepted work images → frozen DINOv2 embeddings → applicant × department cell means
                                             ├─ department prototype contrast
                                             └─ same-applicant direct contrast
                                                   → SC / magnitude ratio R
```

## 최종 원고에 기록된 표본과 결과

2026년 6월 원고 `Recovering Institutional Preference from Selection-Only Admissions: A Dual Estimator Framework` 기준: 4,128 image embedding rows, 1,024 dimensions, 2,022 applicant-department cells, 9 departments. Main Design 6 cluster의 14 viable pairs에서 SC=14/14; primary Design 4 core는 6 pairs, median R=2.72, hierarchical bootstrap median R=3.20, 95% CI [2.82, 3.65]. 원고는 selection-only 자료로 ability selection과 strategic portfolio differentiation을 서로 분리 식별할 수 없다고 제한합니다.

**버전 주의:** 원본 README에는 17 viable pairs라는 이전 수치가 남아 있으나 최종 `main.tex`는 Design 6에서 14 pairs와 Design 4 core 6 pairs를 기술합니다. 이 repo는 최종 원고 수치만 요약하고 이전 README 수치를 최종 결과와 혼합하지 않습니다.

## 코드 / 실행

`research/`에는 estimator, bootstrap 및 공통 유틸리티의 원본 코드 일부만 포함합니다. 원본 실행은 로컬 metadata, image/embedding cache, shared state 경로에 의존하며 이 저장소에는 저작권 있는 작품, applicant-level 자료, embedding cache를 넣지 않았습니다. 따라서 복제 실행은 이 저장소만으로 불가능하고, 개인정보/저작권 검토를 거친 데이터 접근과 경로 구성이 필요합니다. 새로운 분석이나 코드는 이번 정리에서 실행하지 않았습니다.

## 제한 및 공개 상태

결과는 selection-only 관측의 기술적·통계적 비교이며 대학 선호나 인과 메커니즘을 완전히 복원하지 않습니다. GitHub repo만으로 논문 게재·발표 상태를 증명하지 않습니다. 원본 repo에 MIT LICENSE가 있었으나 여기서는 해당 파일을 보존했습니다. 이는 제3자 작품 및 지원자 자료에 대한 재배포 권한을 부여하지 않습니다.

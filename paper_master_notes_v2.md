# 논문 마스터 노트 v2.1

> **v2.1 갱신 사항** (vs v2):
> - 차원 점수 산출 방식 전환: 개별 메트릭 직접 상관 → **percentile rank 정규화 후 차원 평균**
> - D1/D3/D4 r 값 전면 갱신, Total(D1+D3+D4) 유의 상관 신규 추가
> - D2: 전 모델 triangle_ratio=1.0 상수 → 변별력 없음, Total 제외
> - Contribution 2번 문구 수정 (기하 무상관 + 종합 유의 상관 명시)
> - 논문 집필 진행 현황 및 Figure 목록 갱신
> - 데이터 기반: 1,993 메쉬 / 16 ELO 모델 / 4 post-paper 모델 (총 20개)
>
> **v2 갱신 사항** (vs v1):
> - §1: 15-model → 16-model 기준으로 모든 r 값 / 분면 분포 / disconnect 재계산
> - Hi3DGen borderline 처리 메모 추가

---

## 1. 핵심 발견 (반드시 인용) — 16-model 기준

### 차원 점수 산출 방식 (v2.1)

각 메트릭을 16 ELO 모델 내 **percentile rank [0~1]** 로 정규화 후 차원별 평균:

| 규칙 | 적용 메트릭 |
|------|------------|
| Percentile rank (높을수록 좋음) | watertight, manifold_edge_ratio, triangle_ratio, uv_bbox_efficiency, uv_packed_area, pbr_channel_count, texture_resolution |
| Inverse percentile (낮을수록 좋음) | connected_components, non_triangle_ratio |
| Boolean 그대로 (Yes=1, No=0) | has_uv, has_texture |
| 제외 (정보용만) | face_count |

```
D1 = mean(watertight_pct, manifold_pct, inv_components_pct)
D2 = 0.5 상수 (전 모델 triangle_ratio=1.0) → Total 계산 제외
D3 = mean(has_uv, uv_bbox_pct, uv_packed_pct)
D4 = mean(has_texture, pbr_channel_pct, texture_res_pct)
Total = mean(D1, D3, D4)
```

### Spearman 상관계수 (n=16 ELO 모델, score-based; 양수 = "점수↑ → ELO↑")

- **D1 Geometry vs ELO: r = −0.022 (p = 0.935, 무관)**
- **D2 Topology vs ELO: 측정 불가 — 전 모델 삼각형 출력, 상수 (0.5)**
- **D3 UV/Texture vs ELO: r = +0.419 (p = 0.106)**
- **D4 PBR vs ELO: r = +0.415 (p = 0.110)**
- **Total (D1+D3+D4) vs ELO: r = +0.650 (p = 0.006, 유의미 ★)**
- **texture_resolution vs ELO (전체 16개): r = +0.578 (p = 0.019, 유의미 ★)**
- **texture_resolution vs ELO (텍스처 보유, n=9): r = +0.710 (p = 0.032, 유의미 ★)**

> **부호 주의**: `data/elo_correlation.csv`는 **rank-based** (rank 1 = 최고)라 부호가 반대로 표시됨. 동일 데이터.
> - rank-based 전체: r = −0.578 (p = 0.019)
> - rank-based 텍스처-보유: r = −0.710 (p = 0.032)

> **구 수치 (v2, 개별 메트릭 직접 상관)** — 참고용 보존:
> - D1: r = −0.091 (p = 0.737) / D3: r = +0.396 (p = 0.129) / D4: r = +0.338 (p = 0.201)

### 핵심 Finding 4가지

| # | Finding | 수치 | 논문 의미 |
|---|---------|------|----------|
| F1 | D1(기하)-ELO 무상관 | r=−0.022, p=0.935 | 인간 선호가 기하 품질 미반영 |
| F2 | texture_resolution 유의 | r=+0.578, p=0.019 | 텍스처 해상도만 ELO와 단독 유의 |
| F3 | Total 유의 ★ | r=+0.665, p=0.005 | UV/텍스처 없는 7개 모델(InstantMesh, Unique3D, Hi3DGen, MeshFormer, Real3D, TripoSR, IM-MA)이 D2+D3에서 모두 0.150 동점 → Spearman 구분 불가. D1 추가 시 기하 구조 차이로 동점 해소 → r: +0.556 → +0.665. 단 Δr=+0.109, Bootstrap 95% CI [-0.14, +0.40] → 통계적 유의성 입증 어려움. 논문 표현: "개선 경향이 있으나 소표본(n=16)으로 통계적 유의성 확인은 향후 과제" |
| F4 | D2 상수 | 전 모델 삼각형 | 현재 i2-3D 파이프라인 삼각형 표준화 확인 |

### 2×2 Matrix (16 모델, ELO median = 1218.5)

- **Q1 시각+production (n=2)**: InstantMesh, Unique3D
- **Q2 시각형 (n=6) ★ 최대 클러스터**: Strawberrry, Strawb3rry, TRELLIS, Zaohaowu3D, Hunyuan3D-2, Meshy-5
- **Q3 production만 (n=2)**: Hi3DGen, IM-MA
- **Q4 양쪽 미흡 (n=6)**: MeshFormer, SF3D, Real3D, SPAR3D, TripoSR, 3DTopia-XL

> **핵심 메시지**: ELO 상위 8개 중 6개가 watertight=0%, "인기 모델 ≠ production-fit".

### Hi3DGen — borderline case 처리 메모

- ELO = 1207, 16-model median = **1218.5** → **Q3 (ELO-Low / WT>0)** 분류
- 15-model 분석 시 median = 1207 → Q1로 분류되었음
- **본문 처리**: "Hi3DGen은 ELO 정중앙 부근(rank 9/16)에 위치한 borderline case로, watertight=70%로 production 적합성은 명확함" 정도로 다루기
- 분면 메시지의 견고성 보강: borderline을 빼더라도 Q2(6개) > Q1(2개) 패턴 유지

### 극단 사례

- **MeshFormer**: components 평균 ≈ **97,891개** / 중앙값 ≈ 96,814 / ELO 10위 — boundary 99.97%로 사실상 mesh가 아님
- **Disconnect total 상위 3** (|ELO_rank − D_rank| 합):
  - 3DTopia-XL: **Δ = 24** (ELO 16위, D3·D4 4위)
  - Strawberrry: **Δ = 18** (ELO 1위, D1 14위)
  - InstantMesh: **Δ = 18** (ELO 6위, D3·D4 13위)

---

## 2. 데이터 규모

- 20개 image-to-3D 모델 (16개 ELO 보유 + 4개 post-paper)
- **1,993개 메쉬** (load_failures = 0, integrity_check.csv로 검증 완료)
- 12개 자동 메트릭 (4차원: D1 Geometry, D2 Topology, D3 UV, D4 PBR)
- 71개 unit tests + 3개 도구 cross-check

---

## 3. Contribution 4가지

1. 학습 없이 메쉬 구조 결정론적 직접 측정 12개 메트릭 suite
2. **ELO와 메쉬 구조 메트릭 간 복합적 관계 정량 분석: 기하 품질(D1)은 ELO와 무상관(r=−0.022), 다차원 종합(Total)은 ELO와 유의한 상관(r=+0.650, p=0.006)**
3. 2×2 분면 분류, ELO 상위 8개 중 6개가 watertight=0%
4. 학습 기반 평가의 직교적 보완 (annotation/GPU 불필요)

---

## 4. 인용 (안전도 분류)

### Peer-reviewed — 안전

- **GPTEval3D** (Wu et al., CVPR 2024): GPT-4V pairwise, text-to-3D, closed-source 한계
- **HyperScore/MATE-3D** (Zhang et al., ICCV 2025): 1,280 mesh + 107k annotations, hypernetwork, text-to-3D

### arXiv preprint — 명시 인용

- **3D Arena** (Ebert, 2025): polygon count 1차원, "topology future work" 명시
- **3DGen-Bench** (Zhang et al., 2025a): "lack of robust 3D embedding, opt for 2D CLIP"
- **Hi3DEval** (Zhang et al., 2025b): "based on 2D renderings, inherently struggle to capture spatial continuity"
- **T3Bench** (He et al., 2023): multi-view metric

---

## 5. 핵심 인용 문장

- **Hi3DEval § 1**: "based on 2D renderings, inherently struggle to capture spatial continuity and structural complexity of 3D assets"
- **3DGen-Bench § 7**: "due to the lack of a robust 3D embedding model, we opt for 2D CLIP embedding as an alternative... fully leverage the naive 3D data for evaluation"
- **HyperScore § 3.3**: "geometry deformations (e.g., incomplete shape, floaters) directly affect 3D perception... geometry quality is most closely related to the overall quality"

---

## 6. Related Work 4-section 구조

- 2.1 인간 선호 기반: 3D Arena, T3Bench
- 2.2 LLM 기반: GPTEval3D
- 2.3 학습 기반: 3DGen-Bench, HyperScore, Hi3DEval
- 2.4 본 연구의 직교적 위치

---

## 7. Limitations (정직하게)

- 메트릭 industrial validation 부재
- 모델 수 제한 (n = 16 ELO 모델 + 4 post-paper)
- Image-to-3D 한정
- 인간 annotation 0건 → subjective alignment 직접 검증 불가

---

## 8. 주요 표현 톤

- "처음으로 정량화" 회피 → "체계적으로 정량 분석"
- "최초의 평가 도구" 회피 → "보완적 직교 접근"
- "우리가 아는 한 (To the best of our knowledge)" 활용
- D1-ELO 무상관: "기하 품질이 인간 선호에 반영되지 않음" (놀라운 발견으로 강조)
- Total 유의: "차원 합산의 필요성 근거" (단일 차원 평가의 한계 논거)
- D2 상수: "현재 i2-3D 파이프라인의 삼각형 표준화를 실증" (긍정적 발견으로 제시)

---

## 9. 논문 집필 현황

### ✅ 완료 섹션

- §1 서론 v5 (Contribution 2번 소폭 수정 필요 → 새 Contribution 문구로 교체)
- §2 관련 연구 v2
- §3 메쉬 구조 기반 자동 평가 프레임워크 v4
- §4 실험 설정 v8
- §5.1 모델별 종합 메트릭 분포 v2
- §5.2 ELO와 메쉬 메트릭 간 상관관계 v2

### ⬜ 미완료 섹션

- §5.3 시각·production 분면 분류 (Figure 1 기반)
- §5.4 극단 사례 분석 (MeshFormer / 3DTopia-XL / Strawberrry)
- §5.5 ELO 미보유 4개 모델 추세 (선택)
- §6 논의 및 한계
- §7 결론

### ⬜ 후처리

- 인용 번호화: [짧은제목] → [N] 형식
- 최종 docx 변환 (KCGS 템플릿)

---

## 10. Figure 현황

| Figure | 파일 | 상태 | 내용 |
|--------|------|------|------|
| Figure 1 | `outputs/fig1_matrix_v2.png` | ✅ | 2×2 scatter (ELO vs watertight), 16 모델 |
| Figure 2 | `outputs/fig2_correlation_v2.png` | ✅ | D1/D3/D4 + texture_res Spearman bar chart |
| Figure 3 | `outputs/fig3_heatmap_v3.png` | ✅ | D1~D4 + Total 점수 히트맵 (D2 annotation 포함) |

> Figure 2는 v2 기준 수치(개별 메트릭 직접 상관). 필요 시 v2.1 수치로 재생성 검토.

---

## 11. 데이터 갱신 시 재실행 순서 (참고)

1. `python scripts/download_all.py --only <model>`  (필요 시)
2. `python run_metrics.py --only <model>`
3. `python scripts/analyze.py`
4. `python scripts/make_fig3_v3.py`   ← v2.1 기준 Figure 3 + dimension_scores_v3.csv
5. `python scripts/update_v2.py`       ← Figure 2 재생성
6. `python scripts/make_paper_figures.py`  ← Figure 1
7. `python scripts/integrity_check.py`    ← sanity check

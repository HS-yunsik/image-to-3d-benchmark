# 논문 마스터 노트 v2

> **v2 갱신 사항** (vs v1):
> - §1 핵심 발견: 15-model → **16-model 기준**으로 모든 r 값 / 분면 분포 / disconnect 재계산
> - §1 Hi3DGen borderline 처리 메모 추가
> - §9 완료 항목 체크, 신규 펜딩 명시
> - 데이터 기반: 1,993 메쉬 / 16 ELO 모델 / 4 post-paper 모델 (총 20개)

---

## 1. 핵심 발견 (반드시 인용) — 16-model 기준

### Spearman 상관계수 (n=16 ELO 모델, score-based; 양수 = "메트릭↑ → ELO↑")
- **D1 Geometry vs ELO: r = −0.091 (p = 0.737, 무관)**
- **D3 UV/Texture vs ELO: r = +0.396 (p = 0.129)**
- **D4 PBR vs ELO: r = +0.338 (p = 0.201)**
- **texture_resolution vs ELO (전체 16개): r = +0.578 (p = 0.019, 유의미 ★)**
- **texture_resolution vs ELO (텍스처 보유, n=9): r = +0.710 (p = 0.032, 유의미 ★)**

> **부호 주의**: `data/elo_correlation.csv`는 **rank-based** (rank 1 = 최고)라 부호가 반대로 표시됨. 동일 데이터.
> - rank-based 전체: r = −0.578 (p = 0.019)
> - rank-based 텍스처-보유: r = −0.710 (p = 0.032)

### 2×2 Matrix (16 모델, ELO median = 1218.5)
- **Q1 시각+production (n=2)**: InstantMesh, Unique3D
- **Q2 시각형 (n=6) ★ 최대 클러스터**: Strawberrry, Strawb3rry, TRELLIS, **Zaohaowu3D**, Hunyuan3D-2, Meshy-5
- **Q3 production만 (n=2)**: Hi3DGen, IM-MA
- **Q4 양쪽 미흡 (n=6)**: MeshFormer, SF3D, Real3D, SPAR3D, TripoSR, 3DTopia-XL

> **핵심 메시지**: ELO 상위 8개 중 6개가 watertight=0%, "인기 모델 ≠ production-fit".

### Hi3DGen — borderline case 처리 메모
- ELO = 1207, 16-model median = **1218.5** → **Q3 (ELO-Low / WT>0)** 분류
- 15-model 분석 시 median = 1207 → Q1로 분류되었음
- **본문에서 명시**: "Hi3DGen은 ELO 정중앙 부근(rank 9/16)에 위치한 borderline case로, watertight=70%로 production 적합성은 명확함" 정도로 다루기
- 분면 메시지의 견고성 보강: borderline을 빼더라도 Q2(6개) > Q1(2개) 패턴 유지

### 극단 사례
- **MeshFormer**: components 평균 ≈ **97,891개** / 중앙값 ≈ 96,814 / ELO 10위 (15-model 기준 9위) — boundary 99.97%로 사실상 mesh가 아님
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

## 3. Contribution 4가지
1. 학습 없이 메쉬 구조 결정론적 직접 측정 12개 메트릭 suite
2. 20개 모델 cross-evaluation, ELO disconnect 정량 보고
3. 2×2 분면 분류, ELO 상위 8개 중 6개가 watertight=0%
4. 학습 기반 평가의 직교적 보완 (annotation/GPU 불필요)

## 4. 인용 (안전도 분류)

### Peer-reviewed — 안전
- **GPTEval3D** (Wu et al., CVPR 2024): GPT-4V pairwise, text-to-3D, closed-source 한계
- **HyperScore/MATE-3D** (Zhang et al., ICCV 2025): 1,280 mesh + 107k annotations, hypernetwork, text-to-3D

### arXiv preprint — 명시 인용
- **3D Arena** (Ebert, 2025): polygon count 1차원, "topology future work" 명시
- **3DGen-Bench** (Zhang et al., 2025a): "lack of robust 3D embedding, opt for 2D CLIP"
- **Hi3DEval** (Zhang et al., 2025b): "based on 2D renderings, inherently struggle to capture spatial continuity"
- **T3Bench** (He et al., 2023): multi-view metric

## 5. 핵심 인용 문장
- **Hi3DEval § 1**: "based on 2D renderings, inherently struggle to capture spatial continuity and structural complexity of 3D assets"
- **3DGen-Bench § 7**: "due to the lack of a robust 3D embedding model, we opt for 2D CLIP embedding as an alternative... fully leverage the naive 3D data for evaluation"
- **HyperScore § 3.3**: "geometry deformations (e.g., incomplete shape, floaters) directly affect 3D perception... geometry quality is most closely related to the overall quality"

## 6. Related Work 4-section 구조
- 2.1 인간 선호 기반: 3D Arena, T3Bench
- 2.2 LLM 기반: GPTEval3D
- 2.3 학습 기반: 3DGen-Bench, HyperScore, Hi3DEval
- 2.4 본 연구의 직교적 위치

## 7. Limitations (정직하게)
- 메트릭 industrial validation 부재
- 모델 수 제한 (n = 16 ELO 모델 + 4 post-paper)
- Image-to-3D 한정
- 인간 annotation 0건 → subjective alignment 직접 검증 불가

## 8. 주요 표현 톤
- "처음으로 정량화" 회피 → "체계적으로 정량 분석"
- "최초의 평가 도구" 회피 → "보완적 직교 접근"
- "우리가 아는 한 (To the best of our knowledge)" 활용

---

## 9. 펜딩 작업 (v2 시점)

### ✅ 완료
- Figure 1 v2 카드 그리드 → `outputs/fig1_matrix_v2.png`
- 1,893 vs 1,993 검증 → 모든 분석 1,993 기준 확인
- D1/D3/D4 r 값 16-model 재계산 → §1 갱신 완료
- Figure 2 v2 (16-model + p-values) → `outputs/fig2_correlation_v2.png`
- Figure 3 v2 (16 ELO 모델, ELO 정렬) → `outputs/fig3_heatmap_v2.png`
- 데이터 무결성 점검 → `data/integrity_check.csv` (모든 20개 OK)

### ⚠️ 신규 펜딩
- 본문 작성 시 §1 "Hi3DGen borderline" 처리 문구 작성 — borderline을 어떻게 다룰지 결정
- texture_resolution 양수 상관 (r=+0.71) 해석: "고해상도 texture 모델이 시각 품질 우위" → 본문에 1단락 분량으로 다룰지 결정
- D1·D3·D4 모두 p > 0.05 → 본문에서 "통계적 유의성보다 패턴 자체를 강조" 톤 정리
- comparison grid (`outputs/comparison_grid_*.png`) 3개 중 paper figure로 어느 것 사용할지 (castle 추천)

### 🔁 데이터 갱신 시 재실행 순서 (참고)
1. `python scripts/download_all.py --only <model>`  (필요 시)
2. `python run_metrics.py --only <model>`
3. `python scripts/analyze.py`
4. `python scripts/additional_analysis.py`
5. `python scripts/make_paper_figures.py` + `python scripts/update_v2.py`
6. `python scripts/integrity_check.py` (sanity check)

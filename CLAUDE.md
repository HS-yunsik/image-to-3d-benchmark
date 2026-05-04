# Image-to-3D Production-Fitness Evaluation Benchmark

이 문서는 Claude Code가 자동으로 읽는 프로젝트 컨텍스트 파일입니다. 작업 시작 전 반드시 전체를 읽고 진행하세요.

---

## 1. 프로젝트 한 줄 요약

**Image-to-3D 생성 모델의 출력 메쉬를 "프로덕션 적합성" 관점에서 자동으로 평가하는 다차원 메트릭 벤치마크.** 3D Arena의 인간 선호 ELO 순위와 차원별 disconnect를 정량 분석한다.

---

## 2. 데드라인 및 산출물

- **제출 학회**: 한국컴퓨터그래픽스학회 (국내)
- **마감**: 약 12일 남음 (오늘 기준)
- **분량**: 6~8 페이지, 한국어
- **현재 단계**: Phase 0 (검증) 완료, Stage 1 시작 준비

---

## 3. 연구 포지셔닝 (왜 새로운가)

### 가장 가까운 선행 연구

| 논문 | 한계 (우리 빈자리) |
|------|----------------|
| **3D Arena** (Ebert, 2025.6, arXiv 2506.18787) | 인간 voting 중심. 자동 메트릭은 폴리곤 count 1차원만. |
| **3DGen-Bench** (Zhang et al., 2025.3, arXiv 2503.21745) | 5개 차원 평가하지만 모두 **렌더링된 2D 이미지** 기반 (normal map + RGB). 메쉬 구조 직접 측정 안 함. |
| **Hi3DEval** (2025.8, arXiv 2508.05609) | 자기 논문에 명시: *"based on 2D renderings, these methods inherently struggle to capture the spatial continuity and structural complexity of 3D assets"* — 본인들도 한계 인정. |
| **MATE-3D / HyperScore** (2024.12) | CLIP 기반 학습 metric, 렌더링 이미지 평가. |
| **Mesh-RFT** (2025.5) | BER, Topology Score 사용하지만 자체 모델 fine-tuning용 내부 메트릭. Image-to-3D 벤치마크 아님. |

### 우리의 정확한 포지셔닝

> **"우리는 image-to-3D 출력 평가를 '렌더링 기반'에서 '메쉬 구조 직접 측정 기반'으로 전환하는 첫 시도이다."**

3D Arena가 인정한 한계:
- "topology assessment is future work" (Section 5)
- "Multi-criteria assessment can disentangle aesthetic appeal from technical utility" (Section 6)

→ **저자가 future work로 명시한 부분을 우리가 채운다**는 narrative.

### 동시대 작업 (Concurrent work) 처리

3D Arena는 **arXiv preprint 단독** (peer-review 안 됨). BibTeX `@misc` 형식. 우리는 concurrent/complementary work로 자연스럽게 포지셔닝. 직접 데이터셋 활용하므로 정중히 인용.

---

## 4. Contributions (확정)

논문 narrative는 다음 우선순위로:

1. **🥇 다차원 자동 평가 메트릭 Suite (Primary)**
   - Geometry/Topology/UV/PBR 4개 차원 × 약 11개 메트릭
   - 메쉬 구조 직접 측정 (렌더링 안 함)

2. **🥈 ELO와 자동 메트릭의 차원별 Disconnect 분석**
   - Spearman/Kendall correlation
   - 폴리곤 count 단일 차원 (3D Arena 한계) → 다차원 확장

3. **🥉 24+ 모델 × 카테고리별 약점 매핑**
   - 3D Arena 원논문 시점(2025.6) 이후 추가된 SOTA 모델 포함 (TRELLIS.2-4B, Hunyuan3D-2.1, Meshy-6 등)

4. **재현 가능한 자동 파이프라인 공개**
   - Blender / trimesh 기반, 오픈소스

---

## 5. ⭐ Phase 0 결과 (이미 완료, 검증됨)

5개 모델 × 5개 메쉬로 패턴 확정:

```
Model            AvgVerts   AvgFaces    WT%   UV%   Tex%
─────────────────────────────────────────────────────────
Hunyuan3D-2       25,711    40,000      0%   100%  100%
Hunyuan3D-2.1     26,165    40,000      0%   100%  100%
TRELLIS           22,214    31,475      0%   100%  100%
InstantMesh       71,214    142,326    20%     0%  100%
Hi3DGen          484,000    967,959    60%     0%    0%
```

### 검증된 발견 (논문 narrative의 핵심)

**3가지 모델 그룹 패턴 명확:**
- **시각 품질형** (Hunyuan, TRELLIS): 작은 메쉬, UV+Tex 완비, 그러나 0% watertight
- **반-production형** (InstantMesh): 중간 메쉬, Tex만 있음, 20% watertight
- **순수 geometry형** (Hi3DGen): 거대 메쉬, UV/Tex 없음, 60% watertight

**ELO-disconnect 시그널 확인:**
- Hunyuan3D-2 (ELO 7위) vs InstantMesh (8위): ELO 거의 같지만 production 특성 정반대
- Hi3DGen (ELO 11위)이 watertight 비율 가장 높음 — 시각이 약하지만 geometry는 우수

**기술적 특성:**
- 모든 메쉬가 100% triangle (marching cubes 계열)
- 모델별 face budget 거의 deterministic (Hunyuan: 정확히 40000)
- 파일 이름이 prompt 기반 (예: `A_cartoon_house_with_red_roof.glb`)

---

## 6. 작업 환경

- **OS**: Windows (PowerShell)
- **Python**: 3.11
- **Conda env**: `3darena` (활성화 필수)
- **작업 디렉토리**: `C:\Users\HP\3d-arena-eval`
- **이미 설치된 패키지**: `trimesh`, `huggingface_hub`, `pillow`, `numpy`, `pygltflib`

활성화:
```powershell
conda activate 3darena
cd C:\Users\HP\3d-arena-eval
```

기존 파일들:
- `phase0_validation.py` — 5개 메쉬 단일 모델 검증 (완료)
- `phase0_5_compare.py` — 5개 모델 1개씩 비교 (완료)
- `phase0_6_extended.py` — 5개 모델 × 5개 메쉬 (완료)
- `meshes/` 디렉토리 — 다운로드된 샘플 메쉬

---

## 7. 데이터셋

### 입력 (3D Arena 메쉬 출력)
- **HuggingFace**: `3d-arena/3d-arena` (이전 `dylanebert/3d-arena`는 404)
- **라이선스**: MIT
- **구조**:
  ```
  outputs/
    Hunyuan3D-2/{prompt_text}.glb (+ .png 썸네일)
    Hunyuan3D-2.1/...
    TRELLIS/...
    InstantMesh/...
    Hi3DGen/...
    ... (총 24+ 모델 폴더)
  ```
- **각 모델 폴더당**: 약 100개 메쉬 + 100개 썸네일 (총 ~200 files)
- **전체 크기**: ~26GB

### 입력 이미지 (iso3d)
- **HuggingFace**: `dylanebert/iso3d`
- **100개 이미지** (DreamShaper-XL 생성, 흰 배경, 격리 객체)

### ELO 점수 (3D Arena 리더보드)
- **출처**: 3D Arena 논문 Table 1 + HuggingFace Space에서 최신 가져오기
- **현재 시점 ELO를 가져와야 함** (논문은 2025.5.30 스냅샷)
- 19개 모델 ELO 공개됨

---

## 8. 본격 구현 단계 (Claude Code가 진행할 작업)

### Stage 1: 전체 데이터 수집 자동화

**목표**: 모든 mesh-based 모델의 모든 메쉬를 다운로드하고 정리.

**해야 할 일:**
- HuggingFace에서 24+ 모델 폴더 식별
- **Splat 모델 제외**: TRELLIS-3DGS, LGM, SAM-3D-Objects-3DGS는 `.ply`/`.splat` 형식 → 우리 메트릭 적용 불가 → "future work"로 처리
- mesh 형식 (`.glb`, `.obj`) 모델만 선별 (예상: 약 18-20개)
- 다운로드 + 실패 로그 저장
- 총 ~1500-2000개 메쉬 예상

**산출물**: `data/meshes/{model_name}/{prompt_name}.glb`, `data/download_log.csv`

### Stage 2: 메트릭 Suite 구현

**Phase 1 메트릭 (필수, 11개)**:

```python
# D1. Geometry (4개)
- watertight: bool         # trimesh.is_watertight
- manifold_edges_ratio: float  # non-manifold edge / total edge
- connected_components: int    # trimesh.split(only_watertight=False)
- vertex_count, face_count: int

# D2. Topology (2개)
- triangle_ratio: float    # 거의 100% 확정 (그 자체가 finding)
- non_triangle_ratio: float  # quad + ngon (Blender bpy 필요할 수 있음)

# D3. UV (2개)
- has_uv: bool
- uv_bbox_efficiency: float   # UV bbox area / 1.0

# D4. PBR (3개)
- has_texture: bool
- pbr_channel_count: int   # base/normal/roughness/metallic 중 몇 개
- texture_resolution: int  # 평균 해상도 (px)
```

**구현 우선순위:**
1. trimesh로 가능한 것 (D1, D3, D4) — 쉬움
2. bpy(Blender Python) 필요한 것 (D2 quad/ngon) — 중간 난이도, 옵션
3. pygltflib로 PBR 채널 분석

**Phase 2 (시간 되면 추가)**:
- self-intersection (pymeshlab)
- pole vertex ratio
- UV overlap ratio (shapely)

### Stage 3: 분석

**해야 할 일:**
- 모든 메쉬에 메트릭 적용 → 큰 CSV 만들기 (~1500 rows × 12 cols)
- 모델별 평균 점수 계산
- ELO 점수 가져와서 매칭
- **Spearman/Kendall correlation**: ELO ranking vs 각 차원별 ranking
- **카테고리 분류**: 파일명 기반 (예: "house", "vase", "chair", "person", "animal" 등)
- 카테고리 × 모델 매트릭스로 약점 식별

**산출물**: `analysis/results.csv`, 그래프, 표

### Stage 4: 시각화 + 논문

- 핵심 figure 3-4개:
  - 모델별 차원별 점수 heatmap
  - ELO vs 각 차원 scatter plot
  - 3가지 모델 그룹 시각화
- 논문 6-8페이지 구조 (아래 참조)

---

## 9. 함정 및 주의사항

### 형식 호환성
- **Splat 파일 (`.ply`, `.splat`)**: 우리 메트릭 적용 불가. 제외하고 "future work" 명시.
- 일부 모델은 매우 큼 (Hi3DGen: 메쉬당 30MB+) — 다운로드 시간 고려
- 깨진 파일 가능성 → try/except 필수

### trimesh의 자동 삼각화
- `trimesh.load()`는 메쉬를 자동으로 삼각화함
- **Quad ratio 측정하려면 Blender bpy 사용 필요**
- 다만 우리 가설상 모든 출력이 100% triangle이므로 bpy는 검증용으로만 사용

### UV/PBR 정보 추출
- `mesh.visual.uv`로 UV 좌표 접근
- PBR 채널은 GLB material 검사 필요 (`pygltflib` 사용 권장)
- 일부 모델은 텍스처 인덱스만 있고 실제 이미지 없을 수 있음

### HuggingFace 다운로드
- 인증 토큰 없이도 가능하지만 rate limit 있음
- 권장: `huggingface-cli login`으로 토큰 설정
- 환경 변수: `HF_TOKEN`

---

## 10. 평가 가설 (논문 narrative)

검증할 가설들:

| 가설 | 예상 결과 |
|------|---------|
| H1: ELO와 D1(Geometry) 상관 약함 | r ≈ 0.1-0.3 |
| H2: ELO와 D2(Topology) 거의 무상관 | r ≈ 0.0 (모두 triangle이라 변별력 없음) |
| H3: ELO와 D4(PBR) 중간~강한 상관 | r ≈ 0.3-0.5 (텍스처 효과) |
| H4: Hi3DGen이 D1에서 1위지만 ELO 11위 | **결정적 disconnect 사례** |
| H5: 카테고리별로 모델 강점이 다름 | 가구는 X 모델, 캐릭터는 Y 모델 |

---

## 11. 논문 구조 (한국어, 6-8p)

```
1. 서론 (0.7p)
   - 시각 품질 vs 프로덕션 적합성의 disconnect
   - 3D Arena 인용으로 motivation

2. 관련 연구 (0.8p)
   - 시각 기반 평가 (T3Bench, MATE-3D, 3DGen-Bench)
   - 인간 선호 평가 (3D Arena)
   - Hi3DEval의 한계 자기 인정 인용
   - 메쉬 품질 검사 도구 (MeshLab, Blender)

3. 제안 평가 프레임워크 (1.5p)
   - 4차원 메트릭 정의
   - 산업 워크플로우 매핑

4. 실험 설정 (0.5p)
   - 3D Arena 데이터셋
   - 평가 모델 목록
   - 카테고리 분류

5. 결과 및 분석 (2.5p)
   - 모델별 종합 점수
   - 차원별 ELO correlation
   - 3가지 모델 그룹 분석
   - 카테고리별 강약

6. 논의 (0.5p)
   - 시사점, 실무 가이드라인

7. 결론 (0.2p)
```

---

## 12. 인용 핵심 (논문 작성 시)

```bibtex
@misc{ebert20253darena,
  title={3D Arena: An Open Platform for Generative 3D Evaluation},
  author={Dylan Ebert},
  year={2025},
  eprint={2506.18787},
  archivePrefix={arXiv}
}

@article{zhang20253dgenbench,
  title={3DGen-Bench: Comprehensive Benchmark Suite for 3D Generative Models},
  author={Zhang et al.},
  year={2025},
  eprint={2503.21745}
}

@article{hi3deval,
  title={Hi3DEval: Advancing 3D Generation Evaluation with Hierarchical Validity},
  year={2025},
  eprint={2508.05609}
}
```

---

## 13. 작업 시작 시 사용자에게 확인할 것

Claude Code 첫 메시지로 다음을 확인하면 좋음:

1. Stage 1 (다운로드)부터 시작할지, 아니면 다른 우선순위가 있는지
2. 다운로드 범위: 모든 24개 모델 vs 메쉬만 (splat 제외, 약 18개)
3. 카테고리 분류 방식: keyword 기반 자동 vs 수동 검토
4. Blender bpy 도입 여부 (D2 quad ratio 측정용)

---

## 14. 우선 작업 추천

**Day 1 (다음 작업 세션)**:
1. Stage 1 다운로드 스크립트 작성 (`download_all.py`)
2. 모든 mesh 모델 폴더 다운로드 시작 (백그라운드)
3. Stage 2 메트릭 모듈 골격 작성 (`metrics/geometry.py`, `metrics/topology.py`, `metrics/uv.py`, `metrics/pbr.py`)

**Day 2-3**:
4. 메트릭 구현 완료
5. 단일 모델로 전체 파이프라인 테스트
6. 모든 모델에 메트릭 적용

**Day 4-5**:
7. ELO 스크래핑/매칭
8. 분석 + 그래프

**Day 6-12**:
9. 논문 작성

---

## 15. 도움 요청 시 참조

작업 중 막히면:
- 이 문서 다시 읽기
- 사용자가 claude.ai에서 컨텍스트를 가지고 있음 (이 문서 외에도 전체 결정 과정 알고 있음)
- 중요한 결정은 사용자에게 확인 후 진행

---

**마지막 업데이트**: Phase 0 완료 시점, 본격 구현 시작 직전.

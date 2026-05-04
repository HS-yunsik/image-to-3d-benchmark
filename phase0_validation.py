"""Phase 0: 3D Arena 데이터 다운로드 + 기본 메트릭 검증"""
import os
import trimesh
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "3d-arena/3d-arena"
MODEL = "Hunyuan3D-2"
LOCAL_DIR = "./meshes"

os.makedirs(LOCAL_DIR, exist_ok=True)

# Step 1: 폴더 안 파일 목록 확인
print(f"=== {MODEL} 폴더 파일 목록 확인 ===")
try:
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    model_files = [f for f in all_files if f.startswith(f"outputs/{MODEL}/")]
    print(f"총 {len(model_files)}개 파일 발견")
    print("처음 10개 샘플:")
    for f in model_files[:10]:
        print(f"  {f}")
except Exception as e:
    print(f"❌ 파일 목록 조회 실패: {e}")
    raise

# Step 2: 메쉬 파일만 필터링 (.glb, .obj)
mesh_files = [f for f in model_files if f.endswith(('.glb', '.obj'))]
print(f"\n메쉬 파일 (.glb/.obj): {len(mesh_files)}개")

# Step 3: 처음 5개만 다운로드
print(f"\n=== 처음 5개 다운로드 ===")
downloaded = []
for f in mesh_files[:5]:
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f,
            repo_type="dataset",
            local_dir=LOCAL_DIR
        )
        downloaded.append(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"✅ {f}  ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"❌ {f} — {e}")

# Step 4: 메트릭 적용
print(f"\n=== Phase 0 메트릭 결과 ===")
print(f"{'File':<25} {'Verts':>8} {'Faces':>8} {'Tri%':>6} {'WTtight':>8} {'HasUV':>6} {'HasTex':>7}")
print("-" * 75)

for path in downloaded:
    fname = os.path.basename(path)
    try:
        loaded = trimesh.load(path, force='mesh')
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        else:
            mesh = loaded

        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces)
        face_sides = mesh.faces.shape[1] if mesh.faces.ndim == 2 else 3
        tri_pct = 100.0 if face_sides == 3 else 0.0
        
        watertight = mesh.is_watertight
        
        has_uv = (
            hasattr(mesh, 'visual') 
            and hasattr(mesh.visual, 'uv') 
            and mesh.visual.uv is not None
        )
        
        has_texture = False
        try:
            if hasattr(mesh.visual, 'material'):
                mat = mesh.visual.material
                has_texture = (mat is not None and 
                              (hasattr(mat, 'baseColorTexture') or 
                               hasattr(mat, 'image')))
        except:
            pass

        print(f"{fname[:24]:<25} {n_verts:>8} {n_faces:>8} {tri_pct:>5.0f}% {str(watertight):>8} {str(has_uv):>6} {str(has_texture):>7}")

    except Exception as e:
        print(f"{fname[:24]:<25} ❌ {str(e)[:40]}")

print("\n=== Phase 0 완료 ===")
print("위 표가 정상 출력되면 파이프라인 검증 OK")
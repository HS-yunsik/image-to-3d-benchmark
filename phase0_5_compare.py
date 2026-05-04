"""Phase 0.5: 여러 모델의 첫 메쉬 1개씩만 비교"""
import os
import trimesh
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "3d-arena/3d-arena"
LOCAL_DIR = "./meshes"

# 비교할 모델 (mesh-based만 선택)
MODELS = ["Hunyuan3D-2", "Hunyuan3D-2.1", "TRELLIS", "InstantMesh", "Hi3DGen"]

print(f"{'Model':<18} {'File pattern':<30} {'Verts':>8} {'Faces':>8} {'WTtight':>8} {'HasUV':>6} {'HasTex':>7}")
print("-" * 90)

for model in MODELS:
    try:
        all_files = list_repo_files(REPO_ID, repo_type="dataset")
        model_files = [f for f in all_files if f.startswith(f"outputs/{model}/") and f.endswith(('.glb', '.obj'))]
        
        if not model_files:
            print(f"{model:<18} ❌ 메쉬 파일 없음")
            continue
        
        # 첫 파일만 다운로드
        first_file = sorted(model_files)[0]
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=first_file,
            repo_type="dataset",
            local_dir=LOCAL_DIR
        )
        
        # 로드 + 측정
        loaded = trimesh.load(path, force='mesh')
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        else:
            mesh = loaded
        
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces)
        watertight = mesh.is_watertight
        has_uv = (hasattr(mesh, 'visual') and 
                 hasattr(mesh.visual, 'uv') and 
                 mesh.visual.uv is not None)
        has_texture = False
        try:
            if hasattr(mesh.visual, 'material'):
                mat = mesh.visual.material
                has_texture = mat is not None and (
                    hasattr(mat, 'baseColorTexture') or hasattr(mat, 'image')
                )
        except:
            pass
        
        fname_short = os.path.basename(first_file)[:28]
        print(f"{model:<18} {fname_short:<30} {n_verts:>8} {n_faces:>8} {str(watertight):>8} {str(has_uv):>6} {str(has_texture):>7}")
        
    except Exception as e:
        print(f"{model:<18} ❌ {str(e)[:60]}")

print("\n=== 모델 간 비교 완료 ===")
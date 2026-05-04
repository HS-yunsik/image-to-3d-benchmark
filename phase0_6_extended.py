"""Phase 0.6: 5개 모델 × 5개 메쉬 평균 비교"""
import os
import trimesh
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "3d-arena/3d-arena"
LOCAL_DIR = "./meshes"
MODELS = ["Hunyuan3D-2", "Hunyuan3D-2.1", "TRELLIS", "InstantMesh", "Hi3DGen"]
N_SAMPLES = 5

results = {}

print("Loading and analyzing 5 meshes per model...")
for model in MODELS:
    try:
        all_files = list_repo_files(REPO_ID, repo_type="dataset")
        mesh_files = sorted([
            f for f in all_files 
            if f.startswith(f"outputs/{model}/") and f.endswith(('.glb', '.obj'))
        ])[:N_SAMPLES]
        
        verts_list, faces_list, wt_list, uv_list, tex_list = [], [], [], [], []
        
        for f in mesh_files:
            path = hf_hub_download(
                repo_id=REPO_ID, filename=f, 
                repo_type="dataset", local_dir=LOCAL_DIR
            )
            loaded = trimesh.load(path, force='mesh')
            mesh = (trimesh.util.concatenate(list(loaded.geometry.values())) 
                    if isinstance(loaded, trimesh.Scene) else loaded)
            
            verts_list.append(len(mesh.vertices))
            faces_list.append(len(mesh.faces))
            wt_list.append(mesh.is_watertight)
            uv_list.append(
                hasattr(mesh, 'visual') and 
                hasattr(mesh.visual, 'uv') and 
                mesh.visual.uv is not None
            )
            has_tex = False
            try:
                if hasattr(mesh.visual, 'material'):
                    mat = mesh.visual.material
                    has_tex = mat is not None and (
                        hasattr(mat, 'baseColorTexture') or hasattr(mat, 'image')
                    )
            except:
                pass
            tex_list.append(has_tex)
        
        results[model] = {
            'avg_verts': sum(verts_list) // len(verts_list),
            'avg_faces': sum(faces_list) // len(faces_list),
            'wt_rate': sum(wt_list) / len(wt_list),
            'uv_rate': sum(uv_list) / len(uv_list),
            'tex_rate': sum(tex_list) / len(tex_list),
            'n': len(verts_list)
        }
        print(f"  ✅ {model}: {len(verts_list)} meshes processed")
    except Exception as e:
        print(f"  ❌ {model}: {e}")

print(f"\n{'Model':<18} {'AvgVerts':>10} {'AvgFaces':>10} {'WT%':>6} {'UV%':>6} {'Tex%':>6}")
print("-" * 70)
for model, r in results.items():
    print(f"{model:<18} {r['avg_verts']:>10} {r['avg_faces']:>10} "
          f"{r['wt_rate']*100:>5.0f}% {r['uv_rate']*100:>5.0f}% {r['tex_rate']*100:>5.0f}%")
"""Profile compute_all() RSS / wall-clock on small + LARGE meshes.

Targets (from oldest to most demanding):
  - 5 Phase-0 meshes (~tens of K verts)
  - 1 MeshFormer mesh (suspected ~100K connected components)
  - 1 Hi3DGen mesh (1.3M faces, half a million verts -> file-size proxy)

For each: snapshot RSS before, run compute_all, snapshot after, log delta
and whether RSS dropped back close to the pre-call value (good cleanup).

Output: logs/memory_profile.txt
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil

from metrics import compute_all

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "memory_profile.txt"
LOG.parent.mkdir(exist_ok=True)


def find_largest(model: str) -> Path | None:
    folder = ROOT / "meshes" / "outputs" / model
    if not folder.exists():
        return None
    candidates = list(folder.glob("*.glb")) + list(folder.glob("*.obj"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


PHASE0_REL = [
    ("Hunyuan3D-2",   "Hunyuan3D-2/A_cartoon_house_with_red_roof.glb"),
    ("Hunyuan3D-2.1", "Hunyuan3D-2.1/A_cartoon_house_with_red_roof.glb"),
    ("TRELLIS",       "TRELLIS/A_cartoon_house_with_red_roof.glb"),
    ("InstantMesh",   "InstantMesh/A_cartoon_house_with_red_roof.glb"),
    ("Hi3DGen",       "Hi3DGen/A_cartoon_house_with_red_roof.glb"),
]


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def run_one(name: str, path: Path, lines: list[str]) -> None:
    if not path.exists():
        lines.append(f"  {name:<26} (file missing: {path})")
        return

    size_mb = path.stat().st_size / (1024 ** 2)
    gc.collect()
    rss_before = rss_mb()
    t0 = time.time()
    row = compute_all(path, model=name)
    elapsed = time.time() - t0
    rss_during = rss_mb()
    gc.collect()
    rss_after = rss_mb()

    leak = rss_after - rss_before
    lines.append(
        f"  {name:<26} size={size_mb:>7.2f}MB  V={row['vertex_count'] or 0:>7}  "
        f"F={row['face_count'] or 0:>8}  CC={row['connected_components']:>6}  "
        f"t={elapsed:>5.2f}s  "
        f"rss before/peak/after = {rss_before:>5.0f}/{rss_during:>5.0f}/{rss_after:>5.0f}MB  "
        f"leak={leak:+.0f}MB"
    )


def main() -> int:
    lines: list[str] = ["# memory_profile  (compute_all)"]
    lines.append(f"baseline rss = {rss_mb():.0f} MB\n")

    lines.append("Phase-0 small meshes:")
    for name, rel in PHASE0_REL:
        run_one(name, ROOT / "meshes" / "outputs" / rel, lines)

    lines.append("\nLargest mesh per heavy model:")
    for model in ("MeshFormer", "Hi3DGen", "TripoSG", "Meshy-6"):
        biggest = find_largest(model)
        if biggest is None:
            lines.append(f"  {model:<26} (not downloaded yet)")
            continue
        run_one(f"{model}/largest", biggest, lines)

    lines.append(f"\nfinal rss = {rss_mb():.0f} MB")

    out = "\n".join(lines)
    print(out)
    LOG.write_text(out + "\n", encoding="utf-8")
    print(f"\nLog: {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

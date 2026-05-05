"""Step 1 of bottom-up categorization rebuild.

Dump every prompt that exists in the 3D Arena evaluation set, one per line,
to logs/all_prompts.txt. Source priority:
  1. data/manifest.csv prompt_name column  (deduped across models)
  2. dylanebert/iso3d HF dataset filenames  (network fallback)
  3. one downloaded model's .glb/.obj file stems  (last resort)

The point is to ground the keyword dictionary in the *actual* prompt list
rather than guessing categories top-down.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "all_prompts.txt"
LOG.parent.mkdir(exist_ok=True)


def from_manifest() -> list[str]:
    mani = ROOT / "data" / "manifest.csv"
    if not mani.exists():
        return []
    seen: dict[str, None] = {}  # ordered set
    with mani.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            name = r.get("prompt_name") or Path(r.get("filename", "")).stem
            if name and name not in seen:
                seen[name] = None
    return list(seen)


def from_iso3d_repo() -> list[str]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        return []
    try:
        files = list_repo_files("dylanebert/iso3d", repo_type="dataset")
    except Exception as e:
        print(f"[dump_iso3d] HF list failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return []
    prompts = []
    for f in files:
        # iso3d likely has images named after prompts (e.g. images/<prompt>.png)
        p = Path(f).stem
        if p and p not in ("README", "iso3d") and not p.startswith("."):
            prompts.append(p)
    # dedupe preserving order
    seen: dict[str, None] = {}
    for p in prompts:
        seen.setdefault(p, None)
    return list(seen)


def from_disk() -> list[str]:
    meshes = ROOT / "meshes" / "outputs"
    if not meshes.exists():
        return []
    for sub in sorted(meshes.iterdir()):
        if not sub.is_dir():
            continue
        files = sorted(list(sub.glob("*.glb")) + list(sub.glob("*.obj")))
        if files:
            return [f.stem for f in files]
    return []


def main() -> int:
    sources = [
        ("manifest.csv", from_manifest),
        ("iso3d HF",     from_iso3d_repo),
        ("disk scan",    from_disk),
    ]
    prompts: list[str] = []
    used: str = ""
    for name, fn in sources:
        prompts = fn()
        if prompts:
            used = name
            break

    if not prompts:
        print("ERROR: no prompts found from any source", file=sys.stderr)
        return 1

    LOG.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"Source used: {used}")
    print(f"Unique prompts: {len(prompts)}")
    print(f"Wrote: {LOG}")
    print()
    print("First 30:")
    for p in prompts[:30]:
        print(f"  {p}")
    if len(prompts) > 30:
        print(f"  ... ({len(prompts) - 30} more)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

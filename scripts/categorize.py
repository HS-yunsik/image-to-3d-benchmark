"""Keyword-based prompt categorization for 3D Arena outputs.

Reads either data/manifest.csv (one prompt per row) or, if absent, falls
back to listing meshes/outputs/<any-model>/*.{glb,obj}. Strips the file
extension and underscore-tokenizes the name to derive the original prompt
text, then assigns one of 8 categories by keyword match.

Output: data/file_categories.csv
    columns: prompt_name, category, matched_keywords, ambiguous

Suspicious entries (no keyword hit, multi-category hit, very long prompts)
are flagged with ambiguous=True for manual review.

Run:
    conda activate 3darena
    python scripts/categorize.py
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.csv"
MESHES = ROOT / "meshes" / "outputs"
OUT = ROOT / "data" / "file_categories.csv"

CATEGORIES: dict[str, list[str]] = {
    "furniture": [
        "chair", "table", "sofa", "bed", "desk", "shelf", "cabinet",
        "stool", "bench", "couch", "armchair", "dresser", "wardrobe",
    ],
    "architecture": [
        "house", "castle", "building", "tower", "church", "temple",
        "room", "cottage", "mansion", "skyscraper", "bridge", "fortress",
        "cabin", "barn",
    ],
    "vehicle": [
        "car", "plane", "ship", "bike", "truck", "boat", "train",
        "motorcycle", "tank", "helicopter", "rocket", "submarine",
        "airplane", "vehicle", "bus", "scooter",
    ],
    "character": [
        "person", "man", "woman", "child", "robot", "alien", "monster",
        "knight", "warrior", "soldier", "ninja", "samurai", "wizard",
        "mage", "witch", "boy", "girl", "human", "figure",
    ],
    "animal": [
        "dog", "cat", "horse", "bird", "fish", "dragon", "bear",
        "tiger", "deer", "rabbit", "wolf", "lion", "elephant",
        "snake", "frog", "owl", "eagle", "shark", "crab", "turtle",
        "panda", "fox", "monkey", "pig", "cow", "sheep", "duck",
    ],
    "organic": [
        "tree", "flower", "plant", "fruit", "food", "mushroom", "leaf",
        "rose", "cactus", "bush", "vegetable", "apple", "pumpkin",
        "carrot", "berry", "burger", "pizza", "cake", "bread",
    ],
    "object": [
        "vase", "cup", "bottle", "lamp", "tool", "weapon", "instrument",
        "sword", "axe", "hammer", "gun", "shield", "guitar", "violin",
        "drum", "book", "phone", "clock", "key", "ring", "crown",
        "helmet", "mask", "shoe", "boot", "hat", "bag", "box",
        "pot", "bowl", "plate", "candle", "lantern", "statue",
    ],
}

# Build reverse lookup; resolve via order in CATEGORIES dict (priority by order)
KEYWORD_TO_CATS: dict[str, list[str]] = defaultdict(list)
for cat, kws in CATEGORIES.items():
    for kw in kws:
        KEYWORD_TO_CATS[kw].append(cat)


_TOK_RE = re.compile(r"[a-z]+")


def tokenize(name: str) -> list[str]:
    """Lowercase, strip extension, split on non-letters."""
    base = Path(name).stem.lower().replace("_", " ")
    return _TOK_RE.findall(base)


def classify(name: str) -> dict:
    tokens = tokenize(name)
    cats_hit = []
    matched = []
    for tok in tokens:
        if tok in KEYWORD_TO_CATS:
            for c in KEYWORD_TO_CATS[tok]:
                cats_hit.append(c)
            matched.append(tok)

    if not cats_hit:
        return {"category": "other", "matched": [], "ambiguous": True}

    # Take the highest-priority category among hits (priority = first in CATEGORIES)
    cat_priority = list(CATEGORIES.keys())
    chosen = min(set(cats_hit), key=lambda c: cat_priority.index(c))

    # Ambiguous if more than one *distinct* category matched
    distinct = set(cats_hit)
    ambiguous = len(distinct) > 1

    return {
        "category": chosen,
        "matched": matched,
        "ambiguous": ambiguous,
        "all_cats": sorted(distinct),
    }


def collect_prompts() -> list[str]:
    """Return a deduplicated list of prompt names (filenames without ext)."""
    if MANIFEST.exists():
        seen = set()
        out: list[str] = []
        with MANIFEST.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                p = r.get("prompt_name") or Path(r.get("filename", "")).stem
                if p and p not in seen:
                    seen.add(p)
                    out.append(p)
        if out:
            return out

    # Fallback: scan one model's directory
    for m_dir in sorted(MESHES.iterdir() if MESHES.exists() else []):
        if not m_dir.is_dir():
            continue
        files = sorted(list(m_dir.glob("*.glb")) + list(m_dir.glob("*.obj")))
        if files:
            return [f.stem for f in files]
    return []


def main() -> int:
    prompts = collect_prompts()
    if not prompts:
        print("ERROR: no prompts found (manifest empty and no meshes downloaded yet).",
              file=sys.stderr)
        return 1

    print(f"Classifying {len(prompts)} prompts ...\n", flush=True)

    rows = []
    counts = Counter()
    suspicious: list[tuple[str, str]] = []  # (prompt, reason)

    for p in prompts:
        result = classify(p)
        counts[result["category"]] += 1
        rows.append({
            "prompt_name": p,
            "category": result["category"],
            "matched_keywords": "|".join(result["matched"]),
            "all_categories_hit": "|".join(result.get("all_cats", [])),
            "ambiguous": result["ambiguous"],
        })
        # Flag suspicious entries
        if result["category"] == "other":
            suspicious.append((p, "no_keyword_hit"))
        elif result["ambiguous"]:
            suspicious.append((p, f"multi_cat: {result.get('all_cats')}"))
        elif len(p) > 80:
            suspicious.append((p, "very_long_prompt"))

    # Write CSV
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Distribution
    print("Category distribution:")
    print("-" * 40)
    for cat in list(CATEGORIES.keys()) + ["other"]:
        n = counts[cat]
        bar = "#" * min(n, 60)
        print(f"  {cat:<14} {n:>4}  {bar}")

    # Suspicious entries
    print(f"\nSuspicious entries ({len(suspicious)}):")
    print("-" * 40)
    for p, reason in suspicious[:30]:
        flag = "WARN" if reason != "no_keyword_hit" else "MISS"
        print(f"  [{flag}] {p[:70]:<70}  -- {reason}")
    if len(suspicious) > 30:
        print(f"  ... and {len(suspicious) - 30} more")

    print(f"\nWrote: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

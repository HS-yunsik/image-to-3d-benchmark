"""Keyword-based prompt categorization v2 for 3D Arena evaluation.

v2 changes over v1:
  - 9 categories: adds 'abstract' (artificial_intelligence, feeling, insight,
    life, metal, molecule, bond)
  - Substring matching with \\b word boundaries, longest keyword matched first
    to avoid 'car' matching 'cardboard', 'cat' matching 'castle', etc.
  - Priority: character > animal > vehicle > architecture > furniture >
    object > organic > abstract > other
  - 'truncated' column: True when prompt_name is a strict prefix of another
    prompt in the same list (filename truncated at ~50 chars)
  - Source: logs/all_prompts.txt, falling back to manifest.csv
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALL_PROMPTS_TXT = ROOT / "logs" / "all_prompts.txt"
MANIFEST = ROOT / "data" / "manifest.csv"
OUT = ROOT / "data" / "file_categories.csv"

PRIORITY: list[str] = [
    "character", "animal", "vehicle", "architecture",
    "furniture", "object", "organic", "abstract",
]

CATEGORIES: dict[str, list[str]] = {
    "character": [
        "person", "man", "woman", "child", "boy", "girl", "robot", "alien",
        "monster", "knight", "warrior", "soldier", "wizard", "cyborg",
        "anubis", "pharaoh",
        "handpalm", "hand", "palm",
    ],
    "animal": [
        "dog", "cat", "horse", "bird", "fish", "dragon", "bear", "tiger",
        "deer", "lion", "wolf", "fox", "rabbit", "mouse", "elephant", "monkey",
        "giraffe", "penguin", "owl", "butterfly", "snake", "frog", "turtle",
        "octopus", "ostrich", "pig", "capybara", "beetle", "dinosaur", "koi",
    ],
    "vehicle": [
        "airplane", "helicopter", "motorcycle", "motorcyle", "skateboard",
        "spaceship", "submarine",
        "car", "plane", "ship", "bike", "truck", "boat", "train", "rocket",
    ],
    "architecture": [
        # multi-word first (matched before single-word 'tree', 'pearl')
        "christmas tree", "oriental pearl", "store front", "storefront",
        "house", "castle", "building", "tower", "church", "temple", "bridge",
        "monument", "gate", "pyramid",
    ],
    "furniture": [
        # longer multi-word / compound first
        "armchair", "bookshelf", "wardrobe", "cabinet",
        "chair", "table", "sofa", "bed", "desk", "shelf", "stool",
        "bench", "drawer", "couch", "post",
    ],
    "object": [
        # multi-word / compound first
        "headphones", "sunglasses", "ice cream", "coffee maker", "trash bin",
        "t-shirt", "sweatshirt", "sneakers", "teapot", "margarita", "cocktail",
        "painting", "pitcher", "present", "pedestal", "container", "helmet",
        "baseball",
        "vase", "cup", "bowl", "bottle", "lamp", "weapon", "instrument",
        "chest", "ball", "ladder", "basket", "book", "hammer", "sword",
        "shield", "bag", "hat", "watch", "phone", "camera", "key", "statue",
        "harp", "sphere", "box", "beer", "milk", "shoe", "heart",
    ],
    "organic": [
        "green pepper",  # multi-word before 'pepper'
        "mushroom", "vegetable", "avocado", "banana", "pumpkin", "mountain",
        "orange", "pepper", "cactus", "flower", "plant", "fruit", "food",
        "leaf", "bush", "rose", "apple", "tree",
    ],
    "abstract": [
        "artificial intelligence",  # multi-word before single tokens
        "feeling", "insight", "metal", "molecule", "bond", "life",
    ],
}


def _build_matchers() -> list[tuple[str, str, re.Pattern]]:
    """Return (keyword, category, pattern) sorted by keyword length desc."""
    seen: set[str] = set()
    entries: list[tuple[str, str]] = []
    for cat in PRIORITY:
        for kw in CATEGORIES[cat]:
            if kw not in seen:
                seen.add(kw)
                entries.append((kw, cat))
    entries.sort(key=lambda x: -len(x[0]))
    return [
        (kw, cat, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE))
        for kw, cat in entries
    ]


_MATCHERS = _build_matchers()


def classify(name: str) -> dict:
    norm = name.lower().replace("_", " ")
    cat_hits: dict[str, list[str]] = {}
    for kw, cat, pat in _MATCHERS:
        if pat.search(norm):
            cat_hits.setdefault(cat, []).append(kw)

    if not cat_hits:
        return {"category": "other", "matched_keywords": "",
                "all_categories": "", "ambiguous": False}

    best = min(cat_hits, key=lambda c: PRIORITY.index(c))
    return {
        "category": best,
        "matched_keywords": "|".join(cat_hits[best]),
        "all_categories": "|".join(
            sorted(cat_hits, key=lambda c: PRIORITY.index(c))
        ),
        "ambiguous": len(cat_hits) > 1,
    }


def collect_prompts() -> list[str]:
    if ALL_PROMPTS_TXT.exists():
        lines = ALL_PROMPTS_TXT.read_text(encoding="utf-8").splitlines()
        raw = [ln.strip() for ln in lines if ln.strip()]
        seen: dict[str, None] = {}
        for p in raw:
            seen.setdefault(p, None)
        if seen:
            return list(seen)

    if MANIFEST.exists():
        seen2: dict[str, None] = {}
        with MANIFEST.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                p = r.get("prompt_name") or Path(r.get("filename", "")).stem
                if p:
                    seen2.setdefault(p, None)
        return list(seen2)

    return []


def detect_truncated(prompts: list[str]) -> set[str]:
    """Prompts that are a filename-truncated prefix of a longer prompt.

    Requires len >= 40 to exclude short prompts that happen to be substrings
    of longer ones (e.g. 'a_cat' vs 'a_cat_statue').
    """
    truncated: set[str] = set()
    for p in prompts:
        if len(p) >= 40 and any(q.startswith(p) and len(q) > len(p) for q in prompts):
            truncated.add(p)
    return truncated


def main() -> int:
    prompts = collect_prompts()
    if not prompts:
        print("ERROR: no prompts found (run dump_iso3d_prompts.py first).",
              file=sys.stderr)
        return 1

    print(f"Classifying {len(prompts)} prompts ...\n", flush=True)

    truncated_set = detect_truncated(prompts)
    rows: list[dict] = []
    counts: Counter = Counter()

    for p in prompts:
        r = classify(p)
        counts[r["category"]] += 1
        rows.append({
            "prompt_name": p,
            "category": r["category"],
            "matched_keywords": r["matched_keywords"],
            "all_categories": r["all_categories"],
            "ambiguous": r["ambiguous"],
            "truncated": p in truncated_set,
        })

    fieldnames = ["prompt_name", "category", "matched_keywords",
                  "all_categories", "ambiguous", "truncated"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(prompts)
    print("Category distribution (all prompts):")
    print("-" * 54)
    for cat in PRIORITY + ["other"]:
        n = counts[cat]
        bar = "#" * min(n, 40)
        print(f"  {cat:<16} {n:>4}  ({n/total*100:>5.1f}%)  {bar}")

    non_trunc = [r for r in rows if not r["truncated"]]
    nt_counts: Counter = Counter(r["category"] for r in non_trunc)
    n_nt = len(non_trunc)
    other_pct = nt_counts["other"] / max(n_nt, 1) * 100

    print(f"\nNon-truncated prompts ({n_nt}):")
    print("-" * 54)
    for cat in PRIORITY + ["other"]:
        n = nt_counts[cat]
        print(f"  {cat:<16} {n:>4}  ({n/n_nt*100:>5.1f}%)")

    trunc_rows = [r for r in rows if r["truncated"]]
    print(f"\nTruncated entries ({len(trunc_rows)}):")
    for r in trunc_rows:
        print(f"  [{r['category']:<13}] {r['prompt_name'][:65]}")

    other_rows = [r for r in rows if r["category"] == "other"]
    print(f"\nOther / no keyword hit ({len(other_rows)}):")
    for r in other_rows:
        trunc_flag = " [truncated]" if r["truncated"] else ""
        print(f"  {r['prompt_name']}{trunc_flag}")

    amb_rows = [r for r in rows if r["ambiguous"]]
    print(f"\nAmbiguous multi-category hits ({len(amb_rows)}):")
    for r in amb_rows[:25]:
        print(f"  [{r['category']:<13}] {r['prompt_name'][:55]}"
              f"  -- {r['all_categories']}")
    if len(amb_rows) > 25:
        print(f"  ... and {len(amb_rows) - 25} more")

    print(f"\nTarget: other% (non-truncated) < 20%  -->  actual: {other_pct:.1f}%")
    print(f"Wrote: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

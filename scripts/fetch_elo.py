"""Fetch the 3D Arena ELO leaderboard.

Strategy:
  1. Try scraping the HF Space (https://huggingface.co/spaces/3d-arena/3d-arena)
     -- the ranking is rendered there as a Gradio table. We attempt several
     known JSON endpoints exposed by Gradio Spaces.
  2. Fall back to the paper's Table 1 ELO snapshot (2025-05-30) -- BUT THE
     VALUES MUST BE FILLED IN MANUALLY FROM THE PAPER PDF. This script
     emits None for ELO when the live scrape fails so analysis can detect
     the gap and prompt the user to populate the values.
  3. Models that are post-paper get the source tag "post-paper" so analysis
     can flag them.

Output: data/elo_scores.csv with columns (model, elo, rank, n_votes, source).

NOTE on the fallback table: the dictionary below contains the 19 model NAMES
that appear in 3D Arena Table 1 but ELO/RANK = None. Fill these from the
paper PDF or a trusted secondary source before running correlation analysis.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
ELO_CSV = DATA_DIR / "elo_scores.csv"

SPACE_HOSTS = (
    "https://3d-arena-3d-arena.hf.space",
    "https://huggingface.co/spaces/3d-arena/3d-arena",
)

# ---------------------------------------------------------------------------
# Paper Table 1 (2025-05-30, arXiv 2506.18787) -- 19 model names.
# ELO and RANK are intentionally None until verified against the paper PDF.
# DO NOT make up values. If you cannot verify, leave None and the analysis
# step will report which entries are missing.
# ---------------------------------------------------------------------------
PAPER_TABLE1: dict[str, dict] = {
    "TRELLIS":       {"rank": None, "elo": None, "n_votes": None},
    "TripoSG":       {"rank": None, "elo": None, "n_votes": None},
    "Strawb3rry":    {"rank": None, "elo": None, "n_votes": None},
    "Strawberrry":   {"rank": None, "elo": None, "n_votes": None},
    "Unique3D":      {"rank": None, "elo": None, "n_votes": None},
    "Meshy-5":       {"rank": None, "elo": None, "n_votes": None},
    "Hunyuan3D-2":   {"rank": None, "elo": None, "n_votes": None},
    "InstantMesh":   {"rank": None, "elo": None, "n_votes": None},
    "MeshFormer":    {"rank": None, "elo": None, "n_votes": None},
    "SPAR3D":        {"rank": None, "elo": None, "n_votes": None},
    "Hi3DGen":       {"rank": None, "elo": None, "n_votes": None},
    "TRELLIS-3DGS":  {"rank": None, "elo": None, "n_votes": None},
    "SF3D":          {"rank": None, "elo": None, "n_votes": None},
    "Real3D":        {"rank": None, "elo": None, "n_votes": None},
    "404_GEN":       {"rank": None, "elo": None, "n_votes": None},
    "LGM":           {"rank": None, "elo": None, "n_votes": None},
    "TripoSR":       {"rank": None, "elo": None, "n_votes": None},
    "3DTopia-XL":    {"rank": None, "elo": None, "n_votes": None},
    "IM-MA":         {"rank": None, "elo": None, "n_votes": None},
}

# Models known to exist in the dataset but added AFTER 2025-05-30 paper snapshot.
POST_PAPER_MODELS = {
    "Hunyuan3D-2.1", "TRELLIS.2-4B", "Meshy-6", "Zaohaowu3D",
    "PicGen3D", "SAM-3D-Objects-3DGS",
}


def _http_get(url: str, timeout: int = 15) -> bytes | None:
    """GET helper: returns body bytes or None on error."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "3d-arena-eval/0.1 (academic benchmark)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"[fetch_elo]  HTTP fail for {url}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None


def try_scrape_space() -> list[dict] | None:
    """Try to scrape the live leaderboard from the HF Space.

    Tries several common Gradio file-serve URLs that 3D Arena might expose.
    Returns None if nothing parseable found; caller falls back to paper.
    """
    candidates = [
        f"{SPACE_HOSTS[0]}/file=leaderboard.csv",
        f"{SPACE_HOSTS[0]}/file=ranking.csv",
        f"{SPACE_HOSTS[0]}/file=elo.csv",
        f"{SPACE_HOSTS[0]}/file=data/leaderboard.csv",
        f"{SPACE_HOSTS[0]}/api/leaderboard",
    ]
    for url in candidates:
        body = _http_get(url)
        if not body:
            continue
        text = body.decode("utf-8", errors="replace")
        if "model" in text.lower() and ("elo" in text.lower() or "rating" in text.lower()):
            print(f"[fetch_elo]  scraped from {url}", flush=True)
            rows: list[dict] = []
            try:
                reader = csv.DictReader(text.splitlines())
                for r in reader:
                    lower = {k.strip().lower(): v.strip()
                             for k, v in r.items() if k}
                    model = lower.get("model") or lower.get("name")
                    elo = lower.get("elo") or lower.get("rating")
                    rank = lower.get("rank")
                    n_votes = (lower.get("n_votes") or lower.get("votes")
                               or lower.get("battles"))
                    if not model or not elo:
                        continue
                    try:
                        rows.append({
                            "model": model,
                            "rank": int(rank) if rank else None,
                            "elo": float(elo),
                            "n_votes": (int(n_votes)
                                        if n_votes and n_votes.isdigit()
                                        else None),
                        })
                    except (ValueError, TypeError):
                        continue
                if rows:
                    return rows
            except csv.Error:
                continue
    return None


def build_from_paper_table() -> list[dict]:
    """Emit one row per paper model, with ELO/rank=None for manual fill."""
    return [
        {"model": m, **info, "source": "paper_table1_TODO_fill_manually"}
        for m, info in PAPER_TABLE1.items()
    ]


def main() -> None:
    print("[fetch_elo] Trying HF Space scrape ...", flush=True)
    scraped = try_scrape_space()

    rows: list[dict]
    if scraped:
        print(f"[fetch_elo] Got {len(scraped)} rows from Space (live).",
              flush=True)
        for r in scraped:
            r.setdefault("source", "hf_space_live")
        rows = scraped
    else:
        print("[fetch_elo] Scrape failed -- emitting paper-table skeleton "
              "with ELO=None. FILL VALUES MANUALLY from the paper PDF "
              "(Table 1, 2025-05-30 snapshot) before correlation analysis.",
              flush=True)
        rows = build_from_paper_table()

    seen = {r["model"] for r in rows}
    for m in POST_PAPER_MODELS:
        if m not in seen:
            rows.append({
                "model": m, "rank": None, "elo": None, "n_votes": None,
                "source": "post-paper",
            })

    fieldnames = ["model", "rank", "elo", "n_votes", "source"]
    with ELO_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda r: (r["rank"] is None, r["rank"] or 999))
        )

    print(f"[fetch_elo] Wrote {ELO_CSV} ({len(rows)} rows)", flush=True)
    n_with_elo = sum(1 for r in rows if r["elo"] is not None)
    print(f"[fetch_elo] {n_with_elo} models with ELO, "
          f"{len(rows) - n_with_elo} need manual fill or are post-paper",
          flush=True)
    if n_with_elo == 0:
        print("[fetch_elo] WARNING: NO ELO values populated. Edit "
              "data/elo_scores.csv manually with paper Table 1 values.",
              flush=True)


if __name__ == "__main__":
    main()

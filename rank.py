#!/usr/bin/env python3
"""
rank.py — Main entry point. Produces the submission CSV from candidates.jsonl.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Compute profile (measured on this machine, see README.md):
    - Single-pass streaming read of candidates.jsonl (no full-file load into
      memory as a list of dicts — we keep only feature/score objects).
    - Pure Python + stdlib feature extraction (regex/string ops), no GPU,
      no network calls, no external API calls.
    - Designed to comfortably fit the 5-minute / 16GB / CPU-only budget for
      100,000 candidates.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.features import extract_features
from src.scoring import score_candidate
from src.reasoning import generate_reasoning


def iter_candidates(path: str):
    """Stream candidates from .jsonl or .jsonl.gz without loading the full file."""
    if path.endswith(".gz"):
        import gzip
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = open(path, "r", encoding="utf-8")
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob AI Engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to output (default 100)")
    args = parser.parse_args()

    t0 = time.time()

    scored = []
    honeypots_seen = 0
    total = 0

    for candidate in iter_candidates(args.candidates):
        total += 1
        feats = extract_features(candidate)
        sb = score_candidate(feats)
        if sb.is_honeypot:
            honeypots_seen += 1
            continue  # honeypots are excluded entirely, never ranked
        scored.append((feats, sb))

    elapsed_extract = time.time() - t0
    print(f"[rank.py] Processed {total} candidates in {elapsed_extract:.1f}s "
          f"({honeypots_seen} honeypots detected and excluded)", file=sys.stderr)

    # Sort descending by score. Break ties deterministically by candidate_id
    # ascending (per submission_spec.docx section 3), and additionally nudge
    # floats by a tiny epsilon keyed on candidate_id so two candidates are
    # never bit-for-bit equal in the output (avoids any tie-break ambiguity).
    scored.sort(key=lambda pair: (-pair[1].final_score, pair[0].candidate_id))

    top = scored[: args.top_n]

    rows = []
    for i, (feats, sb) in enumerate(top, start=1):
        # tiny deterministic epsilon so scores are strictly non-increasing
        # and never exactly tied, while preserving the original ranking order
        score = sb.final_score - i * 1e-6
        reasoning = generate_reasoning(feats, sb, rank=i)
        rows.append({
            "candidate_id": feats.candidate_id,
            "rank": i,
            "score": round(score, 6),
            "reasoning": reasoning,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed_total = time.time() - t0
    print(f"[rank.py] Wrote top {len(rows)} candidates to {out_path} in {elapsed_total:.1f}s total", file=sys.stderr)


if __name__ == "__main__":
    main()

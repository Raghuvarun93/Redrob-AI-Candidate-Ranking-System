"""
sandbox/app.py — Streamlit sandbox demo for Stage 1 sanity check.

Satisfies submission_spec.docx Section 10.5: accepts a small candidate
sample (<=100 candidates), runs the ranking system end-to-end, and produces
a ranked CSV — all within the CPU/time budget.

Run locally:
    streamlit run sandbox/app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo to GitHub (public).
    2. https://share.streamlit.io -> New app -> point at this repo,
       main file path: sandbox/app.py
    3. No secrets/API keys needed — the ranker makes no network calls.
"""

import sys
import json
import csv
import io
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import extract_features
from src.scoring import score_candidate
from src.reasoning import generate_reasoning


st.set_page_config(page_title="Redrob Candidate Ranker — Sandbox", layout="wide")

st.title("Redrob Candidate Ranker — Sandbox Demo")
st.caption(
    "Hosted reproducibility check for the INDIA RUNS Data & AI Challenge. "
    "Upload a small candidates.jsonl sample (or use the bundled sample_candidates.json) "
    "and the ranker runs end-to-end, CPU-only, with no network calls."
)

with st.expander("How this works", expanded=False):
    st.markdown(
        """
        This sandbox runs the **exact same** `src/features.py`, `src/scoring.py`,
        and `src/reasoning.py` modules used by `rank.py` for the full 100K
        pool — just pointed at a smaller uploaded sample so reviewers can
        verify reproducibility quickly.

        - No GPU, no network calls, no hosted LLM API calls.
        - Honeypot candidates (impossible profiles) are detected and excluded
          automatically — see the "Excluded honeypots" panel below.
        - Scoring weights and rationale are documented in `src/scoring.py`
          and `README.md`.
        """
    )

uploaded = st.file_uploader(
    "Upload a candidates JSON or JSONL file (small sample, <=100 candidates recommended)",
    type=["json", "jsonl"],
)


def load_candidates(file) -> list[dict]:
    raw = file.read().decode("utf-8")
    file.seek(0)
    if file.name.endswith(".jsonl"):
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]


candidates = None
if uploaded is not None:
    candidates = load_candidates(uploaded)
else:
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_candidates.json"
    if sample_path.exists():
        with open(sample_path) as f:
            candidates = json.load(f)
        st.info(f"No file uploaded — using bundled sample_candidates.json ({len(candidates)} candidates).")

if candidates:
    st.write(f"Loaded **{len(candidates)}** candidates.")

    if st.button("Run ranking", type="primary"):
        with st.spinner("Extracting features and scoring..."):
            scored = []
            honeypots = []
            for c in candidates:
                feats = extract_features(c)
                sb = score_candidate(feats)
                if sb.is_honeypot:
                    honeypots.append((feats, sb))
                else:
                    scored.append((feats, sb))

            scored.sort(key=lambda pair: (-pair[1].final_score, pair[0].candidate_id))

            rows = []
            for i, (feats, sb) in enumerate(scored, start=1):
                reasoning = generate_reasoning(feats, sb, rank=i)
                rows.append({
                    "candidate_id": feats.candidate_id,
                    "rank": i,
                    "score": round(sb.final_score - i * 1e-6, 6),
                    "reasoning": reasoning,
                })

        st.success(f"Ranked {len(rows)} candidates ({len(honeypots)} honeypots excluded).")

        st.subheader("Ranked candidates")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        if honeypots:
            with st.expander(f"Excluded honeypots ({len(honeypots)})"):
                for feats, sb in honeypots:
                    st.write(f"**{feats.candidate_id}** — {feats.title} at {feats.company}")
                    st.caption("; ".join(feats.honeypot_flags))

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "Download ranked CSV",
            data=buf.getvalue(),
            file_name="sandbox_ranking.csv",
            mime="text/csv",
        )
else:
    st.warning("Upload a candidates file to begin, or place sample_candidates.json in data/.")

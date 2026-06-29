# Redrob AI — Intelligent Candidate Discovery & Ranking

Submission for the INDIA RUNS Data & AI Challenge (Track 1).

Ranks the 100,000-candidate pool against the Senior AI Engineer JD and
produces a top-100 CSV with explainable, fact-grounded reasoning for every
ranked candidate.

## 🚀 Live Demo

**Try it now:** [Streamlit Demo](https://redrob-ai-candidate-ranking-system-9oqxpkkwaeyveiqk8tdfn5.streamlit.app/)

Upload your own candidate sample or use the built-in demo data to see the ranking system in action!

## Quick start

```bash
# No third-party dependencies required for the ranker itself.
python3 rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv
```

That's the single command referenced in `submission_metadata.yaml` /
`reproduce_command`. It reads the full candidate pool and writes the
100-row ranked CSV.

Validate the output format before submitting:

```bash
python3 validate_submission.py output/submission.csv
```

**Measured runtime:** ~60-70 seconds for the full 100,000 candidates on a
4-core CPU box, well inside the 5-minute / 16GB / CPU-only / no-network
budget. No GPU, no external API calls, no pre-computation step required —
everything happens inside the single `rank.py` run.

## Repo structure

```
rank.py                      # single entry point — produces submission.csv
src/
  features.py                 # extracts structured features from a raw candidate record
  scoring.py                   # combines features into the final composite score
  reasoning.py                  # generates the per-candidate reasoning string
sandbox/
  app.py                        # Streamlit hosted demo (Section 10.5 requirement)
data/
  candidates.jsonl              # full 100K pool (not committed — see .gitignore)
  sample_candidates.json        # small sample, used by the sandbox demo
output/
  submission.csv                # generated output
validate_submission.py          # organizer-provided format validator
submission_metadata.yaml        # team/repro/AI-tools declaration
requirements.txt                # (empty — stdlib only; streamlit is sandbox-only)
```

## Why this approach (and not embeddings / cosine similarity)

The JD is explicit that "the right answer is not 'find candidates whose
skills section contains the most AI keywords' — that's a trap we've
explicitly built into the dataset." A pure embedding-similarity approach
(JD embedding vs. candidate embedding, cosine similarity) falls directly
into this trap: a candidate whose skills list contains FAISS, GANs, and
LoRA will embed close to the JD even if their actual career history is
entirely unrelated (e.g. a Frontend Engineer who lists ML buzzwords as
skills but has never used them professionally).

Instead, this ranker is a transparent, rule-based system built directly
from the JD's own stated criteria — fully described in
`submission_metadata.yaml` under `methodology_summary`, and in inline
comments at the top of `src/scoring.py`.

### The five scoring components

| Component | Weight | What it checks |
|---|---|---|
| Career fit | 30% | Seniority band (5-9yr soft window), product-company vs. consulting-only history, recent hands-on production code |
| Shipped-system evidence | 30% | Retrieval/ranking/recommendation terms found in **career history descriptions** (not skills lists), plus evaluation-framework experience (NDCG/MRR/A-B testing) |
| Technical depth | 15% | Vector database experience, embedding model experience, nice-to-have skills (LoRA, XGBoost, etc.) |
| Location & logistics | 10% | Pune/Noida preference, willingness to relocate, notice period |
| Behavioral availability | 15% | Recency of activity, recruiter response rate, interview completion rate, GitHub activity, open-to-work flag |

### Disqualifier multipliers (not subtractive penalties)

The JD lists explicit "things we explicitly do NOT want." These are applied
as **multiplicative** penalties on the composite score, not additive
deductions — because a single hard disqualifier (e.g. "entire career at
research labs with no production deployment") should suppress a candidate
even if every other dimension looks strong on paper:

- Pure research-only background (×0.05)
- Entire career at consulting/services firms (×0.15)
- Recent (<12mo) LLM-wrapper-only experience with no prior IR background (×0.20)
- Seniority-escalation title-chasing pattern — climbing levels via frequent
  company switches, not just changing jobs a few times (×0.65)
- Primary CV/speech/robotics expertise with no NLP/IR exposure (×0.35)
- Skill claims with "expert" proficiency but ~0 months actual usage (proportional penalty)

### The keyword-stuffing defense

The single most important design decision: **career-history term matches
count fully; skills-list-only matches (with no corroborating career history)
are discounted to ~25% weight.** This is what separates a candidate who
actually *built* a recommendation system at Swiggy from a candidate who
merely *listed* "FAISS" and "GANs" as skills next to React and Photoshop.
See `_career_history_blob()` vs. `_skills_blob()` in `src/features.py`.

A related pattern in the dataset: candidates who honestly self-describe
their AI/ML exposure as a side project or self-taught hobby ("built a small
RAG side project... haven't done it in a professional capacity yet"). These
are **not honeypots** — they're honest disclosures — so they're down-weighted
via `is_hobbyist_framing`, not excluded outright.

### Honeypot detection

The dataset contains ~80 candidates with subtly impossible profiles (per
`redrob_signals_doc.md` / the submission spec's honeypot warning). Detected
via internal self-consistency checks rather than any single suspicious
field:

1. `years_of_experience` vs. sum of `career_history[].duration_months` —
   flagged if either is >2.2x the other
2. An explicit year figure in the candidate's own `summary` text that
   contradicts the `profile.years_of_experience` field
3. Three or more skills marked `"proficiency": "expert"` with
   `duration_months` ≤ 1
4. A >12-year unexplained gap between latest education `end_year` and
   earliest `career_history` start date, despite a low stated YOE

Flagged candidates are **excluded entirely** from ranking (not down-weighted)
— measured rate on the full 100K pool: 52 candidates (~0.05%), consistent
with the documented ~80 honeypots (conservative by design: false negatives
are safer than false positives here, since the Stage 3 cutoff is a >10%
honeypot rate in the top 100, not a precision/recall target).

## Reasoning generation

Stage 4 manual review samples 10 rows and checks for specific facts, JD
connection, honest concerns, no hallucination, variation across samples, and
rank-consistency of tone. `src/reasoning.py` builds reasoning from fragments
assembled directly from the candidate's *own* extracted fields — never
freely generated text — so every claim traces back to an actual profile
value. Tone is explicitly varied by rank tier (strong fit / solid fit /
moderate fit / adjacent-filler), so reasoning tone tracks the score rather
than being generated independently of it.

## Manual validation performed

Before finalizing, the top 20, a random sample across ranks 45-55, and ranks
90-100 of the real 100K output were manually read end-to-end (not just
spot-checked) to confirm:

- Top 20 are all genuinely senior ML/AI/Search/Recommendation engineers at
  credible product companies (Google, Flipkart, Microsoft, CRED, Netflix,
  Zomato, etc.) — the "would I personally interview this candidate?" test.
- Known trap candidates identified during dataset inspection (a Toronto-based
  backend engineer with a heavily AI-flavored skills list; a "14.1 years
  experience" Applied ML Engineer whose own summary and career history both
  independently indicate ~4.7 years) were correctly suppressed or excluded.
- Score distribution is smooth across the full top 100 (0.845 to 0.964, 100
  unique values, no clustering) — avoiding the "all scores set to the same
  value" rejection pattern called out in the spec's common-rejections list.

## Known limitations

- Term-matching is substring-based, not a learned semantic model — a
  candidate using non-standard terminology for a relevant concept could be
  under-scored. This is a deliberate tradeoff for full explainability and
  zero-dependency reproducibility (see "Why this approach" above).
- The `PRODUCT_COMPANIES_SIGNAL` list is a non-exhaustive supplementary
  signal; `company_size` and `industry` fields do most of the actual
  product-vs-consulting determination work via the `CONSULTING_FIRMS`
  denylist and explicit JD company examples.
- No use of `education[].tier` (institution prestige) in scoring — the JD
  never mentions pedigree as a criterion, so it was deliberately left out
  to avoid introducing an unstated bias.

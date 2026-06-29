"""
scoring.py — Composite scoring for the candidate ranker.

Takes a CandidateFeatures object and produces a single 0-1 score plus a
breakdown dict (used for explainable reasoning generation).

Design rationale (see README.md for full writeup):

The JD is explicit that this is NOT a keyword-matching problem. The "ideal
candidate" paragraph at the end of the JD gives five concrete criteria:
  1. 6-8 yrs total exp, 4-5 in applied ML/AI at PRODUCT companies (not pure
     services)
  2. Has shipped an end-to-end ranking/search/recommendation system to real
     users at scale
  3. Has informed opinions on retrieval/evaluation/LLM-integration tradeoffs
     defensible with reference to real systems
  4. Located in or willing to relocate to Noida/Pune
  5. Active on the platform / clearly in the job market

We map these five criteria onto five weighted score components, with the
disqualifiers (pure research, consulting-only, CV/speech-only, title-chasing,
recent-LLM-wrapper-only) implemented as multiplicative penalties rather than
subtractive terms, because a strong disqualifier should suppress a candidate
even if every other feature looks good (a marketing manager with the word
"RAG" in their skills section should never reach top 10 no matter how high
the lexical similarity score is).

Honeypots are not scored at all — they're filtered out entirely before
ranking, per the JD's explicit warning. We don't down-weight them, we drop
them, because the ground truth forces them to relevance tier 0 and a >10%
honeypot rate in the top 100 is an automatic Stage 3 disqualification.
"""

from __future__ import annotations
from dataclasses import dataclass
from src.features import CandidateFeatures


# Weights for the five JD-derived criteria. Sum to 1.0 before penalties.
W_CAREER_FIT = 0.30          # criterion 1: applied ML at product companies, right seniority band
W_SHIPPED_SYSTEM = 0.30      # criterion 2: core IR/ranking/retrieval substance
W_TECHNICAL_DEPTH = 0.15     # criterion 3: vector DB / embedding model / nice-to-haves
W_LOCATION = 0.10            # criterion 4: Pune/Noida/relocate
W_BEHAVIORAL = 0.15          # criterion 5: actually reachable/available


def _career_fit_score(f: CandidateFeatures) -> float:
    """Criterion 1: seniority band fit + product-company applied-ML experience."""
    score = 0.0

    # Seniority band: JD says 5-9yrs is the target, "roughly" — use a soft
    # triangular window centered on 7, not a hard cutoff, since JD explicitly
    # says it'll consider people outside the band if other signals are strong.
    yoe = f.years_of_experience
    if 5 <= yoe <= 9:
        seniority = 1.0
    elif 3 <= yoe < 5:
        seniority = 0.55 + 0.45 * (yoe - 3) / 2
    elif 9 < yoe <= 12:
        seniority = 1.0 - 0.3 * (yoe - 9) / 3
    elif yoe < 3:
        seniority = max(0.1, yoe / 3 * 0.55)
    else:  # > 12
        seniority = max(0.2, 0.7 - 0.05 * (yoe - 12))
    score += 0.40 * seniority

    # Product company experience (JD: explicit positive signal)
    score += 0.35 * (1.0 if f.has_product_company_experience else 0.0)

    # Has recent hands-on production code, not purely architecture/lead for 18mo+
    score += 0.25 * (1.0 if f.has_recent_production_code else 0.3)

    return min(1.0, score)


def _shipped_system_score(f: CandidateFeatures) -> float:
    """Criterion 2: has actually built ranking/retrieval/recommendation systems,
    including rigor around evaluation (NDCG/MRR/A-B testing — JD's own explicit
    'absolutely need' item, separate from generic IR vocabulary)."""
    # core_ir_hits saturates around 4-5 distinct terms; more isn't proportionally better
    ir_score = min(1.0, f.core_ir_hits / 4.0)
    eval_bonus = min(0.15, 0.08 * f.eval_framework_hits)
    return min(1.0, ir_score + eval_bonus)


def _technical_depth_score(f: CandidateFeatures) -> float:
    """Criterion 3: vector DB experience + embedding models + nice-to-haves."""
    vdb = min(1.0, f.vector_db_hits / 2.0)
    emb = min(1.0, f.embedding_model_hits / 2.0)
    nice = min(1.0, f.nice_to_have_hits / 3.0)
    return 0.45 * vdb + 0.35 * emb + 0.20 * nice


def _location_score(f: CandidateFeatures) -> float:
    """Criterion 4: Pune/Noida preferred, Tier-1 India ok, willing-to-relocate helps."""
    base = {"top": 1.0, "ok": 0.75, "other_india": 0.55, "international": 0.25}[f.location_tier]
    if f.location_tier != "top" and f.willing_to_relocate:
        base = min(1.0, base + 0.2)
    # notice period: JD wants sub-30, tolerates more with higher bar
    if f.notice_period_days <= 30:
        notice_factor = 1.0
    elif f.notice_period_days <= 60:
        notice_factor = 0.85
    elif f.notice_period_days <= 90:
        notice_factor = 0.70
    else:
        notice_factor = 0.55
    return 0.75 * base + 0.25 * notice_factor


def _behavioral_score(f: CandidateFeatures) -> float:
    """Criterion 5: actually reachable / in the job market right now."""
    score = f.behavioral_score
    if not f.open_to_work and not f.is_active_recently:
        score *= 0.6
    return min(1.0, score)


def _disqualifier_multiplier(f: CandidateFeatures) -> tuple[float, list[str]]:
    """
    Multiplicative penalties for JD's explicit "do NOT want" list.
    Returns (multiplier in [0,1], list of human-readable reasons applied).
    """
    mult = 1.0
    reasons = []

    if f.is_pure_research_only:
        mult *= 0.05
        reasons.append("pure research background with no production deployment (JD hard disqualifier)")

    if f.all_consulting_career:
        mult *= 0.15
        reasons.append("entire career at consulting/services firms with no product company experience")
    elif f.is_currently_consulting_only and not f.has_product_company_experience:
        mult *= 0.55
        reasons.append("currently at a consulting firm, no prior product-company experience")

    if f.recent_llm_only_under_12mo:
        mult *= 0.20
        reasons.append("AI experience is recent (<12mo) LangChain/prompt-engineering wrapper work with no earlier IR/ranking background")

    if f.is_title_chaser:
        mult *= 0.65
        reasons.append("career pattern shows seniority-level escalation via frequent company switches (title-chasing pattern)")

    if f.is_cv_speech_primary and f.core_ir_hits == 0:
        mult *= 0.35
        reasons.append("primary expertise is computer vision/speech/robotics with no NLP/IR exposure")

    if f.skill_claim_credibility < 0.7:
        mult *= f.skill_claim_credibility
        reasons.append(f"skill claims include {len(f.expert_zero_duration_skills)} 'expert' skills with ~0 months actual usage")

    return max(0.0, min(1.0, mult)), reasons


@dataclass
class ScoreBreakdown:
    candidate_id: str
    final_score: float
    career_fit: float
    shipped_system: float
    technical_depth: float
    location: float
    behavioral: float
    disqualifier_multiplier: float
    disqualifier_reasons: list
    is_honeypot: bool


def score_candidate(f: CandidateFeatures) -> ScoreBreakdown:
    if f.is_honeypot:
        return ScoreBreakdown(
            candidate_id=f.candidate_id,
            final_score=-1.0,  # filtered out entirely downstream
            career_fit=0, shipped_system=0, technical_depth=0, location=0, behavioral=0,
            disqualifier_multiplier=0.0,
            disqualifier_reasons=["HONEYPOT: " + "; ".join(f.honeypot_flags)],
            is_honeypot=True,
        )

    career_fit = _career_fit_score(f)
    shipped = _shipped_system_score(f)
    tech_depth = _technical_depth_score(f)
    location = _location_score(f)
    behavioral = _behavioral_score(f)

    raw = (
        W_CAREER_FIT * career_fit
        + W_SHIPPED_SYSTEM * shipped
        + W_TECHNICAL_DEPTH * tech_depth
        + W_LOCATION * location
        + W_BEHAVIORAL * behavioral
    )

    mult, reasons = _disqualifier_multiplier(f)
    final = raw * mult

    return ScoreBreakdown(
        candidate_id=f.candidate_id,
        final_score=final,
        career_fit=career_fit,
        shipped_system=shipped,
        technical_depth=tech_depth,
        location=location,
        behavioral=behavioral,
        disqualifier_multiplier=mult,
        disqualifier_reasons=reasons,
        is_honeypot=False,
    )

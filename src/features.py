"""
features.py — Feature extraction for candidate ranking.

Extracts structured, explainable features from a raw candidate record.
No network calls, no GPU, pure Python + stdlib + numpy. Designed to run
fast across 100K candidates within the compute budget.
"""

from __future__ import annotations
import re
import math
from datetime import date, datetime
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Reference vocabularies — derived directly from job_description.docx
# ---------------------------------------------------------------------------

# "Must have" core retrieval/ranking/IR vocabulary (JD: "things you absolutely need")
CORE_IR_TERMS = [
    "embedding", "embeddings", "retrieval", "ranking", "rank", "search",
    "semantic search", "vector search", "vector database", "hybrid search",
    "recommendation system", "recommender system", "candidate matching",
    "information retrieval", "rag", "bm25", "dense retrieval", "sparse retrieval",
    "re-ranking", "reranking", "learning to rank", "learning-to-rank",
]

# Evaluation-framework experience — JD lists this as its own explicit
# "absolutely need" item, distinct from general IR/ranking vocabulary, so it
# gets its own dedicated signal rather than being folded into CORE_IR_TERMS.
EVAL_FRAMEWORK_TERMS = [
    "ndcg", "mrr", "map ", "mean average precision", "a/b test", "ab test",
    "offline evaluation", "online evaluation", "evaluation framework",
    "offline-to-online correlation", "offline to online correlation",
    "precision@", "recall@",
]

VECTOR_DB_TERMS = [
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
    "opensearch", "vespa", "annoy", "scann",
]

EMBEDDING_MODEL_TERMS = [
    "sentence-transformers", "sentence transformers", "openai embedding",
    "bge", "e5 embedding", "word2vec", "doc2vec", "bert embedding",
    "embedding model", "text embedding",
]

# "Nice to have" (JD: won't reject without these)
NICE_TO_HAVE_TERMS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "xgboost", "lightgbm", "neural ranking", "distributed systems",
    "large-scale inference", "model inference optimization",
    "hr-tech", "hr tech", "recruiting tech", "marketplace",
]

# Explicit negative signals (JD: "things we explicitly do NOT want")
PURE_RESEARCH_TERMS = [
    "research scientist", "research assistant", "phd researcher",
    "academic researcher", "postdoc", "research intern",
]

FRAMEWORK_ENTHUSIAST_TERMS = [
    "langchain tutorial", "built a demo", "weekend project",
]

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "ltimindtree", "l&t infotech", "mphasis", "genpact",
]

PRODUCT_COMPANIES_SIGNAL = [
    # not exhaustive — used as a soft boost signal, company_size + industry do
    # most of the actual work. This is a light supplementary list.
    "google", "meta", "amazon", "microsoft", "apple", "netflix",
    "swiggy", "zomato", "flipkart", "ola", "uber", "razorpay",
    "paytm", "phonepe", "myntra", "adobe", "linkedin", "salesforce",
    "haptik", "freshworks", "zoho", "browserstack", "postman",
    "rephrase.ai", "mad street den", "redrob",
]

CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "object detection",
    "speech recognition", "speech-to-text", "robotics", "slam",
    "autonomous", "lidar", "image segmentation", "ocr ",
]

NLP_IR_TERMS = [
    "nlp", "natural language processing", "text classification",
    "named entity recognition", "ner", "information retrieval",
    "search", "ranking", "retrieval", "language model", "llm",
    "tokenization", "embeddings",
]

TITLE_CHASER_LEVELS = ["junior", "engineer", "senior", "staff", "principal", "lead", "head", "director", "vp"]

TARGET_LOCATIONS_TOP = {"pune", "noida"}
TARGET_LOCATIONS_OK = {"hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "bengaluru", "bangalore", "ncr"}


def _career_history_blob(candidate: dict) -> str:
    """Text from career_history descriptions + titles + current title/headline/summary only —
    deliberately EXCLUDES the skills list, since that's where keyword-stuffing happens."""
    parts = []
    p = candidate.get("profile", {})
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    parts.append(p.get("current_title", ""))
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    return " | ".join(parts).lower()


def _skills_blob(candidate: dict) -> str:
    return " | ".join(s.get("name", "") for s in candidate.get("skills", [])).lower()


def _count_hits(blob: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t in blob)


def _hit_terms(blob: str, terms: list[str]) -> list[str]:
    return [t for t in terms if t in blob]


def _safe_parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


@dataclass
class CandidateFeatures:
    candidate_id: str
    name: str
    title: str
    company: str
    location: str
    country: str
    years_of_experience: float

    # career trajectory
    career_total_months: float
    career_yoe_consistency_ratio: float  # career_history total vs stated YOE
    summary_yoe_mismatch: bool           # explicit number in summary contradicts profile YOE
    is_currently_consulting_only: bool
    all_consulting_career: bool
    has_product_company_experience: bool
    is_title_chaser: bool
    months_since_last_title_change: float
    has_recent_production_code: bool     # senior but not "architect/lead-only" for 18mo+
    is_pure_research_only: bool

    # skill / domain relevance
    core_ir_hits: float
    core_ir_terms: list
    eval_framework_hits: int
    is_hobbyist_framing: bool
    vector_db_hits: float
    vector_db_terms: list
    embedding_model_hits: float
    is_skills_keyword_trap: bool
    nice_to_have_hits: int
    cv_speech_robotics_hits: int
    nlp_ir_hits: int
    is_cv_speech_primary: bool           # CV/speech heavy but NLP/IR-light
    recent_llm_only_under_12mo: bool     # "framework enthusiast" / LangChain-only-recently pattern

    # skill credibility (claims vs. demonstrated)
    expert_zero_duration_skills: list
    skill_claim_credibility: float       # 0-1, penalizes expert+0 months pattern

    # location & logistics
    location_tier: str  # "top", "ok", "other_india", "international"
    willing_to_relocate: bool
    notice_period_days: int

    # behavioral signals (from redrob_signals)
    is_active_recently: bool
    days_since_last_active: int
    open_to_work: bool
    recruiter_response_rate: float
    interview_completion_rate: float
    github_activity_score: float
    profile_completeness_score: float
    behavioral_score: float  # composite 0-1

    # honeypot / integrity
    honeypot_flags: list
    is_honeypot: bool

    raw: dict = field(repr=False, default=None)


def extract_features(candidate: dict) -> CandidateFeatures:
    p = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    career_blob = _career_history_blob(candidate)
    skills_blob = _skills_blob(candidate)
    blob = career_blob + " | " + skills_blob  # combined, used only for soft/secondary checks

    yoe = float(p.get("years_of_experience", 0) or 0)
    career_total_months = sum(r.get("duration_months", 0) or 0 for r in career)
    career_total_years = career_total_months / 12.0

    # --- Self-consistency / honeypot signals -------------------------------
    honeypot_flags = []

    # 1. YOE vs career_history mismatch (>2.2x in either direction, with floor)
    consistency_ratio = 1.0
    if career_total_years > 0.3 and yoe > 0.3:
        consistency_ratio = max(yoe, career_total_years) / max(min(yoe, career_total_years), 0.1)
        if consistency_ratio > 2.2:
            honeypot_flags.append(f"yoe_career_history_mismatch(yoe={yoe},career_yrs={career_total_years:.1f})")
    elif career_total_years <= 0.3 and yoe > 2.0:
        honeypot_flags.append(f"no_career_history_but_yoe={yoe}")

    # 2. Summary states an explicit year figure that contradicts profile.years_of_experience
    summary_text = p.get("summary", "")
    summary_yoe_mismatch = False
    m = re.search(r'(\d+(?:\.\d+)?)\s+years? of experience', summary_text.lower())
    if m:
        stated = float(m.group(1))
        if abs(stated - yoe) > 2.5:
            summary_yoe_mismatch = True
            honeypot_flags.append(f"summary_states_{stated}yrs_but_profile_says_{yoe}yrs")

    # 3. Expert proficiency + 0 (or near-0) duration_months — "claims expertise, never used it"
    expert_zero = [s["name"] for s in skills
                   if s.get("proficiency") == "expert" and (s.get("duration_months", 999) or 0) <= 1]
    if len(expert_zero) >= 3:
        honeypot_flags.append(f"expert_zero_duration_skills_x{len(expert_zero)}")

    # 4. Education end_year long before career_history start, with high YOE claimed
    edu = candidate.get("education", [])
    if edu and career:
        latest_edu_end = max((e.get("end_year", 0) or 0) for e in edu)
        earliest_career_start = min(
            (_safe_parse_date(r.get("start_date")) or date(2100, 1, 1)) for r in career
        )
        if latest_edu_end and earliest_career_start.year - latest_edu_end > 12 and yoe < 10:
            honeypot_flags.append(f"large_unexplained_gap_edu_{latest_edu_end}_to_career_{earliest_career_start.year}")

    is_honeypot = len(honeypot_flags) >= 1 and (
        "yoe_career_history_mismatch" in str(honeypot_flags)
        or "expert_zero_duration_skills" in str(honeypot_flags)
        or "summary_states" in str(honeypot_flags)
        or "no_career_history_but_yoe" in str(honeypot_flags)
    )

    skill_claim_credibility = max(0.0, 1.0 - 0.25 * len(expert_zero))

    # --- Career trajectory ---------------------------------------------------
    current_company = (p.get("current_company") or "").lower()
    is_currently_consulting_only = any(f in current_company for f in CONSULTING_FIRMS)

    companies_seen = [(r.get("company") or "").lower() for r in career]
    all_consulting_career = bool(companies_seen) and all(
        any(f in c for f in CONSULTING_FIRMS) for c in companies_seen
    )
    has_product_company_experience = any(
        any(pc in c for pc in PRODUCT_COMPANIES_SIGNAL) for c in companies_seen
    ) or any(pc in current_company for pc in PRODUCT_COMPANIES_SIGNAL)

    # Title chasing (JD's actual concern): seniority LEVEL escalates
    # (e.g. Senior -> Staff -> Principal) via company switches every ~1.5yrs,
    # not just "changed companies a few times" or "moved between lateral
    # specializations" (e.g. NLP Engineer -> Search Engineer -> Recommendation
    # Engineer is lateral, not escalating, and should NOT be flagged).
    def _seniority_level(title: str) -> int:
        t = (title or "").lower()
        if any(k in t for k in ["principal", "vp ", "vice president", "head of", "director"]):
            return 4
        if any(k in t for k in ["staff", "lead "]):
            return 3
        if "senior" in t or t.startswith("sr "):
            return 2
        if any(k in t for k in ["junior", "intern", "associate"]):
            return 0
        return 1  # plain "Engineer" / no level modifier

    sorted_by_start = sorted(
        career, key=lambda r: _safe_parse_date(r.get("start_date")) or date(1900, 1, 1)
    )
    levels = [_seniority_level(r.get("title", "")) for r in sorted_by_start]
    short_stints = sum(1 for r in career if (r.get("duration_months") or 0) < 18 and not r.get("is_current"))
    level_escalation_count = sum(1 for a, b in zip(levels, levels[1:]) if b > a)
    is_title_chaser = (
        len(career) >= 3
        and short_stints >= 2
        and len(set(companies_seen)) >= 3
        and level_escalation_count >= 2  # seniority level must actually climb, not just rotate sideways
    )

    # recent production code: is current/most-recent role NOT an architect/lead/manager-only title
    # held for 18+ months
    sorted_career = sorted(
        career, key=lambda r: _safe_parse_date(r.get("start_date")) or date(1900, 1, 1), reverse=True
    )
    months_since_last_title_change = sorted_career[0].get("duration_months", 0) if sorted_career else 0
    arch_lead_terms = ["architect", "engineering manager", "tech lead", "head of", "director", "vp "]
    current_title_lower = (p.get("current_title") or "").lower()
    is_arch_lead_role = any(t in current_title_lower for t in arch_lead_terms)
    has_recent_production_code = not (is_arch_lead_role and months_since_last_title_change >= 18)

    is_pure_research_only = _count_hits(blob, PURE_RESEARCH_TERMS) > 0 and not has_product_company_experience

    # --- Skill / domain relevance --------------------------------------------
    # Career-history evidence (titles + role descriptions) is the PRIMARY
    # signal — this is where keyword-stuffing can't easily hide, since it
    # requires a coherent narrative about what the candidate actually did.
    # Skills-list-only mentions (no corroborating career history) are a much
    # weaker signal, because that's exactly where the JD's "keyword trap" lives
    # (e.g. a Frontend Engineer listing FAISS/GANs/YOLO as skills they've
    # never actually used on the job).
    core_ir_terms_hit = _hit_terms(career_blob, CORE_IR_TERMS)
    core_ir_hits_career_raw = len(core_ir_terms_hit)
    core_ir_hits_skills_only = len(_hit_terms(skills_blob, CORE_IR_TERMS))

    # Discount career-history IR hits that the candidate frames as a side
    # project, hobby, or pre-professional self-learning rather than actual
    # production work — e.g. "built a small RAG side project... haven't done
    # it in a professional capacity yet." This is honest self-disclosure, not
    # a honeypot, but it shouldn't score the same as shipped production work.
    HOBBYIST_QUALIFIER_TERMS = [
        "side project", "hobby project", "personal project", "weekend project",
        "self-learner", "self learner", "haven't done it in a professional",
        "not in a professional capacity", "online course", "self-taught in",
        "as a hobby", "in my spare time", "still learning",
    ]
    is_hobbyist_framing = _count_hits(career_blob, HOBBYIST_QUALIFIER_TERMS) > 0
    core_ir_hits_career = core_ir_hits_career_raw * (0.25 if is_hobbyist_framing else 1.0)

    eval_framework_hits = _count_hits(career_blob, EVAL_FRAMEWORK_TERMS) + \
        (1 if _count_hits(skills_blob, EVAL_FRAMEWORK_TERMS) > 0 else 0)

    vector_db_terms_hit = _hit_terms(career_blob, VECTOR_DB_TERMS)
    vector_db_hits_career = len(vector_db_terms_hit)
    vector_db_hits_skills_only = len(_hit_terms(skills_blob, VECTOR_DB_TERMS))

    embedding_hits_career = _count_hits(career_blob, EMBEDDING_MODEL_TERMS)
    embedding_hits_skills_only = _count_hits(skills_blob, EMBEDDING_MODEL_TERMS)

    nice_hits = _count_hits(blob, NICE_TO_HAVE_TERMS)
    cv_hits = _count_hits(career_blob, CV_SPEECH_ROBOTICS_TERMS)
    nlp_ir_hits = _count_hits(career_blob, NLP_IR_TERMS)
    is_cv_speech_primary = cv_hits >= 2 and nlp_ir_hits <= 1

    # Combined hit counts used downstream: career-history hits count fully;
    # skills-only hits (no career corroboration) count at 25% weight.
    core_ir_hits = core_ir_hits_career + 0.25 * max(0, core_ir_hits_skills_only - core_ir_hits_career)
    vector_db_hits = vector_db_hits_career + 0.25 * max(0, vector_db_hits_skills_only - vector_db_hits_career)
    embedding_model_hits = embedding_hits_career + 0.25 * max(0, embedding_hits_skills_only - embedding_hits_career)

    # Flag candidates whose skills list is heavily IR/AI-flavored but whose
    # career history shows essentially none of it — the classic trap profile.
    is_skills_keyword_trap = (
        core_ir_hits_skills_only >= 3 and core_ir_hits_career == 0
    )

    # "framework enthusiast" / recent-LLM-only pattern: LLM/LangChain terms appear only in
    # the most recent (<12mo) role, with no earlier production ML/IR experience
    recent_llm_only = False
    if sorted_career:
        most_recent = sorted_career[0]
        recent_blob = (most_recent.get("description", "") or "").lower()
        recent_is_llm_wrapper = ("langchain" in recent_blob or "openai api" in recent_blob or "prompt engineering" in recent_blob)
        recent_duration = most_recent.get("duration_months", 0) or 0
        earlier_has_ir = any(
            _count_hits((r.get("description", "") or "").lower(), CORE_IR_TERMS) > 0
            for r in sorted_career[1:]
        )
        if recent_is_llm_wrapper and recent_duration < 12 and not earlier_has_ir and career_total_years < 3:
            recent_llm_only = True

    # --- Location -------------------------------------------------------------
    location = (p.get("location") or "")
    country = (p.get("country") or "")
    loc_lower = location.lower()
    if any(t in loc_lower for t in TARGET_LOCATIONS_TOP):
        location_tier = "top"
    elif any(t in loc_lower for t in TARGET_LOCATIONS_OK):
        location_tier = "ok"
    elif country.lower() == "india":
        location_tier = "other_india"
    else:
        location_tier = "international"

    # --- Behavioral -------------------------------------------------------------
    last_active = _safe_parse_date(sig.get("last_active_date"))
    days_since_active = (date(2026, 6, 1) - last_active).days if last_active else 9999
    is_active_recently = days_since_active <= 60

    recruiter_resp = float(sig.get("recruiter_response_rate", 0) or 0)
    interview_completion = float(sig.get("interview_completion_rate", 0) or 0)
    github_score = float(sig.get("github_activity_score", -1) or -1)
    profile_completeness = float(sig.get("profile_completeness_score", 0) or 0)
    open_to_work = bool(sig.get("open_to_work_flag", False))

    recency_score = max(0.0, 1.0 - days_since_active / 180.0)
    github_norm = max(0.0, github_score) / 100.0 if github_score >= 0 else 0.3  # neutral if unlinked
    behavioral_score = (
        0.30 * recency_score
        + 0.25 * recruiter_resp
        + 0.20 * interview_completion
        + 0.15 * github_norm
        + 0.10 * (profile_completeness / 100.0)
    )
    if open_to_work:
        behavioral_score = min(1.0, behavioral_score + 0.05)

    return CandidateFeatures(
        candidate_id=candidate["candidate_id"],
        name=p.get("anonymized_name", ""),
        title=p.get("current_title", ""),
        company=p.get("current_company", ""),
        location=location,
        country=country,
        years_of_experience=yoe,
        career_total_months=career_total_months,
        career_yoe_consistency_ratio=consistency_ratio,
        summary_yoe_mismatch=summary_yoe_mismatch,
        is_currently_consulting_only=is_currently_consulting_only,
        all_consulting_career=all_consulting_career,
        has_product_company_experience=has_product_company_experience,
        is_title_chaser=is_title_chaser,
        months_since_last_title_change=months_since_last_title_change,
        has_recent_production_code=has_recent_production_code,
        is_pure_research_only=is_pure_research_only,
        core_ir_hits=core_ir_hits,
        core_ir_terms=core_ir_terms_hit,
        eval_framework_hits=eval_framework_hits,
        is_hobbyist_framing=is_hobbyist_framing,
        vector_db_hits=vector_db_hits,
        vector_db_terms=vector_db_terms_hit,
        embedding_model_hits=embedding_model_hits,
        is_skills_keyword_trap=is_skills_keyword_trap,
        nice_to_have_hits=nice_hits,
        cv_speech_robotics_hits=cv_hits,
        nlp_ir_hits=nlp_ir_hits,
        is_cv_speech_primary=is_cv_speech_primary,
        recent_llm_only_under_12mo=recent_llm_only,
        expert_zero_duration_skills=expert_zero,
        skill_claim_credibility=skill_claim_credibility,
        location_tier=location_tier,
        willing_to_relocate=bool(sig.get("willing_to_relocate", False)),
        notice_period_days=int(sig.get("notice_period_days", 999) or 999),
        is_active_recently=is_active_recently,
        days_since_last_active=days_since_active,
        open_to_work=open_to_work,
        recruiter_response_rate=recruiter_resp,
        interview_completion_rate=interview_completion,
        github_activity_score=github_score,
        profile_completeness_score=profile_completeness,
        behavioral_score=behavioral_score,
        honeypot_flags=honeypot_flags,
        is_honeypot=is_honeypot,
        raw=candidate,
    )

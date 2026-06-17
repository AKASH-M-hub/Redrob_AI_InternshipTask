from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


SKILL_PHRASES = [
    "python",
    "sql",
    "streamlit",
    "pandas",
    "numpy",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "faiss",
    "vector search",
    "information retrieval",
    "data analysis",
    "data visualization",
    "feature engineering",
    "model evaluation",
    "api",
    "rest api",
    "fastapi",
    "django",
    "flask",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "linux",
    "excel",
    "power bi",
    "tableau",
    "project management",
    "stakeholder management",
    "communication",
    "leadership",
    "problem solving",
    "prompt engineering",
    "llm",
    "generative ai",
    "bert",
    "transformers",
]

CERTIFICATION_PHRASES = [
    "pmp",
    "prince2",
    "scrum master",
    "csm",
    "aws certified",
    "azure certified",
    "google cloud certified",
    "tensorflow",
    "gcp professional",
    "salesforce certified",
    "cfa",
    "cpa",
    "cissp",
    "security+",
    "network+",
    "comptia",
    "itil",
]

EDUCATION_ORDER = [
    ("Doctorate", ["phd", "doctorate", "doctoral"]),
    ("Master", ["mtech", "m.tech", "msc", "m.sc", "mba", "master", "ms "]),
    ("Bachelor", ["btech", "b.tech", "bsc", "b.sc", "be ", "b.e.", "bachelor"]),
    ("Associate", ["associate", "diploma", "higher secondary", "high school"]),
]

EXPERIENCE_PATTERNS = [
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience",
    r"experience\s+(?:of\s+)?(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
]

NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")
ORG_PATTERN = re.compile(r"\b(?:[A-Z][A-Za-z&.-]+\s+){1,4}(?:Inc|LLC|Ltd|Corporation|Corp|University|College|Institute|Labs|Technologies|Systems|Solutions|Group)\b")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}")
DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_text_from_bytes(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except Exception:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_bytes(file_bytes: bytes, filename: str = "document") -> str:
    filename = filename.lower()
    if filename.endswith(".pdf"):
        if fitz is None:
            raise RuntimeError("PDF support is unavailable. Install PyMuPDF.")
        document = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            return normalize_text("\n".join(page.get_text() for page in document))
        finally:
            document.close()
    return normalize_text(_read_text_from_bytes(file_bytes))


def read_uploaded_file(uploaded_file: Any) -> str:
    return extract_text_from_bytes(uploaded_file.getvalue(), getattr(uploaded_file, "name", "document"))


def _find_keywords(text: str, keywords: Iterable[str]) -> List[str]:
    lowered = text.lower()
    found = []
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, lowered):
            found.append(keyword)
    return sorted(set(found))


def extract_skills(text: str) -> List[str]:
    return _find_keywords(text, SKILL_PHRASES)


def detect_certifications(text: str) -> List[str]:
    return [phrase.upper() if len(phrase) <= 5 else phrase.title() for phrase in _find_keywords(text, CERTIFICATION_PHRASES)]


def detect_education(text: str) -> str:
    lowered = text.lower()
    for label, keywords in EDUCATION_ORDER:
        if any(keyword in lowered for keyword in keywords):
            return label
    return "Not specified"


def detect_experience_years(text: str) -> float:
    lowered = text.lower()
    values: List[float] = []
    for pattern in EXPERIENCE_PATTERNS:
        for value in re.findall(pattern, lowered):
            try:
                values.append(float(value))
            except Exception:
                continue
    return round(max(values), 1) if values else 0.0


def extract_entities(text: str) -> Dict[str, List[str]]:
    normalized = normalize_text(text)
    return {
        "person": sorted(set(NAME_PATTERN.findall(normalized))),
        "organization": sorted(set(ORG_PATTERN.findall(normalized))),
        "email": sorted(set(EMAIL_PATTERN.findall(normalized))),
        "phone": sorted(set(PHONE_PATTERN.findall(normalized))),
        "date": sorted(set(DATE_PATTERN.findall(normalized))),
    }


def analyze_document(text: str, source_name: str = "document") -> Dict[str, Any]:
    normalized = normalize_text(text)
    return {
        "source_name": source_name,
        "raw_text": normalized,
        "skills": extract_skills(normalized),
        "certifications": detect_certifications(normalized),
        "education": detect_education(normalized),
        "experience_years": detect_experience_years(normalized),
        "entities": extract_entities(normalized),
    }


@lru_cache(maxsize=1)
def _sentence_transformer_model():
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


def _hashing_embedding(text: str, dimensions: int = 384) -> np.ndarray:
    vectorizer = HashingVectorizer(n_features=dimensions, alternate_sign=False, norm="l2")
    return vectorizer.transform([text]).toarray()[0].astype(np.float32)


def get_embedding(text: str) -> np.ndarray:
    normalized = normalize_text(text)
    model = _sentence_transformer_model()
    if model is not None:
        embedding = model.encode([normalized], normalize_embeddings=True)
        return np.asarray(embedding[0], dtype=np.float32)
    return _hashing_embedding(normalized)


def cosine_score(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    vector_a = np.asarray(vector_a)
    vector_b = np.asarray(vector_b)
    if vector_a.ndim == 1:
        vector_a = vector_a.reshape(1, -1)
    if vector_b.ndim == 1:
        vector_b = vector_b.reshape(1, -1)
    return float(cosine_similarity(vector_a, vector_b)[0][0])


def _set_overlap_ratio(source: Sequence[str], target: Sequence[str]) -> float:
    source_set = {item.lower() for item in source}
    target_set = {item.lower() for item in target}
    if not target_set:
        return 0.0
    return len(source_set & target_set) / len(target_set)


def _education_score(jd_level: str, resume_level: str) -> float:
    levels = {"Not specified": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}
    return 1.0 if levels.get(resume_level, 0) >= levels.get(jd_level, 0) else 0.0


def _experience_score(jd_years: float, resume_years: float) -> float:
    if jd_years <= 0:
        return 0.5 if resume_years > 0 else 0.0
    if resume_years <= 0:
        return 0.0
    return float(min(resume_years / jd_years, 1.0))


def build_skill_gap_analysis(jd_profile: Dict[str, Any], resume_profile: Dict[str, Any]) -> Dict[str, Any]:
    jd_skills = set(jd_profile.get("skills", []))
    resume_skills = set(resume_profile.get("skills", []))
    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)
    coverage = round((len(matched) / len(jd_skills) * 100) if jd_skills else 0.0, 1)
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "coverage_pct": coverage,
    }


def score_candidate(jd_profile: Dict[str, Any], resume_profile: Dict[str, Any]) -> Dict[str, Any]:
    jd_embedding = get_embedding(jd_profile["raw_text"])
    resume_embedding = get_embedding(resume_profile["raw_text"])

    semantic_score = max(0.0, min(1.0, (cosine_score(jd_embedding, resume_embedding) + 1.0) / 2.0))
    skill_overlap = _set_overlap_ratio(resume_profile.get("skills", []), jd_profile.get("skills", []))
    experience_score = _experience_score(jd_profile.get("experience_years", 0.0), resume_profile.get("experience_years", 0.0))
    education_score = _education_score(jd_profile.get("education", "Not specified"), resume_profile.get("education", "Not specified"))
    certification_score = _set_overlap_ratio(resume_profile.get("certifications", []), jd_profile.get("certifications", []))

    combined_score = (
        0.40 * semantic_score
        + 0.30 * skill_overlap
        + 0.15 * experience_score
        + 0.10 * education_score
        + 0.05 * certification_score
    )

    return {
        "candidate_name": resume_profile.get("source_name", "candidate"),
        "score": round(float(combined_score * 100), 1),
        "semantic_score": round(float(semantic_score * 100), 1),
        "skill_overlap": round(float(skill_overlap * 100), 1),
        "experience_score": round(float(experience_score * 100), 1),
        "education_score": round(float(education_score * 100), 1),
        "certification_score": round(float(certification_score * 100), 1),
        "candidate_profile": resume_profile,
        "skill_gap": build_skill_gap_analysis(jd_profile, resume_profile),
    }


def rank_candidates(jd_profile: Dict[str, Any], resume_profiles: Sequence[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    if not resume_profiles:
        return []

    scored = [score_candidate(jd_profile, profile) for profile in resume_profiles]
    scored.sort(key=lambda item: item["score"], reverse=True)

    if faiss is not None and len(resume_profiles) > 1:
        embeddings = np.vstack([get_embedding(profile["raw_text"]) for profile in resume_profiles]).astype(np.float32)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        query = get_embedding(jd_profile["raw_text"]).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        _, indices = index.search(query, min(top_k, len(resume_profiles)))
        scored_by_name = {item["candidate_name"]: item for item in scored}
        ordered = [scored_by_name[resume_profiles[int(idx)]["source_name"]] for idx in indices[0]]
        return ordered[:top_k]

    return scored[:top_k]


def format_explanation(jd_profile: Dict[str, Any], result: Dict[str, Any]) -> str:
    gap = result["skill_gap"]
    matched = ", ".join(gap["matched_skills"]) if gap["matched_skills"] else "no direct skill matches"
    missing = ", ".join(gap["missing_skills"][:6]) if gap["missing_skills"] else "none"
    return (
        f"{result['candidate_name']} achieved a {result['score']}% fit score. "
        f"Shared skills: {matched}. Coverage of JD skills is {gap['coverage_pct']}%. "
        f"Primary gaps: {missing}. Experience fit: {result['experience_score']}%, "
        f"education fit: {result['education_score']}%, certification fit: {result['certification_score']}%."
    )


def build_profile_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **profile,
        "summary": {
            "skills_count": len(profile.get("skills", [])),
            "certifications_count": len(profile.get("certifications", [])),
            "experience_years": profile.get("experience_years", 0.0),
            "education": profile.get("education", "Not specified"),
        },
    }


def profile_text(profile: Dict[str, Any]) -> str:
    sections = []
    if profile.get("skills"):
        sections.append("Skills: " + ", ".join(profile["skills"]))
    if profile.get("certifications"):
        sections.append("Certifications: " + ", ".join(profile["certifications"]))
    if profile.get("education"):
        sections.append(f"Education: {profile['education']}")
    if profile.get("experience_years"):
        sections.append(f"Experience: {profile['experience_years']} years")
    return " | ".join(sections) if sections else "No structured profile detected"


# Filename and content validation helpers
JD_FILENAME_KEYWORDS = ["job", "jd", "job_description", "job-description", "jobdescription", "opening", "position"]
RESUME_FILENAME_KEYWORDS = ["resume", "cv", "candidate", "application", "applicant"]

JD_SECTION_KEYWORDS = ["responsibilities", "requirements", "qualifications", "skills required", "job description", "role"]
RESUME_SECTION_KEYWORDS = ["experience", "education", "skills", "summary", "contact", "projects"]


def classify_file_type(filename: str, text: str) -> Dict[str, Any]:
    """Heuristic classification using filename tokens and simple content section checks.

    Returns a dict with: filename_hint ('jd'|'resume'|'unknown'), content_hint, final_label, confidence (0-1),
    and counts of detected sections. Content has higher weight than filename to avoid adversarial names.
    """
    fname = (filename or "").lower()
    lowered = (text or "").lower()

    filename_hint = "unknown"
    if any(tok in fname for tok in JD_FILENAME_KEYWORDS):
        filename_hint = "jd"
    if any(tok in fname for tok in RESUME_FILENAME_KEYWORDS):
        # prefer resume if resume keyword present
        filename_hint = "resume"

    jd_sections = sum(1 for kw in JD_SECTION_KEYWORDS if kw in lowered)
    resume_sections = sum(1 for kw in RESUME_SECTION_KEYWORDS if kw in lowered)
    email_found = bool(EMAIL_PATTERN.search(lowered))
    phone_found = bool(PHONE_PATTERN.search(lowered))

    content_hint = "unknown"
    # simple rules: resumes usually have two or more resume sections or contact info
    if resume_sections >= 2 or email_found or phone_found:
        content_hint = "resume"
    # JDs likely contain at least one JD section token and reasonable length
    if jd_sections >= 1 and len(lowered) > 200:
        content_hint = "jd"

    # compute a confidence score: content (70%), filename (30%)
    content_score = min(1.0, (resume_sections + jd_sections) / 3.0)
    filename_score = 1.0 if filename_hint in ("jd", "resume") else 0.0
    # bump content_score if contact info present (strong signal for resume)
    if email_found or phone_found:
        content_score = max(content_score, 0.6)

    # final label: prefer content_hint when available
    final_label = content_hint if content_hint != "unknown" else filename_hint
    confidence = round(0.7 * content_score + 0.3 * filename_score, 2)

    return {
        "filename": filename or "",
        "filename_hint": filename_hint,
        "content_hint": content_hint,
        "jd_sections": jd_sections,
        "resume_sections": resume_sections,
        "email_found": email_found,
        "phone_found": phone_found,
        "final_label": final_label,
        "confidence": confidence,
    }


def validate_document(filename: str, text: str, expected: Optional[str] = None, min_confidence: float = 0.5) -> Dict[str, Any]:
    """Validate a document by filename and content. If `expected` is provided ('jd' or 'resume'),
    returns `is_valid` indicating whether the document likely matches expected type and confidence >= min_confidence.
    """
    info = classify_file_type(filename, text)
    # Strict mode: require both filename and content to agree when an expected type is provided.
    is_valid = True
    if expected is not None:
        is_valid = (
            info["filename_hint"] == expected
            and info["content_hint"] == expected
            and info["final_label"] == expected
            and info["confidence"] >= float(min_confidence)
        )
    else:
        is_valid = info["confidence"] >= float(min_confidence)
    return {**info, "is_valid": is_valid, "min_confidence": float(min_confidence)}

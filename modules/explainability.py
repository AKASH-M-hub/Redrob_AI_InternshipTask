from .pipeline import build_skill_gap_analysis, format_explanation


def generate_reason(score, jd_profile=None, result=None):
    if jd_profile is not None and result is not None:
        return format_explanation(jd_profile, result)

    if score > 80:
        return "Excellent match"
    if score > 60:
        return "Good match"
    return "Low match"


def analyze_skill_gap(jd_skills, resume_skills):
    return build_skill_gap_analysis({"skills": jd_skills}, {"skills": resume_skills})
from .pipeline import cosine_score, rank_candidates


def calculate_score(jd_embedding, resume_embedding):
    return cosine_score(jd_embedding, resume_embedding)
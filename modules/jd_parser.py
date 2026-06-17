from .pipeline import analyze_document


def parse_jd(text):
    return analyze_document(text, source_name="job_description")
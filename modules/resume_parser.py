from .pipeline import analyze_document, extract_text_from_bytes, read_uploaded_file


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as handle:
        return extract_text_from_bytes(handle.read(), pdf_path)


def extract_text_from_upload(uploaded_file):
    return read_uploaded_file(uploaded_file)


def parse_resume(text, source_name="resume"):
    return analyze_document(text, source_name=source_name)
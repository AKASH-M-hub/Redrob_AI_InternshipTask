# app.py

from __future__ import annotations

from io import BytesIO
from importlib import import_module

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    pipeline = import_module("modules.pipeline")
except Exception as import_error:
    raise RuntimeError(f"Unable to import modules.pipeline: {import_error}") from import_error

analyze_document = getattr(pipeline, "analyze_document")
build_profile_summary = getattr(pipeline, "build_profile_summary")
extract_text_from_bytes = getattr(pipeline, "extract_text_from_bytes")
format_explanation = getattr(pipeline, "format_explanation")
profile_text = getattr(pipeline, "profile_text")
rank_candidates = getattr(pipeline, "rank_candidates")
read_uploaded_file = getattr(pipeline, "read_uploaded_file")

def _fallback_validate_document(filename, text, expected=None, min_confidence=0.5):
    label = "resume" if filename and any(token in filename.lower() for token in ("resume", "cv")) else "jd"
    return {
        "filename": filename or "",
        "filename_hint": label,
        "content_hint": label,
        "jd_sections": 0,
        "resume_sections": 0,
        "email_found": False,
        "phone_found": False,
        "final_label": label,
        "confidence": 1.0,
        "is_valid": expected is None or label == expected,
        "min_confidence": float(min_confidence),
    }

validate_document = getattr(pipeline, "validate_document", _fallback_validate_document)


st.set_page_config(page_title="AI Talent Match", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap');

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(244,162,97,0.20), transparent 24%),
            radial-gradient(circle at top right, rgba(42,157,143,0.18), transparent 22%),
            linear-gradient(180deg, #07111f 0%, #0f172a 48%, #111827 100%);
        color: #f8fafc;
        font-family: 'Space Grotesk', sans-serif;
    }
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(12, 18, 32, 0.82);
        box-shadow: 0 24px 70px rgba(0,0,0,0.34);
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        margin: 0;
        font-family: 'Sora', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        line-height: 1.02;
        background: linear-gradient(90deg, #ffffff 0%, #f4a261 48%, #2a9d8f 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .hero p {
        margin-top: 0.5rem;
        color: #cbd5e1;
        max-width: 68rem;
        font-size: 1rem;
    }
    h2, h3, .stMarkdown, .stDataFrame, .stMetric, .stSelectbox, .stSlider {
        font-family: 'Space Grotesk', sans-serif;
    }
    .panel {
        padding: 1rem 1rem 0.6rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(15, 23, 42, 0.85);
        margin-bottom: 1rem;
    }
    .section-label {
        font-family: 'Sora', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.78rem;
        color: #f4a261;
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>AI Talent Match</h1>
        <p>An end-to-end recruiter cockpit with skill extraction, NER, experience detection, education matching, certification detection, FAISS-backed retrieval, explainability, and skill gap analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def parse_uploaded_document(uploaded_file, source_name):
    file_bytes = uploaded_file.getvalue()
    text = extract_text_from_bytes(file_bytes, uploaded_file.name)
    return analyze_document(text, source_name=source_name)


def build_pdf_report(jd_profile, results_df, selected_result=None):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AI Talent Match Title",
        parent=styles["Title"],
        textColor=colors.HexColor("#111827"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "AI Talent Match Heading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "AI Talent Match Body",
        parent=styles["BodyText"],
        leading=14,
        spaceAfter=8,
    )

    elements = [
        Paragraph("AI Talent Match Report", title_style),
        Paragraph(
            "This report summarizes the JD analysis, ranked candidates, and the top-k retrieval output used by the matching engine.",
            body_style,
        ),
        Spacer(1, 0.12 * inch),
        Paragraph("Top-K Retrieval", heading_style),
        Paragraph(
            "Top-k retrieval means the system ranks all candidate resumes by relevance and returns only the k strongest matches. "
            "For example, if top-k is 5, the app surfaces the five most relevant resumes for the given job description.",
            body_style,
        ),
        Paragraph(
            f"Job skills detected: {len(jd_profile['skills'])}. Certifications detected: {len(jd_profile['certifications'])}. Experience requested: {jd_profile['experience_years']} years.",
            body_style,
        ),
    ]

    elements.append(Paragraph("Ranked Candidates", heading_style))

    table_rows = [["Candidate", "Score", "Semantic", "Skill overlap", "Experience", "Education", "Certifications"]]
    for _, row in results_df.iterrows():
        table_rows.append(
            [
                str(row["candidate"]),
                f"{row['score']:.1f}",
                f"{row['semantic']:.1f}",
                f"{row['skill_overlap']:.1f}",
                f"{row['experience']:.1f}",
                f"{row['education']:.1f}",
                f"{row['certifications']:.1f}",
            ]
        )

    ranking_table = Table(table_rows, repeatRows=1)
    ranking_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#e2e8f0")]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    elements.append(ranking_table)

    if selected_result is not None:
        elements.extend(
            [
                Spacer(1, 0.18 * inch),
                Paragraph("Selected Candidate Summary", heading_style),
                Paragraph(f"Candidate: {selected_result['candidate_name']}", body_style),
                Paragraph(f"Fit score: {selected_result['score']}%", body_style),
                Paragraph(format_explanation(jd_profile, selected_result), body_style),
                Paragraph(
                    "Matched skills: " + (", ".join(selected_result["skill_gap"]["matched_skills"]) or "None"),
                    body_style,
                ),
                Paragraph(
                    "Missing skills: " + (", ".join(selected_result["skill_gap"]["missing_skills"]) or "None"),
                    body_style,
                ),
            ]
        )

    document.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


with st.sidebar:
    st.subheader("Pipeline Controls")
    top_k = st.slider("Top-K retrieval", 1, 10, 5)
    match_threshold = st.slider("Minimum match threshold", 0, 100, 0)
    st.caption("Version 2 to 5 modules are active in the workflow.")
    st.markdown(
        "- Skill extraction\n- NER\n- Experience detection\n- Education matching\n- Certification detection\n- FAISS vector search\n- Top-K retrieval\n- LLM-style explainability\n- Skill gap analysis"
    )
    with st.expander("What is top-k retrieval?"):
        st.write(
            "Top-k retrieval ranks every resume against the JD and returns only the top k best matches. "
            "It is the standard way to avoid showing the user every candidate when only the most relevant ones matter."
        )

jd_file = st.file_uploader(
    "Upload Job Description (single file only)",
    type=["pdf", "txt"],
    accept_multiple_files=False,
    help="Upload exactly one job description file. Multiple JDs are not supported in this upload box.",
)
resume_files = st.file_uploader("Upload Resumes", type=["pdf", "txt"], accept_multiple_files=True)

if jd_file and resume_files:
    try:
        # Validate JD file by filename + content before heavy parsing
        jd_bytes = jd_file.getvalue()
        jd_text = extract_text_from_bytes(jd_bytes, jd_file.name)
        jd_check = validate_document(jd_file.name, jd_text, expected="jd", min_confidence=0.5)
        if not jd_check.get("is_valid"):
            st.warning(
                f"Uploaded JD ({jd_file.name}) looks suspicious: detected label={jd_check.get('final_label')} (confidence={jd_check.get('confidence')}).\n"
                "If this is not a JD file, parsing may fail or produce poor results."
            )
            st.error("Validation failed: upload a JD file whose filename and content both look like a JD.")
            st.stop()

        # Validate resumes and filter out files that clearly do not look like resumes
        valid_resume_files = []
        skipped_resumes = []
        for rf in resume_files:
            try:
                rb = rf.getvalue()
                rtext = extract_text_from_bytes(rb, rf.name)
            except Exception:
                rtext = ""
            info = validate_document(rf.name, rtext, expected="resume", min_confidence=0.5)
            if info.get("is_valid"):
                valid_resume_files.append(rf)
            else:
                skipped_resumes.append({"name": rf.name, "label": info.get("final_label"), "confidence": info.get("confidence")})

        if skipped_resumes:
            st.warning(f"{len(skipped_resumes)} uploaded file(s) did not look like resumes and were skipped by default.")
            for s in skipped_resumes:
                st.write(f"- {s['name']}: detected={s['label']}, confidence={s['confidence']}")
            st.info("Skipped files will not be included in matching. Rename them to match the correct resume format and upload again.")

        if not valid_resume_files:
            st.error("No valid resume files to process. Upload at least one resume file.")
            st.stop()

        with st.spinner("Parsing documents and ranking candidates..."):
            # Parse JD and the validated resume files
            jd_profile = analyze_document(jd_text, source_name=jd_file.name)
            resume_profiles = [parse_uploaded_document(resume_file, resume_file.name) for resume_file in valid_resume_files]
            ranked = rank_candidates(jd_profile, resume_profiles, top_k=top_k)

        left, right = st.columns([1.0, 1.35], gap="large")

        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Job Description Analysis</div>', unsafe_allow_html=True)
            st.metric("Skills detected", len(jd_profile["skills"]))
            st.metric("Certifications detected", len(jd_profile["certifications"]))
            st.metric("Experience requested", f"{jd_profile['experience_years']} years")
            st.write("**Skills:**", ", ".join(jd_profile["skills"]) if jd_profile["skills"] else "None detected")
            st.write("**Education:**", jd_profile["education"])
            st.write("**Certifications:**", ", ".join(jd_profile["certifications"]) if jd_profile["certifications"] else "None detected")
            st.write("**Entities:**", jd_profile["entities"])
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Top Matches</div>', unsafe_allow_html=True)
            rows = []
            for item in ranked:
                profile = item["candidate_profile"]
                rows.append(
                    {
                        "candidate": item["candidate_name"],
                        "score": item["score"],
                        "semantic": item["semantic_score"],
                        "skill_overlap": item["skill_overlap"],
                        "experience": item["experience_score"],
                        "education": item["education_score"],
                        "certifications": item["certification_score"],
                        "summary": profile_text(profile),
                        "reason": format_explanation(jd_profile, item),
                    }
                )

            results_df = pd.DataFrame(rows)
            if match_threshold > 0:
                results_df = results_df[results_df["score"] >= float(match_threshold)].reset_index(drop=True)
            st.dataframe(
                results_df[["candidate", "score", "semantic", "skill_overlap", "experience", "education", "certifications"]],
                use_container_width=True,
                hide_index=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        if ranked:
            selected_candidate = st.selectbox("Inspect a candidate", [item["candidate_name"] for item in ranked])
            selected = next(item for item in ranked if item["candidate_name"] == selected_candidate)
            selected_profile = selected["candidate_profile"]
            gap = selected["skill_gap"]
            selected_reason = format_explanation(jd_profile, selected)

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Selected Candidate</div>', unsafe_allow_html=True)
                st.metric("Overall fit", f"{selected['score']}%")
                st.write(selected_reason)
                st.write("**Matched skills:**", ", ".join(gap["matched_skills"]) if gap["matched_skills"] else "None")
                st.write("**Missing skills:**", ", ".join(gap["missing_skills"]) if gap["missing_skills"] else "None")
                st.write("**Extra skills:**", ", ".join(gap["extra_skills"]) if gap["extra_skills"] else "None")
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Resume Profile</div>', unsafe_allow_html=True)
                summary = build_profile_summary(selected_profile)
                st.write("**Skills:**", ", ".join(summary["skills"]) if summary["skills"] else "None detected")
                st.write("**Certifications:**", ", ".join(summary["certifications"]) if summary["certifications"] else "None detected")
                st.write("**Education:**", summary["education"])
                st.write("**Experience:**", summary["experience_years"])
                st.write("**Entities:**", summary["entities"])
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Ranking Snapshot</div>', unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            pdf_bytes = build_pdf_report(jd_profile, results_df, selected_result=selected)
            st.download_button(
                "Download rankings CSV",
                results_df.to_csv(index=False).encode("utf-8"),
                file_name="talent_match_rankings.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download rankings PDF",
                pdf_bytes,
                file_name="talent_match_rankings.pdf",
                mime="application/pdf",
            )
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Pipeline failed: {error}")
else:
    st.info("Upload a job description and at least one resume to run the matching pipeline.")
    st.markdown(
        """
        - Version 2: Skill Extraction, NER, Experience Detection, Education Matching, Certification Detection
        - Version 3: FAISS Vector Search and Top-K Retrieval
        - Version 4: Explainable match narratives
        - Version 5: Skill Gap Analysis
        """
    )
# AI_TalentMatch

AI_TalentMatch is a proof-of-concept end-to-end talent matching pipeline that analyzes job descriptions (JDs) and candidate resumes to produce ranked candidate suggestions, explanations, and skill-gap analysis. It combines heuristic NLP extraction with semantic embeddings and optional FAISS vector search to deliver fast Top-K retrieval and human-friendly explainability.

**Key Features**
- **Skill extraction:** Detects explicit and implicit skills from JDs and resumes.
- **Named-entity heuristics:** Extracts basic PII and organization/education entities.
- **Experience detection:** Estimates years of experience from resume text.
- **Education matching:** Maps education levels and scores closeness to JD requirements.
- **Certification detection:** Identifies common certification mentions.
- **Embeddings & vector search:** Uses `sentence-transformers` embeddings with optional `faiss-cpu` for fast nearest-neighbor retrieval.
- **Top-K retrieval & ranking:** Combines semantic similarity, skill overlap, experience, education and certification signals into a composite score.
- **Explainability & skill-gap analysis:** Human-readable reasons for scores and a list of missing skills relative to the JD.
- **Streamlit UI:** Interactive web interface to upload JDs/resumes, inspect candidates, and download CSV/PDF reports.

**Repository Layout**
- **app.py**: Streamlit app and UI (entrypoint).
- **requirements.txt**: Python dependencies.
- **modules/pipeline.py**: Core pipeline (parsing, extraction, embeddings, scoring, ranking).
- **modules/explainability.py**: Explanation helpers.
- **modules/**: Other helper modules (`embeddings.py`, `ranker.py`, `jd_parser.py`, `resume_parser.py`).
- **data/jds/** and **data/resumes/**: Example JD and resume files for testing.

Quick links:
- Main app: [app.py](app.py)
- Pipeline engine: [modules/pipeline.py](modules/pipeline.py)
- Explainability helpers: [modules/explainability.py](modules/explainability.py)

## Quickstart (Local)

Prerequisites
- Python 3.10+ (3.11 recommended)
- Recommended: create and use a virtual environment

Install dependencies

```bash
python -m venv .venv
.\.venv\Scripts\activate    # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

Run the Streamlit app

```bash
# Use the -m form to ensure the right interpreter runs Streamlit
python -m streamlit run app.py
```

Notes
- On first run the `sentence-transformers` model will be downloaded (internet connection required).
- If `streamlit` is not on your PATH, always use `python -m streamlit run app.py`.
- Optional dependencies:
  - `faiss-cpu` for faster similarity search (fallback implemented when missing).
  - `PyMuPDF` (`fitz`) for PDF parsing of uploaded resumes.
  - `reportlab` for PDF export of ranking reports (already included in `requirements.txt`).

## Running a smoke test (headless)

You can run a quick smoke test by importing pipeline utilities in a Python REPL:

```bash
python -c "from modules.pipeline import analyze_document, rank_candidates; print('pipeline imports OK')"
```

## Docker (basic)

Create a `Dockerfile` (example):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt
EXPOSE 8501
CMD ["python","-m","streamlit","run","app.py","--server.port","8501","--server.headless","true"]
```

Build and run

```bash
docker build -t ai_talentmatch:latest .
docker run --rm -p 8501:8501 ai_talentmatch:latest
```

Notes for production
- Use a file-based or networked vector index (FAISS or Milvus) for scaling beyond a single process.
- Move heavy models to a dedicated model service (REST/gRPC) or use an embeddings API to avoid model downloads and memory pressure in the web process.
- Add caching for sentence-transformer model and generated embeddings to avoid repeated downloads and to speed up processing.
- Securely handle PII: apply redaction or minimization when storing or transmitting resumes.

## Deployment options
- Streamlit Cloud: easiest for demos — connect the repo and set required secrets (HF_TOKEN if private models are used).
- Containerized (Docker) behind a reverse proxy (Nginx) with TLS for production.
- Kubernetes for scale: host a persistent vector store (FAISS on NFS or Milvus), and run workers for heavy embedding tasks.

## Configuration & tuning
- Adjust ranking weights and scoring logic in `modules/pipeline.py` (`score_candidate` and related helpers).
- Add or replace heuristic extractors with spaCy/transformer-based NER models in `modules/` for improved accuracy.

## Troubleshooting
- "`streamlit` command not found": run `python -m streamlit run app.py` with the same interpreter used to install packages.
- Large model downloads: ensure your environment has internet access, or pre-download models in a shared cache.
- FAISS errors: if `faiss-cpu` is unavailable on Windows, use the pure-Python fallback implemented in the pipeline.
- PDF export missing or failing: ensure `reportlab` is installed (`pip install reportlab`).

## Next steps / Recommended improvements
- Replace heuristic NER with a transformer NER (spaCy or Hugging Face) for reliable entity extraction.
- Add unit/integration tests and a CI pipeline (GitHub Actions) to run static checks and smoke tests.
- Add Docker Compose with a managed vector database (Milvus or Weaviate) for persistence.
- Implement authentication, logging, and usage quotas for a multi-user deployment.

If you'd like, I can add a `Dockerfile`, CI workflow, or convert the pipeline to use a hosted embeddings API — tell me which next.

---

File created: [README.md](README.md)
# AI Talent Match

Run the end-to-end talent matching pipeline with:

```bash
streamlit run app.py
```

The app now supports:

- Skill extraction
- NER-style entity extraction
- Experience detection
- Education matching
- Certification detection
- FAISS vector search
- Top-K retrieval
- Explainability
- Skill gap analysis

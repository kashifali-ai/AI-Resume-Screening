"""FastAPI app: serves the landing page and the /api/screen endpoint."""

from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from config import get_settings
from llm_client import LLMError
from logging_config import configure_logging, get_logger
from report_csv import bulk_to_csv
from resume_parser import EmptyResumeError, UnsupportedFileType, extract_text
from screening import screen, screen_bulk

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger(__name__)

app = FastAPI(title="Resume Screening (LLM + Embeddings)")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/")
def landing_page():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mock_llm": settings.mock_llm,
        "llm_provider": "mock" if settings.mock_llm else settings.llm_provider,
        "llm_model": "mock-rule-based" if settings.mock_llm else settings.gemini_model,
        "gemini_key_configured": bool(settings.gemini_api_key),
        "embedding_model": settings.embedding_model,
    }


@app.post("/api/screen")
async def screen_endpoint(
    job_description: str = Form(...),
    resume: UploadFile = File(...),
):
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="Paste a job description.")

    data = await resume.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > settings.max_file_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File too large (max {settings.max_file_mb} MB)."
        )

    try:
        text = extract_text(resume.filename or "", data)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=415, detail=str(e))
    except EmptyResumeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        report = screen(job_description, text, settings=settings)
    except LLMError as e:
        log.error("LLM failure: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # noqa: BLE001 — surface anything else cleanly
        log.exception("Screening failed")
        raise HTTPException(status_code=500, detail=f"Screening failed: {e}")

    return report.model_dump()


@app.post("/api/screen-bulk")
async def screen_bulk_endpoint(
    job_description: str = Form(...),
    resumes: list[UploadFile] = File(...),
    response_format: str = Form("json"),  # "json" (default) or "csv"
):
    """Evaluate many resumes against ONE job description. The JD is extracted
    once (cached); each resume is screened independently and results are ranked
    by score, highest first."""
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="Paste a job description.")
    if not resumes:
        raise HTTPException(status_code=400, detail="Upload at least one resume.")
    if len(resumes) > settings.max_bulk_resumes:
        raise HTTPException(
            status_code=413,
            detail=f"Too many resumes ({len(resumes)}). "
                   f"Max {settings.max_bulk_resumes} per request.",
        )

    parsed: list[tuple[str, str]] = []
    prefailed: list[tuple[str, str]] = []
    for f in resumes:
        name = f.filename or "resume"
        data = await f.read()
        if not data:
            prefailed.append((name, "The uploaded file is empty."))
            continue
        if len(data) > settings.max_file_mb * 1024 * 1024:
            prefailed.append((name, f"File too large (max {settings.max_file_mb} MB)."))
            continue
        try:
            parsed.append((name, extract_text(name, data)))
        except (UnsupportedFileType, EmptyResumeError) as e:
            prefailed.append((name, str(e)))

    try:
        response = screen_bulk(
            job_description, parsed, prefailed=prefailed, settings=settings
        )
    except LLMError as e:
        log.error("LLM failure: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # noqa: BLE001 — surface anything else cleanly
        log.exception("Bulk screening failed")
        raise HTTPException(status_code=500, detail=f"Bulk screening failed: {e}")

    if response_format.lower() == "csv":
        role = (response.role_title or "results").replace(" ", "_")
        return PlainTextResponse(
            bulk_to_csv(response),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="screening_{role}.csv"'},
        )
    return response.model_dump()

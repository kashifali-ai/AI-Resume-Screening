"""FastAPI app: serves the landing page and the /api/screen endpoint."""

from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse

from config import get_settings
from llm_client import LLMError
from logging_config import configure_logging, get_logger
from resume_parser import EmptyResumeError, UnsupportedFileType, extract_text
from screening import screen

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
        "llm_provider": settings.llm_provider,
        "llm_model": settings.gemini_model,
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

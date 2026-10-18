"""
FastAPI backend.   uvicorn app.server:app --port 8000   →  http://localhost:8000

Jobs persist to data/jobs/<id>.json so finished reels survive a restart.
Analysis (slow) and render (fast) are separate, so /recut re-renders with new
weights / length / manual picks in seconds without re-transcribing or re-scoring.
"""

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import PipelineConfig, DOMAIN_PRESETS, normalise, load_weights
from src.pipeline import Analysis, analyse, render
from src.download import download

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
for d in ("input", "output", "transcripts", "jobs"):
    (DATA / d).mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 1500
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mp3", ".m4a", ".wav"}

app = FastAPI(title="Highlight Generator")
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _persist(job: dict) -> None:
    (DATA / "jobs" / f"{job['id']}.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")


def _set(job_id: str, **kw) -> None:
    with LOCK:
        JOBS[job_id].update(kw)
        if JOBS[job_id].get("status") in ("done", "error"):
            _persist(JOBS[job_id])


def _load_jobs() -> None:
    for p in (DATA / "jobs").glob("*.json"):
        if p.name.endswith(".analysis.json"):
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            if j.get("status") == "running":
                j["status"], j["message"] = "error", "server restarted during processing"
            JOBS[j["id"]] = j
        except (json.JSONDecodeError, KeyError):
            continue


_load_jobs()


def _cfg(job: dict, weights: dict | None = None) -> PipelineConfig:
    return PipelineConfig(domain=job["domain"], target_highlight_duration_s=job["duration"],
                          asr_backend=job.get("asr", "groq"), use_visual=job.get("visual", True),
                          burn_subtitles=job.get("burn", False), weights=weights or job.get("weights") or {},
                          output_dir=str(DATA / "output" / job["id"]),
                          transcript_cache_dir=str(DATA / "transcripts"), jobs_dir=str(DATA / "jobs"))


def _public(job: dict) -> dict:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _finish(job_id: str, result: dict) -> None:
    manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
    _set(job_id, status="done", stage="done", percent=100, message="ready",
         weights=manifest["meta"]["weights"], manifest=manifest,
         files={k: f"/api/jobs/{job_id}/file/{k}" for k in ("video", "srt", "chapters", "manifest_txt", "labels")},
         _paths={k: result[k] for k in ("video", "srt", "chapters", "manifest_txt", "labels", "manifest_json")})


def _worker(job_id: str, video_path: str) -> None:
    job = JOBS[job_id]
    prog = lambda stage, pct, msg: _set(job_id, stage=stage, percent=round(pct), message=msg)
    try:
        cfg = _cfg(job)
        a = analyse(video_path, cfg, progress=prog)
        _set(job_id, _analysis=str(Path(cfg.jobs_dir) / f"{a.name}.analysis.json"), source_duration=a.source_duration_s)
        _finish(job_id, render(a, cfg, progress=prog))
    except Exception as e:
        _set(job_id, status="error", message=str(e)[:400])


def _start(job: dict, video_path: str) -> dict:
    with LOCK:
        JOBS[job["id"]] = job
    threading.Thread(target=_worker, args=(job["id"], video_path), daemon=True).start()
    return {"id": job["id"]}


def _new_job(filename: str, domain: str, duration: int, asr: str, visual: bool, burn: bool) -> dict:
    if domain not in DOMAIN_PRESETS:
        raise HTTPException(400, "unknown domain preset")
    if not 15 <= duration <= 1800:
        raise HTTPException(400, "duration must be 15–1800 seconds")
    return {"id": uuid.uuid4().hex[:10], "status": "running", "stage": "queued", "percent": 0,
            "message": "starting", "filename": filename, "domain": domain, "duration": duration,
            "asr": asr, "visual": visual, "burn": burn, "created": time.time(),
            "weights": load_weights(domain) if visual else normalise({**load_weights(domain), "visual": 0})}


@app.get("/api/presets")
def presets():
    return {d: load_weights(d) for d in DOMAIN_PRESETS}


@app.get("/api/jobs")
def list_jobs():
    return sorted((_public({k: v for k, v in j.items() if k != "manifest"}) for j in JOBS.values()),
                  key=lambda j: j.get("created", 0), reverse=True)[:50]


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), domain: str = Form("general"), duration: int = Form(120),
                     asr: str = Form("groq"), visual: bool = Form(True), burn: bool = Form(False)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"unsupported file type {ext or '(none)'}")
    job = _new_job(file.filename, domain, duration, asr, visual, burn)
    dest = DATA / "input" / f"{job['id']}_{Path(file.filename).stem[:60]}{ext}"
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                out.close(); dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
            out.write(chunk)
    return _start(job, str(dest))


class UrlJob(BaseModel):
    url: str
    domain: str = "general"
    duration: int = 120
    asr: str = "groq"
    visual: bool = True
    burn: bool = False


@app.post("/api/jobs/from-url")
def create_job_from_url(body: UrlJob):
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(400, "url must start with http(s)://")
    job = _new_job(body.url, body.domain, body.duration, body.asr, body.visual, body.burn)

    def run():
        try:
            _set(job["id"], stage="download", percent=1, message="downloading video")
            path = download(body.url, str(DATA / "input"))
            _worker(job["id"], path)
        except Exception as e:
            _set(job["id"], status="error", message=str(e)[:400])

    with LOCK:
        JOBS[job["id"]] = job
    threading.Thread(target=run, daemon=True).start()
    return {"id": job["id"]}


class Recut(BaseModel):
    weights: dict[str, float] | None = None
    duration: int | None = None
    force_include: list[int] = []
    force_exclude: list[int] = []
    burn: bool | None = None


@app.post("/api/jobs/{job_id}/recut")
def recut(job_id: str, body: Recut):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done" or not job.get("_analysis"):
        raise HTTPException(409, "job is not ready to re-cut")
    if body.duration:
        job["duration"] = body.duration
    if body.burn is not None:
        job["burn"] = body.burn
    weights = normalise(body.weights) if body.weights else job.get("weights")
    _set(job_id, status="running", stage="fusion", percent=86, message="re-cutting")

    def run():
        try:
            a = Analysis.load(job["_analysis"])
            cfg = _cfg(job, weights)
            prog = lambda stage, pct, msg: _set(job_id, stage=stage, percent=round(pct), message=msg)
            _finish(job_id, render(a, cfg, body.force_include, body.force_exclude, prog))
        except Exception as e:
            _set(job_id, status="error", message=str(e)[:400])

    threading.Thread(target=run, daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return _public(job)


@app.get("/api/jobs/{job_id}/file/{kind}")
def get_file(job_id: str, kind: str):
    job = JOBS.get(job_id)
    paths = (job or {}).get("_paths") or {}
    if job is None or job.get("status") != "done" or kind not in paths:
        raise HTTPException(404, "file not ready")
    media = {"video": "video/mp4", "srt": "text/plain", "chapters": "text/plain",
             "manifest_txt": "text/plain", "labels": "application/json"}[kind]
    name = {"video": "highlight.mp4", "srt": "highlight.srt", "chapters": "chapters.txt",
            "manifest_txt": "manifest.txt", "labels": "labels.json"}[kind]
    return FileResponse(paths[kind], media_type=media, filename=name)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(404, "job not found")
    (DATA / "jobs" / f"{job_id}.json").unlink(missing_ok=True)
    shutil.rmtree(DATA / "output" / job_id, ignore_errors=True)
    return {"deleted": job_id}


app.mount("/", StaticFiles(directory=str(ROOT / "app" / "static"), html=True), name="static")

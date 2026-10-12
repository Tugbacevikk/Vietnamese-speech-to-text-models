"""HTTP API (FastAPI).

    POST /jobs          ses dosyası yükle -> 202 + job_id (iş kuyruğa girer)
                        form alanları: file (zorunlu), language (vi | tr | auto, varsayılan auto),
                        callback_url (isteğe bağlı: iş bitince buraya POST edilir)
    GET  /jobs/{id}     durum; bittiyse sonuç (konuşmacılı satırlar)
    GET  /health        veritabanı ve kuyruk erişimi, durum sayıları

Çalıştırma:  uvicorn viet_asr.interface.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from viet_asr.application.jobs import SubmitJob
from viet_asr.application.ports import JobQueue, JobStore
from viet_asr.application.serialization import record_to_dict
from viet_asr.domain import AudioJob, JobStatus, UnsupportedLanguageError
from viet_asr.interface.settings import Settings

log = logging.getLogger(__name__)
_ALLOWED = {".wav", ".flac", ".ogg", ".mp3", ".opus"}


def create_app(store: JobStore, queue: JobQueue, settings: Settings,
               health_checks: dict | None = None) -> FastAPI:
    api = FastAPI(title="Ses -> metin + konuşmacı (vi, tr)", version="0.2.0")
    submit = SubmitJob(store, queue, settings.languages)
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    @api.post("/jobs", status_code=202)
    def create_job(file: UploadFile | None = File(None), language: str = Form("auto"),
                   callback_url: str | None = Form(None), audio_path: str | None = Form(None)):
        if callback_url and not callback_url.startswith(("http://", "https://")):
            raise HTTPException(422, "callback_url http:// ya da https:// ile başlamalı")
        job_id = uuid.uuid4().hex
        if file is not None:
            ext = Path(file.filename or "").suffix.lower()
            if ext not in _ALLOWED:
                raise HTTPException(415, f"desteklenmeyen dosya türü '{ext}'; izinli: {sorted(_ALLOWED)}")
            path = data_dir / f"{job_id}{ext}"
            with path.open("wb") as out:
                shutil.copyfileobj(file.file, out, length=1 << 20)
            if path.stat().st_size > max_bytes:
                path.unlink()
                raise HTTPException(413, f"dosya {settings.max_upload_mb} MB'tan büyük")
        elif audio_path and settings.allow_local_paths:
            path = Path(audio_path)
            if not path.is_file():
                raise HTTPException(404, f"dosya yok: {audio_path}")
        else:
            raise HTTPException(422, "file alanı zorunlu")
        try:
            info = sf.info(str(path))  # bozuk dosya kuyruğa girmesin
        except Exception:
            if file is not None:
                path.unlink(missing_ok=True)
            raise HTTPException(400, "ses dosyası okunamadı (bozuk ya da desteklenmeyen biçim)")
        try:
            submit(AudioJob(job_id, str(path.resolve()), language, callback_url))
        except UnsupportedLanguageError as e:
            if file is not None:
                path.unlink(missing_ok=True)
            raise HTTPException(422, str(e))
        return {"job_id": job_id, "status": JobStatus.QUEUED.value,
                "duration": round(info.duration, 3), "language": language}

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str):
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(404, "iş bulunamadı")
        tr = store.get_transcript(job_id) if rec.status == JobStatus.DONE else None
        return record_to_dict(rec, tr)

    @api.get("/health")
    def health():
        checks, ok = {}, True
        for name, fn in (health_checks or {}).items():
            try:
                fn()
                checks[name] = "ok"
            except Exception as e:
                checks[name], ok = f"hata: {type(e).__name__}", False
        body = {"status": "ok" if ok else "bozuk", "checks": checks,
                "languages": settings.languages}
        try:
            body["jobs"] = store.counts()
        except Exception:
            pass
        return JSONResponse(body, status_code=200 if ok else 503)

    return api


def _production_app() -> FastAPI:
    import redis

    from viet_asr.infrastructure.postgres import PostgresJobStore
    from viet_asr.interface.worker import CeleryJobQueue

    s = Settings()
    store = PostgresJobStore(s.database_url)
    store.migrate()
    r = redis.Redis.from_url(s.redis_url)

    def db_check():
        with store.pool.connection() as c:
            c.execute("SELECT 1")

    return create_app(store, CeleryJobQueue(), s, {"veritabani": db_check, "kuyruk": r.ping})


def __getattr__(name):  # uvicorn viet_asr.interface.api:app -> ilk erişimde kurulur
    if name == "app":
        globals()["app"] = _production_app()
        return globals()["app"]
    raise AttributeError(name)

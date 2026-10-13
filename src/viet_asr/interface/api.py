"""HTTP API (FastAPI).

    POST /jobs          ses dosyası yükle -> 202 + job_id (iş kuyruğa girer)
                        form alanları: file (zorunlu), language (vi | tr | auto, varsayılan auto),
                        callback_url (isteğe bağlı: iş bitince buraya POST edilir),
                        num_speakers (isteğe bağlı: biliniyorsa konuşmacı sayısı; yoksa sesten bulunur)
                        external_id (isteğe bağlı: çağıran sistemin kendi kimliği; aynısı tekrar
                        gelirse yeni iş açılmaz, var olan iş 200 ile döner -> ağ hatasında
                        güvenle tekrar gönderilebilir)
    GET  /jobs/{id}     durum; bittiyse sonuç (konuşmacılı satırlar)
    GET  /health        veritabanı ve kuyruk erişimi, durum sayıları (anahtar istemez)
    GET  /metrics       gecikme ve "1 dk ses <= 1 dk" hedefi, kuyruk doluluğu

API_KEY tanımlıysa /jobs istekleri "X-API-Key" başlığı ister (production'da zorunlu).

Çalıştırma:  uvicorn viet_asr.interface.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from viet_asr.application.jobs import SubmitJob
from viet_asr.application.maintenance import latency_stats
from viet_asr.application.ports import JobQueue, JobStore
from viet_asr.application.serialization import record_to_dict
from viet_asr.domain import AudioJob, JobStatus, UnsupportedLanguageError
from viet_asr.infrastructure.audio import FFMPEG_ONLY, AudioDecodeError, ffmpeg_bin, probe_duration
from viet_asr.interface.security import api_key_dependency, callback_problem
from viet_asr.interface.settings import Settings

log = logging.getLogger(__name__)
_ALLOWED = {".wav", ".flac", ".ogg", ".mp3", ".opus"} | FFMPEG_ONLY


def create_app(store: JobStore, queue: JobQueue, settings: Settings,
               health_checks: dict | None = None, queue_stats=None) -> FastAPI:
    api = FastAPI(title="Ses -> metin + konuşmacı (vi, tr)", version="0.3.0")
    auth = Depends(api_key_dependency(settings))
    submit = SubmitJob(store, queue, settings.languages)
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    @api.post("/jobs", status_code=202, dependencies=[auth])
    def create_job(file: UploadFile | None = File(None), language: str = Form("auto"),
                   callback_url: str | None = Form(None), audio_path: str | None = Form(None),
                   num_speakers: int | None = Form(None), external_id: str | None = Form(None)):
        problem = callback_problem(callback_url, settings)
        if problem:
            raise HTTPException(422, problem)
        job_id = uuid.uuid4().hex
        if file is not None:
            ext = Path(file.filename or "").suffix.lower()
            if ext not in _ALLOWED:
                raise HTTPException(415, f"desteklenmeyen dosya türü '{ext}'; izinli: {sorted(_ALLOWED)}")
            if ext in FFMPEG_ONLY and ffmpeg_bin() is None:
                raise HTTPException(415, f"'{ext}' için sunucuda ffmpeg kurulu olmalı")
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
        def reject(code, msg):
            if file is not None:
                path.unlink(missing_ok=True)
            raise HTTPException(code, msg)

        try:
            duration = probe_duration(str(path))  # bozuk dosya kuyruğa girmesin
        except (AudioDecodeError, Exception):
            reject(400, "ses dosyası okunamadı (bozuk ya da desteklenmeyen biçim)")
        if duration > settings.max_audio_minutes * 60:
            reject(413, f"ses {settings.max_audio_minutes:g} dakikadan uzun")
        try:
            got = submit(AudioJob(job_id, str(path.resolve()), language, callback_url, num_speakers,
                                  external_id))
        except (UnsupportedLanguageError, ValueError) as e:
            reject(422, str(e))
        if got != job_id:  # aynı external_id daha önce gönderilmiş: yeni dosya gereksiz
            if file is not None:
                path.unlink(missing_ok=True)
            rec = store.get(got)
            return JSONResponse({"job_id": got, "status": rec.status.value if rec else None,
                                 "tekrar": True}, status_code=200)
        return {"job_id": job_id, "status": JobStatus.QUEUED.value,
                "duration": round(duration, 3), "language": language}

    @api.get("/jobs/{job_id}", dependencies=[auth])
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

    @api.get("/metrics", dependencies=[auth])
    def metrics():
        """Son 500 biten iş: istekten bitişe süre ve 'bitiş süresi / ses süresi' oranı.
        Hedef: oran_p95 <= 1 (1 dk ses en geç 1 dk'da biter)."""
        body = {"gecikme": latency_stats(store.recent_done(500)), "isler": store.counts()}
        if queue_stats is not None:
            try:
                body["kuyruk"] = queue_stats()
            except Exception as e:
                body["kuyruk"] = f"hata: {type(e).__name__}"
        return body

    return api


def _production_app() -> FastAPI:
    import redis

    from viet_asr.infrastructure.db import SqlAlchemyJobStore
    from viet_asr.infrastructure.redis_streams import RedisStreamJobQueue

    s = Settings()
    store = SqlAlchemyJobStore(s.database_url)
    store.migrate()
    r = redis.Redis.from_url(s.redis_url)
    queue = RedisStreamJobQueue(r)
    return create_app(store, queue, s, {"veritabani": store.ping, "kuyruk": r.ping},
                      queue_stats=lambda: queue.stats(["lid", *s.languages]))


def __getattr__(name):  # uvicorn viet_asr.interface.api:app -> ilk erişimde kurulur
    if name == "app":
        globals()["app"] = _production_app()
        return globals()["app"]
    raise AttributeError(name)

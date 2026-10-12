"""Celery worker'ları.

Kuyruklar:
    lid      dil tespiti (Whisper-tiny/small, hafif)       -> detect_language
    asr.vi   Vietnamca metin + konuşmacı (Zipformer)      -> transcribe
    asr.tr   Türkçe metin + konuşmacı (Omnilingual/Whisper) -> transcribe

Çalıştırma (her biri ayrı süreç; Docker'da ayrı servis):
    celery -A viet_asr.interface.worker worker -Q lid    -c 1 -n lid@%h
    celery -A viet_asr.interface.worker worker -Q asr.vi -c 4 -n vi@%h
    celery -A viet_asr.interface.worker worker -Q asr.tr -c 2 -n tr@%h
-c (eşzamanlılık) x ONNX_THREADS <= çekirdek sayısı olmalı; yoksa süreçler
birbirini yavaşlatır (2 çekirdekte 4 thread ile Whisper 2,3 kat yavaşladı).

Güvenilirlik: acks_late + prefetch 1 -> worker iş ortasında ölürse iş kuyruğa
geri döner, başka worker alır. Aynı iş iki kez gelirse RunTranscription
bitmiş işi atlar.
"""
from __future__ import annotations

import logging

from celery import Celery
from celery.signals import worker_process_init

from viet_asr.application.jobs import DetectLanguage, RunTranscription
from viet_asr.infrastructure.audio import SoundfileLoader
from viet_asr.infrastructure.postgres import PostgresJobStore
from viet_asr.infrastructure.sherpa_vad import SileroVad
from viet_asr.interface.composition import build_pipeline, build_resolver, default_config
from viet_asr.interface.settings import Settings

log = logging.getLogger(__name__)
settings = Settings()

app = Celery("viet_asr", broker=settings.redis_url)
app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,       # uzun işler: bir worker fazladan iş kapmasın
    task_default_queue="lid",
    broker_connection_retry_on_startup=True,
    task_ignore_result=True,            # sonuç PostgreSQL'de, Celery sonucuna gerek yok
)

DETECT = "viet_asr.detect_language"
TRANSCRIBE = "viet_asr.transcribe"


class CeleryJobQueue:
    def __init__(self, celery_app: Celery = app):
        self.app = celery_app

    def enqueue_detection(self, job_id: str) -> None:
        self.app.send_task(DETECT, args=[job_id], queue="lid")

    def enqueue_transcription(self, job_id: str, language: str) -> None:
        self.app.send_task(TRANSCRIBE, args=[job_id], queue=f"asr.{language}")


# --- süreç başına bir kez kurulan bileşenler ---
_state: dict = {}


def _config():
    return default_config(settings.models_dir, tuple(settings.languages), settings.models,
                          settings.threads, settings.speakers, settings.hotwords)


def _store() -> PostgresJobStore:
    if "store" not in _state:
        _state["store"] = PostgresJobStore(settings.database_url, max_size=2)
    return _state["store"]


def _detector() -> DetectLanguage:
    if "detect" not in _state:
        cfg = _config()
        _state["detect"] = DetectLanguage(_store(), CeleryJobQueue(), SoundfileLoader(),
                                          SileroVad(cfg.vad), build_resolver(cfg))
    return _state["detect"]


def _transcriber() -> RunTranscription:
    if "run" not in _state:
        pipeline = build_pipeline(_config(), _store(), with_detection=False)
        _state["pipeline"] = pipeline
        _state["run"] = RunTranscription(_store(), pipeline)
    return _state["run"]


@worker_process_init.connect
def _warm_up(**_):
    if settings.warmup_lid:
        for det in _detector().resolver.detectors:
            getattr(det, "load", lambda: None)()
        log.info("dil tespiti modelleri önceden yüklendi")
    if settings.warmup:
        run = _transcriber()
        for lang in settings.warmup:
            run.pipeline.engines[lang]
            log.info("model önceden yüklendi: %s", lang)


@app.task(name=DETECT)
def detect_language(job_id: str) -> str | None:
    return _detector()(job_id)


@app.task(name=TRANSCRIBE)
def transcribe(job_id: str) -> None:
    _transcriber()(job_id)

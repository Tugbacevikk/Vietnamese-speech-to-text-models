"""Yerel mod: tek bilgisayarda, tek komutla tüm sistem (Redis, PostgreSQL, Celery GEREKMEZ).

    python -m viet_asr.interface.local_server               # http://localhost:8000
    python -m viet_asr.interface.local_server --port 9000 --workers 2

Aynı API (POST /jobs, GET /jobs/{id}, /health) ve aynı iş akışı; farkı:
  - veritabanı: SQLite dosyası (DATA_DIR/jobs.db)
  - kuyruk: süreç içi iş parçacığı havuzları
  - webhook göndericisi: arka planda bir iş parçacığı
Açılışta yarım kalmış işler (QUEUED/RUNNING) yeniden kuyruğa alınır.
Windows'ta da çalışır (Celery Windows'u desteklemiyor, bu mod onun yerine).
"""
from __future__ import annotations

import argparse
import logging
import threading
import time

import uvicorn

from viet_asr.application.jobs import DetectLanguage, RunTranscription
from viet_asr.infrastructure.audio import SoundfileLoader
from viet_asr.infrastructure.sherpa_vad import SileroVad
from viet_asr.infrastructure.thread_queue import ThreadJobQueue
from viet_asr.infrastructure.webhook import OutboxDispatcher
from viet_asr.interface.api import create_app
from viet_asr.interface.composition import (
    build_maintenance, build_pipeline, build_resolver, config_from_settings, store_from_settings,
    verify_models_or_exit,
)
from viet_asr.interface.settings import Settings

log = logging.getLogger(__name__)


def build_local(settings: Settings, workers: int | None = None, warm_up: bool = True):
    """Yerel sistemi kurar; (app, store, queue, dispatcher) döndürür (testler de kullanır)."""
    store = store_from_settings(settings, url=settings.local_db)
    store.migrate()
    n = workers or settings.concurrency
    cfg = config_from_settings(settings, workers=n)
    queue = ThreadJobQueue(asr_workers=n,
                           max_attempts=settings.max_job_attempts)
    pipeline = build_pipeline(cfg, store, with_detection=False)
    detect = DetectLanguage(store, queue, SoundfileLoader(), SileroVad(cfg.vad), build_resolver(cfg))
    queue.bind(detect, RunTranscription(store, pipeline), give_up=store.fail)
    if warm_up:  # ilk işte model yükleme gecikmesi olmasın
        for lang in settings.languages:
            pipeline.engines[lang]

    for rec in store.unfinished():  # önceki çalıştırmadan yarım kalanlar
        code = rec.decision.code if rec.decision else rec.job.language
        if code:
            queue.enqueue_transcription(rec.job.job_id, code)
        else:
            queue.enqueue_detection(rec.job.job_id)
        log.info("yarım kalan iş yeniden kuyrukta: %s", rec.job.job_id)

    dispatcher = OutboxDispatcher(store, secret=settings.webhook_secret,
                                  max_attempts=settings.webhook_max_attempts)
    app = create_app(store, queue, settings, {"veritabani": store.ping}, queue_stats=queue.stats)
    app.state.maintenance = build_maintenance(store, settings)
    return app, store, queue, dispatcher


def main(argv=None):
    p = argparse.ArgumentParser(description="Yerel mod: tek komutla tüm sistem")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=None, help="aynı anda işlenen ses sayısı")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s = Settings()
    verify_models_or_exit(s)
    app, store, queue, dispatcher = build_local(s, a.workers)
    threading.Thread(target=dispatcher.run_forever, daemon=True, name="webhook").start()

    def bakim():
        while True:
            try:
                app.state.maintenance.run_once()
            except Exception:
                log.exception("bakım turu başarısız")
            time.sleep(60)

    threading.Thread(target=bakim, daemon=True, name="bakim").start()
    n, why = (a.workers, "--workers") if a.workers else s.concurrency_plan()
    log.info("hazır: http://%s:%d  (diller: %s, eşzamanlı iş: %d — %s)", a.host, a.port,
             ",".join(s.languages), n, why)
    try:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    finally:
        dispatcher.stop()
        queue.shutdown(wait=False)


if __name__ == "__main__":
    main()

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


def build_local(settings: Settings, workers: int | None = None, warm_up: bool = True,
                live_engines=None, live_detector=None, live_detect=None, live_speakers=None):
    """Yerel sistemi kurar; (app, store, queue, dispatcher) döndürür (testler de kullanır).
    live_*: testler için sahte canlı altyazı motorları (dil -> motor), algılayıcı, dil tespiti."""
    store = store_from_settings(settings, url=settings.local_db)
    store.migrate()
    n = workers or settings.concurrency
    cfg = config_from_settings(settings, workers=n)
    queue = ThreadJobQueue(asr_workers=n,
                           max_attempts=settings.max_job_attempts)
    pipeline = build_pipeline(cfg, store, with_detection=False)
    resolver = build_resolver(cfg)
    detect = DetectLanguage(store, queue, SoundfileLoader(), SileroVad(cfg.vad), resolver)
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
    def models_ready():
        loaded = set(pipeline.engines.loaded())
        missing = [lang for lang in settings.languages if lang not in loaded]
        return not missing, (f"yüklenmemiş: {missing}" if missing else f"yüklü: {sorted(loaded)}")

    streams = None
    if settings.enable_streams:
        from viet_asr.application.jobs import SubmitJob
        from viet_asr.interface.streams import StreamManager

        streams = StreamManager(pipeline.engines, cfg.vad, settings.data_dir,
                                submit=SubmitJob(store, queue, settings.languages),
                                languages=list(settings.languages), max_sessions=settings.stream_max_sessions,
                                allow_files=settings.allow_local_paths, max_speech=settings.stream_max_speech)
    app = create_app(store, queue, settings, {"veritabani": store.ping}, queue_stats=queue.stats,
                     readiness={"modeller": models_ready}, streams=streams)
    app.state.streams = streams
    app.state.maintenance = build_maintenance(store, settings)

    # canlı altyazı (/canli, /ws/live): aynı süreç, aynı modeller; oturum kaydı bitince toplu işe
    import uuid

    from viet_asr.application.jobs import SubmitJob as _Submit
    from viet_asr.domain import AudioJob
    from viet_asr.interface.composition import build_live, build_live_speakers
    from viet_asr.interface.live_server import add_live_routes

    engines, detector, detect_lang = build_live(settings, cfg, pipeline.engines, resolver)
    if live_engines is not None:
        engines, detect_lang = live_engines, live_detect
    submit = _Submit(store, queue, settings.languages)

    def submit_recording(path: str, language: str | None, callback_url: str | None = None,
                         session_id: str | None = None) -> str:
        """Oturum kaydı toplu işe: konuşmacı ayrımı; bitince 'job.done' aynı callback adresine."""
        return submit(AudioJob(uuid.uuid4().hex, path, language, callback_url,
                               external_id=f"live:{session_id}" if session_id else None))

    def notify_ended(url: str, payload: dict, job_id: str | None) -> None:
        store.add_event("live.ended", url, payload, job_id)

    speakers = live_speakers if live_speakers is not None else build_live_speakers(settings, cfg)
    add_live_routes(app, settings, engines, live_detector or detector, submit_recording, detect_lang,
                    notify_ended, speaker_factory=speakers or None)
    if warm_up and live_engines is None:  # canlı motorlar açılışta arka planda hazırlanır
        for e in engines.values():
            e.start_loading()
    return app, store, queue, dispatcher


def main(argv=None):
    s = Settings()
    p = argparse.ArgumentParser(description="Yerel mod: tek komutla tüm sistem")
    p.add_argument("--host", default=s.host)
    p.add_argument("--port", type=int, default=s.port)
    p.add_argument("--workers", type=int, default=None, help="aynı anda işlenen ses sayısı")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    verify_models_or_exit(s)
    app, store, queue, dispatcher = build_local(s, a.workers)
    threading.Thread(target=dispatcher.run_forever, daemon=True, name="webhook").start()

    def bakim():
        while True:
            try:
                app.state.maintenance.run_once()
                from viet_asr.infrastructure.recorder import delete_old_recordings

                n_old = delete_old_recordings(s.recordings_dir, s.recordings_retention_days)
                if n_old:
                    log.info("bakım: %d eski canlı oturum dosyası silindi", n_old)
            except Exception:
                log.exception("bakım turu başarısız")
            time.sleep(60)

    threading.Thread(target=bakim, daemon=True, name="bakim").start()
    n, why = (a.workers, "--workers") if a.workers else s.concurrency_plan()
    log.info("hazır: http://%s:%d  (diller: %s, eşzamanlı iş: %d — %s)", a.host, a.port,
             ",".join(s.languages), n, why)
    log.info("canlı altyazı: http://%s:%d/canli", a.host, a.port)
    if "tr" in s.languages and s.model_for("tr").startswith("faster-whisper"):
        log.info("Türkçe: Whisper %s, %d iş parçacığı, beam_size=%d, aynı anda %d çözüm (WHISPER_THREADS)",
                 s.model_for("tr"), s.whisper_threads, s.whisper_beam_size, s.whisper_workers(n))
    try:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    finally:
        dispatcher.stop()
        queue.shutdown(wait=False)


if __name__ == "__main__":
    main()

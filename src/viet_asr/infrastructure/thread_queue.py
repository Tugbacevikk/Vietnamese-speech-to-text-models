"""Yerel mod kuyruğu: Redis/Celery yerine aynı süreçte iş parçacığı havuzları.

İki havuz: dil tespiti (1 iş parçacığı) ve metne çevirme (çekirdek / ONNX_THREADS).
Modeller süreç içinde bir kez yüklenir ve iş parçacıkları arasında paylaşılır
(ONNX Runtime oturumları aynı anda çağrılabilir).
Süreç kapanırsa kuyruktaki işler kaybolmaz: kayıtları veritabanında QUEUED/RUNNING
olarak kalır ve yerel sunucu açılışta onları yeniden kuyruğa alır.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)


class ThreadJobQueue:
    def __init__(self, asr_workers: int, lid_workers: int = 1):
        self._lid = ThreadPoolExecutor(lid_workers, thread_name_prefix="lid")
        self._asr = ThreadPoolExecutor(asr_workers, thread_name_prefix="asr")
        self._detect: Callable[[str], object] | None = None
        self._run: Callable[[str], object] | None = None

    def bind(self, detect: Callable[[str], object], run: Callable[[str], object]) -> None:
        """İş akışı nesneleri kuyruğa ihtiyaç duyduğu için kurulumdan sonra bağlanır."""
        self._detect, self._run = detect, run

    @staticmethod
    def _safe(fn, job_id):
        try:
            fn(job_id)
        except Exception:  # durum zaten FAILED/REJECTED yazıldı; havuz ölmesin
            log.exception("iş başarısız: %s", job_id)

    def enqueue_detection(self, job_id: str) -> None:
        self._lid.submit(self._safe, self._detect, job_id)

    def enqueue_transcription(self, job_id: str, language: str) -> None:
        self._asr.submit(self._safe, self._run, job_id)

    def shutdown(self, wait: bool = True) -> None:
        self._lid.shutdown(wait=wait, cancel_futures=not wait)
        self._asr.shutdown(wait=wait, cancel_futures=not wait)

"""Bakım: saklama süreleri ve askıda kalan işler.

fine-tuning reposundaki sweeper + retention_service'ten uyarlandı:
  - biten işlerin ses dosyası audio_hours sonra silinir (hasta kaydı: gereğinden uzun tutulmaz)
  - biten işin kaydı ve sonucu result_days sonra silinir
  - RUNNING'de stale_sec'ten uzun kalan iş FAILED olur ve (callback varsa) bildirim gider
Kuyruktaki çöken işlerin devralınması bakımın değil kuyruğun işi (Redis XAUTOCLAIM).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)


class Maintenance:
    def __init__(self, store, remove_file: Callable[[str], bool], audio_hours: int,
                 result_days: int, stale_sec: int,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.store, self.remove_file, self.now = store, remove_file, now
        self.audio_hours, self.result_days, self.stale_sec = audio_hours, result_days, stale_sec

    def run_once(self) -> dict[str, int]:
        now = self.now()
        out = {"ses_silindi": 0, "kayit_silindi": 0, "askida_kaldi": 0}
        if self.audio_hours > 0:
            for job_id, path in self.store.finished_with_audio_before(now - timedelta(hours=self.audio_hours)):
                if self.remove_file(path):
                    out["ses_silindi"] += 1
                self.store.clear_audio(job_id)
        if self.result_days > 0:
            out["kayit_silindi"] = self.store.delete_finished_before(now - timedelta(days=self.result_days))
        for job_id in self.store.stale_running(now - timedelta(seconds=self.stale_sec)):
            self.store.fail(job_id, f"askıda kaldı: {self.stale_sec} sn'den uzun süre işleniyor görünüyordu")
            out["askida_kaldi"] += 1
        if any(out.values()):
            log.info("bakım: %s", out)
        return out


def latency_stats(samples: list[tuple[float, float]]) -> dict[str, float]:
    """samples: (ses süresi sn, istekten bitişe sn). Hedef: 1 dk ses en geç 1 dk'da biter,
    yani 'oran' = bitiş süresi / ses süresi <= 1."""
    if not samples:
        return {"n": 0}
    lat = sorted(s[1] for s in samples)
    ratio = sorted(s[1] / max(s[0], 1e-6) for s in samples if s[0] > 0)

    def q(xs, p):
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0

    return {"n": len(samples), "gecikme_p50_sn": round(q(lat, 0.5), 2),
            "gecikme_p95_sn": round(q(lat, 0.95), 2), "oran_p50": round(q(ratio, 0.5), 3),
            "oran_p95": round(q(ratio, 0.95), 3),
            "hedef_disi": sum(1 for r in ratio if r > 1.0)}

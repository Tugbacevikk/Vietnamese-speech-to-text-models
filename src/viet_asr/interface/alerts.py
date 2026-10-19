"""Uyarılar: bir şey ters gidince arayüzde kırmızı şerit ve /health "uyarilar" listesi.

Yalnızca sayı ve süre içerir (kişisel veri yok): anahtarsız /health'te durabilir. Her biri ne
yapılacağını da söyler. Uyarı kaybolunca (sorun çözülünce) listeden kendiliğinden çıkar.
"""
from __future__ import annotations

import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def collect(store, settings) -> list[dict]:
    """[{"tur", "mesaj"}]; tür: bildirim | kuyruk | hata | yedek | disk."""
    out: list[dict] = []
    now = datetime.now(timezone.utc)

    def add(kind: str, msg: str) -> None:
        out.append({"tur": kind, "mesaj": msg})

    try:
        dead = store.outbox_stats(settings.webhook_max_attempts).get("olu", 0)
        if dead:
            add("bildirim", f"{dead} bildirim teslim edilemedi (alıcı yanıt vermiyor). "
                            "Liste: GET /webhooks/dead, yeniden dene: POST /webhooks/{id}/retry")
    except Exception:
        pass
    try:
        waiting = len(store.stale_queued(now - timedelta(minutes=settings.alert_queue_wait_min)))
        if waiting:
            add("kuyruk", f"{waiting} iş {settings.alert_queue_wait_min:g} dakikadan uzun süredir kuyrukta: "
                          "işçi çalışıyor mu (/health/ready), yük çok mu (/metrics)?")
    except Exception:
        pass
    try:
        failed = store.failed_since(now - timedelta(hours=24))
        if failed:
            add("hata", f"son 24 saatte {failed} iş başarısız oldu (geçmişte kırmızı olanların hata mesajına bakın)")
    except Exception:
        pass
    if settings.backup_every_hours > 0:
        files = [p for p in Path(settings.backup_dir).glob("yedek_*") if not p.name.endswith(".part")]
        age = (time.time() - max(p.stat().st_mtime for p in files)) / 3600 if files else None
        late = max(settings.backup_every_hours * 1.5, settings.backup_every_hours + 2)
        if age is None:
            try:
                has_jobs = sum(store.counts().values()) > 0
            except Exception:
                has_jobs = False
            if has_jobs:
                add("yedek", f"henüz veritabanı yedeği yok ({settings.backup_dir}); sunucu günlüğünde "
                             "'yedeği alınamadı' satırına bakın ya da yedekle.bat ile deneyin")
        elif age > late:
            add("yedek", f"son veritabanı yedeği {age:.0f} saat önce (beklenen {settings.backup_every_hours:g} "
                         "saatte bir); sunucu günlüğüne bakın ya da yedekle.bat ile deneyin")
    try:
        where = Path(settings.data_dir).resolve()
        while not where.exists() and where != where.parent:  # klasör henüz yoksa üstü (aynı disk)
            where = where.parent
        free = shutil.disk_usage(where).free / 1e9
        if free < settings.alert_disk_gb:
            add("disk", f"diskte {free:.1f} GB boş yer kaldı (sınır {settings.alert_disk_gb:g} GB): "
                        "eski yedekleri başka diske taşıyın, AUDIO_RETENTION_HOURS'u kısaltın")
    except OSError:
        pass
    return out

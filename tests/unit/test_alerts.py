"""Uyarılar: /health 'uyarilar' (ölü bildirim, uzun bekleyen iş, başarısız iş, yedek, disk)."""
import os
import time
from collections import namedtuple
from datetime import datetime, timedelta, timezone

from viet_asr.domain import AudioJob, JobStatus
from viet_asr.infrastructure.memory_store import MemoryJobStore
from viet_asr.interface.alerts import collect
from viet_asr.interface.settings import Settings


def kinds(store, s):
    return sorted(a["tur"] for a in collect(store, s))


def test_sorun_yoksa_uyari_yok(tmp_path):
    s = Settings(data_dir=str(tmp_path / "up"), backup_every_hours=24, alert_disk_gb=0)
    assert collect(MemoryJobStore(), s) == []                       # boş sistem: yedek uyarısı da yok


def test_her_uyari_turu(tmp_path, monkeypatch):
    s = Settings(data_dir=str(tmp_path / "up"), backup_every_hours=24, alert_disk_gb=0,
                 alert_queue_wait_min=30)
    store = MemoryJobStore()
    store.create(AudioJob("bekleyen", "/a.wav", "vi"))
    store.create(AudioJob("bozuk", "/b.wav", "vi"))
    store.set_status("bozuk", JobStatus.FAILED)
    assert kinds(store, s) == ["hata", "yedek"]                     # iş var ama hiç yedek yok
    r = store.jobs["bekleyen"]
    store.jobs["bekleyen"] = r.__class__(**{**r.__dict__, "updated_at": datetime.now(timezone.utc) - timedelta(hours=1)})
    monkeypatch.setattr(store, "outbox_stats", lambda n: {"olu": 2})
    (tmp_path / "yedek").mkdir()
    old = tmp_path / "yedek" / "yedek_eski.dump"
    old.write_text("x")
    os.utime(old, (time.time() - 50 * 3600,) * 2)                  # 50 saat önce: gecikmiş
    al = collect(store, s)
    assert sorted(a["tur"] for a in al) == ["bildirim", "hata", "kuyruk", "yedek"]
    assert any("2 bildirim" in a["mesaj"] for a in al) and any("50 saat" in a["mesaj"] for a in al)
    old.touch()                                                      # yeni yedek: uyarı kalkar
    assert "yedek" not in kinds(store, s)
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr("shutil.disk_usage", lambda p: Usage(100e9, 98e9, 2e9))
    assert "disk" in kinds(store, Settings(data_dir=str(tmp_path / "yok" / "up"), alert_disk_gb=5))


def test_health_uyarilari_dondurur(tmp_path):
    from fastapi.testclient import TestClient
    from test_api import Queue

    from viet_asr.interface.api import create_app

    store = MemoryJobStore()
    store.create(AudioJob("x", "/a.wav", "vi"))
    store.set_status("x", JobStatus.FAILED)
    c = TestClient(create_app(store, Queue(), Settings(data_dir=str(tmp_path), languages=["vi"],
                                                      backup_every_hours=0, alert_disk_gb=0), {}))
    body = c.get("/health").json()
    assert [a["tur"] for a in body["uyarilar"]] == ["hata"]


def test_kontrol_edilemeyen_sey_sessiz_kalmaz(tmp_path):
    """Veritabanı erişilemezse "sorun yok" değil, "kontrol edilemedi" uyarısı çıkar."""
    class Bozuk(MemoryJobStore):
        def outbox_stats(self, *a, **k):
            raise RuntimeError("baglanti yok")
    s = Settings(data_dir=str(tmp_path / "up"), backup_every_hours=0, alert_disk_gb=0)
    uyarilar = collect(Bozuk(), s)
    assert [a["tur"] for a in uyarilar] == ["izleme"]
    assert "RuntimeError" in uyarilar[0]["mesaj"]

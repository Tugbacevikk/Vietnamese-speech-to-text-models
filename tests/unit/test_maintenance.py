from datetime import datetime, timedelta, timezone

from viet_asr.application.maintenance import Maintenance, latency_stats
from viet_asr.domain import (
    AudioJob, JobStatus, LanguageDecision, LanguageSource, Transcript,
)
from viet_asr.infrastructure.db import SqlAlchemyJobStore
from viet_asr.infrastructure.webhook import sign, verify_signature
from viet_asr.interface.composition import safe_file_remover


def test_saklama_ve_askidaki_isler(tmp_path):
    up = tmp_path / "up"
    up.mkdir()
    store = SqlAlchemyJobStore(f"sqlite:///{tmp_path / 'j.db'}")
    store.migrate()
    disari = tmp_path / "kullanici.wav"
    disari.write_bytes(b"x")
    for jid, path in (("eski", up / "eski.wav"), ("yeni", up / "yeni.wav"), ("dis", disari)):
        if path.parent == up:
            path.write_bytes(b"x")
        store.create(AudioJob(jid, str(path), None, "http://a/h"))
        store.save(Transcript(jid, 1.0, [], "m", LanguageDecision("vi", 1.0, LanguageSource.REQUEST)))
    store.create(AudioJob("takili", str(up / "t.wav"), None, "http://a/h"))
    store.set_status("takili", JobStatus.RUNNING)

    now = datetime.now(timezone.utc)
    m = Maintenance(store, safe_file_remover(up), audio_hours=1, result_days=30, stale_sec=600,
                    now=lambda: now + timedelta(hours=2))
    out = m.run_once()
    assert not (up / "eski.wav").exists() and not (up / "yeni.wav").exists()
    assert disari.exists()                   # yükleme klasörü dışındaki dosyaya dokunulmaz
    assert out["ses_silindi"] == 2 and out["askida_kaldi"] == 1
    assert store.get("takili").status == JobStatus.FAILED
    assert store.get("eski").job.audio_path == "" and store.get_transcript("eski") is not None
    assert ("takili", "job.failed") in [(o["job_id"], o["event"]) for o in store.outbox_rows()]

    m2 = Maintenance(store, safe_file_remover(up), 1, 30, 600, now=lambda: now + timedelta(days=31))
    assert m2.run_once()["kayit_silindi"] == 4 and store.counts() == {}


def test_gecikme_istatistigi():
    s = latency_stats([(60, 20), (60, 30), (30, 45), (10, 5)])
    assert s["n"] == 4 and s["hedef_disi"] == 1 and s["oran_p95"] == 1.5


def test_webhook_imzasi():
    body = b'{"job_id":"x"}'
    sig = sign("s3cr3t", "1700000000", body)
    assert sig.startswith("sha256=")
    assert verify_signature("s3cr3t", "1700000000", body, sig, now=1700000100)
    assert not verify_signature("baska", "1700000000", body, sig, now=1700000100)
    assert not verify_signature("s3cr3t", "1700000000", body + b" ", sig, now=1700000100)
    assert not verify_signature("s3cr3t", "1700000000", body, sig, now=1700009999)  # eski

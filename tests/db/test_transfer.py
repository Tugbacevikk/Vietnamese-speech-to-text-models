"""SQLite -> PostgreSQL taşıma (scripts/postgrese_tasi.py). TEST_DATABASE_URL yoksa atlanır.

Kaynak gerçek kullanımı taklit eder: biten iş (JSON alanları, konuşmacı profili), düzeltilmiş ve
silinmiş satır, veri seti onayı, bekleyen ve gönderilmiş bildirim, işe bağlı olmayan canlı oturum
bildirimi, kuyrukta bekleyen iş. Taşınınca hedef depo aynı sonuçları vermeli.
"""
import os
import uuid

import pytest

from viet_asr.domain import AudioJob, JobStatus, LanguageDecision, LanguageSource, Transcript, Utterance
from viet_asr.domain.models import SpeakerInfo
from viet_asr.infrastructure.db import SqlAlchemyJobStore
from viet_asr.infrastructure.db.transfer import transfer

PG = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not PG, reason="TEST_DATABASE_URL yok")


@pytest.fixture
def pg_url():
    import psycopg

    schema = "tasi_" + uuid.uuid4().hex[:8]
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute(f"CREATE SCHEMA {schema}")
    yield f"{PG}{'&' if '?' in PG else '?'}options=-csearch_path%3D{schema}"
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute(f"DROP SCHEMA {schema} CASCADE")


def fill(src: SqlAlchemyJobStore):
    src.create(AudioJob("bitti", "/veri/a.wav", "vi", "http://alici/hook", 2, external_id="dis-1"))
    src.save(Transcript(job_id="bitti", duration=12.5, model="zipformer-68m",
                        language=LanguageDecision("vi", 0.97, LanguageSource.DETECTED, True),
                        utterances=[Utterance("speaker_00", 0.1, 2.0, "xin chào", conf=0.91,
                                              low_conf=(1,), breaks=(1,)),
                                    Utterance("speaker_01", 2.5, 4.0, "chào bác sĩ"),
                                    Utterance("speaker_00", 4.2, 5.0, "uydurma")],
                        timings={"asr": 1.5}, quality={"kapsama_yuzde": 98.1},
                        speakers=(SpeakerInfo("speaker_00", 120.0, "erkek", 3.0),)))
    src.update_utterance("bitti", 1, text="chào bác sĩ ạ")
    src.delete_utterance("bitti", 2)
    src.set_consent("bitti", "Tuğba", "test")
    src.create(AudioJob("bekliyor", "/veri/b.wav", None))
    src.add_event("live.ended", "http://alici/canli", {"event": "live.ended", "text": "merhaba"})
    rows = src.claim_outbox(10, 8, __import__("datetime").timedelta(seconds=30))
    src.mark_sent(rows[0]["id"])  # biri gönderilmiş, biri bekliyor


def test_sqlite_verisi_postgresqle_eksiksiz_tasinir(tmp_path, pg_url):
    src_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    src = SqlAlchemyJobStore(src_url)
    src.migrate()
    fill(src)
    before = (src.get_transcript("bitti"), src.get("bitti"), src.corrections(), src.outbox_rows())

    dry = transfer(src_url, pg_url, dry_run=True)
    assert dry.source == {"jobs": 2, "utterances": 3, "outbox": 2} and dry.copied == {}

    rep = transfer(src_url, pg_url)
    assert rep.copied == {"jobs": 2, "utterances": 3, "outbox": 2} and rep.complete
    dst = SqlAlchemyJobStore(pg_url)
    t, rec = dst.get_transcript("bitti"), dst.get("bitti")
    assert t == before[0]                                   # metin, düzeltme, silinen satır, JSON
    assert (rec.status, rec.created_at, rec.decision) == (before[1].status, before[1].created_at,
                                                          before[1].decision)
    assert dst.find_by_external_id("dis-1") == "bitti"
    assert dst.corrections() == before[2]                   # ince ayar verisi (onaylı düzeltme)
    assert dst.get("bekliyor").status == JobStatus.QUEUED
    assert [x.job.job_id for x in dst.unfinished()] == ["bekliyor"]
    out = sorted(dst.outbox_rows(), key=lambda r: r["event"])
    assert [(r["event"], r["job_id"], r["sent_at"] is not None) for r in out] == [
        ("job.done", "bitti", True), ("live.ended", None, False)]       # gönderilen tekrar gitmez
    assert out[1]["payload"]["text"] == "merhaba"

    again = transfer(src_url, pg_url)                       # tekrar çalıştırmak çoğaltmaz
    assert again.copied == {"jobs": 0, "utterances": 0, "outbox": 0} and again.target == rep.target
    dst.add_event("live.ended", "http://x/y", {"k": 1})     # hedefin sayacı çakışmadan devam eder
    assert len(dst.outbox_rows()) == 3


def test_kaynak_ve_hedef_ayni_olamaz(tmp_path):
    url = f"sqlite:///{tmp_path / 'a.db'}"
    with pytest.raises(ValueError):
        transfer(url, url)

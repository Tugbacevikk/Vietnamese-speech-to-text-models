"""Gerçek PostgreSQL ile depo ve outbox testleri.

    TEST_DATABASE_URL=postgresql://postgres@localhost/asr_test pytest tests/db
Değişken yoksa atlanır. Her test kendi şemasında çalışır, sonra silinir.
"""
import os
import uuid

from datetime import timedelta

import httpx
import pytest

from viet_asr.domain import (
    AudioJob, JobStatus, LanguageDecision, LanguageSource, Transcript, Utterance,
)

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL yok")


@pytest.fixture
def store():
    import psycopg

    from viet_asr.infrastructure.postgres import PostgresJobStore

    schema = "t_" + uuid.uuid4().hex[:8]
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute(f"CREATE SCHEMA {schema}")
    s = PostgresJobStore(f"{DSN}{'&' if '?' in DSN else '?'}options=-csearch_path%3D{schema}")
    s.migrate()
    s.migrate()  # iki kez çalıştırmak zararsız olmalı
    yield s
    s.close()
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute(f"DROP SCHEMA {schema} CASCADE")


def transcript(job_id):
    return Transcript(job_id=job_id, duration=12.5, model="zipformer-68m",
                      language=LanguageDecision("vi", 0.97, LanguageSource.DETECTED),
                      utterances=[Utterance("speaker_00", 0.1, 2.0, "xin chào"),
                                  Utterance("speaker_01", 2.5, 4.0, "chào bác sĩ")])


def outbox(store):
    with store.pool.connection() as c:
        return c.execute("SELECT * FROM outbox ORDER BY id").fetchall()


def test_is_yasam_dongusu_ve_sonuc(store):
    store.create(AudioJob("j1", "/data/a.wav", None, "http://alici/hook"))
    r = store.get("j1")
    assert r.status == JobStatus.QUEUED and r.decision is None and r.job.callback_url

    store.set_language("j1", LanguageDecision("vi", 0.97, LanguageSource.DETECTED))
    assert store.get("j1").decision.code == "vi"
    store.set_status("j1", JobStatus.RUNNING)
    assert outbox(store) == []  # ara durumlar bildirim üretmez

    store.save(transcript("j1"))
    r = store.get("j1")
    assert r.status == JobStatus.DONE
    t = store.get_transcript("j1")
    assert [u.text for u in t.utterances] == ["xin chào", "chào bác sĩ"]
    assert t.language.code == "vi" and t.model == "zipformer-68m"

    ob = outbox(store)
    assert len(ob) == 1 and ob[0]["event"] == "job.done" and ob[0]["url"] == "http://alici/hook"
    assert ob[0]["payload"]["utterances"][1]["speaker"] == "speaker_01"

    store.save(transcript("j1"))  # tekrar işlenirse satırlar çoğalmaz
    assert len(store.get_transcript("j1").utterances) == 2


def test_ret_ve_hata_bildirim_uretir_callback_yoksa_uretmez(store):
    store.create(AudioJob("j2", "/a.wav", None, "http://alici/hook"))
    store.set_status("j2", JobStatus.REJECTED, "ses desteklenmeyen dilde (en)")
    store.create(AudioJob("j3", "/b.wav", "tr", None))
    store.set_status("j3", JobStatus.FAILED, "bozuk dosya")
    ob = outbox(store)
    assert [(o["job_id"], o["event"]) for o in ob] == [("j2", "job.rejected")]
    assert store.get_transcript("j2") is None
    assert store.counts() == {"rejected": 1, "failed": 1}


def test_webhook_gonderimi_basari_ve_tekrar(store):
    from viet_asr.infrastructure.webhook import OutboxDispatcher

    store.create(AudioJob("ok", "/a.wav", None, "http://alici/ok"))
    store.create(AudioJob("bozuk", "/b.wav", None, "http://alici/bozuk"))
    store.save(transcript("ok"))
    store.save(transcript("bozuk"))
    gelen = []

    def handler(req: httpx.Request):
        gelen.append((req.url.path, req.headers["X-Event"], req.headers["X-Job-Id"]))
        return httpx.Response(200 if req.url.path == "/ok" else 503)

    d = OutboxDispatcher(store, httpx.Client(transport=httpx.MockTransport(handler)), max_attempts=2)
    assert d.run_once() == 1
    assert ("/ok", "job.done", "ok") in gelen
    ob = {o["job_id"]: o for o in outbox(store)}
    assert ob["ok"]["sent_at"] is not None
    assert ob["bozuk"]["sent_at"] is None and ob["bozuk"]["attempts"] == 1
    assert ob["bozuk"]["last_error"] == "HTTP 503"

    gelen.clear()
    assert d.run_once() == 0 and gelen == []  # bekleme süresi dolmadan tekrar denenmez
    assert store.claim_outbox(10, 8, timedelta(seconds=60)) == []
    with store.pool.connection() as c:
        c.execute("UPDATE outbox SET next_attempt_at = now()")
    d.run_once()
    d.run_once()
    assert len(gelen) == 1  # max_attempts=2: ikinciden sonra bırakılır

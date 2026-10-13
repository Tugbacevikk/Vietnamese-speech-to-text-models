"""SQLite deposu: PostgreSQL deposuyla aynı davranış (tests/db ile aynı senaryolar)."""
from datetime import timedelta

import httpx
import pytest

from viet_asr.domain import AudioJob, JobStatus, LanguageDecision, LanguageSource, Transcript, Utterance
from viet_asr.infrastructure.sqlite_store import SqliteJobStore
from viet_asr.infrastructure.webhook import OutboxDispatcher


@pytest.fixture
def store(tmp_path):
    s = SqliteJobStore(tmp_path / "jobs.db")
    s.migrate()
    s.migrate()
    yield s
    s.close()


def transcript(job_id):
    return Transcript(job_id=job_id, duration=12.5, model="zipformer-68m",
                      language=LanguageDecision("vi", 0.97, LanguageSource.DETECTED),
                      utterances=[Utterance("speaker_00", 0.1, 2.0, "xin chào"),
                                  Utterance("speaker_01", 2.5, 4.0, "chào bác sĩ")])


def test_yasam_dongusu(store):
    store.create(AudioJob("j1", "/a.wav", None, "http://alici/hook", 2))
    r = store.get("j1")
    assert r.status == JobStatus.QUEUED and r.job.num_speakers == 2 and r.decision is None
    store.set_language("j1", LanguageDecision("vi", 0.9, LanguageSource.DETECTED, True))
    assert store.get("j1").decision.uncertain is True
    assert [x.job.job_id for x in store.unfinished()] == ["j1"]
    store.save(transcript("j1"))
    store.save(transcript("j1"))
    assert store.get("j1").status == JobStatus.DONE and store.unfinished() == []
    t = store.get_transcript("j1")
    assert [u.text for u in t.utterances] == ["xin chào", "chào bác sĩ"]
    ob = store.outbox_rows()
    assert len(ob) == 2 and ob[0]["event"] == "job.done"  # iki kez kaydedildi -> iki olay


def test_ret_bildirimi_ve_bildirimsiz_is(store):
    store.create(AudioJob("r", "/a.wav", None, "http://alici/hook"))
    store.set_status("r", JobStatus.REJECTED, "desteklenmeyen dil")
    store.create(AudioJob("f", "/b.wav"))
    store.set_status("f", JobStatus.FAILED, "bozuk")
    assert [(o["job_id"], o["event"]) for o in store.outbox_rows()] == [("r", "job.rejected")]
    assert store.counts() == {"rejected": 1, "failed": 1}


def test_webhook_gonderimi(store):
    store.create(AudioJob("ok", "/a.wav", None, "http://alici/ok"))
    store.create(AudioJob("bozuk", "/b.wav", None, "http://alici/bozuk"))
    store.save(transcript("ok"))
    store.save(transcript("bozuk"))
    gelen = []

    def handler(req: httpx.Request):
        gelen.append(req.url.path)
        return httpx.Response(200 if req.url.path == "/ok" else 500)

    d = OutboxDispatcher(store, httpx.Client(transport=httpx.MockTransport(handler)))
    assert d.run_once() == 1 and sorted(gelen) == ["/bozuk", "/ok"]
    ob = {o["job_id"]: o for o in store.outbox_rows()}
    assert ob["ok"]["sent_at"] and ob["bozuk"]["attempts"] == 1 and ob["bozuk"]["last_error"] == "HTTP 500"
    assert d.run_once() == 0  # bekleme süresi dolmadan tekrar yok
    assert store.claim_outbox(10, 8, timedelta(seconds=60)) == []

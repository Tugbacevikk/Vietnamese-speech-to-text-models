"""Gerçek Redis ile kuyruk testleri. TEST_REDIS_URL yoksa atlanır.

    TEST_REDIS_URL=redis://localhost:6379/15 pytest tests/db/test_redis_streams.py
"""
import os
import time
import uuid

import pytest

URL = os.environ.get("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_REDIS_URL yok")


@pytest.fixture
def r():
    import redis

    c = redis.Redis.from_url(URL)
    c.flushdb()
    yield c
    c.flushdb()


def consumer(r, q, handler, gave_up, **kw):
    from viet_asr.infrastructure.redis_streams import StreamConsumer

    c = StreamConsumer(r, q, handler, on_give_up=lambda j, e: gave_up.append((j, e)),
                       block_ms=100, **kw)
    c.ensure_group()
    return c


def test_yonlendirme_ve_onay(r):
    from viet_asr.infrastructure.redis_streams import RedisStreamJobQueue

    q = RedisStreamJobQueue(r)
    q.enqueue_detection("a")
    q.enqueue_transcription("b", "vi")
    done, gave = [], []
    for name in ("lid", "vi"):
        c = consumer(r, name, done.append, gave)
        c.run_once()
    assert done == ["a", "b"] and gave == []
    assert q.stats(["lid", "vi", "tr"]) == {
        "lid": {"islenmekte": 0, "bekleyen": 0}, "vi": {"islenmekte": 0, "bekleyen": 0},
        "tr": {"islenmekte": 0, "bekleyen": 0}}


def test_hata_tekrar_sonra_birakma(r):
    from viet_asr.infrastructure.redis_streams import DLQ, RedisStreamJobQueue

    RedisStreamJobQueue(r).enqueue_transcription("x", "vi")
    calls, gave = [], []

    def boom(job_id):
        calls.append(job_id)
        raise RuntimeError("kalıcı hata")

    c = consumer(r, "vi", boom, gave, max_attempts=3)
    for _ in range(4):
        c.run_once()
    assert calls == ["x", "x", "x"] and len(gave) == 1 and "kalıcı" in gave[0][1]
    assert r.xlen(DLQ) == 1


def test_coken_worker_isini_baskasi_devralir(r):
    from viet_asr.infrastructure.redis_streams import GROUP, RedisStreamJobQueue, stream_name

    RedisStreamJobQueue(r).enqueue_transcription("y", "vi")
    c1 = consumer(r, "vi", lambda j: None, [], consumer="olu-" + uuid.uuid4().hex[:4],
                  claim_idle_ms=200)
    # c1 mesajı alır ama işlemeden "çöker" (onay yok)
    assert c1._read_new()
    assert r.xpending(stream_name("vi"), GROUP)["pending"] == 1
    time.sleep(0.3)
    done = []
    c2 = consumer(r, "vi", done.append, [], claim_idle_ms=200)
    c2.run_once()
    assert done == ["y"] and r.xpending(stream_name("vi"), GROUP)["pending"] == 0


def test_redis_istek_siniri():
    import os

    import redis

    from viet_asr.infrastructure.rate_limit import RedisRateLimiter

    r = redis.Redis.from_url(os.environ["TEST_REDIS_URL"])
    now = [600.0]
    lim = RedisRateLimiter(r, 2, clock=lambda: now[0], prefix="test:rl")
    r.delete(*[k for k in r.scan_iter("test:rl:*")] or ["yok"])
    assert [lim.check("a") for _ in range(3)][:2] == [None, None] and lim.check("a") == 61
    assert lim.check("b") is None
    now[0] += 60
    assert lim.check("a") is None

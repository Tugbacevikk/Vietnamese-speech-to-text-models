import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from viet_asr.application.jobs import DetectLanguage, RunTranscription
from viet_asr.application.fusion import FusionEngine
from viet_asr.application.language import LanguageResolver
from viet_asr.application.transcribe import TranscribeAndDiarize
from viet_asr.domain import JobStatus, SpeakerSegment, VadSegment, Word
from viet_asr.infrastructure.engines import LazyAsrEngines
from viet_asr.infrastructure.memory_store import MemoryJobStore
from viet_asr.interface.api import create_app
from viet_asr.interface.settings import Settings


class Queue:
    def __init__(self):
        self.sent = []

    def enqueue_detection(self, job_id):
        self.sent.append(("lid", job_id))

    def enqueue_transcription(self, job_id, language):
        self.sent.append((f"asr.{language}", job_id))


def wav_bytes(sec=1.0):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * sec), dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


@pytest.fixture
def ctx(tmp_path):
    store, queue = MemoryJobStore(), Queue()
    s = Settings(data_dir=str(tmp_path), languages=["vi", "tr"], max_upload_mb=1)
    return store, queue, TestClient(create_app(store, queue, s, {"sahte": lambda: None}))


def post(client, data=None, content=None, name="a.wav"):
    files = {"file": (name, content if content is not None else wav_bytes(), "audio/wav")}
    return client.post("/jobs", files=files, data=data or {})


def test_dil_verilmezse_tespit_kuyruguna(ctx):
    store, queue, c = ctx
    r = post(c)
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    assert queue.sent == [("lid", jid)]
    assert store.get(jid).status == JobStatus.QUEUED and r.json()["duration"] == 1.0


def test_dil_verilirse_o_dilin_kuyruguna(ctx):
    _, queue, c = ctx
    jid = post(c, {"language": "TR", "callback_url": "https://x/hook"}).json()["job_id"]
    assert queue.sent == [("asr.tr", jid)]


def test_ayni_external_id_tekrar_is_acmaz(ctx, tmp_path):
    _, queue, c = ctx
    first = post(c, {"external_id": "mesaj-42"})
    assert first.status_code == 202
    again = post(c, {"external_id": "mesaj-42"})
    assert again.status_code == 200 and again.json()["job_id"] == first.json()["job_id"]
    assert again.json()["tekrar"] is True and len(queue.sent) == 1   # kuyruğa bir kez girdi
    assert len(list(tmp_path.glob("*.wav"))) == 1                    # ikinci dosya silindi


def test_hatali_istekler(ctx):
    _, queue, c = ctx
    assert post(c, {"language": "en"}).status_code == 422          # desteklenmeyen dil
    assert post(c, name="a.xyz").status_code == 415                 # dosya türü
    assert post(c, content=b"ses degil").status_code == 400          # bozuk dosya
    assert post(c, content=wav_bytes(40)).status_code == 413         # 1 MB sınırı
    assert post(c, {"callback_url": "ftp://x"}).status_code == 422
    assert c.post("/jobs", data={"language": "vi"}).status_code == 422  # dosya yok
    assert queue.sent == []


def test_sonuc_sorgulama_ve_uctan_uca_akis(ctx):
    """API -> lid kuyruğu -> asr kuyruğu, sahte modellerle, gerçek iş akışı sınıflarıyla."""
    store, queue, c = ctx

    class Loader:
        def load(self, p):
            return sf.read(p, dtype="float32")[0]

    class Vad:
        def split(self, a):
            return [VadSegment(0.0, a)]

    class Det:
        def probabilities(self, a):
            return {"vi": 0.95, "tr": 0.03, "en": 0.02}

    class Asr:
        name = "sahte-vi"

        def transcribe(self, segs):
            return [Word("xin", 0.1, 0.3), Word("chào", 0.3, 0.6)]

    class Diar:
        def diarize(self, a, segments=None, num_speakers=None):
            return [SpeakerSegment("speaker_00", 0.0, 1.0)]

    jid = post(c).json()["job_id"]
    assert c.get(f"/jobs/{jid}").json()["status"] == "queued"
    resolver = LanguageResolver(["vi", "tr"], [Det()])
    DetectLanguage(store, queue, Loader(), Vad(), resolver)(jid)
    assert queue.sent[-1] == ("asr.vi", jid)
    assert c.get(f"/jobs/{jid}").json()["language"]["code"] == "vi"

    pipeline = TranscribeAndDiarize(Loader(), Vad(), LanguageResolver(["vi", "tr"], None),
                                    LazyAsrEngines({"vi": Asr}), Diar(), FusionEngine(), store)
    run = RunTranscription(store, pipeline)
    run(jid)
    body = c.get(f"/jobs/{jid}").json()
    assert body["status"] == "done"
    assert body["result"]["text"] == "xin chào"
    assert body["result"]["language"] == {"code": "vi", "confidence": 0.9694, "source": "tespit",
                                          "uncertain": False}
    assert body["result"]["utterances"][0]["speaker"] == "speaker_00"
    assert run(jid) is None  # aynı iş tekrar gelirse atlanır

    assert c.get("/jobs/yok").status_code == 404
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["jobs"] == {"done": 1}


def test_konusmaci_sayisi_ve_sure_siniri(tmp_path):
    store, queue = MemoryJobStore(), Queue()
    s = Settings(data_dir=str(tmp_path), languages=["vi", "tr"], max_audio_minutes=0.5)
    c = TestClient(create_app(store, queue, s))
    r = post(c, {"num_speakers": "3"})
    assert r.status_code == 202
    assert store.get(r.json()["job_id"]).job.num_speakers == 3
    assert c.get(f"/jobs/{r.json()['job_id']}").json()["num_speakers"] == 3
    assert post(c, {"num_speakers": "0"}).status_code == 422
    assert post(c, content=wav_bytes(31)).status_code == 413   # 0,5 dakikadan uzun
    assert len(list(tmp_path.iterdir())) == 1                   # reddedilenlerin dosyası silindi


def test_api_anahtari(tmp_path):
    store, queue = MemoryJobStore(), Queue()
    s = Settings(data_dir=str(tmp_path), languages=["vi", "tr"], api_key="gizli")
    c = TestClient(create_app(store, queue, s))
    assert post(c).status_code == 401
    ok = c.post("/jobs", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                headers={"X-API-Key": "gizli"})
    assert ok.status_code == 202
    assert c.get(f"/jobs/{ok.json()['job_id']}").status_code == 401
    assert c.get(f"/jobs/{ok.json()['job_id']}", headers={"X-API-Key": "gizli"}).status_code == 200
    assert c.get("/health").status_code == 200   # sağlık kontrolü anahtar istemez
    with pytest.raises(RuntimeError):            # production'da anahtarsız açılmaz
        create_app(store, queue, Settings(data_dir=str(tmp_path), app_env="production", api_key=""))


def test_callback_dogrulama(tmp_path):
    store, queue = MemoryJobStore(), Queue()
    prod = Settings(data_dir=str(tmp_path), app_env="production", api_key="k", languages=["vi"])
    c = TestClient(create_app(store, queue, prod))
    h = {"X-API-Key": "k"}

    def send(url):
        return c.post("/jobs", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                      data={"callback_url": url}, headers=h).status_code

    assert send("http://127.0.0.1:9000/hook") == 422      # iç adres: SSRF
    assert send("http://10.0.0.5/hook") == 422
    allow = Settings(data_dir=str(tmp_path), app_env="production", api_key="k", languages=["vi"],
                     webhook_allowed_hosts=["10.0.0.5"])
    c2 = TestClient(create_app(store, queue, allow))
    r = c2.post("/jobs", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                data={"callback_url": "http://10.0.0.5/hook"}, headers=h)
    assert r.status_code == 202                           # izin listesindeki iç ağ alıcısı
    r = c2.post("/jobs", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                data={"callback_url": "http://baska.local/hook"}, headers=h)
    assert r.status_code == 422


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg yok")
def test_m4a_ffmpeg_ile_okunur(tmp_path):
    import subprocess

    from viet_asr.infrastructure.audio import SoundfileLoader

    wav, m4a = tmp_path / "a.wav", tmp_path / "a.m4a"
    t = np.arange(16000 * 2) / 16000
    sf.write(wav, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(m4a)], check=True)
    a = SoundfileLoader().load(str(m4a))
    assert abs(len(a) / 16000 - 2.0) < 0.1
    store, queue = MemoryJobStore(), Queue()
    c = TestClient(create_app(store, queue, Settings(data_dir=str(tmp_path / "up"), languages=["vi"])))
    assert post(c, content=m4a.read_bytes(), name="kayit.m4a").status_code == 202


def test_metrikler(ctx):
    store, queue, c = ctx
    m = c.get("/metrics").json()
    assert m["gecikme"] == {"n": 0}


def test_arayuz_gecmis_ses_ve_silme(ctx, tmp_path):
    store, queue, c = ctx
    assert c.get("/").status_code == 200 and "Ses Analizi" in c.get("/").text
    jid = post(c, {"language": "vi"}).json()["job_id"]
    assert [j["job_id"] for j in c.get("/jobs").json()] == [jid]
    audio = c.get(f"/jobs/{jid}/audio")
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert c.delete(f"/jobs/{jid}").status_code == 409          # kuyruktaki iş silinmez
    store.set_status(jid, JobStatus.FAILED, "x")
    assert c.delete(f"/jobs/{jid}").status_code == 200
    assert c.get(f"/jobs/{jid}").status_code == 404 and list(tmp_path.glob("*.wav")) == []
    assert c.get("/jobs/yok/audio").status_code == 404


def test_asiri_yuk_ve_istek_siniri(tmp_path):
    from viet_asr.infrastructure.rate_limit import MemoryRateLimiter

    store, queue = MemoryJobStore(), Queue()
    s = Settings(data_dir=str(tmp_path), languages=["vi", "tr"], max_pending_jobs=2,
                 rate_limit_per_minute=0)
    c = TestClient(create_app(store, queue, s))
    assert post(c).status_code == 202 and post(c).status_code == 202
    r = post(c)
    assert r.status_code == 503 and r.headers["Retry-After"] == "60"     # 2 iş bekliyor: dolu
    assert len(list(tmp_path.glob("*.wav"))) == 2                         # reddedilen dosya yazılmadı
    for j in list(store.jobs):
        store.set_status(j, JobStatus.FAILED, "x")
    assert post(c).status_code == 202                                     # yer açıldı

    now = [120.0]
    lim = MemoryRateLimiter(2, clock=lambda: now[0])
    s2 = Settings(data_dir=str(tmp_path), languages=["vi"], max_pending_jobs=0)
    c2 = TestClient(create_app(MemoryJobStore(), Queue(), s2, rate_limiter=lim))
    assert [post(c2).status_code for _ in range(3)] == [202, 202, 429]
    assert c2.post("/jobs", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                   headers={"X-API-Key": "baska"}).status_code == 202     # her istemcinin kendi sayacı
    now[0] += 60                                                          # yeni dakika
    assert post(c2).status_code == 202


def test_satir_duzeltme(ctx):
    from viet_asr.domain import LanguageDecision, LanguageSource, Transcript, Utterance

    store, _, c = ctx
    jid = post(c, {"language": "tr"}).json()["job_id"]
    assert c.patch(f"/jobs/{jid}/utterances/0", json={"text": "x"}).status_code == 404   # bitmedi
    store.save(Transcript(jid, 3.0, [Utterance("speaker_00", 0, 1, "merhba doktor")], "m",
                          LanguageDecision("tr", 1.0, LanguageSource.REQUEST)))
    r = c.patch(f"/jobs/{jid}/utterances/0", json={"text": "  merhaba   doktor ", "speaker": "hasta"})
    assert r.status_code == 200 and r.json()["text"] == "merhaba doktor" and r.json()["speaker"] == "hasta"
    assert c.get(f"/jobs/{jid}").json()["result"]["utterances"][0]["edited"] is True
    assert c.patch(f"/jobs/{jid}/utterances/0", json={}).status_code == 422
    assert c.patch(f"/jobs/{jid}/utterances/0", json={"speaker": "<script>"}).status_code == 422
    (k,) = c.get("/corrections").json()
    assert k["original_text"] == "merhba doktor" and k["text"] == "merhaba doktor" and k["language"] == "tr"

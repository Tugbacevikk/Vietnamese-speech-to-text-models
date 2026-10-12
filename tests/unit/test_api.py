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


def test_hatali_istekler(ctx):
    _, queue, c = ctx
    assert post(c, {"language": "en"}).status_code == 422          # desteklenmeyen dil
    assert post(c, name="a.m4a").status_code == 415                 # dosya türü
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

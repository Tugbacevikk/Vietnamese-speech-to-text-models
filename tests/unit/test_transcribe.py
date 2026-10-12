import numpy as np
import pytest

from viet_asr.application.fusion import FusionEngine
from viet_asr.application.transcribe import TranscribeAndDiarize
from viet_asr.domain import AudioJob, JobStatus, SpeakerSegment, Syllable, VadSegment


class Loader:
    def load(self, path):
        return np.zeros(16000 * 3, dtype=np.float32)


class Vad:
    def split(self, audio):
        return [VadSegment(start=0.5, samples=np.zeros(16000, dtype=np.float32))]


class Asr:
    name = "sahte"

    def transcribe(self, segs):
        return [Syllable("xin", 0.5, 0.8), Syllable("chào", 0.8, 1.2)]


class Diar:
    def diarize(self, audio):
        return [SpeakerSegment("speaker_00", 0.4, 1.5)]


class BozukDiar:
    def diarize(self, audio):
        raise RuntimeError("model yok")


class Repo:
    def __init__(self):
        self.history, self.saved = [], None

    def set_status(self, job_id, status, error=None):
        self.history.append((status, error))

    def save(self, t):
        self.saved = t


def test_mutlu_yol():
    repo = Repo()
    uc = TranscribeAndDiarize(Loader(), Vad(), Asr(), Diar(), FusionEngine(), repo)
    t = uc.run(AudioJob("j1", "x.wav"))
    assert t.text == "xin chào" and t.utterances[0].speaker == "speaker_00"
    assert t.duration == 3.0 and t.model == "sahte"
    assert [s for s, _ in repo.history] == [JobStatus.RUNNING, JobStatus.DONE]
    assert repo.saved is t


def test_hata_durumu_kaydedilir():
    repo = Repo()
    uc = TranscribeAndDiarize(Loader(), Vad(), Asr(), BozukDiar(), FusionEngine(), repo)
    with pytest.raises(RuntimeError):
        uc.run(AudioJob("j2", "x.wav"))
    assert repo.history[-1][0] == JobStatus.FAILED and "model yok" in repo.history[-1][1]
    assert repo.saved is None

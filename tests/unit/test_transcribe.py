import numpy as np
import pytest

from viet_asr.application.fusion import FusionEngine
from viet_asr.application.language import LanguageResolver
from viet_asr.application.transcribe import TranscribeAndDiarize
from viet_asr.domain import (
    AudioJob, JobStatus, LanguageSource, SpeakerSegment, UnsupportedLanguageError, VadSegment, Word,
)
from viet_asr.infrastructure.engines import LazyAsrEngines


class Loader:
    def load(self, path):
        return np.zeros(16000 * 3, dtype=np.float32)


class Vad:
    def split(self, audio):
        return [VadSegment(start=0.5, samples=np.zeros(16000, dtype=np.float32))]


class Asr:
    def __init__(self, name, words):
        self.name, self.words = name, words

    def transcribe(self, segs):
        return [Word(t, 0.5 + i * 0.3, 0.8 + i * 0.3) for i, t in enumerate(self.words)]


class Det:
    def __init__(self, probs):
        self.p = probs

    def probabilities(self, audio):
        return self.p


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
        self.history.append((JobStatus.DONE, None))


def make(probs, diar=None, loaded=None):
    loaded = loaded if loaded is not None else []

    def f(code, words):
        def build():
            loaded.append(code)
            return Asr(f"asr-{code}", words)
        return build

    engines = LazyAsrEngines({"vi": f("vi", ["xin", "chào"]), "tr": f("tr", ["merhaba"])})
    repo = Repo()
    uc = TranscribeAndDiarize(Loader(), Vad(), LanguageResolver(["vi", "tr"], Det(probs)),
                              engines, diar or Diar(), FusionEngine(), repo)
    return uc, repo, loaded


def test_tespit_edilen_dilin_motoru_kullanilir_digeri_yuklenmez():
    uc, repo, loaded = make({"vi": 0.9, "tr": 0.05, "en": 0.05})
    t = uc.run(AudioJob("j1", "x.wav"))
    assert t.text == "xin chào" and t.model == "asr-vi"
    assert t.language.code == "vi" and t.language.source == LanguageSource.DETECTED
    assert loaded == ["vi"]
    assert [s for s, _ in repo.history] == [JobStatus.RUNNING, JobStatus.DONE]
    assert repo.saved is t and t.utterances[0].speaker == "speaker_00"


def test_istekteki_dil_tespiti_gecersiz_kilar():
    uc, _, loaded = make({"vi": 1.0})
    t = uc.run(AudioJob("j2", "x.wav", language="tr"))
    assert t.text == "merhaba" and t.language.source == LanguageSource.REQUEST and loaded == ["tr"]


def test_desteklenmeyen_dil_reddedilir_motor_yuklenmez():
    uc, repo, loaded = make({"en": 0.99, "vi": 0.01})
    with pytest.raises(UnsupportedLanguageError):
        uc.run(AudioJob("j3", "x.wav"))
    assert repo.history[-1][0] == JobStatus.REJECTED and "en" in repo.history[-1][1]
    assert loaded == [] and repo.saved is None


def test_hata_durumu_kaydedilir():
    uc, repo, _ = make({"vi": 1.0}, diar=BozukDiar())
    with pytest.raises(RuntimeError):
        uc.run(AudioJob("j4", "x.wav"))
    assert repo.history[-1][0] == JobStatus.FAILED and "model yok" in repo.history[-1][1]
    assert repo.saved is None

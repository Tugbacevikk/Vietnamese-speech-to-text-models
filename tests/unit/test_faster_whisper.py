"""faster-whisper adaptörü (Türkçe): gerçek model yerine sahte nesnelerle.

Gerçek modelle doğrulama: tests/integration (models/faster-whisper-medium ya da -small varsa).
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from viet_asr.application.config import AsrParams, ModelRef, WhisperParams
from viet_asr.domain import VadSegment, Word
from viet_asr.infrastructure.faster_whisper_asr import FasterWhisperAsr, drop_repeats, resolve_device
from viet_asr.infrastructure.normalizer import normalizer_for

REF = ModelRef("faster-whisper-small", "faster-whisper", "models/faster-whisper-small", {})


def seg(start, end):
    return VadSegment(start, np.full(int(round((end - start) * 16000)), 0.1, np.float32))


def w(text, a, b):
    return NS(word=text, start=a, end=b)


class FakeBatched:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def transcribe(self, audio, clip_timestamps, batch_size, **opt):
        self.calls.append((audio, clip_timestamps, batch_size, opt))

        def gen():  # gerçek kütüphane gibi tembel: hata döngüde çıkar
            if self.fail:
                raise RuntimeError("bellek yetmedi")
            yield NS(text=" Merhaba, nasılsınız?", words=[w(" Merhaba,", 1.0, 1.4), w(" nasılsınız?", 1.5, 2.1)])
            yield NS(text=" İzlediğiniz için teşekkür ederim.", words=[w(" İzlediğiniz", 5.0, 5.5)])
            yield NS(text=" evet evet evet evet", words=[w(" evet", 6.0, 6.2), w(" evet,", 6.2, 6.4),
                                                       w(" evet", 6.4, 6.6), w(" Evet", 6.6, 6.8)])
        return gen(), NS(language="tr")


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples, vad_filter, **opt):
        self.calls.append(len(samples))
        return iter([NS(text=" iyi günler", words=[w(" İyi", 0.1, 0.3), w(" günler", 0.4, 0.8)])]), None


def engine(batched=None, model=None, **wp):
    params = AsrParams(drop_phrases=("İzlediğiniz için teşekkür ederim.",), whisper=WhisperParams(**wp))
    return FasterWhisperAsr(REF, params, normalizer_for("tr"), "tr",
                            whisper_model=model or FakeModel(), batched=batched)


def test_toplu_cozme_mutlak_zaman_ve_suzgecler():
    b = FakeBatched()
    words = engine(b).transcribe([seg(1.0, 2.2), seg(5.0, 7.0)])
    # sistem çıktısı küçük harf, noktalamasız (iki dilde aynı); tekrar süzgeci noktalamaya bakmaz
    assert [x.text for x in words] == ["merhaba", "nasılsınız", "evet", "evet"]
    assert words[0].start == 1.0 and words[1].end == 2.1
    audio, clips, bs, opt = b.calls[0]
    assert clips == [{"start": 1.0, "end": 7.0}] and bs == 8   # 2,8 sn boşluk: tek pencerede paketlenir
    assert audio[int(1.5 * 16000)] == pytest.approx(0.1) and audio[int(3 * 16000)] == 0  # boşluk sessiz
    assert opt["language"] == "tr" and opt["word_timestamps"] is True
    assert opt["condition_on_previous_text"] is False and opt["log_prob_threshold"] == -1.0


def test_toplu_cozme_hatasinda_tek_tek_cozer():
    m = FakeModel()
    words = engine(FakeBatched(fail=True), m).transcribe([seg(10.0, 11.0), seg(20.0, 21.5)])
    assert m.calls == [16000, 24000]
    # parçaya göreli zaman mutlak zamana çevrilir
    assert [(x.text, x.start) for x in words] == [("iyi", 10.1), ("günler", 10.4), ("iyi", 20.1), ("günler", 20.4)]


def test_istem_ciktida_tekrar_edilirse_atilir():
    e = engine(None, initial_prompt="Türkçe ses kaydı.")
    e.model = NS(transcribe=lambda s, vad_filter, **o: (iter([NS(text=" Türkçe ses kaydı.",
                                                                    words=[w(" Türkçe", 0, 1)])]), None))
    assert e.transcribe([seg(0, 1)]) == []


def test_tekrar_suzgeci():
    ws = [Word(t, i, i + 1) for i, t in enumerate(["a", "a", "a", "b", "a", "a", "a", "a"])]
    assert [x.text for x in drop_repeats(ws, 2)] == ["a", "a", "b", "a", "a"]
    assert drop_repeats([], 2) == []


def test_katalog_ve_profil():
    from viet_asr.infrastructure.catalog import DEFAULT_MODELS, language_profile

    assert DEFAULT_MODELS["tr"] == "faster-whisper-medium"
    p = language_profile("tr", "models")
    assert p.segmentation == "vad" and p.asr_model.kind == "faster-whisper"
    assert p.asr_model.path("model").replace("\\", "/").endswith("faster-whisper-medium/model.bin")


class RescueBatched:
    """Toplu çözümde yalnızca 1-3 sn aralığına kelime döner; diğer aralıklar boş kalır."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, clip_timestamps, batch_size, **opt):
        self.calls.append((clip_timestamps, opt))
        return iter([NS(text=" tamam", words=[w(" tamam", 1.2, 1.6)])]), None


def test_guvenlik_agi_tek_toplu_cozum_ve_sinirli_tekrar():
    b, m = RescueBatched(), FakeModel()
    segs = [seg(1.0, 3.0)] + [seg(10.0 + 5 * i, 12.0 + 5 * i + 0.1 * i) for i in range(6)]
    words = engine(b, m).rescue(segs)
    clips = b.calls[0][0]
    assert len(b.calls) == 1 and len(clips) < 7                  # tüm aralıklar tek çağrıda (yakınlar paketli)
    assert all(any(c["start"] <= s.start and s.end <= c["end"] + 1e-6 for c in clips) for s in segs)  # hiçbiri dışarıda kalmaz
    opt = b.calls[0][1]
    assert opt["temperature"] == 0.0 and opt["no_speech_threshold"] == 0.8
    # boş kalan 6 aralığın en uzun 4'ü tek tek, düşük maliyetle yeniden denenir
    assert len(m.calls) == FasterWhisperAsr.RESCUE_MAX_RETRY == 4
    assert sorted(m.calls, reverse=True) == m.calls
    assert words[0].text == "tamam" and [x.text for x in words[1:3]] == ["iyi", "günler"]


def test_guvenlik_agi_tekrar_ayarlari():
    e = engine(None)
    o = e._options(rescue=True, retry=True)
    assert o["temperature"] == (0.2, 0.4) and o["best_of"] == 1 and o["beam_size"] == 1
    assert engine(None)._options(rescue=True)["beam_size"] == 5


def test_yakin_parcalar_tek_pencerede_paketlenir():
    """Whisper her pencereyi 30 sn'lik işler: yakın parçalar birleşmeli, uzak olanlar ve sınır aşanlar ayrı."""
    p = FasterWhisperAsr.pack_clips
    assert p([]) == []
    assert p([(0, 2), (2.5, 5), (6, 9)]) == [(0, 9)]                 # yakın: tek pencere
    assert p([(0, 2), (6, 8)]) == [(0, 2), (6, 8)]                   # boşluk 4 sn > 3 sn: ayrı
    assert p([(0, 10), (10.5, 20), (20.5, 30)]) == [(0, 20), (20.5, 30)]   # 30 sn'yi aşarsa yeni pencere
    assert p([(0, 40)]) == [(0, 40)]                                 # tek uzun parça bölünmez
    assert p([(6, 9), (0, 2), (2.5, 5)]) == [(0, 9)]                 # sıra karışık gelse de


def test_toplu_cozum_paketlenmis_pencereleri_kullanir_kelimeler_kaybolmaz():
    batched = FakeBatched()
    asr = engine(batched=batched)
    asr.transcribe([seg(1.0, 3.0), seg(3.5, 6.0), seg(40.0, 42.0)])
    clips = batched.calls[0][1]
    assert clips == [{"start": 1.0, "end": 6.0}, {"start": 40.0, "end": 42.0}]   # 3 parça -> 2 pencere


def test_cihaz_ve_hesap_turu_cozumu():
    # CUDA istendi ama yok + .env'de float16: eskiden CPU'da float16 ile çöküyordu
    assert resolve_device("cuda", "float16", cuda_ok=False) == ("cpu", "int8")
    assert resolve_device("cuda", "int8_float16", cuda_ok=False) == ("cpu", "int8")
    assert resolve_device("cpu", "float16", cuda_ok=False) == ("cpu", "int8")
    assert resolve_device("auto", "auto", cuda_ok=False) == ("cpu", "int8")
    assert resolve_device("auto", "auto", cuda_ok=True) == ("cuda", "float16")
    assert resolve_device("cuda", "float16", cuda_ok=True) == ("cuda", "float16")
    assert resolve_device("cpu", "int8", cuda_ok=False) == ("cpu", "int8")

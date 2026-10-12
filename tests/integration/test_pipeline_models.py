"""Gerçek modellerle uçtan uca testler. Gereken model yoksa test atlanır.

    python scripts/download_models.py
    pytest -m models
"""
from pathlib import Path

import numpy as np
import pytest

M = Path(__file__).resolve().parents[2] / "models"
Z68 = M / "sherpa-onnx-zipformer-vi-int8-2025-04-20"
LIDW = M / "spoken-language-identification-test-wavs"
TINY = M / "sherpa-onnx-whisper-tiny"
SMALL = M / "sherpa-onnx-whisper-small"
TURBO = M / "sherpa-onnx-whisper-turbo"
OMNI = M / "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"


def need(*paths):
    missing = [p for p in paths if not Path(p).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"model yok: {[str(p) for p in missing]}")


pytestmark = pytest.mark.models
VI_OK = need(Z68 / "tokens.txt", M / "silero_vad.onnx")
LID_OK = need(TINY / "tiny-encoder.int8.onnx", SMALL / "small-encoder.int8.onnx",
              LIDW / "tr-turkish.wav", Z68 / "test_wavs")
DIAR = [M / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"]


def load(p):
    from viet_asr.infrastructure.audio import SoundfileLoader

    return SoundfileLoader().load(str(p))


def vi_wavs():
    return [load(Z68 / "test_wavs" / f"{i}.wav") for i in range(3)]


@VI_OK
def test_vad_ofseti_ve_asr_zamanlari():
    from viet_asr.application.config import VadConfig
    from viet_asr.infrastructure.catalog import language_profile
    from viet_asr.infrastructure.normalizer import ViNormalizer
    from viet_asr.infrastructure.sherpa_asr import build_asr
    from viet_asr.infrastructure.sherpa_vad import SileroVad

    audio = np.concatenate([np.zeros(16000 * 3, dtype=np.float32), vi_wavs()[0]])
    segs = SileroVad(VadConfig(model=str(M / "silero_vad.onnx"))).split(audio)
    assert segs and 2.5 < segs[0].start < 3.6

    words = build_asr(language_profile("vi", M), ViNormalizer()).transcribe(segs)
    assert len(words) > 3
    assert words[0].start >= 2.5  # ofset eklenmiş olmalı
    assert all(a.start <= b.start for a, b in zip(words, words[1:]))
    assert all(w.text == w.text.lower() for w in words)


@VI_OK
def test_hotword_bpe_vocab_olmadan_hata_verir(tmp_path):
    from dataclasses import replace

    from viet_asr.infrastructure.catalog import language_profile
    from viet_asr.infrastructure.normalizer import ViNormalizer
    from viet_asr.infrastructure.sherpa_asr import build_asr

    hw = tmp_path / "h.txt"
    hw.write_text("BÁC SĨ\n", encoding="utf-8")
    prof = language_profile("vi", M, hotwords_file=str(hw))
    files = {**prof.asr_model.files, "bpe_vocab": "yok.vocab"}
    prof = replace(prof, asr_model=replace(prof.asr_model, files=files))
    with pytest.raises(FileNotFoundError):
        build_asr(prof, ViNormalizer())


@LID_OK
def test_dil_tespiti_gercek_seslerde():
    from viet_asr.application.config import LidConfig
    from viet_asr.application.language import LanguageResolver
    from viet_asr.domain import UnsupportedLanguageError, VadSegment
    from viet_asr.infrastructure.catalog import lid_stages
    from viet_asr.infrastructure.whisper_lid import WhisperLanguageDetector

    tiny, small = (WhisperLanguageDetector(e, d) for e, d in lid_stages(M))
    p = tiny.probabilities(load(LIDW / "tr-turkish.wav"))
    assert abs(sum(p.values()) - 1) < 1e-6 and len(p) == 99

    # 2.wav yalnızca 1,6 sn: tiny tek başına reddediyor, kademe small'a devredip doğru buluyor
    with pytest.raises(UnsupportedLanguageError):
        LanguageResolver(["vi", "tr"], [tiny]).resolve(None, [VadSegment(0.0, vi_wavs()[2])])

    res = LanguageResolver(["vi", "tr"], [tiny, small], LidConfig())
    seg = lambda a: [VadSegment(0.0, a)]  # noqa: E731
    assert res.resolve(None, seg(load(LIDW / "tr-turkish.wav"))).code == "tr"
    for a in vi_wavs():
        assert res.resolve(None, seg(a)).code == "vi"
    for f in ("en-english.wav", "de-german.wav", "zh-chinese.wav"):
        with pytest.raises(UnsupportedLanguageError):
            res.resolve(None, seg(load(LIDW / f)))


@need(TURBO / "turbo-encoder.int8.onnx", LIDW / "tr-turkish.wav", M / "silero_vad.onnx")
def test_turkce_whisper_ve_uydurma_filtresi():
    from viet_asr.domain import VadSegment
    from viet_asr.infrastructure.catalog import language_profile
    from viet_asr.infrastructure.normalizer import TrNormalizer
    from viet_asr.infrastructure.sherpa_asr import build_asr

    asr = build_asr(language_profile("tr", M, "whisper-turbo"), TrNormalizer())
    words = asr.transcribe([VadSegment(4.0, load(LIDW / "tr-turkish.wav"))])
    text = " ".join(w.text for w in words).lower()
    assert "hayal" in text and "yapabilirsin" in text
    assert words[0].start >= 4.0 and words[-1].end <= 4.0 + len(load(LIDW / "tr-turkish.wav")) / 16000 + 1e-6
    # 10 sn sessizlikte Whisper "İzlediğiniz için teşekkür ederim." uyduruyor -> atılmalı
    assert asr.transcribe([VadSegment(0.0, np.zeros(16000 * 10, dtype=np.float32))]) == []


@need(OMNI / "model.int8.onnx", LIDW / "tr-turkish.wav")
def test_turkce_omnilingual_gercek_zaman_damgasi():
    from viet_asr.domain import VadSegment
    from viet_asr.infrastructure.catalog import language_profile
    from viet_asr.infrastructure.normalizer import TrNormalizer
    from viet_asr.infrastructure.sherpa_asr import build_asr

    a = load(LIDW / "tr-turkish.wav")
    words = build_asr(language_profile("tr", M), TrNormalizer()).transcribe([VadSegment(4.0, a)])
    assert [w.text for w in words] == ["hayal", "edebiliyorsan", "yapabilirsin"]
    assert 4.0 <= words[0].start < words[1].start < words[2].start < 4.0 + len(a) / 16000


@VI_OK
@LID_OK
@need(*DIAR)
def test_uctan_uca_dil_tespitli(tmp_path):
    import soundfile as sf

    from viet_asr.domain import AudioJob, LanguageSource
    from viet_asr.interface.cli import MemoryRepo, build, default_config

    w = vi_wavs()
    gap = np.zeros(8000, dtype=np.float32)
    p = tmp_path / "konusma.wav"
    sf.write(p, np.concatenate([w[0], gap, w[1], gap, w[2], gap, w[0]]), 16000)

    uc = build(default_config(M), MemoryRepo())
    t = uc.run(AudioJob("e2e", str(p)))
    assert t.language.code == "vi" and t.language.source == LanguageSource.DETECTED
    assert t.utterances and len(t.text.split()) > 10
    # 0.wav, 1.wav, 2.wav üç farklı kişi (3D-Speaker uzaklığı 0,84-1,07); sıra 0,1,2,0
    assert [u.speaker for u in t.utterances] == ["speaker_00", "speaker_01", "speaker_02", "speaker_00"]
    assert uc.engines.loaded() == ["vi"]  # Türkçe model yüklenmedi


@need(*DIAR, LIDW / "tr-turkish.wav", Z68 / "test_wavs", M / "silero_vad.onnx")
def test_konusmaci_sayisi_otomatik_ve_verilen():
    from viet_asr.application.config import VadConfig
    from viet_asr.infrastructure.sherpa_vad import SileroVad
    from viet_asr.interface.composition import build_diarizer, default_config

    tr, vi = load(LIDW / "tr-turkish.wav"), vi_wavs()[0]
    gap = np.zeros(8000, dtype=np.float32)
    iki = np.concatenate([tr, gap, vi, gap, tr, gap, vi])
    tek = np.concatenate([vi, gap, vi, gap, vi])
    vad = SileroVad(VadConfig(model=str(M / "silero_vad.onnx")))
    d = build_diarizer(default_config(M).diarizer)

    segs = d.diarize(iki, vad.split(iki))
    assert len({s.speaker for s in segs}) == 2
    assert [s.speaker for s in segs][:2] == ["speaker_00", "speaker_01"]
    assert len({s.speaker for s in d.diarize(tek, vad.split(tek))}) == 1
    assert len({s.speaker for s in d.diarize(iki, vad.split(iki), num_speakers=1)}) == 1
    assert d.diarize(iki, []) == []

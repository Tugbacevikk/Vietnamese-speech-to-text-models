"""Gerçek modellerle uçtan uca test. Modeller yoksa atlanır.

    python scripts/download_models.py
    pytest -m models
"""
from pathlib import Path

import numpy as np
import pytest

M = Path(__file__).resolve().parents[2] / "models"
Z68 = M / "sherpa-onnx-zipformer-vi-int8-2025-04-20"
need = pytest.mark.skipif(not (Z68 / "tokens.txt").exists() or not (M / "silero_vad.onnx").exists(),
                          reason="modeller indirilmemiş")
pytestmark = [pytest.mark.models, need]


def _wavs():
    from viet_asr.infrastructure.audio import SoundfileLoader

    return [SoundfileLoader().load(str(Z68 / "test_wavs" / f"{i}.wav")) for i in range(3)]


def _cfg(hot=None):
    from viet_asr.application.config import ZIPFORMER_68M, AsrParams, ModelRef

    return ModelRef(folder=Z68, **ZIPFORMER_68M), AsrParams(hotwords_file=hot)


def test_vad_ofseti_ve_asr_zamanlari():
    from viet_asr.application.config import VadConfig
    from viet_asr.infrastructure.normalizer import ViNormalizer
    from viet_asr.infrastructure.sherpa_asr import SherpaOnnxAsr
    from viet_asr.infrastructure.sherpa_vad import SileroVad

    w = _wavs()[0]
    sessiz = np.zeros(16000 * 3, dtype=np.float32)
    audio = np.concatenate([sessiz, w])  # konuşma 3. saniyede başlıyor
    segs = SileroVad(VadConfig(model=str(M / "silero_vad.onnx"))).split(audio)
    assert segs and 2.5 < segs[0].start < 3.6

    asr = SherpaOnnxAsr(*_cfg(), ViNormalizer())
    syl = asr.transcribe(segs)
    assert len(syl) > 3
    assert syl[0].start >= 2.5  # ofset eklenmiş olmalı
    assert all(a.start <= b.start for a, b in zip(syl, syl[1:]))
    assert all(s.text == s.text.lower() for s in syl)


def test_hotword_bpe_vocab_olmadan_hata_verir(tmp_path):
    from viet_asr.application.config import ZIPFORMER_68M, AsrParams, ModelRef
    from viet_asr.infrastructure.normalizer import ViNormalizer
    from viet_asr.infrastructure.sherpa_asr import SherpaOnnxAsr

    hw = tmp_path / "h.txt"
    hw.write_text("BÁC SĨ\n", encoding="utf-8")
    ref = ModelRef(folder=Z68, **{**ZIPFORMER_68M}, bpe_vocab="yok.vocab")
    with pytest.raises(FileNotFoundError):
        SherpaOnnxAsr(ref, AsrParams(hotwords_file=str(hw)), ViNormalizer())


@pytest.mark.skipif(not (M / "wespeaker_en_voxceleb_resnet34_LM.onnx").exists(), reason="diarization modeli yok")
def test_uctan_uca_iki_konusmaci(tmp_path):
    import soundfile as sf

    from viet_asr.application.config import DiarizerConfig, PipelineConfig, VadConfig
    from viet_asr.domain import AudioJob
    from viet_asr.interface.cli import MemoryRepo, build

    w = _wavs()
    gap = np.zeros(8000, dtype=np.float32)
    audio = np.concatenate([w[0], gap, w[1], gap, w[2], gap, w[0]])
    p = tmp_path / "konusma.wav"
    sf.write(p, audio, 16000)

    model, params = _cfg()
    cfg = PipelineConfig(
        asr_model=model, asr=params, vad=VadConfig(model=str(M / "silero_vad.onnx")),
        diarizer=DiarizerConfig(
            segmentation_model=str(M / "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            embedding_model=str(M / "wespeaker_en_voxceleb_resnet34_LM.onnx")))
    t = build(cfg, MemoryRepo()).run(AudioJob("e2e", str(p)))
    assert t.utterances and len(t.text.split()) > 10
    assert {u.speaker for u in t.utterances} <= {"speaker_00", "speaker_01", "BELIRSIZ"}

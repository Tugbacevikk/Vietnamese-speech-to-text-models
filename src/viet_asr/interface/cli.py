"""Komut satırı: tek bir ses dosyasını konuşmacılı metne çevirir.

    python -m viet_asr.interface.cli kayit.wav                 # dil sesten tespit edilir
    python -m viet_asr.interface.cli kayit.wav --lang vi       # dil verilir, tespit atlanır
    python -m viet_asr.interface.cli kayit.wav --tr-model omnilingual-300m --json sonuc.json

API (FastAPI) ve kuyruk (Celery) Faz 1'de bu katmana eklenecek; ikisi de
aynı build() ile kurulan TranscribeAndDiarize senaryosunu çağıracak.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from viet_asr.application.config import (
    DiarizerConfig, LidConfig, PipelineConfig, VadConfig,
)
from viet_asr.application.fusion import FusionEngine
from viet_asr.application.language import LanguageResolver
from viet_asr.application.transcribe import TranscribeAndDiarize
from viet_asr.domain import AudioJob, JobStatus, Transcript, UnsupportedLanguageError
from viet_asr.infrastructure.audio import SoundfileLoader
from viet_asr.infrastructure.catalog import MODEL_CHOICES, language_profile, lid_stages
from viet_asr.infrastructure.engines import LazyAsrEngines
from viet_asr.infrastructure.normalizer import normalizer_for
from viet_asr.infrastructure.sherpa_asr import build_asr
from viet_asr.infrastructure.sherpa_diarizer import SherpaOnnxDiarizer
from viet_asr.infrastructure.sherpa_vad import SileroVad
from viet_asr.infrastructure.whisper_lid import LazyDetector, WhisperLanguageDetector


class MemoryRepo:
    def __init__(self):
        self.status, self.results = {}, {}

    def set_status(self, job_id, status: JobStatus, error=None):
        self.status[job_id] = (status, error)

    def save(self, transcript: Transcript):
        self.results[transcript.job_id] = transcript


def build(cfg: PipelineConfig, repo) -> TranscribeAndDiarize:
    def factory(profile):
        return lambda: build_asr(profile, normalizer_for(profile.code))

    engines = LazyAsrEngines({code: factory(p) for code, p in cfg.languages.items()})
    detectors = []
    if cfg.lid is not None and len(cfg.languages) > 1:
        for i, (enc, dec) in enumerate(cfg.lid.stages):
            make = (lambda e=enc, d=dec: WhisperLanguageDetector(e, d, cfg.lid.num_threads))
            detectors.append(make() if i == 0 else LazyDetector(make))
    return TranscribeAndDiarize(
        loader=SoundfileLoader(), vad=SileroVad(cfg.vad),
        language=LanguageResolver(list(cfg.languages), detectors, cfg.lid),
        engines=engines, diarizer=SherpaOnnxDiarizer(cfg.diarizer),
        fusion=FusionEngine(cfg.fusion), repo=repo)


def default_config(models_dir: str | Path = "models", languages=("vi", "tr"),
                   models: dict[str, str] | None = None, threads: int = 2,
                   speakers: int = 2, hotwords: str | None = None) -> PipelineConfig:
    d, models = Path(models_dir), models or {}
    profiles = {}
    for code in languages:
        extra = {"num_threads": threads}
        if code == "vi" and hotwords:
            extra["hotwords_file"] = hotwords
        profiles[code] = language_profile(code, d, models.get(code), **extra)
    return PipelineConfig(
        languages=profiles,
        lid=LidConfig(stages=lid_stages(d), num_threads=threads),
        vad=VadConfig(model=str(d / "silero_vad.onnx")),
        diarizer=DiarizerConfig(
            segmentation_model=str(d / "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            embedding_model=str(d / "wespeaker_en_voxceleb_resnet34_LM.onnx"),
            num_speakers=speakers, num_threads=threads))


def main(argv=None):
    p = argparse.ArgumentParser(description="Ses -> konuşmacılı metin (Vietnamca, Türkçe)")
    p.add_argument("audio")
    p.add_argument("--lang", default="auto", help="vi, tr ya da auto (sesten tespit)")
    p.add_argument("--languages", default="vi,tr", help="sistemde açık diller")
    p.add_argument("--vi-model", choices=MODEL_CHOICES["vi"], default=None)
    p.add_argument("--tr-model", choices=MODEL_CHOICES["tr"], default=None)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--speakers", type=int, default=2, help="-1: otomatik")
    p.add_argument("--hotwords", default=None, help="Vietnamca terim listesi (BÜYÜK HARF)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    models = {k: v for k, v in (("vi", a.vi_model), ("tr", a.tr_model)) if v}
    cfg = default_config(a.models_dir, tuple(a.languages.split(",")), models, a.threads,
                         a.speakers, a.hotwords)
    repo = MemoryRepo()
    job = AudioJob(job_id=Path(a.audio).stem, audio_path=a.audio,
                   language=None if a.lang == "auto" else a.lang)
    try:
        t = build(cfg, repo).run(job)
    except UnsupportedLanguageError as e:
        print(f"REDDEDİLDİ: {e}", file=sys.stderr)
        return 2
    lang = t.language
    print(f"# dil: {lang.code} ({lang.source.value}, güven {lang.confidence:.2f}"
          f"{', BELİRSİZ' if lang.uncertain else ''}) model: {t.model}")
    for u in t.utterances:
        print(f"[{u.start:7.2f} - {u.end:7.2f}] {u.speaker}: {u.text}")
    if a.json:
        Path(a.json).write_text(json.dumps(asdict(t), ensure_ascii=False, indent=1, default=str),
                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

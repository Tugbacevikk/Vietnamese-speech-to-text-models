"""Komut satırı: tek bir ses dosyasını konuşmacılı metne çevirir.

    python -m viet_asr.interface.cli ses.wav --model 68m
    python -m viet_asr.interface.cli ses.wav --model 30m --json sonuc.json

API (FastAPI) ve kuyruk (Celery) Faz 1'de bu katmana eklenecek; ikisi de
aynı TranscribeAndDiarize senaryosunu çağıracak.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from viet_asr.application.config import (
    ZIPFORMER_30M, ZIPFORMER_68M, AsrParams, DiarizerConfig, ModelRef, PipelineConfig, VadConfig,
)
from viet_asr.application.fusion import FusionEngine
from viet_asr.application.transcribe import TranscribeAndDiarize
from viet_asr.domain import AudioJob, JobStatus, Transcript
from viet_asr.infrastructure.audio import SoundfileLoader
from viet_asr.infrastructure.normalizer import ViNormalizer
from viet_asr.infrastructure.sherpa_asr import SherpaOnnxAsr
from viet_asr.infrastructure.sherpa_diarizer import SherpaOnnxDiarizer
from viet_asr.infrastructure.sherpa_vad import SileroVad

MODELS = {
    "68m": ("sherpa-onnx-zipformer-vi-int8-2025-04-20", ZIPFORMER_68M),
    "30m": ("sherpa-onnx-zipformer-vi-30M-int8-2026-02-09", ZIPFORMER_30M),
}


class MemoryRepo:
    def __init__(self):
        self.status, self.results = {}, {}

    def set_status(self, job_id, status: JobStatus, error=None):
        self.status[job_id] = (status, error)

    def save(self, transcript: Transcript):
        self.results[transcript.job_id] = transcript


def build(cfg: PipelineConfig, repo) -> TranscribeAndDiarize:
    return TranscribeAndDiarize(
        loader=SoundfileLoader(), vad=SileroVad(cfg.vad),
        asr=SherpaOnnxAsr(cfg.asr_model, cfg.asr, ViNormalizer()),
        diarizer=SherpaOnnxDiarizer(cfg.diarizer), fusion=FusionEngine(cfg.fusion), repo=repo)


def main(argv=None):
    p = argparse.ArgumentParser(description="Vietnamca ses -> konuşmacılı metin")
    p.add_argument("audio")
    p.add_argument("--model", choices=MODELS, default="68m")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--speakers", type=int, default=2, help="-1: otomatik")
    p.add_argument("--hotwords", default=None, help="terim listesi (BÜYÜK HARF)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    d = Path(a.models_dir)
    folder, spec = MODELS[a.model]
    cfg = PipelineConfig(
        asr_model=ModelRef(folder=d / folder, **spec),
        asr=AsrParams(num_threads=a.threads, hotwords_file=a.hotwords),
        vad=VadConfig(model=str(d / "silero_vad.onnx")),
        diarizer=DiarizerConfig(
            segmentation_model=str(d / "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            embedding_model=str(d / "wespeaker_en_voxceleb_resnet34_LM.onnx"),
            num_speakers=a.speakers, num_threads=a.threads))

    repo = MemoryRepo()
    t = build(cfg, repo).run(AudioJob(job_id=Path(a.audio).stem, audio_path=a.audio))
    for u in t.utterances:
        print(f"[{u.start:7.2f} - {u.end:7.2f}] {u.speaker}: {u.text}")
    if a.json:
        Path(a.json).write_text(json.dumps(asdict(t), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

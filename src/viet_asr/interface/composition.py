"""Bileşenleri birbirine bağlayan tek yer (CLI, API ve worker'lar buradan kurar)."""
from __future__ import annotations

from pathlib import Path

from viet_asr.application.config import DiarizerConfig, LidConfig, PipelineConfig, VadConfig
from viet_asr.application.fusion import FusionEngine
from viet_asr.application.language import LanguageResolver
from viet_asr.application.ports import ResultRepository
from viet_asr.application.transcribe import TranscribeAndDiarize
from viet_asr.infrastructure.audio import SoundfileLoader
from viet_asr.infrastructure.catalog import language_profile, lid_stages
from viet_asr.infrastructure.engines import LazyAsrEngines
from viet_asr.infrastructure.normalizer import normalizer_for
from viet_asr.infrastructure.sherpa_asr import build_asr
from viet_asr.infrastructure.cluster_diarizer import EmbeddingClusterDiarizer
from viet_asr.infrastructure.sherpa_diarizer import SherpaOnnxDiarizer
from viet_asr.infrastructure.sherpa_vad import SileroVad
from viet_asr.infrastructure.whisper_lid import LazyDetector, WhisperLanguageDetector


def default_config(models_dir: str | Path = "models", languages=("vi", "tr"),
                   models: dict[str, str] | None = None, threads: int = 2,
                   speakers: int = 0, hotwords: str | None = None,
                   diarizer: str = "cluster") -> PipelineConfig:
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
            kind=diarizer,
            segmentation_model=str(d / "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            embedding_model=str(d / ("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
                                     if diarizer == "cluster" else "wespeaker_en_voxceleb_resnet34_LM.onnx")),
            num_speakers=speakers, num_threads=threads))


def build_diarizer(cfg: DiarizerConfig):
    if cfg.kind == "cluster":
        return EmbeddingClusterDiarizer(cfg)
    if cfg.kind == "pyannote":
        return SherpaOnnxDiarizer(cfg)
    raise ValueError(f"bilinmeyen diarizer: {cfg.kind}")


def build_resolver(cfg: PipelineConfig, with_detection: bool = True) -> LanguageResolver:
    """with_detection=False: ASR worker'ları dili kayıttan alır, Whisper LID yüklemez."""
    detectors = []
    if with_detection and cfg.lid is not None and len(cfg.languages) > 1:
        for i, (enc, dec) in enumerate(cfg.lid.stages):
            make = (lambda e=enc, d=dec: WhisperLanguageDetector(e, d, cfg.lid.num_threads))
            detectors.append(make() if i == 0 else LazyDetector(make))  # small yalnızca gerekince
    return LanguageResolver(list(cfg.languages), detectors, cfg.lid)


def build_engines(cfg: PipelineConfig) -> LazyAsrEngines:
    def factory(profile):
        return lambda: build_asr(profile, normalizer_for(profile.code))

    return LazyAsrEngines({code: factory(p) for code, p in cfg.languages.items()})


def build_pipeline(cfg: PipelineConfig, repo: ResultRepository,
                   with_detection: bool = True) -> TranscribeAndDiarize:
    return TranscribeAndDiarize(
        loader=SoundfileLoader(), vad=SileroVad(cfg.vad),
        language=build_resolver(cfg, with_detection), engines=build_engines(cfg),
        diarizer=build_diarizer(cfg.diarizer), fusion=FusionEngine(cfg.fusion), repo=repo)


# geriye dönük ad (CLI ve testler)
build = build_pipeline

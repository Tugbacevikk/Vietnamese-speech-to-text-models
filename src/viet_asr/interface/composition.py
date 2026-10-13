"""Bileşenleri birbirine bağlayan tek yer (CLI, API ve worker'lar buradan kurar)."""
from __future__ import annotations

from pathlib import Path

from viet_asr.application.config import DiarizerConfig, LidConfig, PipelineConfig, VadConfig
from viet_asr.application.language import LanguageResolver
from viet_asr.application.ports import ResultRepository
from viet_asr.application.transcribe import TranscribeAndDiarize
from viet_asr.infrastructure.audio import SoundfileLoader
from viet_asr.infrastructure.chunker import EnergyChunker
from viet_asr.infrastructure.native_fusion import build_fusion
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
            embedding_model=str(d / ("3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
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
        diarizer=build_diarizer(cfg.diarizer), fusion=build_fusion(cfg.fusion), repo=repo,
        chunker=EnergyChunker(cfg.chunk.max_len, cfg.chunk.min_len))


# geriye dönük ad (CLI ve testler)
build = build_pipeline


def model_files(cfg: PipelineConfig) -> list[Path]:
    """Bu yapılandırmanın gerçekten kullandığı model dosyaları."""
    files = [Path(cfg.vad.model), Path(cfg.diarizer.embedding_model)]
    if cfg.diarizer.kind == "pyannote":
        files.append(Path(cfg.diarizer.segmentation_model))
    if cfg.lid is not None and len(cfg.languages) > 1:
        files += [Path(p) for stage in cfg.lid.stages for p in stage]
    for prof in cfg.languages.values():
        files += [Path(prof.asr_model.path(role)) for role in prof.asr_model.files
                  if role != "bpe_vocab" or prof.asr.hotwords_file]
    return files


def verify_models(models_dir: str | Path, files: list[Path]) -> list[str]:
    """MANIFEST.json'daki sha256 ile karşılaştırır. Sorun listesi döndürür (boş = tamam).
    İnternetsiz sunucuya taşınan modellerin bozulmadığını/değiştirilmediğini garanti eder."""
    import hashlib
    import json

    md = Path(models_dir)
    mpath = md / "MANIFEST.json"
    if not mpath.exists():
        return [f"{mpath} yok: scripts/download_models.py ile üretin"]
    expected = {}
    for pack in json.loads(mpath.read_text(encoding="utf-8")).values():
        expected.update(pack.get("sha256", {}))
    problems = []
    for f in files:
        if not f.exists():
            problems.append(f"dosya yok: {f}")
            continue
        try:
            rel = f.resolve().relative_to(md.resolve()).as_posix()
        except ValueError:
            continue  # models klasörü dışında (ör. ince ayarlı model): manifest kapsamaz
        if rel not in expected:
            if f.suffix in (".onnx", ".bin"):  # ağırlık dosyası doğrulanmadan çalışmaz
                problems.append(f"MANIFEST'te yok: {rel}")
            continue
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected[rel]:
            problems.append(f"sha256 uyuşmuyor (dosya bozuk ya da değişmiş): {rel}")
    return problems


def verify_models_or_exit(settings) -> None:
    import logging

    if not settings.verify_models:
        return
    cfg = default_config(settings.models_dir, tuple(settings.languages), settings.models,
                         settings.threads, settings.speakers, settings.hotwords, settings.diarizer)
    problems = verify_models(settings.models_dir, model_files(cfg))
    if problems:
        raise SystemExit("Model doğrulaması başarısız (VERIFY_MODELS=0 ile atlanabilir):\n  "
                         + "\n  ".join(problems))
    logging.getLogger(__name__).info("modeller doğrulandı (sha256)")


def safe_file_remover(data_dir: str | Path):
    """Yalnızca yükleme klasörü içindeki dosyaları siler (ALLOW_LOCAL_PATHS ile verilen
    kullanıcı dosyalarına asla dokunmaz)."""
    root = Path(data_dir).resolve()

    def remove(path: str) -> bool:
        try:
            p = Path(path).resolve()
            if root not in p.parents:
                return False
            p.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    return remove


def build_maintenance(store, settings):
    from viet_asr.application.maintenance import Maintenance

    return Maintenance(store, safe_file_remover(settings.data_dir), settings.audio_retention_hours,
                       settings.result_retention_days, settings.processing_stale_sec)

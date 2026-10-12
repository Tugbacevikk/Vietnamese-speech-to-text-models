"""Hattın tüm ayarları tek yerde. Kodda sabit sayı yerine buradan okunur."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelRef:
    """Bir ASR model paketinin dosyaları (klasöre göre)."""

    name: str
    folder: Path
    encoder: str
    decoder: str
    joiner: str
    tokens: str = "tokens.txt"
    bpe_vocab: str | None = "bpe.vocab"  # hotword için zorunlu
    license: str = ""

    def path(self, file: str) -> str:
        return str(self.folder / file)


ZIPFORMER_68M = dict(
    name="zipformer-vi-68m-int8-2025-04-20",
    encoder="encoder-epoch-12-avg-8.int8.onnx",
    decoder="decoder-epoch-12-avg-8.onnx",
    joiner="joiner-epoch-12-avg-8.int8.onnx",
    license="Apache-2.0",
)
ZIPFORMER_30M = dict(
    name="zipformer-vi-30m-int8-2026-02-09",
    encoder="encoder.int8.onnx",
    decoder="decoder.onnx",
    joiner="joiner.int8.onnx",
    license="CC BY-NC-ND 4.0 (ticari kullanım ve ince ayar yasak)",
)


@dataclass(frozen=True)
class AsrParams:
    num_threads: int = 2
    decoding_method: str = "modified_beam_search"
    hotwords_file: str | None = None  # terimler BÜYÜK HARFLE yazılmalı
    hotwords_score: float = 1.5
    max_syllable_duration: float = 1.0  # tek hece en fazla bu kadar sürebilir


@dataclass(frozen=True)
class VadConfig:
    model: str = "models/silero_vad.onnx"
    threshold: float = 0.5
    min_silence_duration: float = 0.25
    min_speech_duration: float = 0.25
    max_speech_duration: float = 20.0
    sample_rate: int = 16000


@dataclass(frozen=True)
class DiarizerConfig:
    segmentation_model: str = "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    embedding_model: str = "models/wespeaker_en_voxceleb_resnet34_LM.onnx"
    num_speakers: int = 2  # doktor + hasta; bilinmiyorsa -1 ve threshold kullanılır
    cluster_threshold: float = 0.5
    min_duration_on: float = 0.3
    min_duration_off: float = 0.5
    num_threads: int = 2


@dataclass(frozen=True)
class FusionConfig:
    """Heceleri konuşmacılara atama ayarları."""

    # hecenin orta noktası iki konuşmacı aralığına birden düşüyorsa:
    # ikisinin çakışma payı bu oranın altındaysa BELIRSIZ işaretlenir
    min_dominance: float = 0.6
    # orta nokta hiçbir aralığa düşmüyorsa en yakın aralığa bu mesafeye kadar atanır (sn)
    max_gap: float = 0.5
    # aynı konuşmacının arasında bu kadar sessizlik varsa yeni satır başlar (sn)
    new_utterance_gap: float = 1.5


@dataclass(frozen=True)
class PipelineConfig:
    asr_model: ModelRef
    asr: AsrParams = field(default_factory=AsrParams)
    vad: VadConfig = field(default_factory=VadConfig)
    diarizer: DiarizerConfig = field(default_factory=DiarizerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

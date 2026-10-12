"""Hattın tüm ayarları tek yerde. Kodda sabit sayı yerine buradan okunur.

Her dil bir LanguageProfile'dır: hangi ASR modeli, hangi ayarlarla.
Yeni dil eklemek = yeni bir profil eklemek; uygulama kodu değişmez.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelRef:
    """Bir model paketinin dosyaları.

    kind: altyapıda hangi adaptörün kullanılacağı
          ("zipformer-transducer", "whisper", "omnilingual-ctc")
    files: rol -> dosya adı (klasöre göre), ör. {"encoder": "encoder.int8.onnx"}
    """

    name: str
    kind: str
    folder: Path
    files: dict[str, str]
    license: str = ""

    def path(self, role: str) -> str:
        return str(Path(self.folder) / self.files[role])

    def has(self, role: str) -> bool:
        return role in self.files and (Path(self.folder) / self.files[role]).exists()


@dataclass(frozen=True)
class AsrParams:
    num_threads: int = 2
    decoding_method: str = "modified_beam_search"  # yalnızca zipformer
    hotwords_file: str | None = None  # yalnızca zipformer; terimler BÜYÜK HARFLE
    hotwords_score: float = 1.5
    max_word_duration: float = 1.0  # tek kelime/hece en fazla bu kadar sürebilir (sn)
    # Whisper sessizlik/gürültüde bu cümleleri uyduruyor; parça yalnızca bundan ibaretse atılır
    drop_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class LanguageProfile:
    code: str  # ISO 639-1: "vi", "tr"
    asr_model: ModelRef
    asr: AsrParams = field(default_factory=AsrParams)


@dataclass(frozen=True)
class LidConfig:
    """Dil tespiti (Whisper encoder + tek adım decoder), ucuzdan pahalıya kademeli.

    stages: (encoder, decoder) çiftleri. İlki karar veremezse (ret ya da düşük
    güven) sıradaki denenir. Varsayılan: tiny (~0,25 sn) -> small (~1 sn).
    """

    stages: tuple[tuple[str, str], ...] = (
        ("models/sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx",
         "models/sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx"),
        ("models/sherpa-onnx-whisper-small/small-encoder.int8.onnx",
         "models/sherpa-onnx-whisper-small/small-decoder.int8.onnx"),
    )
    num_threads: int = 2
    max_seconds: float = 30.0      # Whisper penceresi 30 sn; ilk 30 sn konuşma yeter
    min_supported: float = 0.5     # desteklenen dillerin toplam olasılığı bunun altındaysa ret
    min_confidence: float = 0.8    # kısıtlı seçimde güven bunun altındaysa "uncertain"


@dataclass(frozen=True)
class VadConfig:
    model: str = "models/silero_vad.onnx"
    threshold: float = 0.5
    min_silence_duration: float = 0.25
    min_speech_duration: float = 0.25
    max_speech_duration: float = 20.0  # Whisper 30 sn'den uzun parçayı kesiyor; 20 güvenli
    # Silero parça sonunu erken kesiyor: Türkçede "mı?", "alın" gibi son kelimeler
    # kayboldu (12 Ekim). Her parça iki yandan bu kadar genişletilir (komşuya taşmadan).
    pad: float = 0.2
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
    """Kelimeleri konuşmacılara atama ayarları."""

    # kelimenin orta noktası iki konuşmacı aralığına birden düşüyorsa:
    # baskın konuşmacının örtüşme payı bu oranın altındaysa BELIRSIZ
    min_dominance: float = 0.6
    # orta nokta hiçbir aralığa düşmüyorsa en yakın aralığa bu mesafeye kadar atanır (sn)
    max_gap: float = 0.5
    # aynı konuşmacının arasında bu kadar sessizlik varsa yeni satır başlar (sn)
    new_utterance_gap: float = 1.5


@dataclass(frozen=True)
class PipelineConfig:
    languages: dict[str, LanguageProfile]
    lid: LidConfig | None = field(default_factory=LidConfig)  # None: dil istekte zorunlu
    vad: VadConfig = field(default_factory=VadConfig)
    diarizer: DiarizerConfig = field(default_factory=DiarizerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

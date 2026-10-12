"""Model kataloğu: indirme adresleri, klasör adları, dosya rolleri, lisanslar.

Hem scripts/download_models.py hem de CLI buradan okur; model bilgisi tek yerde.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from viet_asr.application.config import AsrParams, LanguageProfile, ModelRef

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class Pack:
    url_path: str           # BASE'e göre
    target: str             # açılınca oluşan klasör ya da dosya adı
    license: str
    kind: str | None = None  # ASR ise adaptör türü
    files: dict[str, str] = field(default_factory=dict)
    needs_bpe_vocab: bool = False
    keep: tuple[str, ...] = ()  # boş değilse açıldıktan sonra yalnızca bu dosyalar kalır


ASR_PACKS: dict[str, Pack] = {
    "zipformer-68m": Pack(
        "asr-models/sherpa-onnx-zipformer-vi-int8-2025-04-20.tar.bz2",
        "sherpa-onnx-zipformer-vi-int8-2025-04-20", "Apache-2.0", "zipformer-transducer",
        {"encoder": "encoder-epoch-12-avg-8.int8.onnx", "decoder": "decoder-epoch-12-avg-8.onnx",
         "joiner": "joiner-epoch-12-avg-8.int8.onnx", "tokens": "tokens.txt", "bpe_vocab": "bpe.vocab"},
        needs_bpe_vocab=True),
    "zipformer-30m": Pack(
        "asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2",
        "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        "CC BY-NC-ND 4.0 (ticari kullanım ve ince ayar yasak)", "zipformer-transducer",
        {"encoder": "encoder.int8.onnx", "decoder": "decoder.onnx", "joiner": "joiner.int8.onnx",
         "tokens": "tokens.txt", "bpe_vocab": "bpe.vocab"},
        needs_bpe_vocab=True),
    "whisper-turbo": Pack(
        "asr-models/sherpa-onnx-whisper-turbo.tar.bz2", "sherpa-onnx-whisper-turbo", "MIT",
        "whisper",
        {"encoder": "turbo-encoder.int8.onnx", "decoder": "turbo-decoder.int8.onnx",
         "tokens": "turbo-tokens.txt"}),
    "omnilingual-300m": Pack(
        "asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2",
        "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12", "Apache-2.0",
        "omnilingual-ctc", {"model": "model.int8.onnx", "tokens": "tokens.txt"}),
}

AUX_PACKS: dict[str, Pack] = {
    "silero-vad": Pack("asr-models/silero_vad.onnx", "silero_vad.onnx", "MIT"),
    "pyannote-seg": Pack(
        "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "sherpa-onnx-pyannote-segmentation-3-0", "MIT"),
    "wespeaker": Pack(
        # 'recongition' yazım hatası sherpa-onnx'in kendi adresinde var, düzeltmeyin
        "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
        "wespeaker_en_voxceleb_resnet34_LM.onnx",
        "WeSpeaker (Apache-2.0 kod; VoxCeleb ile eğitilmiş ağırlık lisansı doğrulanacak)"),
    "3dspeaker": Pack(
        # 'recongition' yazım hatası sherpa-onnx'in kendi adresinde var, düzeltmeyin
        "speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "3D-Speaker / ModelScope (Apache-2.0 olduğu doğrulanacak)"),
    "whisper-tiny-lid": Pack(
        "asr-models/sherpa-onnx-whisper-tiny.tar.bz2", "sherpa-onnx-whisper-tiny", "MIT",
        keep=("tiny-encoder.int8.onnx", "tiny-decoder.int8.onnx", "tiny-tokens.txt")),
    "whisper-small-lid": Pack(
        "asr-models/sherpa-onnx-whisper-small.tar.bz2", "sherpa-onnx-whisper-small", "MIT",
        keep=("small-encoder.int8.onnx", "small-decoder.int8.onnx", "small-tokens.txt")),
    "lid-test-wavs": Pack(
        "asr-models/spoken-language-identification-test-wavs.tar.bz2",
        "spoken-language-identification-test-wavs", "test sesleri (sherpa-onnx)"),
}

# Her dilin varsayılan modeli. Ölçüm (12 Ekim, 2 çekirdek):
#   Türkçe, 24 sentetik cümle:  whisper-turbo CER %3,0 | omnilingual-300m CER %3,8
#   Türkçe, 60 sn iki konuşmacılı diyalog (23 VAD parçası), tüm hat:
#     whisper-turbo 63,5 sn (ASR 53,7)  | omnilingual-300m 24,9 sn (ASR 12,1)
#   Whisper her parçayı 30 sn'ye doldurup işliyor; kısa konuşma sıralarında
#   maliyet parça sayısıyla artıyor ve kelime zamanı vermiyor. Omnilingual
#   gerçek zaman damgası veriyor, ama "varmı", "birkan" gibi bitişik yazım
#   hataları ve noktalamasız küçük harf çıktı var.
#   -> varsayılan hız hedefini (1 dk ses <= 1 dk) karşılayan omnilingual-300m;
#      doğruluk öncelikliyse whisper-turbo seçilebilir.
DEFAULT_MODELS = {"vi": "zipformer-68m", "tr": "omnilingual-300m"}
MODEL_CHOICES = {"vi": ["zipformer-68m", "zipformer-30m"],
                 "tr": ["omnilingual-300m", "whisper-turbo"]}

# Whisper'ın sessizlik/gürültüde uydurduğu, gözlenmiş cümleler
DROP_PHRASES = {"tr": ("İzlediğiniz için teşekkür ederim.",)}


def lid_stages(models_dir: str | Path) -> tuple[tuple[str, str], ...]:
    d = Path(models_dir)
    return tuple((str(d / f"sherpa-onnx-whisper-{s}/{s}-encoder.int8.onnx"),
                  str(d / f"sherpa-onnx-whisper-{s}/{s}-decoder.int8.onnx")) for s in ("tiny", "small"))


def model_ref(key: str, models_dir: str | Path) -> ModelRef:
    p = ASR_PACKS[key]
    return ModelRef(name=key, kind=p.kind, folder=Path(models_dir) / p.target,
                    files=dict(p.files), license=p.license)


def language_profile(code: str, models_dir: str | Path, model: str | None = None,
                     **params) -> LanguageProfile:
    key = model or DEFAULT_MODELS[code]
    if key not in MODEL_CHOICES.get(code, []):
        raise ValueError(f"'{key}' modeli '{code}' dili için tanımlı değil: {MODEL_CHOICES.get(code)}")
    params.setdefault("drop_phrases", DROP_PHRASES.get(code, ()))
    ref = model_ref(key, models_dir)
    # Whisper sessizlik/müzikte metin uyduruyor: ona yalnızca VAD parçaları
    seg = "vad" if ref.kind == "whisper" else "full"
    return LanguageProfile(code=code, asr_model=ref, asr=AsrParams(**params), segmentation=seg)

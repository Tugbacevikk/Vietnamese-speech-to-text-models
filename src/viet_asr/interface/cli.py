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
from dataclasses import asdict, replace
from pathlib import Path

from viet_asr.domain import AudioJob, UnsupportedLanguageError
from viet_asr.infrastructure.catalog import MODEL_CHOICES
from viet_asr.infrastructure.memory_store import MemoryJobStore
from viet_asr.interface.composition import build, config_from_settings, default_config  # noqa: F401  (testler kullanır)
from viet_asr.interface.settings import Settings

MemoryRepo = MemoryJobStore  # geriye dönük ad
__all__ = ["MemoryRepo", "build", "default_config", "main"]


def main(argv=None):
    p = argparse.ArgumentParser(description="Ses -> konuşmacılı metin (Vietnamca, Türkçe)")
    p.add_argument("audio")
    p.add_argument("--lang", default="auto", help="vi, tr ya da auto (sesten tespit)")
    p.add_argument("--languages", default=None, help="sistemde açık diller (boşsa LANGUAGES, varsayılan vi,tr)")
    p.add_argument("--vi-model", default=None, help=f"{MODEL_CHOICES['vi']} ya da model klasörü")
    p.add_argument("--tr-model", default=None, help=f"{MODEL_CHOICES['tr']} ya da model klasörü")
    p.add_argument("--models-dir", default=None)
    p.add_argument("--speakers", type=int, default=None, help="konuşmacı sayısı; 0: sesten bul")
    p.add_argument("--diarizer", choices=["cluster", "pyannote"], default=None)
    p.add_argument("--hotwords", default=None, help="Vietnamca terim listesi (BÜYÜK HARF)")
    p.add_argument("--threads", type=int, default=None, help="ONNX iş parçacığı (boşsa ONNX_THREADS)")
    p.add_argument("--whisper-device", choices=["auto", "cpu", "cuda"], default=None,
                   help="Whisper cihazı (boşsa WHISPER_DEVICE)")
    p.add_argument("--whisper-compute-type", default=None, help="float16, int8, auto (boşsa WHISPER_COMPUTE_TYPE)")
    p.add_argument("--whisper-beam-size", type=int, default=None, help="boşsa WHISPER_BEAM_SIZE")
    p.add_argument("--whisper-threads", type=int, default=None, help="boşsa WHISPER_THREADS (auto: fiziksel çekirdek)")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Ayarlar: komut satırı > ortam değişkeni / .env > varsayılan (sunucuyla aynı kaynak)
    s = Settings()
    over = {"models_dir": a.models_dir, "speakers": a.speakers, "diarizer": a.diarizer,
            "hotwords": a.hotwords, "threads": a.threads, "vi_model": a.vi_model,
            "tr_model": a.tr_model, "whisper_device": a.whisper_device,
            "whisper_compute_type": a.whisper_compute_type,
            "whisper_beam_size": a.whisper_beam_size,
            "whisper_threads_raw": None if a.whisper_threads is None else str(a.whisper_threads),
            "languages": None if a.languages is None else a.languages.split(",")}
    s = replace(s, **{k: v for k, v in over.items() if v is not None})
    cfg = config_from_settings(s)
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

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

from viet_asr.domain import AudioJob, UnsupportedLanguageError
from viet_asr.infrastructure.catalog import MODEL_CHOICES
from viet_asr.infrastructure.memory_store import MemoryJobStore
from viet_asr.interface.composition import build, default_config  # noqa: F401  (testler kullanır)

MemoryRepo = MemoryJobStore  # geriye dönük ad


def main(argv=None):
    p = argparse.ArgumentParser(description="Ses -> konuşmacılı metin (Vietnamca, Türkçe)")
    p.add_argument("audio")
    p.add_argument("--lang", default="auto", help="vi, tr ya da auto (sesten tespit)")
    p.add_argument("--languages", default="vi,tr", help="sistemde açık diller")
    p.add_argument("--vi-model", choices=MODEL_CHOICES["vi"], default=None)
    p.add_argument("--tr-model", choices=MODEL_CHOICES["tr"], default=None)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--speakers", type=int, default=0, help="konuşmacı sayısı; 0: sesten bul")
    p.add_argument("--diarizer", choices=["cluster", "pyannote"], default="cluster")
    p.add_argument("--hotwords", default=None, help="Vietnamca terim listesi (BÜYÜK HARF)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    models = {k: v for k, v in (("vi", a.vi_model), ("tr", a.tr_model)) if v}
    cfg = default_config(a.models_dir, tuple(a.languages.split(",")), models, a.threads,
                         a.speakers, a.hotwords, a.diarizer)
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

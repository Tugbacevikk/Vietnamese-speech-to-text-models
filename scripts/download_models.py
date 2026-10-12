"""Tüm modelleri models/ klasörüne indirir, açar ve bpe.vocab üretir.

    python scripts/download_models.py            # her şey
    python scripts/download_models.py --asr 68m  # yalnızca 68M ASR + VAD + diarization

Windows ve Linux'ta aynı çalışır (yalnızca Python standart kütüphanesi + sentencepiece).
İnternetsiz (air-gapped) sunucu için: bunu internetli makinede çalıştırıp models/
klasörünü olduğu gibi taşıyın. MANIFEST.json içindeki sha256 değerleri ile kontrol edin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
ASR = {
    "68m": ("asr-models/sherpa-onnx-zipformer-vi-int8-2025-04-20.tar.bz2", "Apache-2.0"),
    "30m": ("asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2",
            "CC BY-NC-ND 4.0"),
}
COMMON = {
    "silero_vad.onnx": ("asr-models/silero_vad.onnx", "MIT"),
    "pyannote-segmentation-3-0": (
        "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2", "MIT"),
    "wespeaker_en_voxceleb_resnet34_LM.onnx": (
        # 'recongition' yazım hatası sherpa-onnx'in kendi adresinde var, düzeltmeyin
        "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
        "WeSpeaker (Apache-2.0 kod; VoxCeleb ile eğitilmiş ağırlık lisansı doğrulanacak)"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel: str, dest: Path) -> Path:
    out = dest / rel.rsplit("/", 1)[1]
    if not out.exists():
        print(f"indiriliyor: {rel}")
        urllib.request.urlretrieve(f"{BASE}/{rel}", out)
    if out.suffix == ".bz2":
        with tarfile.open(out, "r:bz2") as t:
            t.extractall(dest, filter="data")
        out.unlink()
        out = dest / out.name.removesuffix(".tar.bz2")
    return out


def make_bpe_vocab(model_dir: Path) -> None:
    """Hotword için gerekli bpe.vocab (yoksa sherpa-onnx hotword'leri sessizce yok sayar)."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_dir / "bpe.model"))
    with (model_dir / "bpe.vocab").open("w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asr", choices=[*ASR, "all"], default="all")
    p.add_argument("--dest", default="models")
    a = p.parse_args()
    dest = Path(a.dest)
    dest.mkdir(exist_ok=True)

    manifest = {}
    items = {**COMMON, **{k: v for k, v in ASR.items() if a.asr in (k, "all")}}
    for key, (rel, lic) in items.items():
        path = fetch(rel, dest)
        if key in ASR:
            make_bpe_vocab(path)
        files = [path] if path.is_file() else sorted(f for f in path.rglob("*.onnx"))
        manifest[path.name] = {"url": f"{BASE}/{rel}", "license": lic,
                               "sha256": {f.relative_to(dest).as_posix(): sha256(f) for f in files}}
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"Tamam. {dest}/MANIFEST.json yazıldı.")


if __name__ == "__main__":
    main()

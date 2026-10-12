"""Modelleri models/ klasörüne indirir, açar, bpe.vocab ve MANIFEST.json üretir.

    python scripts/download_models.py                          # varsayılanlar: vi=zipformer-68m, tr=omnilingual-300m
    python scripts/download_models.py --vi zipformer-30m --tr whisper-turbo
    python scripts/download_models.py --vi all --tr all        # tüm adaylar (karşılaştırma için)
    python scripts/download_models.py --tr none                # yalnızca Vietnamca

Her zaman inenler: Silero VAD, pyannote segmentation, WeSpeaker, Whisper-tiny ve -small (dil tespiti).
İnternetsiz (air-gapped) sunucu için: bunu internetli makinede çalıştırıp models/
klasörünü olduğu gibi taşıyın. MANIFEST.json içindeki sha256 değerleri ile kontrol edin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from viet_asr.infrastructure.catalog import ASR_PACKS, AUX_PACKS, BASE, MODEL_CHOICES, Pack  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(pack: Pack, dest: Path) -> Path:
    out = dest / pack.target
    if out.exists():
        print(f"var, atlanıyor: {pack.target}")
        return out
    name = pack.url_path.rsplit("/", 1)[1]
    tmp = dest / name
    print(f"indiriliyor: {name}")
    urllib.request.urlretrieve(f"{BASE}/{pack.url_path}", tmp)
    if name.endswith(".tar.bz2"):
        with tarfile.open(tmp, "r:bz2") as t:
            t.extractall(dest, filter="data")
        tmp.unlink()
    if pack.keep and out.is_dir():
        for f in out.iterdir():
            if f.is_file() and f.name not in pack.keep:
                f.unlink()  # ör. fp32 kopyalar: yalnızca int8 kullanılıyor
    return out


def make_bpe_vocab(model_dir: Path) -> None:
    """Hotword için gerekli bpe.vocab (yoksa sherpa-onnx hotword'leri sessizce yok sayar)."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_dir / "bpe.model"))
    with (model_dir / "bpe.vocab").open("w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")


def choose(arg: str, lang: str) -> list[str]:
    if arg == "none":
        return []
    if arg == "all":
        return MODEL_CHOICES[lang]
    return [arg]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vi", choices=[*MODEL_CHOICES["vi"], "all", "none"], default="zipformer-68m")
    p.add_argument("--tr", choices=[*MODEL_CHOICES["tr"], "all", "none"], default="omnilingual-300m")
    p.add_argument("--dest", default="models")
    a = p.parse_args()
    dest = Path(a.dest)
    dest.mkdir(exist_ok=True)

    packs = dict(AUX_PACKS)
    for key in choose(a.vi, "vi") + choose(a.tr, "tr"):
        packs[key] = ASR_PACKS[key]

    mpath = dest / "MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    for key, pack in packs.items():
        path = fetch(pack, dest)
        if pack.needs_bpe_vocab and not (path / "bpe.vocab").exists():
            make_bpe_vocab(path)
        files = [path] if path.is_file() else sorted(path.rglob("*.onnx"))
        manifest[key] = {"url": f"{BASE}/{pack.url_path}", "license": pack.license,
                         "sha256": {f.relative_to(dest).as_posix(): sha256(f) for f in files}}
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Tamam. {mpath} yazıldı.")


if __name__ == "__main__":
    main()

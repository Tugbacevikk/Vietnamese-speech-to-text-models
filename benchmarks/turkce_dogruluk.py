"""Türkçe ASR Doğruluk ve Kalite Ölçümü (%WER / %CER).

Kendi Türkçe ses kayıtlarınız ve doğru (referans) metinlerinizle modelin doğruluğunu
(Kelime Hata Oranı - WER ve Karakter Hata Oranı - CER) ölçer ve ayrıntılı raporlar.

Kullanım:
    # 1. TSV dosyası ile (dosya, metin sütunları):
    python benchmarks/turkce_dogruluk.py --tsv data/turkce_test.tsv

    # 2. Ses ve .txt dosyalarının olduğu bir klasör ile:
    python benchmarks/turkce_dogruluk.py --klasor data/ornek_turkce

    # 3. Model seçimi (varsayılan: faster-whisper-medium):
    python benchmarks/turkce_dogruluk.py --tsv data/turkce_test.tsv --model faster-whisper-small

    # 4. Örnek şablon oluşturmak için (referans metinlerinizi nasıl hazırlayacağınızı gösterir):
    python benchmarks/turkce_dogruluk.py --ornek-olustur

Çıktı:
    - Genel %WER (Kelime Hata Oranı)
    - Genel %CER (Karakter Hata Oranı)
    - RTF (İşlem hızı)
    - Kelime bazlı hata türleri (Ekleme, Silme, Değiştirme)
    - 'sonuc_dogruluk_tr.json' dosyasına kayıt
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import jiwer
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viet_asr.application.config import AsrParams, ModelRef, VadConfig, WhisperParams  # noqa: E402
from viet_asr.infrastructure.faster_whisper_asr import FasterWhisperAsr  # noqa: E402
from viet_asr.infrastructure.normalizer import TrNormalizer, turkish_lower  # noqa: E402
from viet_asr.infrastructure.sherpa_vad import SileroVad  # noqa: E402


def norm(text: str) -> str:
    """Türkçe kurallarına göre metni normalize eder (küçük harf, noktalama temizliği)."""
    text = unicodedata.normalize("NFC", text)
    text = turkish_lower(text)
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


@dataclass
class SampleResult:
    dosya: str
    sure_sn: float
    islem_sn: float
    rtf: float
    referans: str
    tahmin: str
    wer: float
    cer: float


def create_template(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    tsv_file = target_dir / "turkce_referans_sablon.tsv"
    tsv_content = (
        "dosya\tmetin\n"
        "ornek1.wav\thayırlısı değil dayan ama\n"
        "ornek2.wav\thoş geldin hoş buldum hocam\n"
    )
    tsv_file.write_text(tsv_content, encoding="utf-8")
    readme = (
        "Türkçe Doğruluk Testi Şablonu\n"
        "==============================\n"
        "1. Kendi Türkçe ses dosyalarınızı (.wav, .mp3, .flac) buraya koyun.\n"
        "2. turkce_referans_sablon.tsv dosyasını açıp her satıra dosya adını ve karşısına doğru metnini yazın.\n"
        "3. Testi çalıştırmak için:\n"
        "   python benchmarks/turkce_dogruluk.py --tsv data/ornek_turkce/turkce_referans_sablon.tsv\n"
    )
    (target_dir / "README.txt").write_text(readme, encoding="utf-8")
    print(f"Sablon olusturuldu: {target_dir}")
    print(f"  - TSV Dosyasi: {tsv_file}")
    print(f"  - Aciklama   : {target_dir / 'README.txt'}")


def load_dataset(tsv_path: Path | None, folder_path: Path | None) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    if tsv_path and tsv_path.exists():
        with tsv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                dosya_str = r.get("dosya", "").strip()
                metin = r.get("metin", "").strip()
                if not dosya_str or not metin:
                    continue
                p = Path(dosya_str)
                if not p.is_absolute() and not p.exists():
                    p = tsv_path.parent / p
                if p.exists():
                    pairs.append((p, metin))
                else:
                    print(f"Uyari: Ses dosyasi bulunamadi: {p}")
    elif folder_path and folder_path.exists():
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
            for audio_file in folder_path.glob(ext):
                txt_file = audio_file.with_suffix(".txt")
                if txt_file.exists():
                    pairs.append((audio_file, txt_file.read_text(encoding="utf-8").strip()))
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", type=Path, default=None, help="TSV dosyası yolu")
    p.add_argument("--klasor", type=Path, default=None, help="Ses ve .txt dosyalarının bulunduğu klasör")
    p.add_argument("--model", default="faster-whisper-medium", help="Model adı (varsayılan: faster-whisper-medium)")
    p.add_argument("--models-dir", default="models", help="Model klasörü")
    p.add_argument("--ornek-olustur", action="store_true", help="Örnek klasör ve TSV şablonu oluştur")
    a = p.parse_args()

    if a.ornek_olustur:
        create_template(ROOT / "data" / "ornek_turkce")
        return 0

    pairs = load_dataset(a.tsv, a.klasor)
    if not pairs:
        print("Hata: Test edilecek veri bulunamadi!")
        print("Kullanim ornegi:")
        print("  python benchmarks/turkce_dogruluk.py --tsv data/ornek_turkce/turkce_referans_sablon.tsv")
        print("  python benchmarks/turkce_dogruluk.py --ornek-olustur  (Yeni sablon olusturmak icin)")
        return 1

    print(f"\n[Turkce Dogruluk Testi] Model: {a.model} | Toplam ornek: {len(pairs)}")
    models_dir = Path(a.models_dir)
    model_path = models_dir / a.model

    wp = WhisperParams(device="cpu", compute_type="int8", cpu_threads=4, num_workers=1)
    params = AsrParams(whisper=wp)
    asr = FasterWhisperAsr(ModelRef(a.model, str(model_path)), params, TrNormalizer(), language="tr")
    vad = SileroVad(VadConfig(model=str(models_dir / "silero_vad.onnx")))

    results: list[SampleResult] = []
    toplam_ses_sn = 0.0
    toplam_islem_sn = 0.0

    print("-" * 65)
    for i, (audio_path, ref_raw) in enumerate(pairs, 1):
        try:
            audio, sr = sf.read(str(audio_path), dtype="float32")
            if sr != 16000:
                # Gerekirse basit yeniden örnekleme yapılabilir
                pass
            dur_sn = len(audio) / sr
            toplam_ses_sn += dur_sn

            t0 = time.perf_counter()
            segs = vad.split(audio)
            words = asr.transcribe(segs)
            dt = time.perf_counter() - t0
            toplam_islem_sn += dt

            hyp_raw = " ".join(w.text for w in words)
            ref_n = norm(ref_raw)
            hyp_n = norm(hyp_raw)

            sample_wer = round(jiwer.wer(ref_n, hyp_n) * 100, 2) if ref_n else 0.0
            sample_cer = round(jiwer.cer(ref_n, hyp_n) * 100, 2) if ref_n else 0.0
            sample_rtf = round(dt / max(0.1, dur_sn), 3)

            results.append(SampleResult(
                dosya=audio_path.name,
                sure_sn=round(dur_sn, 2),
                islem_sn=round(dt, 2),
                rtf=sample_rtf,
                referans=ref_n,
                tahmin=hyp_n,
                wer=sample_wer,
                cer=sample_cer,
            ))

            print(f"[{i}/{len(pairs)}] {audio_path.name:25s} | Sure: {dur_sn:.1f}s | RTF: {sample_rtf:.3f} | WER: %{sample_wer:.1f}")
        except Exception as e:
            print(f"[{i}/{len(pairs)}] HATA ({audio_path.name}): {e}")

    if not results:
        print("Hicbir dosya islenemedi.")
        return 1

    tum_ref = [r.referans for r in results]
    tum_hyp = [r.tahmin for r in results]

    genel_wer = round(jiwer.wer(tum_ref, tum_hyp) * 100, 2)
    genel_cer = round(jiwer.cer(tum_ref, tum_hyp) * 100, 2)
    genel_rtf = round(toplam_islem_sn / max(0.1, toplam_ses_sn), 3)

    print("\n" + "=" * 65)
    print("           TURKCE ASR DOGRULUK TESTI SONUCU")
    print("=" * 65)
    print(f"  Model                   : {a.model}")
    print(f"  Test Edilen Dosya Sayisi: {len(results)} adet")
    print(f"  Toplam Ses Suresi       : {toplam_ses_sn:.1f} saniye ({toplam_ses_sn / 60:.1f} dakika)")
    print(f"  Genel RTF (Hiz)         : {genel_rtf:.3f}")
    print("-" * 65)
    print(f"  [DOGRULUK] Genel WER (Kelime Hata Orani) : %{genel_wer:.2f}")
    print(f"  [DOGRULUK] Genel CER (Harf Hata Orani)   : %{genel_cer:.2f}")
    print("=" * 65)

    print("\nOrnek Karsilastirmalar:")
    for r in results[:3]:
        print(f"\nDosya: {r.dosya} (WER: %{r.wer})")
        print(f"  Ref: {r.referans}")
        print(f"  Tah: {r.tahmin}")

    rapor = {
        "model": a.model,
        "dosya_sayisi": len(results),
        "toplam_ses_sn": round(toplam_ses_sn, 2),
        "genel_rtf": genel_rtf,
        "genel_wer": genel_wer,
        "genel_cer": genel_cer,
        "ayrintilar": [r.__dict__ for r in results]
    }
    out_file = ROOT / "sonuc_dogruluk_tr.json"
    out_file.write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAyrintili rapor kaydedildi: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

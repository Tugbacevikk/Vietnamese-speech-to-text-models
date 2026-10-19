"""4000 Mesaj Yük ve Kapasite Testi.

Sistemin büyük ölçekli mesaj yükünü (örneğin 4000 sesli mesaj) ne kadar sürede,
nasıl bir işlemci/bellek tüketimiyle ve hangi hata oranıyla tamamlayacağını ölçer.

Kullanım:
    # 1. Canlı çalışan yerel sunucuya karşı (HTTP API ile):
    python benchmarks/yuk_testi.py --api http://127.0.0.1:8001 --ornek 20 --hedef-toplam 4000

    # 2. Doğrudan yerel hat üzerinde (API olmadan, çekirdek performansını ölçmek için):
    python benchmarks/yuk_testi.py --yerel --ornek 20 --hedef-toplam 4000 --es-zamanli 4

    # 3. Gerçek ses klasörü vererek:
    python benchmarks/yuk_testi.py --klasor data/uploads --ornek 30 --hedef-toplam 4000

Ölçülen Değerler:
    - İstek alma ve kuyruk kapasitesi (istek/saniye)
    - Gerçek Zaman Faktörü (RTF = işlem süresi / ses süresi)
    - İşlem hızı (mesaj/dakika ve ses saati/duvar saati)
    - 4000 mesajın tahmini toplam bitiş süresi (saat:dakika)
    - Başarı/Hata oranı ve tepe bellek (RAM) kullanımı

Sonuçlar 'yuk_testi_raporu.json' dosyasına yazılır ve konsola özet tablo basılır.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SR = 16000


def create_dummy_wav(duration_sec: float) -> bytes:
    """Belirtilen sürede 16kHz mono WAV verisi üretir (sentetik konuşma/ton simülasyonu)."""
    t = np.arange(int(duration_sec * SR)) / SR
    audio = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@dataclass
class JobMetric:
    job_id: str
    duration_sec: float
    elapsed_sec: float
    rtf: float
    status: str
    error: str | None = None


def find_sample_audios(folder: Path, limit: int = 10) -> list[Path]:
    """Mevcut klasörden örnek wav/flac seslerini bulur."""
    exts = ("*.wav", "*.flac", "*.mp3", "*.ogg")
    found = []
    for ext in exts:
        found.extend(list(folder.glob(ext)))
    return found[:limit]


def run_api_load_test(api_url: str, samples: list[tuple[str, bytes, float]], total_target: int,
                      concurrency: int = 4, language: str = "vi", is_synthetic: bool = False) -> dict:
    import urllib.error
    import urllib.request
    import uuid

    api_url = api_url.rstrip("/")
    # Health kontrolü
    try:
        req = urllib.request.urlopen(f"{api_url}/health", timeout=5)
        req.read()
    except Exception as e:
        raise RuntimeError(f"Sunucuya bağlanılamadı ({api_url}/health): {e}")

    print(f"\n[API Testi Başlatılıyor] Sunucu: {api_url} (Eşzamanlı işçi: {concurrency}, Dil: {language})")
    print(f"Örnek havuzu: {len(samples)} ses | Hedef projeksiyon: {total_target} mesaj\n")

    metrics: list[JobMetric] = []
    t_start = time.perf_counter()

    def send_and_wait(name: str, wav_data: bytes, audio_sec: float) -> JobMetric:
        job_id = f"bench_{uuid.uuid4().hex[:12]}"
        boundary = "----BenchBoundary" + uuid.uuid4().hex[:8]
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'.encode())
        body.extend(b"Content-Type: audio/wav\r\n\r\n")
        body.extend(wav_data)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(b'Content-Disposition: form-data; name="language"\r\n\r\n')
        body.extend(f"{language}\r\n".encode())
        body.extend(f"--{boundary}--\r\n".encode())

        req = urllib.request.Request(
            f"{api_url}/jobs",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                resp = json.loads(res.read().decode("utf-8"))
                jid = resp.get("job_id")
        except Exception as e:
            return JobMetric(job_id, audio_sec, time.perf_counter() - t0, 0.0, "failed", str(e))

        # İşi tamamlanana kadar bekle (polling)
        poll_url = f"{api_url}/jobs/{jid}"
        while True:
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(poll_url, timeout=10) as res:
                    st = json.loads(res.read().decode("utf-8"))
                    if st.get("status") in ("done", "failed"):
                        dt = time.perf_counter() - t0
                        rtf = round(dt / max(0.1, audio_sec), 3)
                        return JobMetric(jid, audio_sec, dt, rtf, st.get("status"), st.get("error"))
            except Exception as e:
                return JobMetric(jid, audio_sec, time.perf_counter() - t0, 0.0, "error", str(e))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i, (name, wav_data, sec) in enumerate(samples):
            futures.append(pool.submit(send_and_wait, f"bench_{i}.wav", wav_data, sec))

        done_count = 0
        for f in as_completed(futures):
            m = f.result()
            metrics.append(m)
            done_count += 1
            durumu = "TAMAM" if m.status == "done" else f"HATA ({m.error})"
            print(f"  [{done_count}/{len(samples)}] Ses: {m.duration_sec:.1f}s | Süre: {m.elapsed_sec:.1f}s | RTF: {m.rtf:.3f} | {durumu}")

    total_wall_sec = time.perf_counter() - t_start
    return evaluate_metrics(metrics, total_wall_sec, total_target)


def run_local_load_test(samples: list[tuple[str, bytes, float]], total_target: int,
                        concurrency: int = 4, language: str = "vi", is_synthetic: bool = False) -> dict:
    from viet_asr.domain import AudioJob
    from viet_asr.infrastructure.memory_store import MemoryJobStore
    from viet_asr.interface.composition import build_pipeline, config_from_settings
    from viet_asr.interface.settings import Settings

    print(f"\n[Yerel Hat Testi Başlatılıyor] (İş parçacığı: {concurrency}, Dil: {language})")
    print(f"Örnek havuzu: {len(samples)} ses | Hedef projeksiyon: {total_target} mesaj\n")

    settings = Settings(concurrency=concurrency)
    cfg = config_from_settings(settings, workers=concurrency)
    store = MemoryJobStore()
    pipeline = build_pipeline(cfg, store, with_detection=False)

    # Geçici dosyalar oluştur
    tmp_paths = []
    tmp_dir = ROOT / "data" / "yuk_testi_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, (_, wav_data, _) in enumerate(samples):
        p = tmp_dir / f"bench_{i}.wav"
        p.write_bytes(wav_data)
        tmp_paths.append(p)

    metrics: list[JobMetric] = []
    t_start = time.perf_counter()

    def process_file(idx: int, path: Path, audio_sec: float) -> JobMetric:
        jid = f"local_{idx}"
        t0 = time.perf_counter()
        try:
            job = AudioJob(jid, str(path), language=language)
            pipeline(job)
            dt = time.perf_counter() - t0
            rtf = round(dt / max(0.1, audio_sec), 3)
            return JobMetric(jid, audio_sec, dt, rtf, "done")
        except Exception as e:
            return JobMetric(jid, audio_sec, time.perf_counter() - t0, 0.0, "failed", str(e))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(process_file, i, tmp_paths[i], samples[i][2]) for i in range(len(samples))]
        done_count = 0
        for f in as_completed(futures):
            m = f.result()
            metrics.append(m)
            done_count += 1
            durumu = "TAMAM" if m.status == "done" else f"HATA ({m.error})"
            print(f"  [{done_count}/{len(samples)}] Ses: {m.duration_sec:.1f}s | Süre: {m.elapsed_sec:.1f}s | RTF: {m.rtf:.3f} | {durumu}")

    # Temizlik
    for p in tmp_paths:
        p.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    total_wall_sec = time.perf_counter() - t_start
    return evaluate_metrics(metrics, total_wall_sec, total_target, is_synthetic=is_synthetic, language=language)


def evaluate_metrics(metrics: list[JobMetric], total_wall_sec: float, total_target: int,
                     is_synthetic: bool = False, language: str = "vi") -> dict:
    successes = [m for m in metrics if m.status == "done"]
    failures = [m for m in metrics if m.status != "done"]

    if not successes:
        return {"hata": "Hiçbir ses başarıyla işlenemedi", "detay": [asdict(m) for m in failures]}

    toplam_ses_sn = sum(m.duration_sec for m in successes)
    ort_ses_sn = toplam_ses_sn / len(successes)
    ort_islem_sn = sum(m.elapsed_sec for m in successes) / len(successes)
    ort_rtf = sum(m.rtf for m in successes) / len(successes)

    # Duvar saatine göre verim (throughput)
    mesaj_basi_duvar_sn = total_wall_sec / len(successes)
    dakikada_mesaj = round(60.0 / mesaj_basi_duvar_sn, 1)
    saatte_mesaj = round(3600.0 / mesaj_basi_duvar_sn, 1)

    # 4000 mesaj için projeksiyon
    tahmini_toplam_sn = mesaj_basi_duvar_sn * total_target
    saat = int(tahmini_toplam_sn // 3600)
    dakika = int((tahmini_toplam_sn % 3600) // 60)
    saniye = int(tahmini_toplam_sn % 60)

    # Sistem bilgileri
    ram_gb = "Bilinmiyor"
    try:
        import psutil
        ram_gb = f"{round(psutil.virtual_memory().used / 2**30, 2)} GB / {round(psutil.virtual_memory().total / 2**30, 1)} GB"
    except Exception:
        pass

    rapor = {
        "tarih": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "islemci": platform.processor(),
        "cekirdek": os.cpu_count(),
        "dil": language,
        "sentetik_ses_kullanildi": is_synthetic,
        "ram_kullanimi": ram_gb,
        "test_edilen_mesaj_sayisi": len(metrics),
        "basarili": len(successes),
        "hatali": len(failures),
        "hedef_toplam_mesaj": total_target,
        "ort_mesaj_suresi_sn": round(ort_ses_sn, 2),
        "ort_islem_suresi_sn": round(ort_islem_sn, 2),
        "ort_rtf": round(ort_rtf, 3),
        "dakikada_islenen_mesaj": dakikada_mesaj,
        "saatte_islenen_mesaj": saatte_mesaj,
        "toplam_4000_mesaj_tahmini_sure": f"{saat} saat {dakika} dakika {saniye} saniye",
        "toplam_4000_mesaj_tahmini_saniye": round(tahmini_toplam_sn, 1),
    }

    # Tablo olarak konsola bas
    print("\n" + "=" * 65)
    print("           YÜK VE KAPASİTE TESTİ SONUÇ RAPORU")
    print("=" * 65)
    print(f"  Dil                      : {language.upper()} ({'Zipformer' if language == 'vi' else 'Whisper'})")
    print(f"  Ses Kaynağı              : {'Sentetik Ton (UYARI: Konuşma yok!)' if is_synthetic else 'Gerçek Konuşma Sesi'}")
    print(f"  Test Edilen Örnek Sayısı : {len(metrics)} adet")
    print(f"  Başarılı / Hatalı        : {len(successes)} / {len(failures)}")
    print(f"  Ortalama Mesaj Süresi    : {ort_ses_sn:.1f} saniye")
    print(f"  Ortalama RTF             : {ort_rtf:.3f} (1 sn ses = {ort_rtf:.2f} sn işlem)")
    print(f"  İşlem Hızı (Throughput)  : {dakikada_mesaj} mesaj/dk  ({saatte_mesaj} mesaj/saat)")
    print("-" * 65)
    print(f"  [HEDEF] {total_target} MESAJ ICIN TAHMINI SURE: {saat} SAAT {dakika} DAKIKA")
    if is_synthetic:
        print("\n  [ONEMLI UYARI] Testte gercek konusma yerine yapay ton kullanilmistir.")
        print("  ASR sessizlik/ton algiladiginda cozumlemeyi hizla gecebilir.")
        print("  Gercek uretim olcumu icin: --ses-dosyasi <dosya.wav> veya --klasor ile gercek konusmali ses veriniz.")
    print("=" * 65 + "\n")

    return rapor


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default=None, help="HTTP API adresi (ör. http://127.0.0.1:8001)")
    p.add_argument("--yerel", action="store_true", help="API yerine doğrudan Python süreç içi hattan çalıştır")
    p.add_argument("--dil", default="vi", choices=["vi", "tr"], help="İşlenecek dil (vi: Zipformer, tr: Whisper; varsayılan: vi)")
    p.add_argument("--ses-dosyasi", type=str, default=None, help="Doğrudan kullanılacak gerçek konuşma ses dosyası (WAV/FLAC)")
    p.add_argument("--ornek", type=int, default=15, help="Test için koşturulacak mesaj sayısı (varsayılan: 15)")
    p.add_argument("--hedef-toplam", type=int, default=4000, help="Kapasite hesabı yapılacak toplam iş (varsayılan: 4000)")
    p.add_argument("--es-zamanli", type=int, default=4, help="Aynı anda koşan işçi sayısı (varsayılan: 4)")
    p.add_argument("--klasor", type=str, default="data/uploads", help="Gerçek seslerin okunacağı klasör")
    a = p.parse_args()

    samples: list[tuple[str, bytes, float]] = []
    is_synthetic = False

    # 1. Tekil ses dosyası belirtilmişse
    if a.ses_dosyasi and Path(a.ses_dosyasi).is_file():
        sf_path = Path(a.ses_dosyasi)
        try:
            info = sf.info(str(sf_path))
            raw_b = sf_path.read_bytes()
            for i in range(a.ornek):
                samples.append((f"{sf_path.stem}_{i}{sf_path.suffix}", raw_b, float(info.duration)))
        except Exception as e:
            print(f"Belirtilen ses dosyası okunamadı: {e}")

    # 2. Klasörden gerçek sesleri tara
    if not samples:
        sample_files = find_sample_audios(Path(a.klasor), limit=a.ornek)
        if sample_files:
            for f in sample_files:
                try:
                    info = sf.info(str(f))
                    samples.append((f.name, f.read_bytes(), float(info.duration)))
                except Exception:
                    pass

    # 3. Yeterli ses yoksa sentetik seslerle tamamla ve uyar
    if len(samples) < a.ornek:
        is_synthetic = True
        durations = [3.0, 7.5, 12.0, 18.0, 25.0]
        while len(samples) < a.ornek:
            d = durations[len(samples) % len(durations)]
            data = create_dummy_wav(d)
            samples.append((f"sentetik_{d:.0f}s.wav", data, d))

    samples = samples[: a.ornek]

    if a.yerel:
        rapor = run_local_load_test(samples, a.hedef_toplam, a.es_zamanli, language=a.dil, is_synthetic=is_synthetic)
    else:
        api_url = a.api or "http://127.0.0.1:8001"
        try:
            rapor = run_api_load_test(api_url, samples, a.hedef_toplam, a.es_zamanli, language=a.dil, is_synthetic=is_synthetic)
        except Exception as e:
            print(f"API testi çalıştırılamadı: {e}\nİpucu: Sunucu açık değilse --yerel parametresiyle deneyebilirsiniz.")
            return 1

    out_file = ROOT / "yuk_testi_raporu.json"
    out_file.write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapor dosyası kaydedildi: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

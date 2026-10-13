# Ses → Metin + Konuşmacı Ayrıştırma (Türkçe, Vietnamca)

Türkçe ve Vietnamca ses kayıtlarını metne çevirir ve her cümlenin hangi konuşmacıya
ait olduğunu bulur. Dil istekte seçilir (`tr`, `vi`) ya da sesten bulunur; konuşmacı sayısı
istekte verilmezse sesten bulunur. Yalnızca CPU'da, internetsiz sunucuda çalışacak şekilde
tasarlandı. Hedef: **1 dakikalık ses en geç 1 dakikada biter.**

Mimari, fine-tuning projesindekiyle aynı fikirde (API → kuyruk → işçi → PostgreSQL → outbox →
imzalı webhook); oradaki hatalar düzeltildi, kararlar [docs/adr](docs/adr) altında.

## Hızlı başlangıç (Windows, tek bilgisayar)

```bat
baslat.bat        :: modelleri indirir (ilk sefer), sunucuyu açar
```

Tarayıcıda **http://127.0.0.1:8000**: dosya yükle, dili seç (Otomatik / Türkçe / Vietnamca),
sonucu dinle, TXT/SRT/JSON/RTTM indir. API belgesi: http://127.0.0.1:8000/docs.
Durdurmak: `durdur.bat`. Yerel modda Redis, PostgreSQL ya da Docker gerekmez (SQLite + iş
parçacıkları). Çok makineli tam mod: `baslat_tam.bat` ya da `docker compose up`.

## Hat

```
ses ─► 16 kHz mono ─► VAD (Silero) ─► dil kararı ─► ASR (dile göre) ─────────► kelimeler + zaman
                         │                                                            │
                         └─► konuşmacı ayrıştırma (CAM++ imzası + spektral kümeleme) ─┤
                                                                                      ▼
                                              birleştirme ─► [0.1 - 3.7] speaker_00: ...
```

| Dil | Varsayılan model | Seçenekler | Not |
|---|---|---|---|
| Vietnamca | Zipformer-68M int8 (Apache-2.0) | Zipformer-30M (CC BY-NC-ND) | Sesin tamamı en sessiz anlardan bölünür |
| Türkçe | **faster-whisper small** (fine-tuning projesinin modeli, MIT) | Omnilingual-300M, Whisper-turbo | Yalnızca VAD parçaları (Whisper sessizlikte uyduruyor) |

**Ölçümler** (ayrıntı: [docs/olcumler.md](docs/olcumler.md)):
- Vietnamca, VietMed 50 dosya: WER %10,1, CER %7,3. Dil tespiti 85/85 doğru.
- Konuşmacı ayırma, 9 gerçek görüşme: DER %10,2 (sayı istekte verilirse %5,2).
- Hız: 3 dk'lık diyalog 2 çekirdekte 28 sn (oran 0,15). 1 dk ses ≈ 9 sn.
- Türkçe faster-whisper small'un hızı bu ortamda ölçülemedi (model indirilemedi).
  Kendi bilgisayarında `python benchmarks/kabul_testi.py --dil tr kayit.wav` ile ölç.
  Oran 1'i aşarsa `.env` içinde `TR_MODEL=omnilingual-300m` yap (24 sentetik cümlede
  CER %3,8; 60 sn diyalog 25 sn).

### Dil seçimi

İstekte `language=tr|vi` varsa tespit atlanır. `auto` ise önce Whisper-tiny, emin olamazsa
Whisper-small; yalnızca desteklenen diller arasında. Desteklenen dillerin toplam olasılığı
%50'nin altındaysa iş **reddedilir** (`rejected`, yanlış modelle anlamsız metin üretilmez);
güven %80'in altındaysa sonuçta `uncertain: true`.

### Konuşmacı ayrıştırma

1,5 sn'lik pencerelerden CAM++ ses imzası → NME-SC spektral kümeleme (kişi sayısını kendisi
bulur) → çok yakın kümeler birleştirilir → konuşmanın %8'inden azını kaplayan "kişi" (gülme,
gürültü) en yakın kişiye katılır. `num_speakers` verilirse tam o sayı. Üst üste konuşma ayrılmaz.

## API

API anahtarı tanımlıysa (`API_KEY`, üretimde zorunlu) istekler `X-API-Key` başlığı ister.

```bash
curl -H "X-API-Key: $K" -F file=@kayit.wav -F callback_url=https://ornek.com/hook http://localhost:8000/jobs
# -> 202 {"job_id": "...", "status": "queued", "duration": 61.2, "language": "auto"}
curl -H "X-API-Key: $K" -F file=@kayit.wav -F language=tr -F num_speakers=2 http://localhost:8000/jobs
curl -H "X-API-Key: $K" -F file=@kayit.wav -F external_id=mesaj-4711 http://localhost:8000/jobs
curl -H "X-API-Key: $K" http://localhost:8000/jobs/<job_id>        # durum + sonuç
```

| Uç | Ne yapar |
|---|---|
| `POST /jobs` | ses yükle; `language` (tr, vi, auto), `num_speakers` (1–20), `callback_url`, `external_id` |
| `GET /jobs/{id}` | durum; bittiyse konuşmacılı satırlar |
| `GET /jobs?limit=20` | son işler |
| `GET /jobs/{id}/audio` | yüklenen ses (saklama süresi dolmadıysa) |
| `DELETE /jobs/{id}` | bitmiş işi, sonucunu ve sesini siler |
| `GET /health` | veritabanı ve kuyruk (anahtar istemez) |
| `GET /metrics` | gecikme oranı (hedef p95 ≤ 1), kuyruk doluluğu |
| `GET /` | web arayüzü |

- **`external_id`:** entegre olan sistemin kendi kimliği. Aynısı tekrar gelirse yeni iş açılmaz,
  var olan iş `200` ile döner; ağ hatasında güvenle tekrar gönderilebilir.
- **Durumlar:** `queued` → `running` → `done` | `failed` | `rejected`.
- **Webhook** (`X-Event: job.done | job.failed | job.rejected`, `X-Delivery-Id`): en az bir kez
  gider; `WEBHOOK_SECRET` varsa `X-Signature: sha256=HMAC(secret, "<X-Timestamp>.<gövde>")`.
  Callback adresi `WEBHOOK_ALLOWED_HOSTS` ile sınırlanabilir (üretimde iç ağ adresleri kapalı).

```json
{"status": "done", "job_id": "...", "duration": 61.2, "model": "zipformer-68m",
 "language": {"code": "vi", "confidence": 0.99, "source": "tespit", "uncertain": false},
 "speakers": 2, "text": "...",
 "utterances": [{"speaker": "speaker_00", "start": 0.1, "end": 3.5, "text": "..."}]}
```

## Tam mod (çok süreç / çok makine)

```
istemci ─► api ─► PostgreSQL (iş) + Redis Streams: asr:lid ─► asr:vi | asr:tr ─► işçiler
                      ▲                                                            │
                      └──── sonuç + bildirim satırı AYNI işlemde (outbox) ◄─────────┘
                                        dispatcher ─► imzalı webhook (artan aralıkla tekrar)
```

- Dil başına kuyruk: yalnızca yoğun dilin işçisi artırılır (`stream_worker --kuyruk vi --surec N`).
- Çöken işçinin işi 2 dk sonra başkasına geçer; 3 denemeden sonra `asr:dlq` ve `job.failed`.
- Süreç sayısı otomatik: kullanılabilir çekirdek / `ONNX_THREADS` (Docker `cpus:` sınırı dahil).
- `bakim` süreci: 24 saat sonra ses, 30 gün sonra kayıt silinir; askıda kalan iş `failed` olur.

## Veritabanı (code-first)

Şema `src/viet_asr/infrastructure/db/models.py` içinde (SQLAlchemy 2.0); değişiklikler Alembic
göçüyle, uygulama açılışta kendisi uygular. SQLite (yerel) ve PostgreSQL (tam) aynı kod.
Eski sürümün veritabanı ilk açılışta veri kaybı olmadan aktarılır.

```bash
# models.py'yi değiştir -> göç üret -> gözden geçir -> test et
set DATABASE_URL=sqlite:///data/gelistirme.db
alembic upgrade head
alembic revision --autogenerate -m "aciklama"
pytest tests/db          # göç unutulursa test_sema_modellerle_ayni kırılır
```

## Vietnamca ince ayar (Faz 2)

Zipformer-68M tıbbi veriyle WSL2 + GPU'da ince ayarlanır; çıkan model Kapı 2'yi geçerse
`VI_MODEL=models/<klasör>` ile kullanılır. Adımlar: [finetune/README.md](finetune/README.md).

## Rust'a geçiş

Parça parça: Python sürümü başvuru kaynağı, Rust sürümü `native/` içinde, rastgele binlerce
girdide birebir aynı sonuç şartı. İlk parça birleştirme (`pip install ./native`; kurulu değilse
Python sürümü çalışır). Sıra ve kurallar: [docs/adr/0005-rust-gecisi.md](docs/adr/0005-rust-gecisi.md).

## Geliştirme

```bash
pip install -e ".[dev,bench,server]"
python scripts/download_models.py      # models/ + MANIFEST.json (sha256; internetsiz sunucuya taşınır)
pytest                                 # birim + (modeller varsa) entegrasyon
TEST_DATABASE_URL=postgresql://postgres@localhost/asr_test TEST_REDIS_URL=redis://localhost:6379/15 pytest tests/db
lint-imports                           # katman kuralları
python benchmarks/kabul_testi.py a.wav b.wav --es-zamanli 4   # 1 dk ses <= 1 dk mı (bu donanımda)
```

Diğer ölçümler: `benchmarks/diyalog_testi.py` (DER, cpWER), `yerel_dogruluk_testi.py` (WER/CER),
`dil_tespiti_testi.py`, `konusmaci_testi.py`, `turkce_dogruluk_testi.py`.
VietMed'den bölüm bazında test seti: `scripts/vietmed_hazirla.py`.

```
src/viet_asr/
  domain/          veri tipleri; dış kütüphane yok
  application/     iş kuralları (TranscribeAndDiarize, LanguageResolver, FusionEngine, SubmitJob...)
                   ve portlar; sherpa-onnx, SQLAlchemy, Redis, FastAPI bilmez (lint-imports)
  infrastructure/  sherpa-onnx ve faster-whisper adaptörleri, kümeleme, db/ (modeller + göçler +
                   depo), Redis Streams, iş parçacığı kuyruğu, webhook, Rust birleştirme sarmalayıcısı
  interface/       api.py, local_server.py, stream_worker.py, dispatcher.py, bakim.py, cli.py,
                   static/index.html; composition.py hepsini aynı şekilde kurar
native/            Rust çekirdeği (PyO3)
finetune/          Vietnamca ince ayar hattı ve Kapı 2
benchmarks/ scripts/ tests/{unit,db,integration} docs/{adr,olcumler.md}
```

CI (GitHub Actions): Linux (PostgreSQL + Redis ile), Windows (yerel mod), Rust (eşlik testi).

## Yol haritası

- [x] Faz 0 — hat, dil seçimi, konuşmacı sayısı otomatik, DER/cpWER ölçütleri, VietMed bölümlü test seti
- [x] Faz 1 — API, Redis Streams, PostgreSQL/SQLite (code-first), outbox + imzalı webhook, güvenlik,
      saklama, `/metrics`, web arayüzü, kabul testi, CI, ADR'ler
- [x] Türkçe — fine-tuning projesindeki faster-whisper small (kelime zamanlı)
- [x] Faz 2 hazırlığı — ince ayar hattı + Kapı 2 (eğitim GPU'lu makinede çalıştırılacak)
- [ ] Kabul testi hedef bilgisayarda (özellikle Türkçe faster-whisper hızı) ve 4000 mesajlık yük testi
- [ ] Türkçe gerçek kayıtlarla WER/DER ölçümü
- [ ] İnce ayarın çalıştırılması ve Kapı 2 kararı
- [ ] Rust adım 2: `tokens_to_words`, `EnergyChunker`

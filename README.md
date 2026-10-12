# Ses → Metin + Konuşmacı Ayrıştırma (Vietnamca, Türkçe)

Vietnamca ve Türkçe ses kayıtlarını metne çevirir ve her cümlenin hangi konuşmacıya
(ör. doktor / hasta) ait olduğunu bulur. Dil istekte verilebilir ya da sesten tespit edilir. Yalnızca CPU'da, internetsiz
(air-gapped) sunucuda çalışacak şekilde tasarlandı. PyTorch gerekmez;
tüm modeller [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ile çalışır.

## Hat (pipeline)

```
ses dosyası ─► 16 kHz mono ─► VAD (Silero) ─► konuşma parçaları ─► dil kararı ─► ASR (dile göre) ─► kelimeler + zaman
                    │                                                                     │
                    └──────────► Diarization (pyannote seg-3.0 + WeSpeaker) ─► konuşmacı aralıkları
                                                                                          │
                                                  Birleştirme (Fusion) ◄──────────────────┘
                                                          │
                                       [0.07 - 3.71] speaker_00: xin chào bác sĩ ...
```

| Adım | Model | Lisans | Hız (RTF, 2 çekirdek) |
|---|---|---|---|
| VAD | Silero VAD | MIT | 0,006 |
| ASR | Zipformer-68M int8 (2025-04-20) | Apache-2.0 | 0,028–0,039 |
| ASR (aday) | Zipformer-30M int8 (2026-02-09) | CC BY-NC-ND 4.0 | 0,024–0,029 |
| Diarization | pyannote segmentation-3.0 + WeSpeaker ResNet34 | MIT / WeSpeaker (ağırlık lisansı doğrulanacak) | 0,14–0,18 |

RTF = işlem süresi / ses süresi. 0,03 → 1 dakikalık ses yaklaşık 2 saniyede metne çevrilir.
Hangi Zipformer'ın kullanılacağına `benchmarks/dogruluk_testi.py` sonucuna göre karar verilecek.

## Dil seçimi

İstekte dil verilirse (`--lang vi` / `--lang tr`) o kullanılır. Verilmezse dil sesten
tespit edilir: önce Whisper-tiny (~0,25 sn), emin olamazsa Whisper-small (~1 sn).
Tespit yalnızca desteklenen diller arasında yapılır; desteklenen dillerin toplam
olasılığı %50'nin altındaysa iş **reddedilir** (`rejected`), yanlış modelle anlamsız
metin üretilmez. Güven %80'in altındaysa sonuç "belirsiz" işaretlenir.

| Dil | Varsayılan model | Seçenek | 60 sn diyalog, tüm hat (2 çekirdek) |
|---|---|---|---|
| Vietnamca | Zipformer-68M | Zipformer-30M | 13–14 sn |
| Türkçe | Omnilingual-300M (Meta, Apache-2.0) | Whisper large-v3-turbo (MIT) | 25 sn (turbo: 63 sn) |

Türkçe doğruluk henüz yalnızca 24 sentetik cümlede ölçüldü (CER: turbo %3,0,
Omnilingual %3,8). Gerçek ölçüm: `benchmarks/turkce_dogruluk_testi.py` (FLEURS tr).

## Sistem (servisler)

```
istemci ──POST /jobs──► api ──► PostgreSQL (iş kaydı)
                         │
                         └──► Redis kuyrukları
                                ├─ lid     ─► worker-lid : dil tespiti ─► asr.<dil> kuyruğuna aktarır
                                ├─ asr.vi  ─► worker-vi  : Zipformer + konuşmacı ayrıştırma
                                └─ asr.tr  ─► worker-tr  : Omnilingual + konuşmacı ayrıştırma
                                                   │
                       sonuç + "bildirim gönder" satırı AYNI işlemde yazılır (outbox)
                                                   │
                         dispatcher ──► callback_url'e POST (başarısızsa artan aralıkla tekrar)
```

- **Dil başına kuyruk:** her ASR worker'ı yalnızca kendi dilinin modelini yükler; diller
  birbirinden bağımsız ölçeklenir (ör. 4000 Vietnamca mesajlık patlamada yalnızca `worker-vi` artırılır).
- **Dil verilmişse** iş doğrudan `asr.<dil>` kuyruğuna gider, tespit atlanır.
- **Güvenilirlik:** worker iş ortasında ölürse iş kuyruğa geri döner (acks_late); aynı iş
  iki kez gelirse bitmiş iş atlanır; webhook en az bir kez gider (`X-Delivery-Id` ile ayıklanır).

### Docker ile çalıştırma

```bash
python scripts/download_models.py          # ./models (yaklaşık 1,5 GB; bir kez)
docker compose up -d --build
curl http://localhost:8000/health
```

Çekirdek bütçesi: worker süreci sayısı (`-c`) x `ONNX_THREADS` makinedeki çekirdek sayısını
geçmemeli; geçerse süreçler birbirini yavaşlatır. `docker-compose.yml` 8 çekirdeğe göre ayarlı.

### API

```bash
# dil otomatik, iş bitince webhook
curl -F file=@kayit.wav -F callback_url=https://ornek.com/hook http://localhost:8000/jobs
# -> 202 {"job_id": "...", "status": "queued", "duration": 61.2, "language": "auto"}

curl -F file=@kayit.wav -F language=tr http://localhost:8000/jobs   # dil verilir
curl http://localhost:8000/jobs/<job_id>                              # durum + sonuç
```

Durumlar: `queued` → `running` → `done` | `failed` | `rejected` (desteklenmeyen dil).
Hata kodları: 415 dosya türü (wav, flac, ogg, mp3, opus), 400 bozuk ses, 413 boyut, 422 dil.

Webhook gövdesi (`X-Event: job.done`):

```json
{"status": "done", "job_id": "...", "duration": 61.2, "model": "zipformer-68m",
 "language": {"code": "vi", "confidence": 0.99, "source": "tespit", "uncertain": false},
 "text": "...",
 "utterances": [{"speaker": "speaker_00", "start": 0.1, "end": 3.5, "text": "..."}]}
```

`job.failed` ve `job.rejected` olaylarında gövde `{"job_id", "status", "error"}`.

Uçtan uca deneme: `python scripts/smoke_test.py --api http://localhost:8000 a.wav b.wav`
(dosyaları gönderir, webhook'ları dinler, sonuçları yazar).

## Kurulum (geliştirme)

```bash
pip install -e ".[dev,bench,server]"
python scripts/download_models.py     # models/ klasörüne indirir, bpe.vocab ve MANIFEST.json üretir
```

## Komut satırı (tek dosya, servis gerekmez)

```bash
python -m viet_asr.interface.cli kayit.wav                    # dil otomatik
python -m viet_asr.interface.cli kayit.wav --lang tr --tr-model whisper-turbo --json sonuc.json
```

Hotword (özel terim) listesi: her satıra bir terim, **BÜYÜK HARFLE** yazılmalı.
`--hotwords terimler.txt` ile verilir. `bpe.vocab` yoksa program hata verir
(sherpa-onnx bu durumda hotword'leri sessizce yok sayıyor, bunu engellemek için).

## Klasör yapısı (temiz mimari)

```
src/viet_asr/
  domain/          veri tipleri (Word, SpeakerSegment, Utterance, Transcript, JobRecord). Dış kütüphane yok.
  application/     iş kuralları: TranscribeAndDiarize, LanguageResolver, FusionEngine,
                   SubmitJob / DetectLanguage / RunTranscription, portlar (arayüzler)
  infrastructure/  gerçeklemeler: sherpa-onnx (VAD, ASR, diarization), Whisper dil tespiti,
                   PostgreSQL deposu + outbox, webhook göndericisi, model kataloğu
  interface/       dışarıya açılan kapılar: api.py (FastAPI), worker.py (Celery),
                   dispatcher.py, cli.py; composition.py hepsini aynı şekilde kurar
benchmarks/        dogruluk_testi.py, turkce_dogruluk_testi.py, dil_tespiti_testi.py, hiz_testi.py
scripts/           download_models.py, smoke_test.py
tests/unit         modelsiz hızlı testler (API dahil)
tests/integration  gerçek modellerle testler (modeller yoksa atlanır)
tests/db           gerçek PostgreSQL ile depo + outbox + webhook (TEST_DATABASE_URL yoksa atlanır)
```

Katman kuralı `lint-imports` ile denetlenir: içteki katman dıştakini import edemez,
domain ve application `sherpa_onnx`, `onnxruntime`, `psycopg`, `celery`, `fastapi` bilmez. Model değişirse yalnızca
`infrastructure/` altındaki bir sınıf değişir.

## Test

```bash
pytest            # birim + (modeller varsa) entegrasyon testleri
TEST_DATABASE_URL=postgresql://postgres@localhost/asr_test pytest tests/db
lint-imports      # katman kuralları
```

## Ölçüm

```bash
python benchmarks/dogruluk_testi.py   # VIVOS test + VietMed dev, iki Zipformer: WER, CER, RTF
python benchmarks/hiz_testi.py        # üretim sunucusunda hız ve bellek
python benchmarks/turkce_dogruluk_testi.py            # FLEURS tr: Omnilingual ve Whisper turbo
python benchmarks/dil_tespiti_testi.py liste.tsv      # kendi seslerinde dil tespiti doğruluğu
```

Seçim kuralı: VIVOS (tarafsız set) CER'i düşük olan model; fark 1 puandan azsa daha hızlı olan.
Zipformer-30M, VietMed ve FLEURS ile eğitildiği için onun VietMed sonucu iyimser çıkar.

## Yol haritası

- [x] Faz 0.1 — iskelet, domain, portlar, PipelineConfig, katman denetimi
- [x] Faz 0.2 — VAD, ASR, diarization adaptörleri + birleştirme motoru + testler
- [x] Faz 0.3 — model indirme betiği (sha256 manifest), doğruluk ve hız testleri
- [ ] Faz 0.4 — doğruluk testini çalıştır, 30M/68M seçimi, kabul eşiği
- [x] Dil seçimi — istekte dil ya da otomatik tespit (tiny → small kademesi), Türkçe desteği
- [x] Faz 1 — FastAPI, Celery + Redis (`lid`, `asr.vi`, `asr.tr`), PostgreSQL + outbox, webhook, Docker
- [ ] Faz 1.1 — gerçek verilerle ölçüm (VietMed, FLEURS tr), 4000 mesajlık yük testi
- [ ] Faz 2 — tıbbi hotword sözlüğü, noktalama geri getirme, DER/cpWER ölçümü

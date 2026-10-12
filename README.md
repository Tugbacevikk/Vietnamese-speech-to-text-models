# Ses → Metin + Konuşmacı Ayrıştırma (Vietnamca, Türkçe)

Vietnamca ses kayıtlarını metne çevirir ve her cümlenin hangi konuşmacıya
(ör. doktor / hasta) ait olduğunu bulur. Yalnızca CPU'da, internetsiz
(air-gapped) sunucuda çalışacak şekilde tasarlandı. PyTorch gerekmez;
tüm modeller [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ile çalışır.

## Hat (pipeline)

```
ses dosyası ─► 16 kHz mono ─► VAD (Silero) ─► konuşma parçaları ─► ASR (Zipformer) ─► heceler + zaman
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

## Kurulum

```bash
pip install -e ".[dev,bench]"
python scripts/download_models.py     # models/ klasörüne indirir, bpe.vocab ve MANIFEST.json üretir
```

## Kullanım

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
  domain/          veri tipleri (Syllable, SpeakerSegment, Utterance, Transcript). Dış kütüphane yok.
  application/     iş kuralları: TranscribeAndDiarize, FusionEngine, PipelineConfig, portlar (arayüzler)
  infrastructure/  arayüzlerin gerçek hali: SileroVad, SherpaOnnxAsr, SherpaOnnxDiarizer, ViNormalizer
  interface/       dışarıya açılan kapı: şimdilik CLI; sonra FastAPI + Celery
benchmarks/        dogruluk_testi.py (WER/CER), hiz_testi.py (RTF, bellek)
scripts/           download_models.py
tests/unit         modelsiz hızlı testler
tests/integration  gerçek modellerle testler (modeller yoksa atlanır)
```

Katman kuralı `lint-imports` ile denetlenir: içteki katman dıştakini import edemez,
domain ve application `sherpa_onnx` bilmez. Model değişirse yalnızca
`infrastructure/` altındaki bir sınıf değişir.

## Test

```bash
pytest            # birim + (modeller varsa) entegrasyon testleri
lint-imports      # katman kuralları
```

## Ölçüm

```bash
python benchmarks/dogruluk_testi.py   # VIVOS test + VietMed dev, iki Zipformer: WER, CER, RTF
python benchmarks/hiz_testi.py        # üretim sunucusunda hız ve bellek
```

Seçim kuralı: VIVOS (tarafsız set) CER'i düşük olan model; fark 1 puandan azsa daha hızlı olan.
Zipformer-30M, VietMed ve FLEURS ile eğitildiği için onun VietMed sonucu iyimser çıkar.

## Yol haritası

- [x] Faz 0.1 — iskelet, domain, portlar, PipelineConfig, katman denetimi
- [x] Faz 0.2 — VAD, ASR, diarization adaptörleri + birleştirme motoru + testler
- [x] Faz 0.3 — model indirme betiği (sha256 manifest), doğruluk ve hız testleri
- [ ] Faz 0.4 — doğruluk testini çalıştır, 30M/68M seçimi, kabul eşiği
- [ ] Faz 1 — FastAPI `POST /jobs`, Celery + Redis (`asr.vi` kuyruğu), PostgreSQL + outbox, webhook
- [ ] Faz 2 — tıbbi hotword sözlüğü, noktalama geri getirme, DER/cpWER ölçümü

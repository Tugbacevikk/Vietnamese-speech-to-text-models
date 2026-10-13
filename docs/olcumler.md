# Ölçümler

Her sayı bir betikten gelir; betik adı ve veri yanında yazılıdır. Donanım: 2 çekirdekli
sanal makine (Xeon), ONNX_THREADS=2, yalnızca CPU.

## VietMed örneği — 50 dosya, 77,2 dk (12 Ekim 2026)

Kullanıcının `vietmed_1_2dk` klasöründen rastgele 50 dosya (seed 42), 8 kHz, 1–2 dk'lık
birleştirilmiş kayıtlar. Roller: Doctor 33, Host 6, Narrator 6, Podcaster 3, Lecturer 2
(hasta kaydı çıkmadı). **Uyarı:** 626 dosyalık küme VietMed train/dev/test bölümlerini
karıştırıyor; Zipformer-30M VietMed ile eğitildiği için onun sonucu iyimser olabilir.

### ASR — `benchmarks/yerel_dogruluk_testi.py`

| Parçalama | Model | WER (hece) | CER | Eksik hece | RTF |
|---|---|---|---|---|---|
| VAD (Silero, eşik 0,5) — eski | 68M | %20,8 | %18,2 | %13,1 | 0,025 |
| VAD eşik 0,2 | 68M | %15,6 | – | %7,3 | 0,031 |
| VAD yok, sabit 15 sn | 68M | %10,8 | – | %1,9 | 0,025 |
| **Tüm ses, en sessiz anda kesim (15 sn)** | **68M** | **%10,1** | **%7,3** | %1,8 | 0,026 |
| Tüm ses, en sessiz anda kesim (15 sn) | 30M | %9,7 | %7,6 | %1,6 | 0,024 |

Bulgu: VAD müzik altındaki / gürültülü konuşmayı atıyordu (en kötü dosyada 95 sn'nin
yalnızca 3,8 sn'si ASR'a gitti). Zipformer sessizlik ve müzikte metin uydurmuyor
(eklenen hece +%0,5), bu yüzden ona artık sesin tamamı veriliyor.
30M ile 68M arasındaki 0,4 puan fark, 30M'in VietMed'i eğitimde görmüş olmasıyla
açıklanabilir; seçim tarafsız VIVOS testine bırakıldı.

Bellek: parçaları hepsini birden çözmek 92 sn'lik seste 1,7 GB'a, 77 dk'lık testte 6 GB'a
(süreç öldürüldü) çıkıyordu. 10 sn'lik gruplarla 460 MB ve daha hızlı (RTF 0,029; 60 sn
gruplarda 2,2 GB, RTF 0,043).

### Dil tespiti — `benchmarks/dil_tespiti_testi.py`

50/50 doğru (vi), 1 dosya Whisper-small'a devredildi, dosya başına 0,29 sn.

### Konuşmacı ayrıştırma — tek kişilik dosyalarda fazladan kişi

Her dosya tek konuşmacının birleştirilmiş kayıtları; doğru cevap çoğunlukla 1.

| Küçük küme kuralı | Gerçek: 1 kişi bulunan | Sentetik 28 diyalog: doğru sayı |
|---|---|---|
| en az 1 sn | 34/50 | 27/28 |
| en az 1,5 sn ve konuşmanın %8'i | **48/50** | **28/28** |

## VietMed hasta örneği — 35 dosya, 54,1 dk (13 Ekim 2026)

15 hasta, 13 doktor, 7 sunucu kaydı; VietMed_011 görüşmesinde sunucu, doktor ve hasta aynı kayıtta.
Hastaların çoğu telefonla bağlanıyor (8 kHz, hat gürültüsü).

### ASR

| Model | Genel WER | Doktor | Sunucu | Hasta | Hasta (duraksama sesleri çıkarılınca) |
|---|---|---|---|---|---|
| 68M | %10,5 | %6,8 | %7,0 | %17,3 | %14,7 |
| 30M | %10,0 | %7,3 | %6,7 | %15,5 | – |

Hasta hatalarının bir kısmı modelin "ờ", "ơ", "à" gibi duraksama seslerini yazmasından:
referansta hastaların bunları yazılmamış (en sık eklenen hece "ờ", 100 kez). Bunlar
çıkarılınca da hasta %14,7: kendiliğinden konuşma ve yöresel ağız gerçekten daha zor.

### Dil tespiti

35/35 doğru, 1 dosya Whisper-small'a devredildi.

### Konuşmacı ayrıştırma — gerçek seslerden kurulan diyaloglar

`konusmaci_testi.py --kaynak tsv`: 4 gerçek konuşmacının (sunucu, doktor, iki hasta)
8 sn'lik kesitlerinden 1-3 kişilik diyaloglar, duraksamalı ve duraksamasız, 2 tohum = 56 diyalog.

| İmza modeli | Eşik | Sayı doğru | Cümle doğru kişide | Sayı verilirse cümle | RTF |
|---|---|---|---|---|---|
| ERes2Net base (önceki) | 0,65 | 16/56 | %80,8 | %86,2 | 0,049 |
| ERes2Net base | 0,75 | 40/56 | %85,6 | %86,2 | 0,049 |
| ERes2NetV2 | 0,70 | 40/56 | %87,1 | %87,2 | 0,103 |
| TitaNet large | 0,80 | 38/56 | %85,0 | %84,5 | 0,055 |
| **CAM++ common** | **0,70** | **47/56** | **%91,7** | %89,9 | **0,030** |

CAM++ 0,70 ile tek kişilik dosyalarda: hasta 10/15, diğer roller 68/70 tek kişi buldu.
Sentetik (temiz TTS) seste 22/28 (eşik 0,65'te 25/28): ayar gerçek sese göre yapıldı.
**Sınır:** veri küçük (4 konuşmacı); eşik 0,65/0,70/0,75 arasında sonuç çok değişiyor,
daha fazla gerçek diyalogla yeniden ayarlanmalı. Hasta seslerinde tek kişiyi iki kişi
sanma hâlâ var (5/15).

## Sentetik diyaloglar — `benchmarks/konusmaci_testi.py`

Piper sesleri (vi ve tr, her dilde 3 ses), 1–3 kişi, 12 konuşma sırası.

| Yöntem | Sayı doğru | Cümle doğru kişide | RTF |
|---|---|---|---|
| pyannote seg-3.0 + WeSpeaker (eski) | 8/14 | %81,0 | 0,121 |
| parça imzası + kümeleme (yeni) | 14/14 | %100 | 0,037 |
| aynısı, sıra değişiminde duraksama yok | 14/14 | %100 | 0,046 |

## Henüz yapılmadı

- VIVOS test (tarafsız) — `benchmarks/dogruluk_testi.py`; bu ortamdan Hugging Face'e erişim yok.
- Türkçe gerçek ses (FLEURS tr) — `benchmarks/turkce_dogruluk_testi.py`.
- Gerçek, kesintisiz çekilmiş doktor-hasta görüşmesi (şimdiki diyaloglar kesitlerden kuruldu;
  üst üste konuşma yok).

# 0004 — Dil seçimi ve dil başına model

**Tarih:** 13 Ekim 2026 · **Durum:** Kabul

## Bağlam
Sistem Türkçe ve Vietnamca sesleri işler. Dil istekte gelebilir ya da gelmeyebilir.
Fine-tuning projesinin Türkçe modeli faster-whisper small; Vietnamca için Zipformer-68M
(Apache-2.0, ince ayar serbest). Hedef: 1 dk ses en geç 1 dk'da biter (CPU).

## Karar
- Dil istekte varsa (`tr`, `vi`) tespit atlanır; yoksa Whisper-tiny → small kademeli tespit,
  yalnızca desteklenen diller arasında. Desteklenen dillerin toplam olasılığı < 0,5 ise ret
  (`job.rejected`), güven < 0,8 ise sonuçta "emin değil" işareti.
- Her dilin ayrı kuyruğu ve ayrı ASR motoru (dil profili). Yeni dil = yeni profil.
- Vietnamca: Zipformer-68M; sesin tamamı en sessiz anlardan bölünür (VAD konuşmayı atıyordu).
- Türkçe: faster-whisper small (proje modeli, kelime zamanlı), yalnızca VAD parçaları
  (Whisper sessizlikte metin uyduruyor). Hız bu makinede ölçülemedi (Hugging Face erişimi yok);
  `benchmarks/kabul_testi.py --dil tr` oranı 1'i aşarsa `TR_MODEL=omnilingual-300m`.

## Sonuçlar
Türkçe model değişse de arayüz (Word listesi) aynı kalır; birleştirme ve konuşmacı ayırma
dile bağlı değildir.

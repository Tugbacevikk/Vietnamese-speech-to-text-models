# Yapay zeka asistanları için yönerge

Bu depoda çalışan ya da bu servisi kullanacak kod yazan bir yapay zeka sohbet botuna (Claude, ChatGPT,
Copilot vb.) verilecek kurallar. Aşağıdaki metin olduğu gibi sohbetin başına yapıştırılabilir.
Kısa kurallar için depo kökündeki `CLAUDE.md` / `AGENTS.md` dosyalarına da konabilir.

```text
Rol: Ses Analizi Sistemi (Vietnamca + Türkçe konuşma metne çevirme ve konuşmacı ayırma) deposunda
çalışıyorsun. Sistem air-gapped (internetsiz), yalnızca ticari olmayan kullanım için, CPU öncelikli.
Kullanıcı bilgisayar mühendisliği öğrencisi; açıklamalarını Türkçe, az jargonla yaz.

1. Önce oku
- README.md, docs/entegrasyon.md (API sözleşmesi), docs/olcumler.md (ölçümler), docs/adr/ (kararlar).
- Bir şeyi değiştirmeden önce ilgili kodu ve testini oku. Sürüm, port, ayar adı ya da uç adını
  tahmin etme; kodda ve .env.example'da doğrula.

2. Mimari sınırlar (lint-imports bunları CI'da zorlar)
- Katmanlar: interface -> infrastructure -> application -> domain. Alttaki üsttekini import etmez.
- domain ve application model/altyapı kütüphanesi (sherpa_onnx, onnxruntime, faster_whisper,
  fastapi, sqlalchemy, redis ...) import etmez; bunlar infrastructure'a gider.
- Veritabanı şeması yalnızca Alembic göçüyle değişir (yeni dosya: alembic/versions/); `alembic check`
  CI'da çalışır.

3. Metin kuralı
- Veritabanındaki ham metin küçük harfli ve noktalamasızdır (fusion.py üretir). Büyük harf ve
  noktalama yalnızca gösterim katmanında eklenir (application/transcript_view.py, punctuator.py).
  Ham metne noktalama ya da büyük harf yazma.

4. Dokunma
- Kullanıcının kendi ince ayar hattı `finetune/` ve ayrı "fine-tuning" deposu kullanıcıya aittir:
  yalnızca oku; istenmedikçe değiştirme, o depoya hiçbir şey ekleme.
- Kullanıcının elle geri aldığı değişiklikleri (ör. LLM düzeltici varsayılanını kapatma) yeniden yapma.
- Model dosyalarını, .env'i, data/ ve recordings/ klasörlerini commit etme (.gitignore).

5. Ölçüm dürüstlüğü
- Ölçmediğin sayıyı yazma. Ölçülmemişse "ölçülmedi" ya da "tahmin" diye işaretle.
- Yük ve hız iddiası için sentetik ton/sinüs kullanma; gerçek konuşma sesi kullan (benchmarks/yuk_testi.py
  sentetik sesi yalnızca --sentetik bayrağıyla çalıştırır).
- Her sayının yanına betiği ve veriyi yaz (docs/olcumler.md biçimi). Büyük veri setini her seferinde
  baştan analiz etme; küçük örnek al (ör. rastgele 50 dosya, sabit tohum) ve örnek boyutunu belirt.
- Türkçe için --dil tr ile benchmarks/yerel_dogruluk_testi.py; hız için benchmarks/kabul_testi.py
  (hedef: 1 dk ses en geç 1 dk).

6. Göndermeden önce (CI ile aynı)
  python -m pyflakes src tests scripts benchmarks
  lint-imports
  pytest -q tests/unit tests/db      # PostgreSQL ve Redis çalışıyor olmalı
  alembic upgrade head && alembic check
- Davranış değiştiriyorsan test ekle. Yeşil olmadan push etme.
- Kullanıcı istedi: doğrudan main'e commit ve push (pull request açma). Commit mesajı Türkçe, ne ve neden.

7. Güvenlik
- Dışarı bağlanan adresler (callback_url, canli_url, http/rtsp/rtmp/srt kaynağı) security.py'deki
  SSRF denetiminden geçmeli; WEBHOOK_ALLOWED_HOSTS ve STREAM_ALLOWED_HOSTS'u atlama.
- Webhook imzası HMAC-SHA256(secret, "{X-Timestamp}.{ham gövde}"); ham baytlarla doğrula.
- API anahtarı, sır ve gerçek kişi verisi (ses, transkript) loga ya da commit'e yazma.

8. Servisi kullanan kod yazarken
- Dosya: POST /jobs (external_id ver, 202 sonrası callback_url ya da GET /jobs/{id}).
- Canlı: POST /streams; ses RTP ile ya da kaynak "push" ile POST /streams/{id}/audio?seq=N
  (ham PCM16 little-endian, mono, ses_hizi; 429'da Retry-After sonra AYNI parçayı yolla;
  409'da X-Expected-Seq'ten devam); sonuç GET /streams/{id}/events (SSE), WS /streams/{id}/ws
  ya da canli_url webhook'u. Örnek: scripts/dis_servis_ornek.py.
- Canlı oturumlar API sürecinin belleğindedir; birden çok kopyada sticky session gerekir.

9. Yanıt biçimi
- İşin sonunda neyin değiştiğini, neyi çalıştırıp doğruladığını ve neyi doğrulayamadığını 3-5 cümleyle yaz.
- Emin olmadığın yerde "bilmiyorum, şu komutla ölçülür" de; uydurma.
```

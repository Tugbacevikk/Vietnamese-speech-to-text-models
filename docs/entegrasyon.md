# Entegrasyon rehberi

Bu sistemi başka bir uygulamaya (ör. mesajlaşma sistemi) bağlayan geliştirici için. Kısaca:
ses dosyasını `POST /jobs` ile gönderin, `job_id` alın; sonucu ya `GET /jobs/{id}` ile sorgulayın
ya da `callback_url` verip iş bitince size gelen bildirimi bekleyin (önerilen).

## 1. İş gönderme

```
POST /jobs            (multipart/form-data)
X-API-Key: <API_KEY>  (API_KEY tanımlıysa; production'da zorunlu)
```

| Alan | Zorunlu | Açıklama |
|---|---|---|
| `file` | evet | Ses dosyası. WAV, MP3, FLAC, OGG, OPUS; M4A/AAC/AMR/WEBM için sunucuda ffmpeg gerekir |
| `language` | hayır | `vi`, `tr` ya da `auto` (varsayılan). `auto`: dil sesten bulunur |
| `num_speakers` | hayır | Biliniyorsa konuşmacı sayısı (1–20). `1` verilirse konuşmacı ayırma hiç çalışmaz (daha hızlı) |
| `callback_url` | hayır | İş bitince sonucun POST edileceği adres |
| `notify` | hayır | Hangi olaylarda bildirim gitsin, virgülle: `running,done,failed,rejected`. Boşsa `NOTIFY_EVENTS` (varsayılan `done,failed,rejected`) |
| `external_id` | hayır | Sizin sistemdeki kimlik (ör. mesaj kimliği). Aynısı tekrar gelirse yeni iş açılmaz |
| `profile` | hayır | `full` (varsayılan) ya da `feedback`: kısa sesli mesajlar için hızlı mod (10 sn altında konuşmacı ayırma yok, Whisper tek ışın) |

Yanıtlar:

| Kod | Anlamı |
|---|---|
| 202 | İş kuyruğa girdi: `{"job_id", "status": "queued", "duration", "language"}` |
| 200 | Aynı `external_id` daha önce gelmiş: `{"job_id": <eski iş>, "status", "tekrar": true}`. Dosya ikinci kez saklanmaz |
| 400 | Ses okunamadı (bozuk dosya) |
| 401 | API anahtarı yok/yanlış |
| 413 | Dosya `MAX_UPLOAD_MB`'tan büyük ya da ses `MAX_AUDIO_MINUTES`'tan uzun |
| 415 | Desteklenmeyen dosya türü |
| 422 | Geçersiz alan (dil, profil, notify, callback adresi) |
| 429 | İstek sınırı (`RATE_LIMIT_PER_MINUTE`, varsayılan kapalı). `Retry-After` başlığına uyun |
| 503 | Sistem dolu (`MAX_PENDING_JOBS` iş bekliyor). `Retry-After` (60 sn) bekleyip tekrar gönderin |

**Ağ hatasında tekrar gönderme:** her isteğe `external_id` koyun. Yanıt gelmeden bağlantı koparsa
aynı `external_id` ile tekrar gönderin; iş iki kez açılmaz (veritabanı kısıtıyla garanti, yarış
durumunda da).

**Ani yük (4000 mesaj):** hepsini aynı anda gönderebilirsiniz; işler kuyrukta sırayla işlenir.
503 alan isteği `Retry-After` sonra yeniden deneyin.

## 2. İş durumları

```
queued ──► running ──► done
   │          │
   │          └──────► failed     (deneme hakkı bitti; error alanında neden)
   └─────────────────► rejected   (bilinçli ret: ör. desteklenmeyen dil tespit edildi)
```

`GET /jobs/{id}` her zaman şu alanları döner: `job_id, status, requested_language, num_speakers,
external_id, profile, notify, language, error, created_at, updated_at`; `done` ise ayrıca `result`.

## 3. Sonuç biçimi (`result`, ve `job.done` bildiriminin gövdesi)

```json
{
  "job_id": "6f26e5a7…",
  "status": "done",
  "duration": 60.0,
  "model": "zipformer-68m",
  "language": {"code": "vi", "confidence": 0.98, "source": "tespit", "uncertain": false},
  "speakers": 2,
  "text": "tüm metin",
  "utterances": [
    {"index": 0, "speaker": "speaker_00", "start": 0.0, "end": 4.3, "text": "…",
     "confidence": 0.91, "low_confidence_words": [3]}
  ],
  "quality": {"kapsama_yuzde": 98.1, "kurtarilan_kelime": 0}
}
```

- `speaker`: `speaker_00`, `speaker_01`… Kime ait olduğu bulunamayan kısa parçalar `BELIRSIZ`.
- `confidence`: satırın ortalama kelime güveni (0–1). `low_confidence_words`: güveni 0,5'in altındaki
  kelimelerin satır içindeki sıra numaraları. Bu kelimelerde hata oranı ~3 kat (olcumler.md);
  arayüzde altı çizili gösterilir, kontrol eden kişi önce bunlara bakmalı.
- `quality.kapsama_yuzde`: konuşma olan sürenin ne kadarında metin çıktı. Düşükse kayıp olabilir.
- `edited: true`: satır bir kişi tarafından düzeltilmiş.
- Noktalama yoktur, metin küçük harftir (istenen biçim).

## 4. Bildirimler (webhook)

İş durumu değişince `callback_url`'e POST gönderilir. Bildirimler veritabanında "outbox" tablosuna
iş durumuyla aynı işlemde yazılır: sunucu çökse de kaybolmaz.

| Olay (`X-Event`) | Ne zaman | Gövde |
|---|---|---|
| `job.running` | İşlenmeye başladı (varsayılanda kapalı) | `{"job_id", "status": "running", "error": null}` |
| `job.done` | Bitti | Bölüm 3'teki sonuç |
| `job.failed` | Hata, deneme hakkı bitti | `{"job_id", "status": "failed", "error"}` |
| `job.rejected` | Bilinçli ret | `{"job_id", "status": "rejected", "error"}` |

Başlıklar:

```
Content-Type: application/json; charset=utf-8
User-Agent: viet-asr-webhook/1
X-Event: job.done
X-Job-Id: <job_id>
X-Delivery-Id: <bu teslimin kimliği>
X-Timestamp: <unix saniye>                     (WEBHOOK_SECRET tanımlıysa)
X-Signature: sha256=<hex HMAC>                 (WEBHOOK_SECRET tanımlıysa)
```

**Teslim kuralları**

- 2xx yanıt = teslim edildi. Başka her şey (hata kodu, zaman aşımı, bağlantı yok) = tekrar:
  10 sn, 20 sn, 40 sn … en fazla 1 saat aralıkla, `WEBHOOK_MAX_ATTEMPTS` (varsayılan 8) kez.
- Aynı bildirim birden fazla gelebilir (ağ koparsa). Alıcı `X-Delivery-Id`'yi saklayıp tekrarı atlamalı.
- Yeniden analiz edilen iş (Bölüm 6) aynı `job_id` ile yeni bir `job.done` gönderir (farklı
  `X-Delivery-Id`). Alıcı son geleni geçerli saymalı.
- Hızlı yanıt verin (10 sn içinde); ağır işi kendi kuyruğunuza bırakın.
- Deneme hakkı biten bildirimler "ölü" sayılır: `GET /webhooks/dead` listeler,
  `POST /webhooks/{id}/retry` yeniden kuyruğa koyar. `/metrics/prometheus`'ta
  `asr_webhook_deliveries{state="olu"}` izlenebilir.
- Kurulum sırasında: `POST /webhooks/test {"url": "..."}` imzalı bir `ping` gönderir ve alıcının
  yanıt kodunu döner.

### İmza doğrulama

İmza `HMAC_SHA256(WEBHOOK_SECRET, "{X-Timestamp}.{ham gövde}")`. Gövdeyi JSON olarak çözüp yeniden
yazmadan, **gelen ham baytlar** üzerinden hesaplayın. Zaman damgası 5 dakikadan eskiyse reddedin
(eski isteğin tekrar oynatılmasına karşı).

Python (FastAPI):

```python
import hashlib, hmac, time
from fastapi import FastAPI, Request, HTTPException

SECRET = b"..."  # WEBHOOK_SECRET
app = FastAPI()
seen = set()     # gerçekte veritabanı

@app.post("/asr-callback")
async def callback(req: Request):
    body = await req.body()
    ts, sig = req.headers.get("X-Timestamp", ""), req.headers.get("X-Signature", "")
    good = "sha256=" + hmac.new(SECRET, ts.encode() + b"." + body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(good, sig) or abs(time.time() - int(ts or 0)) > 300:
        raise HTTPException(401)
    if req.headers["X-Delivery-Id"] in seen:
        return {"ok": True}                       # tekrar: yine 2xx dön
    seen.add(req.headers["X-Delivery-Id"])
    data = await req.json()
    ...                                           # kendi kuyruğunuza bırakın
    return {"ok": True}
```

Node.js (Express):

```js
const crypto = require("crypto");
app.post("/asr-callback", express.raw({ type: "application/json" }), (req, res) => {
  const ts = req.get("X-Timestamp") || "", sig = req.get("X-Signature") || "";
  const good = "sha256=" + crypto.createHmac("sha256", process.env.WEBHOOK_SECRET)
    .update(ts + ".").update(req.body).digest("hex");
  const ok = sig.length === good.length && crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(good));
  if (!ok || Math.abs(Date.now() / 1000 - Number(ts)) > 300) return res.sendStatus(401);
  const data = JSON.parse(req.body.toString("utf8"));
  res.sendStatus(200);
});
```

C# (ASP.NET Core):

```csharp
app.MapPost("/asr-callback", async (HttpRequest req) => {
    using var ms = new MemoryStream(); await req.Body.CopyToAsync(ms); var body = ms.ToArray();
    var ts = req.Headers["X-Timestamp"].ToString(); var sig = req.Headers["X-Signature"].ToString();
    using var h = new HMACSHA256(Encoding.UTF8.GetBytes(secret));
    var good = "sha256=" + Convert.ToHexString(
        h.ComputeHash(Encoding.UTF8.GetBytes(ts + ".").Concat(body).ToArray())).ToLowerInvariant();
    if (!CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(good), Encoding.UTF8.GetBytes(sig))
        || Math.Abs(DateTimeOffset.UtcNow.ToUnixTimeSeconds() - long.Parse(ts)) > 300)
        return Results.Unauthorized();
    return Results.Ok();
});
```

## 5. Sorgulama ile çalışma (callback kullanmadan)

`GET /jobs/{id}`'yi 2–5 sn aralıkla sorgulayın; `status` `done`, `failed` ya da `rejected` olunca
durun. Çok sayıda iş için callback daha verimlidir.

Beklerken yanıtta ek alanlar:

| Durum | Alan | Anlamı |
|---|---|---|
| `queued` | `sira` | Önündeki bekleyen iş sayısı (0 = sıradaki) |
| `running` | `asama` | `ses_okunuyor`, `konusma_bulunuyor`, `dil_belirleniyor`, `metne_cevriliyor`, `eksikler_kurtariliyor`, `konusmacilar_ayriliyor`, `birlestiriliyor` |
| `running` | `ilerleme` | Yaklaşık yüzde (aşamaya göre; dile ve donanıma göre sapar) |

## 6. Sonucu düzeltme ve yeniden analiz

| İstek | Ne yapar |
|---|---|
| `PATCH /jobs/{id}/utterances/{n}` `{"text"?, "speaker"?}` | Satırı düzeltir; modelin ilk hali saklanır (ince ayar verisi) |
| `DELETE /jobs/{id}/utterances/{n}` | Uydurma satırı sonuçtan kaldırır (kayıtta kalır) |
| `POST /jobs/{id}/reanalyze` `{"num_speakers": 0–20}` | Aynı sesi yeni konuşmacı sayısıyla yeniden analiz eder (0 = sesten bul). **Aynı iş** kuyruğa döner, yeni iş açılmaz; elle düzeltmeler yeni sonuçla değişir. 409: iş bitmemiş ya da sesi silinmiş |
| `DELETE /jobs/{id}` | Bitmiş işi, sonucunu ve sesini siler |

Sesler `AUDIO_RETENTION_HOURS` (varsayılan 24) sonra silinir (düzeltilmiş işlerinki hariç); sonrasında `reanalyze` ve
ses dinleme 409/410 döner.

## 7. Temiz transkript (okunacak metin)

`GET /jobs/{id}/transcript` kayıttaki satırları DEĞİŞTİRMEDEN okunur bir görünüm verir:

```
[00:00 - 00:57] [Sunucu - Kadın]: Vừa rồi và chúc bác nhiều sức khỏe. Vâng mỗi người thì lại có...
[00:57 - 01:06] [Katılımcı 1 - Erkek]: Bác sĩ ạ vâng trước hết thì cũng xin hỏi bác là...
```

| Parametre | Değerler | Varsayılan |
|---|---|---|
| `etiket` | `genel` (Konuşmacı 1, 2…), `studyo` (en çok söz alan Sunucu, diğerleri Katılımcı 1, 2…), `yok` | `TRANSCRIPT_LABELS` |
| `temizle` | dolgu sesleri (á, ờ, ừm; ıı, ee) ve kekeleme tekrarı ("thì thì") atılır; anlamlı ikilemeler ("từ từ", "yavaş yavaş") kalır | `true` |
| `noktalama` | cümle sınırı = konuşmadaki gerçek sessizlik (VAD); büyük harf, sonda `.` ya da soru kalıbıyla `?` | `true` |
| `bicim` | `txt` ya da `json` (`[{speaker, label, start, end, text}]`) | `txt` |

- Konuşmacı değişince yeni satır; aynı kişinin art arda satırları tek blok.
- **Cinsiyet** yalnızca sesten ölçülür (ses perdesi, `speaker_info.f0`): ≤155 Hz erkek,
  ≥185 Hz kadın, arası belirsiz (etikete yazılmaz). Rolden ya da metinden tahmin edilmez.
- `studyo` rolü bir varsayımdır (en çok söz alan kişi sunucu sayılır); gerçek stüdyo
  kayıtları için uygundur, telefon görüşmesinde `genel` kullanın.
- Sonuç JSON'unda da `speaker_labels` (genel) ve `speaker_info` ({f0, gender}) var.

## 8. Veri setine aktarma (isteğe bağlı)

Kontrol edilmiş sonuçlar modelin doğruluğunu ölçmek (test) ya da ince ayar (eğitim) için
kullanılabilir. Varsayılan **kapalı**; `ENABLE_DATASET_EXPORT=1` ile açılır.

1. Arayüzde "Veri setine ekle" ya da `POST /jobs/{id}/dataset`
   `{"reviewer": "Ad", "consent": true, "use": "test"|"egitim", "notes": "..."}`.
   `consent: true` ve kontrol eden kişinin adı zorunlu; kim, ne zaman, hangi amaçla onay verdiği
   veritabanına yazılır.
2. Kayıt `DATASET_DIR/bekleyen/<iş>/` klasörüne düşer (repo dışında; ses commit edilmez).
3. Sorumlu kişi `python scripts/veri_seti.py listele` ile bakar, `onayla <iş>` ya da `reddet <iş>`.
   - `test` → `test/<dil>/diyalog.tsv` (doğrudan `benchmarks/diyalog_testi.py` ile ölçülür)
   - `egitim` → `egitim/<dil>/liste.tsv` (1–20 sn satır kesitleri; `finetune/veri_hazirla.py` girdisi)
   - Aynı ses iki bölüme giremez (ses özeti `ozetler.tsv`'de tutulur): test sonucu şişmez.
4. `scripts/duzeltmeleri_disa_aktar.py` yalnızca onay verilmiş işlerin düzeltmelerini dışa aktarır.

## 9. Canlı akış (RTP, yayın)

Ses dosyası yerine canlı akış: metin konuşma sırasında gelir, akış bitince kaydedilen ses normal
işe dönüşür (konuşmacı ayırma, temiz transkript).

```
POST /streams {"kaynak": "rtp://0.0.0.0:5004", "dil": "vi"}  -> 201 {"id", "port", "durum": "dinleniyor"}
WS   /streams/{id}/ws  (?key=API_KEY)                      -> {"tur": "parca", ...} ... {"tur": "bitti", "son_is"}
GET  /streams/{id}                                         -> o ana kadarki parçalar
DELETE /streams/{id}                                       -> durdurur (ses yine kaydedilir)
```

| Kaynak | Ne |
|---|---|
| `rtp://0.0.0.0:5004` | UDP portunda RTP dinle: G.711 PCMU/PCMA (telefon, 8 kHz) ve L16. Sıra bozulması ve paket kaybı düzeltilir; başka akış (SSRC) karışırsa atılır. Port 0 = boş port seçilir, yanıtta döner |
| `{"sdp": "v=0 …"}` | **SDP metni** (santralin SIP ile verdiği): port ve biçim buradan okunur. PCMU, PCMA ve L16 (WAV'ın başlıksız ham hali, mono/stereo) bizim alıcıyla; Opus (OGG'nin içindeki ses), G.722 vb. ffmpeg ile. Tuş sesi (telephone-event) paketleri atılır. `kaynak: "rtp://0.0.0.0:0"` ile birlikte verilirse dinleme adresi/portu oradan (yalnızca bizim alıcının biçimlerinde) |
| `/yol/cagri.sdp` | SDP dosyası (aynısı; `ALLOW_LOCAL_PATHS=1`) |
| `http(s)://…/akim.ogg` | OGG/MP3/WAV yayını; `rtsp://`, `srt://`, `udp://` de olur (ffmpeg) |
| `/yol/kayit.wav` | dosyayı gerçek zaman hızında oynat (deneme; `ALLOW_LOCAL_PATHS=1`) |

Parça olayı: `{"tur": "parca", "index", "start", "end", "text", "final", "confidence",
"low_confidence_words", "latency"}`. `final: false` ara sonuçtur (konuşma sürerken ~1,5 sn'de
bir); aynı `index`in `final: true` olayı gelince yerine geçer. `latency`: parçanın sonundan
metnin hazır olmasına kadar geçen süre.

- **Vietnamca (Zipformer):** parçalar bitişik, hiçbir ses atılmaz (VAD yalnızca kesim yerini
  seçer; en az 5, en fazla 8 sn). Ara sonuç var.
- **Türkçe (Whisper):** yalnızca konuşma parçaları, en fazla `STREAM_MAX_SPEECH` sn; ara sonuç
  yok (Whisper ara çözümleri akışı geride bırakır).
- **Konuşmacı:** canlı metinde yok. Santraller her tarafı genelde ayrı RTP akışı (ayrı port)
  olarak verir: taraf = akış. Tek akışta birden çok kişi varsa konuşmacılar akış bitince gelen
  tam analizde (`son_is`).
- Tam modda (API ayrı süreç) canlı akış CLI ile: `python -m viet_asr.interface.akis --kaynak
  rtp://0.0.0.0:5004 --dil vi --api http://127.0.0.1:8001`.
- SDP örneği (L16, 16 kHz):

  ```
  v=0
  c=IN IP4 10.0.0.5
  m=audio 5004 RTP/AVP 96
  a=rtpmap:96 L16/16000/1
  ```

  Opus: `m=audio 5004 RTP/AVP 111` + `a=rtpmap:111 opus/48000/2`. Komut satırı:
  `python -m viet_asr.interface.akis --sdp cagri.sdp --dil vi`.
- Deneme göndericisi: `python scripts/rtp_gonder.py kayit.wav --hedef 127.0.0.1:5004
  [--kayip 0.02 --karistir 0.05]` ya da `ffmpeg -re -i kayit.wav -ar 8000 -ac 1 -c:a pcm_mulaw
  -f rtp rtp://127.0.0.1:5004`.

### Dışarıdaki servis için: ses nasıl girer, sonuç nasıl alınır

Karşıdaki servis iki yoldan birini seçer; sonucu almanın üç yolu her ikisinde de aynıdır.

**Ses girişi**

| Yol | Ne zaman | Nasıl |
|---|---|---|
| **RTP** | santral / VoIP / medya sunucusu zaten RTP yayınlıyor | `POST /streams {"kaynak": "rtp://0.0.0.0:0"}` → yanıttaki `rtp_adres` (`STREAM_PUBLIC_HOST:port`) adresine UDP gönderilir. Opus için `sdp` alanı |
| **Parça parça HTTP** | servis RTP kuramıyor, sesi kendisi bölüyor | `{"kaynak": "push", "ses_hizi": 16000}` açılır; her parça `POST /streams/{id}/audio?seq=N` (gövde: ham PCM16 little-endian, mono) |
| **WebSocket** | tek bağlantıda hem ses hem metin | `WS /streams/{id}/audio?key=…`: ikili mesaj = PCM16 parçası, `"son"` metni = bitir; metin olayları aynı bağlantıdan döner |

Parça süresi 0,5–2 sn önerilir (en çok 2 MB/istek). Aynı `seq` iki kez gelirse ikincisi yok sayılır
(zaman aşımı sonrası güvenle tekrar gönderilir); atlanan numara `409` + `X-Expected-Seq`.
Sunucu yetişemezse `429` + `Retry-After`: **aynı parçayı** bekleyip yeniden gönderin
(en çok ~30 sn ses bekletilir; sınırsız büyütmek belleği doldururdu). Bitirmek için
`POST …/audio?son=true`, ya da DELETE, ya da `idle_timeout` sn ses gelmemesi.

**Sonuç alma**

| Yol | Ne |
|---|---|
| `GET /streams/{id}/events` (SSE) | her kesin parça `id: <index>` ile; koparsa `Last-Event-ID` ile kaldığı yerden. Ara sonuçlar da gelir |
| `WS /streams/{id}/ws` | aynı olaylar WebSocket ile |
| **`canli_url`** (webhook) | akış açılırken verilir; her **kesin** parça `stream.segment`, sonda `stream.ended` olarak POST edilir. İmza ve başlıklar iş bildirimleriyle aynı (`WEBHOOK_SECRET`, `X-Signature`). Teslim edilemezse 3 deneme (0,5 / 1 / 2 sn), sonra o parça atılır: canlı akış beklemez, kesin metin `GET /streams/{id}` ve `son_is`te durur. `callback_url` ise akış bitince oluşan **tam analiz işinin** bildirimidir (iki ayrı şey) |

**Sunucu tarafı**

- `STREAM_RTP_PORTS=40000-40100`: port 0 istenince bu aralıktan boş port verilir; Docker/güvenlik
  duvarında yalnızca bu UDP aralığı açılır. `STREAM_PUBLIC_HOST`: karşı servise söylenecek adres.
- `STREAM_ALLOWED_HOSTS`: sunucunun dışarı bağlanacağı kaynaklar (`http`, `rtsp`, `rtmp`, `srt`) yalnızca bu adreslere;
  boşsa production'da iç ağ adresi `422`. `rtp://`, `push` ve `udp://` sunucu dinlediği için denetlenmez.
  `STREAM_MAX_BACKLOG` (varsayılan 30 sn): push'ta bekleyen ses bunu aşınca `429`.
- Özet (`ozet`) sayaçları kaynağa göre `rtp` ya da `push` anahtarı altında gelir.
- `STREAM_MAX_SESSIONS` aşılırsa `422`. Canlı oturumlar bu sürecin belleğindedir: birden çok API
  kopyası arkasında aynı akışın istekleri aynı kopyaya gitmeli (yapışkan oturum).
- API anahtarı: HTTP'de `X-API-Key`; tarayıcı dışı istemcide WebSocket için de başlık olur,
  tarayıcıda `?key=`.
- Deneme istemcisi (servisin yazacağı kodun örneği): `python scripts/dis_servis_ornek.py kayit.wav
  --api http://sunucu:8000 [--uc ws] [--hiz 0]`.

## 10. İzleme

| Uç | Ne için |
|---|---|
| `GET /health` | Veritabanı/kuyruk erişimi, iş sayıları, açık özellikler (anahtar istemez) |
| `GET /health/ready` | Modeller yüklü ve her dil kuyruğunda çalışan işçi var mı (yük dengeleyici için) |
| `GET /metrics` | Gecikme, "1 dk ses ≤ 1 dk" hedefine uyum, kuyruk doluluğu (JSON) |
| `GET /metrics/prometheus` | Aynısı + aşama süreleri, webhook teslim/ölü sayıları |
| `GET /quality?gun=90` | Kişilerin düzeltmelerinden hafta / dil / model başına düzeltme oranı (yaklaşık WER) |
| `GET /audit?is_id=&kim=&limit=` | Erişim kaydı: kim, ne zaman, hangi işte ne yaptı (`AUDIT_READERS` doluysa yalnızca onlar) |

`GET /health` yanıtındaki `uyarilar` listesi sorun olunca dolar (arayüzde kırmızı şerit):
teslim edilemeyen bildirim, `ALERT_QUEUE_WAIT_MIN`'den uzun bekleyen iş, son 24 saatte başarısız
iş, gecikmiş ya da hiç alınmamış yedek, `ALERT_DISK_GB`'tan az boş disk. Her uyarı ne yapılacağını
söyler; sorun çözülünce kendiliğinden kalkar.

**Anahtarlar ve erişim kaydı.** `API_KEY` entegre olan sistemin anahtarıdır (kayıtta "sistem").
Arayüzü kullanan her kişiye `API_KEYS=Ad:anahtar,...` ile ayrı anahtar verilir; böylece şu eylemler
kişi adıyla kaydedilir: iş oluşturma, sonucu açma, sesi dinleme, transkript alma, satır düzeltme ve
silme, işi silme, yeniden analiz, veri setine aktarma, düzeltmeleri dışa alma, erişim kaydını okuma.
Aynı kişinin aynı işteki aynı eylemi 60 sn'de bir yazılır (arayüzün sorgulaması kaydı şişirmesin).
Kayıt işe bağlı değildir: iş silinse de kalır, `AUDIT_RETENTION_DAYS` (730) gün sonra silinir.

## 11. Canlı altyazı (mikrofon, `/ws/live`)

Tarayıcı ya da başka bir istemci sesi WebSocket ile gönderir, metin konuşma sırasında gelir.
Hazır ekran: `GET /canli`. Diller ve model durumu: `GET /live/health`.

```
WS /ws/live?language=vi            (tr | vi | auto; API_KEY varsa ?key=... ya da X-API-Key)
  istemci -> ikili mesaj: Int16 little-endian PCM, 16 kHz, mono (her 100-250 ms bir parça)
  istemci -> {"type": "stop"}       son parçayı işle ve kapat
  sunucu  -> {"type": "ready", "session_id", "resume_token", "language", "model", "device"}
             {"type": "partial", "start", "text"}          ara sonuç (konuşma sürerken)
             {"type": "final", "id", "start", "end", "text", "speaker"}
             {"type": "discard", "start"}                  ara sonuç geçersiz (gürültüydü)
             {"type": "language", "code", "confidence"}    yalnızca auto'da, ilk cümleden sonra
             {"type": "stats", "lag_sec"}                  işleme gecikmesi
             {"type": "done", "session_id", "job_id", "webhook"}
```

İsteğe bağlı sorgu parametreleri: `silence_ms` (cümle sonu sessizliği, 700), `partial_sec` (ara
sonuç sıklığı), `max_sec` (cümle üst sınırı), `save=0` (sesi kaydetme), `analyze=0` (oturum sonu
toplu iş açma), `callback_url`.

- **Vietnamca (Zipformer):** parçalar bitişik, 5-8 sn, kısa bir duraklamada kesilir; hiçbir ses
  atılmaz. Live-stream'in VAD bölücüsü cümleler arasını atıyordu; Zipformer'da bu sözcük
  kaybettirdi (bkz. ölçümler).
- **Türkçe (Whisper):** VAD ile cümle cümle (700 ms sessizlikte cümle biter, en fazla 12 sn);
  Whisper'ın uydurduğu kalıp cümleler ("Altyazı M.K.", "Abone olun." …) atılır.
- **Konuşmacı:** her kesin satıra CAM++ ses izinden canlı etiket (`speaker_00`…; kısa satır
  önceki kişiye verilir). Kesin konuşmacı ayrımı oturum sonundaki toplu iştedir (`job_id`).
- **Kopma:** bağlantı koparsa oturum `LIVE_RESUME_SEC` (30 sn) bekler; istemci
  `?resume=<session_id>&token=<resume_token>&after=<son final id>` ile döner, kaçırdığı satırlar
  yeniden gelir. `canli.html` bunu kendisi yapar.
- **Bildirim:** oturum bitince (Durdur ya da kopma) `callback_url`'e ya da `LIVE_WEBHOOK_URL`'e
  imzalı `live.ended` gider (outbox; aynı imza ve yeniden deneme kuralları, `X-Job-Id` boş):

  ```json
  {"event": "live.ended", "session_id": "…", "reason": "stop",
   "started_at": "…", "ended_at": "…", "duration": 94.2,
   "language": {"code": "vi", "source": "secim", "confidence": null},
   "lines": [{"id": 0, "start": 0.4, "end": 6.1, "text": "…", "speaker": "speaker_00"}],
   "text": "…", "recording": "2026-10-16_14-27-55", "job_id": "…"}
  ```

  `job_id` dolu ise toplu iş bitince aynı adrese `job.done` da gelir (konuşmacılı satırlar).
- Kayıtlar `RECORDINGS_DIR`'de (wav + txt + jsonl), `RECORDINGS_RETENTION_DAYS` gün sonra silinir.

### WebRTC ile (başka cihaz, kayıplı ağ)

Aynı oturum, ses WebSocket yerine WebRTC ile: tarayıcı sesi Opus ile UDP'den yollar, mesajlar
"altyazi" veri kanalından gelir (yukarıdaki JSON'ların aynısı, `stop` da oradan gider).

```
POST /webrtc/teklif   (X-API-Key gerekiyorsa başlıkta)
  {"sdp": "<tarayıcının teklifi>", "type": "offer",
   "ayarlar": {"language": "vi", "silence_ms": 700, ...}}      # /ws/live sorgu parametreleri
-> {"sdp": "<sunucunun yanıtı>", "type": "answer", "session_id": "..."}
```

- İstemci önce `createDataChannel("altyazi")` ve mikrofon izini ekler, ICE adaylarını toplar
  (aynı ağda ~0,1 sn), sonra teklifi gönderir. STUN/TURN yok: kapalı ağ, aynı ağdaki adresler yeter.
- Sunucu Opus'u 16 kHz'e çevirip WebSocket'le aynı bölücüye verir; kayıt, toplu iş, `live.ended`
  aynı. Kopunca `ayarlar`'a `resume`, `token`, `after` eklenip yeni teklifle aynı oturuma dönülür.
- `GET /live/health` -> `"webrtc": true` (sunucuda `aiortc` kurulu). Hazır istemci:
  `GET /canli_istemci.js` (`LiveClient`; ana sayfa ve /canli bunu kullanır).
- Hata: 400 (ayar, geçersiz teklif), 401 (anahtar), 503 (aiortc yok ya da 16 bağlantı dolu).

### Başka cihazdan erişim (HTTPS)

Tarayıcı mikrofonu yalnızca güvenli sayfada verir (`https://` ya da bu bilgisayarda `localhost`).
`.env`: `HOST=0.0.0.0`, `HTTPS=1` -> açılışta `data/tls`'de bu bilgisayarın adları ve ağ adresleri
için kendinden imzalı sertifika üretilir, adresler günlüğe yazılır. Kurumun sertifikası varsa
`TLS_CERT` / `TLS_KEY`. Tam modda da aynı: `baslat_tam.bat` API'yi `python -m
viet_asr.interface.api` ile başlatır (HTTPS ve canlı altyazı dahil; konuşmacı ayrımı işçilerde).


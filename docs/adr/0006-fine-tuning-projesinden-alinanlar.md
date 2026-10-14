# 0006 — Fine-tuning projesinden alınanlar, düzeltilenler, alınmayanlar

**Tarih:** 14 Ekim 2026 · **Durum:** Kabul

Fine-tuning projesindeki (Tugbacevikk/fine-tuning) yapıların bu sistemdeki karşılığı.

## Alındı (düzeltilerek)

| Projede | Burada | Düzeltilen |
|---|---|---|
| FasterWhisperAdapter | `faster_whisper_asr.py` (Türkçe, medium) | Ayarlar kurucuda; tembel sonuç yüzünden yakalanamayan toplu çözme hatası; noktalamaya takılan tekrar süzgeci; `log_prob_threshold=None` ile sessiz konuşmanın atılması; istemin çıktıya sızması |
| Celery + Redis | Redis Streams (0001) | Windows'ta çalışır |
| SQLAlchemy + Alembic | code-first depo (0002) | SQLite için toplu değişiklik modu; `external_id` yarışı hata metni yerine kısıtla |
| `external_id` | aynı | — |
| Webhook outbox + HMAC | aynı (+ deneme bildirimi, ölü teslim listesi, yeniden deneme, iş başına olay seçimi, `job.running`) | — |
| SSRF denetimi | `security.callback_problem` | — |
| Rate limiter (Redis) | `rate_limit.py` (varsayılan kapalı) + bekleyen iş üst sınırı (503) | Tek entegre sistemin 4000'lik ani yükü reddedilmesin diye sınır kapalı, koruma kuyruk derinliğiyle |
| Sweeper + kilit | `bakim` + `RedisLock` | Kilit yenileme Lua ile atomik; QUEUED işi yalnızca kuyruk boşsa yeniden koyar (çift işleme yok) |
| Retention | `Maintenance` | Düzeltilmiş işin sesi ince ayar verisi olduğu için kayıtla birlikte silinir |
| `PIPELINE_PROFILE=feedback\|full` | aynı, istekte `profile=` ile iş başına da | — |
| Prometheus `/metrics` | `/metrics/prometheus` | Aşama süreleri veritabanından: ayrı işçi süreçlerindeki işler de görünür |
| `/health/ready` | aynı (modeller / canlı işçiler) | — |
| Arayüz: satır düzeltme/silme, RTTM | aynı + düzeltmelerden ince ayar verisi | Silme kalıcı değil (kayıtta kalır) |
| SemanticRefiner "sandwich" | `FusionConfig.sandwich_max` | Ölçüldü; varsayılan kapalı (olcumler.md) |
| Rust DSP (PyO3) | `native/` (birleştirme) | Python sürümüyle eşlik testi zorunlu (0005) |

## Alınmadı

- **Gürültü giderme (DeepFilterNet):** PyTorch ister; internetsiz CPU kurulumuna ve "1 dk ses ≤ 1 dk"
  hedefine uymaz. ASR modelleri gürültülü veriyle eğitildiği için ön temizlik çoğu zaman yarardan
  çok zarar verir; gerekirse ölçülerek eklenmeli.
- **Üst üste konuşma özeti (overlap):** buradaki konuşmacı ayırma çakışmasız aralık üretir; özet hep
  sıfır çıkardı. pyannote yolu (`DIARIZER=pyannote`) çakışma verebilir; ihtiyaç olursa onunla.
- **LLM ile düzeltme (Ollama):** projede de deneysel; internetsiz sunucuda ayrı model servisi ister.
- **S3 depolama:** tek makine, internetsiz kurulum; yükleme klasörü yeterli.
- **ASR ve konuşmacı ayırmayı aynı iş içinde paralel çalıştırma:** eşzamanlı iş sayısı zaten
  çekirdekleri dolduruyor; iş içi paralellik çekirdekleri aşırı paylaştırır.
- **Escalating/fallback diarizer, SpeechBrain:** spektral kümeleme gerçek veride daha iyi ölçüldü (0003).

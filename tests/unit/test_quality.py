"""Kalite takibi: düzeltmelerden düzeltme oranı (yaklaşık kelime hata oranı)."""
import random
from datetime import datetime, timezone

import jiwer

from viet_asr.application.quality import edit_distance, quality_report, words
from viet_asr.domain import ReviewLine


def test_kelime_mesafesi_jiwer_ile_ayni():
    rng = random.Random(0)
    vocab = ["xin", "chào", "bác", "sĩ", "merhaba", "nasılsın", "evet", "hayır", "ạ"]
    for _ in range(300):
        a = [rng.choice(vocab) for _ in range(rng.randint(1, 12))]
        b = [rng.choice(vocab) for _ in range(rng.randint(1, 12))]
        m = jiwer.process_words(" ".join(a), " ".join(b))
        assert edit_distance(a, b) == m.substitutions + m.deletions + m.insertions
    assert edit_distance([], ["a", "b"]) == 2 and edit_distance(["a"], []) == 1


def test_noktalama_ve_harf_buyuklugu_sayilmaz():
    assert words("Merhaba, nasılsın?") == words("merhaba nasılsın")


def L(job, orig, final, week_day=19, lang="vi", spk=False, model="zipformer-68m"):
    return ReviewLine(job, lang, model, datetime(2026, 10, week_day, tzinfo=timezone.utc), orig, final, spk)


def test_rapor_hafta_dil_model_bazinda():
    rows = quality_report([
        L("a", "xin chào bác sĩ", "xin chào bác sĩ ạ"),          # 1 ekleme
        L("a", "tôi bị đau", "tôi bị đau"),                       # dokunulmamış: doğru sayılır
        L("a", "uydurma satır", ""),                               # silindi: 2 hata
        L("b", "chào", "chào", spk=True),                          # konuşmacı düzeltmesi
        L("c", "merhaba nasılsın", "merhaba nasılsın", lang="tr", model="medium"),
        L("d", "eski hafta", "eski hafta değişti", week_day=5),
    ])
    vi = next(r for r in rows if r["dil"] == "vi" and r["hafta"] == "2026-H43")
    assert (vi["is_sayisi"], vi["kelime"], vi["duzeltilen_kelime"]) == (2, 10, 3)
    assert vi["duzeltme_orani"] == 30.0 and vi["silinen_satir"] == 1 and vi["duzeltilen_satir"] == 1
    assert vi["konusmaci_duzeltme"] == 1
    assert [r["hafta"] for r in rows] == ["2026-H43", "2026-H43", "2026-H41"]  # en yeni önce
    assert next(r for r in rows if r["dil"] == "tr")["duzeltme_orani"] == 0.0


def test_kalite_ucu(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from test_live import make_local  # aynı klasör

    app, store, _ = make_local(tmp_path, monkeypatch)
    from viet_asr.domain import AudioJob, LanguageDecision, LanguageSource, Transcript, Utterance

    store.create(AudioJob("j", "/a.wav", "vi"))
    store.save(Transcript(job_id="j", duration=3.0, model="zipformer-68m",
                          language=LanguageDecision("vi", 1.0, LanguageSource.REQUEST),
                          utterances=[Utterance("speaker_00", 0, 1, "xin chào"),
                                      Utterance("speaker_00", 1, 2, "bác sĩ")]))
    with TestClient(app) as c:
        assert c.get("/quality").json()["satirlar"] == []          # henüz kimse dokunmadı
        assert c.patch("/jobs/j/utterances/0", json={"text": "xin chào ạ"}).status_code == 200
        r = c.get("/quality?gun=30").json()["satirlar"]
    assert len(r) == 1 and (r[0]["kelime"], r[0]["duzeltilen_kelime"], r[0]["duzeltme_orani"]) == (4, 1, 25.0)


def test_islem_hatti_asamalari_yazar_ve_api_gosterir(tmp_path, monkeypatch):
    """Pipeline her aşamada ilerleme yazar; GET /jobs/{id} çalışırken aşama/yüzde, beklerken sıra."""
    from types import SimpleNamespace as NS

    import numpy as np

    from viet_asr.application.config import FusionConfig
    from viet_asr.application.fusion import FusionEngine
    from viet_asr.application.transcribe import TranscribeAndDiarize
    from viet_asr.domain import AudioJob, JobStatus, LanguageDecision, LanguageSource, SpeakerSegment, VadSegment, Word
    from viet_asr.infrastructure.memory_store import MemoryJobStore

    store = MemoryJobStore()
    seen = []
    real = store.set_progress
    store.set_progress = lambda j, s, p: (seen.append((s, p)), real(j, s, p))
    audio = np.zeros(16000 * 4, np.float32)
    seg = VadSegment(0.0, audio[:16000 * 3])
    asr = NS(name="sahte", segmentation="vad", transcribe=lambda segs, **k: [Word("xin", 0.1, 0.5), Word("chào", 0.6, 1.0)])
    p = TranscribeAndDiarize(
        loader=NS(load=lambda path: audio), vad=NS(split=lambda a: [seg]),
        language=NS(resolve=lambda req, segs: LanguageDecision("vi", 1.0, LanguageSource.REQUEST)),
        engines={"vi": asr}, diarizer=NS(diarize=lambda a, s, n: [SpeakerSegment("speaker_00", 0, 4)]),
        fusion=FusionEngine(FusionConfig()), repo=store)
    store.create(AudioJob("j", "/a.wav", "vi"))
    p.run(AudioJob("j", "/a.wav", "vi"))
    stages = [s for s, _ in seen]
    assert stages[:2] == ["ses_okunuyor", "konusma_bulunuyor"] and "konusmacilar_ayriliyor" in stages
    assert [x for _, x in seen] == sorted(x for _, x in seen)          # ilerleme hep artar
    assert store.get("j").status == JobStatus.DONE and store.get("j").progress == 1.0

    from viet_asr.application.serialization import record_to_dict
    store.create(AudioJob("k", "/a.wav", "vi"))
    store.set_status("k", JobStatus.RUNNING)
    store.set_progress("k", "metne_cevriliyor", 0.12)
    d = record_to_dict(store.get("k"))
    assert (d["asama"], d["ilerleme"]) == ("metne_cevriliyor", 12)

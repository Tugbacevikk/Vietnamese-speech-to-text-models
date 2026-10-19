"""Kişi başına anahtar ve erişim kaydı (KVKK): kim, ne zaman, hangi işte ne yaptı."""
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from test_api import Queue, post  # aynı klasör

from viet_asr.domain import AudioJob, LanguageDecision, LanguageSource, Transcript, Utterance
from viet_asr.infrastructure.memory_store import MemoryJobStore
from viet_asr.interface.api import create_app
from viet_asr.interface.security import identify
from viet_asr.interface.settings import Settings

TUGBA, AYSE = {"X-API-Key": "t-111"}, {"X-API-Key": "a-222"}


def make(tmp_path, **kw):
    store = MemoryJobStore()
    s = Settings(data_dir=str(tmp_path), languages=["vi"], api_key="sis-000",
                 api_keys_setting="Tuğba Çevik:t-111, Ayşe:a-222", **kw)
    return store, TestClient(create_app(store, Queue(), s, {"sahte": lambda: None}))


def done_job(store, jid="j", audio=""):
    store.create(AudioJob(jid, audio, "vi"))
    store.save(Transcript(job_id=jid, duration=3.0, model="m",
                          language=LanguageDecision("vi", 1.0, LanguageSource.REQUEST),
                          utterances=[Utterance("speaker_00", 0, 1, "xin chào"),
                                      Utterance("speaker_01", 1, 2, "chào bác sĩ")]))


def test_anahtarlar_kisiye_baglanir():
    s = Settings(api_key="sis-000", api_keys_setting="Tuğba Çevik:t-111, Ayşe:a-222,bozuk")
    assert s.api_keys == {"sis-000": "sistem", "t-111": "Tuğba Çevik", "a-222": "Ayşe"}
    assert identify(s, "t-111") == "Tuğba Çevik" and identify(s, "sis-000") == "sistem"
    assert identify(s, "yanlis") is None and identify(s, None) is None
    assert identify(Settings(api_key="", api_keys_setting=""), None) == "anahtarsız"


def test_kim_ne_yapti_kaydedilir(tmp_path):
    store, c = make(tmp_path)
    done_job(store)
    assert c.get("/jobs/j").status_code == 401                          # anahtarsız
    assert c.get("/jobs/j", headers={"X-API-Key": "yanlis"}).status_code == 401
    for _ in range(5):                                                   # arayüz saniyede bir sorar
        assert c.get("/jobs/j", headers=TUGBA).status_code == 200
    c.patch("/jobs/j/utterances/0", json={"text": "xin chào ạ"}, headers=AYSE)
    c.delete("/jobs/j/utterances/1", headers=AYSE)
    c.get("/jobs/j/transcript", headers=TUGBA)
    jid = post(c.__class__(c.app, headers={"X-API-Key": "sis-000"})).json()["job_id"]  # entegre sistem
    c.delete("/jobs/j", headers=TUGBA)
    rows = c.get("/audit", headers=TUGBA).json()
    got = [(r["kim"], r["eylem"], r["job_id"]) for r in reversed(rows)]
    assert got == [("Tuğba Çevik", "sonuc_goruntulendi", "j"),           # 5 sorgu -> 1 kayıt
                   ("Ayşe", "satir_duzeltildi", "j"), ("Ayşe", "satir_silindi", "j"),
                   ("Tuğba Çevik", "transkript_alindi", "j"), ("sistem", "is_olusturuldu", jid),
                   ("Tuğba Çevik", "is_silindi", "j"), ("Tuğba Çevik", "erisim_kaydi_okundu", None)]
    assert rows[-2]["ayrinti"] == "satır 0 metin" and rows[0]["ip"]
    # iş silindi ama kaydı duruyor; işe göre süzülür (okuma da kaydedilir: en üstte)
    by_job = c.get("/audit?is_id=j", headers=AYSE).json()
    assert len(by_job) == 6 and (by_job[0]["kim"], by_job[0]["eylem"]) == ("Ayşe", "erisim_kaydi_okundu")


def test_erisim_kaydini_yalnizca_yetkililer_okur(tmp_path):
    store, c = make(tmp_path, audit_readers=["tuğba çevik"])
    assert c.get("/audit", headers=AYSE).status_code == 403
    assert c.get("/audit", headers=TUGBA).status_code == 200


def test_eski_kayitlar_bakimda_silinir(tmp_path):
    from viet_asr.application.maintenance import Maintenance

    store = MemoryJobStore()
    store.audit("Ayşe", "ses_dinlendi", "j")
    store.audit_log[0]["at"] = datetime.now(timezone.utc) - timedelta(days=800)
    store.audit("Ayşe", "ses_dinlendi", "k")
    out = Maintenance(store, lambda p: True, 0, 0, 10**9, audit_days=730).run_once()
    assert out["erisim_kaydi_silindi"] == 1 and [r["job_id"] for r in store.audit_rows()] == ["k"]


def test_canli_baglanti_kisi_anahtarini_kabul_eder():
    from viet_asr.interface.security import key_ok

    s = Settings(api_key="", api_keys_setting="Ayşe:a-222")
    assert key_ok(s, "a-222") and not key_ok(s, "x") and s.auth_enabled

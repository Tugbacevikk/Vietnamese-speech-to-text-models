from fastapi.testclient import TestClient

from tests.unit.test_live import make_local


def test_finetune_durum_onayli_satirlari_sayar(tmp_path, monkeypatch):
    app, store, _ = make_local(tmp_path, monkeypatch)
    from viet_asr.domain import AudioJob, LanguageDecision, LanguageSource, Transcript, Utterance

    store.create(AudioJob("j", "/a.wav", "vi"))
    store.save(Transcript(job_id="j", duration=6.0, model="zipformer-68m",
                          language=LanguageDecision("vi", 1.0, LanguageSource.REQUEST),
                          utterances=[Utterance("speaker_00", 0, 3, "xin chào"),
                                      Utterance("speaker_00", 3, 3.5, "ạ")]))
    with TestClient(app) as c:
        assert c.get("/finetune/durum").json()["diller"] == {}   # henüz düzeltme yok
        c.patch("/jobs/j/utterances/0", json={"text": "xin chào ạ"})
        c.patch("/jobs/j/utterances/1", json={"text": "dạ"})
        assert c.get("/finetune/durum").json()["diller"] == {}   # onay (veri setine ekle) verilmedi
        store.set_consent("j", "Dr. B", "egitim")
        r = c.get("/finetune/durum").json()
    assert r["esik_satir"] == 500 and r["esik_sure_sn"] == 3600
    assert r["diller"] == {"vi": {"satir": 1, "sure_sn": 3.0, "esik_asildi": False}}   # 0,5 sn'lik satır sayılmaz


def test_bos_metinle_duzeltme_reddedilir(tmp_path, monkeypatch):
    app, store, _ = make_local(tmp_path, monkeypatch)
    from viet_asr.domain import AudioJob, LanguageDecision, LanguageSource, Transcript, Utterance

    store.create(AudioJob("j", "/a.wav", "vi"))
    store.save(Transcript(job_id="j", duration=3.0, model="m",
                          language=LanguageDecision("vi", 1.0, LanguageSource.REQUEST),
                          utterances=[Utterance("speaker_00", 0, 3, "xin chào")]))
    with TestClient(app) as c:
        assert c.patch("/jobs/j/utterances/0", json={"text": "   "}).status_code == 422
        assert c.patch("/jobs/j/utterances/0", json={"text": "xin chào ạ"}).status_code == 200

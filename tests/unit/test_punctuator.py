"""Vietnamca özel isim büyük harfi: gerçek VietMed metinlerinden alınan sorunlu örneklerle."""
import pytest

from viet_asr.application.punctuator import capitalize_proper_nouns as cap


def test_cok_heceli_adlar_her_yerde_buyur():
    out = cap("tôi sinh ra ở hà nội việt nam và làm việc tại bệnh viện bạch mai của bộ y tế")
    assert "Hà Nội" in out and "Việt Nam" in out and "Bạch Mai" in out and "Bộ Y tế" in out
    assert cap("đức phật có dạy") == "Đức Phật có dạy"


@pytest.mark.parametrize("metin", [
    "sẽ có một cái phương pháp mổ",       # VietMed: "pháp" 340 kez, hepsi bileşik
    "ta hiểu rồi còn liệu pháp hoocmon",
    "gây rất là mất thẩm mỹ trên khuôn mặt",
    "bệnh viện hoàn mỹ thành",
    "mà bên anh á hiện thực",             # anh = zamir
    "ví dụ như anh xây một ngôi nhà",
    "chăm sóc mỹ phẩm và pháp luật",
    "đạo đức nghề nghiệp",
])
def test_tek_heceli_siradan_kelimeler_buyumez(metin):
    assert cap(metin) == metin


@pytest.mark.parametrize("girdi,beklenen", [
    ("kể cả ở mỹ người ta cũng", "kể cả ở Mỹ người ta cũng"),
    ("người đức thì ông nói", "người Đức thì ông nói"),
    ("sang pháp du học", "sang Pháp du học"),
    ("học tiếng anh", "học tiếng Anh"),
    ("tại nga có thuốc", "tại Nga có thuốc"),
])
def test_tek_heceli_ulke_adi_yalnizca_baglamla_buyur(girdi, beklenen):
    assert cap(girdi) == beklenen


def test_turkce_ve_bos_metne_dokunulmaz():
    assert cap("hastanın ateşi var mı", lang="tr") == "hastanın ateşi var mı"
    assert cap("") == ""

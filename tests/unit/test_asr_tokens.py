import unicodedata

from viet_asr.infrastructure.normalizer import ViNormalizer
from viet_asr.infrastructure.sherpa_asr import tokens_to_syllables

N = ViNormalizer()


def test_bpe_parcalari_heceye_birlesir_ve_ofset_eklenir():
    tokens = ["▁BÁC", "▁S", "Ĩ", "▁ƠI"]
    ts = [0.0, 0.4, 0.5, 0.9]
    s = tokens_to_syllables(tokens, ts, seg_start=10.0, seg_end=11.5, normalizer=N)
    assert [x.text for x in s] == ["bác", "sĩ", "ơi"]
    assert [x.start for x in s] == [10.0, 10.4, 10.9]
    assert [round(x.end, 2) for x in s] == [10.4, 10.9, 11.5]  # son hece parça sonuna kadar


def test_hece_suresi_sinirlanir():
    s = tokens_to_syllables(["▁A", "▁B"], [0.0, 3.0], 0.0, 5.0, N, max_duration=1.0)
    assert s[0].end == 1.0 and s[1].end == 4.0


def test_bosluk_ile_baslayan_parca_da_yeni_hece():
    s = tokens_to_syllables([" XIN", " CHÀO"], [0.0, 0.3], 0.0, 1.0, N)
    assert [x.text for x in s] == ["xin", "chào"]


def test_bos_cikti():
    assert tokens_to_syllables([], [], 0.0, 1.0, N) == []


def test_normalizer_nfc_ve_kucuk_harf():
    ayrik = unicodedata.normalize("NFD", "BỆNH  VIỆN")
    assert N.normalize(ayrik) == unicodedata.normalize("NFC", "bệnh viện")
    assert ViNormalizer(strip_punct=True).normalize("Xin chào, bác sĩ!") == "xin chào bác sĩ"

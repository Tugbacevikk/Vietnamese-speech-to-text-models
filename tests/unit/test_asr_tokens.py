import unicodedata

from viet_asr.infrastructure.normalizer import ViNormalizer
from viet_asr.infrastructure.sherpa_asr import tokens_to_words

N = ViNormalizer()


def test_bpe_parcalari_heceye_birlesir_ve_ofset_eklenir():
    tokens = ["▁BÁC", "▁S", "Ĩ", "▁ƠI"]
    ts = [0.0, 0.4, 0.5, 0.9]
    s = tokens_to_words(tokens, ts, seg_start=10.0, seg_end=11.5, normalizer=N)
    assert [x.text for x in s] == ["bác", "sĩ", "ơi"]
    assert [x.start for x in s] == [10.0, 10.4, 10.9]
    assert [round(x.end, 2) for x in s] == [10.4, 10.9, 11.5]  # son hece parça sonuna kadar


def test_hece_suresi_sinirlanir():
    s = tokens_to_words(["▁A", "▁B"], [0.0, 3.0], 0.0, 5.0, N, max_duration=1.0)
    assert s[0].end == 1.0 and s[1].end == 4.0


def test_bosluk_ile_baslayan_parca_da_yeni_hece():
    s = tokens_to_words([" XIN", " CHÀO"], [0.0, 0.3], 0.0, 1.0, N)
    assert [x.text for x in s] == ["xin", "chào"]


def test_bos_cikti():
    assert tokens_to_words([], [], 0.0, 1.0, N) == []


def test_normalizer_nfc_ve_kucuk_harf():
    ayrik = unicodedata.normalize("NFD", "BỆNH  VIỆN")
    assert N.normalize(ayrik) == unicodedata.normalize("NFC", "bệnh viện")
    assert ViNormalizer(strip_punct=True).normalize("Xin chào, bác sĩ!") == "xin chào bác sĩ"


# --- Türkçe / çok dilli ---
from viet_asr.infrastructure.normalizer import TrNormalizer, turkish_lower  # noqa: E402
from viet_asr.infrastructure.sherpa_asr import spread_words  # noqa: E402


def test_harf_tabanli_ctc_ciktisi_kelimeye_birlesir():
    # Omnilingual: her harf ayrı token, " " ayırıcı
    toks = list("iki gün")
    ts = [0.1 * i for i in range(len(toks))]
    w = tokens_to_words(toks, ts, 5.0, 6.0, TrNormalizer())
    assert [x.text for x in w] == ["iki", "gün"]
    assert w[0].start == 5.0 and round(w[1].start, 2) == 5.4


def test_zaman_bilgisi_yoksa_harf_sayisiyla_yayilir():
    w = spread_words("Merhaba doktor bey.", 2.0, 4.0, TrNormalizer())
    assert [x.text for x in w] == ["Merhaba", "doktor", "bey."]
    assert w[0].start == 2.0 and abs(w[-1].end - 4.0) < 1e-9
    assert w[0].end - w[0].start > w[2].end - w[2].start  # uzun kelime daha uzun sürer
    assert spread_words("   ", 0, 1, TrNormalizer()) == []


def test_turkce_kucuk_harf():
    assert turkish_lower("IŞIK İLAÇ") == "ışık ilaç"
    assert TrNormalizer().normalize("İzmir'de  doktor.") == "İzmir'de doktor."  # büyük harf korunur
    assert TrNormalizer(lowercase=True, strip_punct=True).normalize("İLAÇ, IŞIK!") == "ilaç ışık"

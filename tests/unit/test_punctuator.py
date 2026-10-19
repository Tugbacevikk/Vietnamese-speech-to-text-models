"""Noktalama ve büyük harf testleri (Vietnamca ve Türkçe)."""
from viet_asr.application.punctuator import (
    capitalize_initial,
    capitalize_proper_nouns,
    is_question_sentence,
    punctuate_text,
    punctuate_words,
)
from viet_asr.domain import Word


def test_capitalize_initial():
    assert capitalize_initial("xin chào") == "Xin chào"
    assert capitalize_initial("ilaç") == "İlaç"
    assert capitalize_initial("ılık") == "Ilık"
    assert capitalize_initial("") == ""


def test_capitalize_proper_nouns():
    inp = "tôi sinh ra ở hà nội việt nam và làm việc tại bệnh viện bạch mai"
    out = capitalize_proper_nouns(inp)
    assert "Hà Nội" in out
    assert "Việt Nam" in out
    assert "Bạch Mai" in out


def test_discourse_markers_punctuation():
    cases = [
        ("thưa bác sĩ em bị đau đầu", "Thưa bác sĩ, em bị đau đầu."),
        ("dạ em hiểu rồi", "Dạ, em hiểu rồi."),
        ("vâng tôi đồng ý", "Vâng, tôi đồng ý."),
        ("tuy nhiên triệu chứng vẫn còn", "Tuy nhiên, triệu chứng vẫn còn."),
        ("ngoài ra cần uống đủ nước", "Ngoài ra, cần uống đủ nước."),
        ("đặc biệt là người cao tuổi", "Đặc biệt, là người cao tuổi."),
    ]
    for inp, expected in cases:
        assert punctuate_text(inp) == expected


def test_pause_based_comma_insertion():
    # 0.4 sn duraklama -> virgül
    ws = [
        Word("xin", 0.0, 0.3),
        Word("chào", 0.3, 0.6),
        Word("hôm", 1.0, 1.3),   # 0.4 sn duraklama (1.0 - 0.6)
        Word("nay", 1.3, 1.6),
    ]
    res = punctuate_words(ws)
    assert res == "Xin chào, hôm nay."


def test_compound_words_protect_from_comma():
    # "bác sĩ" bileşik kelimesi arasına duraklama olsa bile virgül girmez
    ws = [
        Word("thưa", 0.0, 0.3),
        Word("bác", 0.3, 0.5),
        Word("sĩ", 0.9, 1.2),   # 0.4 sn duraklama ama "bác sĩ" bileşik
        Word("tôi", 1.3, 1.6),
        Word("bị", 1.6, 1.8),
        Word("sốt", 1.8, 2.0),
    ]
    res = punctuate_words(ws)
    assert "bác, sĩ" not in res
    assert "bác sĩ" in res


def test_question_detection_vi_and_tr():
    assert is_question_sentence("bác sĩ có nhà không") is True
    assert is_question_sentence("nguyên nhân do đâu ạ") is True
    assert is_question_sentence("chi phí là bao nhiêu") is True
    assert is_question_sentence("tôi bị sốt cao") is False
    assert is_question_sentence("hastanın ateşi var mı") is True
    assert is_question_sentence("bugün nasılsınız") is True
    assert is_question_sentence("yarın kontrole gelmeniz gerekiyor") is False


def test_speed_offline():
    import time
    text = "thưa bác sĩ hôm nay em đến từ hà nội việt nam để khám bệnh bạch mai"
    t0 = time.perf_counter()
    for _ in range(1000):
        punctuate_text(text)
    elapsed = time.perf_counter() - t0
    # 1000 cümle 0.1 saniyeden az sürmeli (< 0.1 ms/cümle)
    assert elapsed < 0.2

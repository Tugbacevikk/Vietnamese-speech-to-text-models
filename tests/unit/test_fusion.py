from viet_asr.application.config import FusionConfig
from viet_asr.application.fusion import FusionEngine
from viet_asr.domain import BELIRSIZ, SpeakerSegment as S, Word as Y

A, B = "speaker_00", "speaker_01"


def eng(**kw):
    return FusionEngine(FusionConfig(**kw))


def test_orta_nokta_tek_konusmaciya_duser():
    segs = [S(A, 0, 2), S(B, 2, 4)]
    syl = [Y("chào", 0.2, 0.5), Y("bác", 1.8, 2.4), Y("sĩ", 2.5, 2.8)]
    # "bác" orta noktası 2.1 -> B
    assert eng().assign(syl, segs) == [A, B, B]


def test_ust_uste_konusmada_baskin_konusmaci():
    segs = [S(A, 0, 3), S(B, 1.0, 1.25)]
    # hece 0.9-1.3, orta 1.1 iki aralıkta; A örtüşmesi 0.4, B 0.25 -> A payı 0.62
    assert eng(min_dominance=0.6).assign([Y("x", 0.9, 1.3)], segs) == [A]
    assert eng(min_dominance=0.7).assign([Y("x", 0.9, 1.3)], segs) == [BELIRSIZ]


def test_bosluga_dusen_hece_en_yakina_ya_da_belirsiz():
    segs = [S(A, 0, 1), S(B, 3, 4)]
    assert eng(max_gap=0.5).assign([Y("a", 1.1, 1.3)], segs) == [A]      # 0.2 uzak
    assert eng(max_gap=0.5).assign([Y("b", 2.6, 2.8)], segs) == [B]      # 0.3 uzak
    assert eng(max_gap=0.5).assign([Y("c", 1.9, 2.1)], segs) == [BELIRSIZ]



def test_uzun_aralik_icindeki_hece_bulunur():
    # A'nın uzun aralığı, B'nin kısa aralıklarından önce başlıyor: geriye bakma doğru çalışmalı
    segs = [S(A, 0, 10), S(B, 1, 1.5), S(B, 2, 2.5)]
    assert eng().assign([Y("x", 5, 5.2)], segs) == [A]


def test_satirlara_toplama():
    segs = [S(A, 0, 2), S(B, 2, 5)]
    syl = [Y("xin", 0.1, 0.3), Y("chào", 0.3, 0.6), Y("dạ", 2.1, 2.3), Y("vâng", 4.5, 4.8)]
    u = eng(new_utterance_gap=1.5).fuse(syl, segs)
    assert [(x.speaker, x.text) for x in u] == [(A, "xin chào"), (B, "dạ"), (B, "vâng")]
    assert u[0].start == 0.1 and u[0].end == 0.6


def test_sirasiz_heceler_siralanir():
    segs = [S(A, 0, 5)]
    u = eng().fuse([Y("b", 1, 1.2), Y("a", 0, 0.2)], segs)
    assert u[0].text == "a b"


def test_tek_konusmaci_varsa_tum_kelimeler_ona():
    # VAD'ın kaçırdığı (aralık dışı) kelimeler bile BELIRSIZ değil, tek konuşmacıya gider
    w = [Y("a", 0.1, 0.3), Y("b", 10.0, 10.2)]
    assert eng(max_gap=0.5).assign(w, [S(A, 0, 1)]) == [A, A]
    assert eng().assign(w, []) == ["speaker_00", "speaker_00"]


def test_uzun_satir_en_uzun_duraklamadan_bolunur():
    # ~60 sn, kelimeler 0,5 sn arayla; iki yerde 1 sn'lik duraklama
    ws, t = [], 0.0
    while t < 60:
        ws.append(Y("x", t, t + 0.4))
        t += 1.4 if len(ws) in (37, 83) else 0.5   # 18. ve 41,9. sn'den sonra duraklama
    utts = eng(max_utterance=25).fuse(ws, [S(A, 0, 61)])
    assert [round(u.start, 1) for u in utts] == [0, 19.4, 43.3]   # iki duraklamadan
    assert all(u.end - u.start <= 25 for u in utts) and {u.speaker for u in utts} == {A}
    assert sum(len(u.text.split()) for u in utts) == len(ws)       # kelime kaybı yok
    assert len(eng(max_utterance=0).fuse(ws, [S(A, 0, 61)])) == 1  # 0 = bölme yok


def test_duraklamasiz_uzun_satir_sinirdan_bolunur():
    ws = [Y("x", i * 0.5, i * 0.5 + 0.5) for i in range(120)]       # 60 sn, hiç boşluk yok
    utts = eng(max_utterance=25).fuse(ws, [S(A, 0, 61)])
    assert all(u.end - u.start <= 25 for u in utts) and len(utts) == 3


def test_araya_sikismis_kisa_konusmaci_duzeltmesi():
    ws = [Y("a", 0, .5), Y("b", .6, 1), Y("x", 1.1, 1.4), Y("c", 1.5, 2), Y("d", 2.1, 2.5),
          Y("u", 2.6, 4.5), Y("e", 4.6, 5)]
    segs = [S(A, 0, 1.05), S(B, 1.05, 1.45), S(A, 1.45, 2.55), S(B, 2.55, 4.55), S(A, 4.55, 6)]
    off = eng().fuse(ws, segs)
    assert [u.speaker for u in off] == [A, B, A, B, A]                       # kapalıyken olduğu gibi
    on = eng(sandwich_max=1.0).fuse(ws, segs)
    assert [(u.speaker, u.text) for u in on] == [(A, "a b x c d"), (B, "u"), (A, "e")]  # 1,9 sn'lik B kalır


def test_clean_words_stutters_and_repeats():
    cases = [
        ("thì thì cái mắc cúm a", "thì cái mắc cúm a"),
        ("với với cái triệu chứng", "với cái triệu chứng"),
        ("đặc đặc biệt", "đặc biệt"),
        ("người người có bệnh lý", "người có bệnh lý"),
        ("để phòng trừ các các các biến chứng", "để phòng trừ các biến chứng"),
        ("tức ngực khó thở ngực", "tức ngực khó thở"),
        ("những cái trẻ em", "những trẻ em"),
        ("các cái người có bệnh", "các người có bệnh"),
        ("bu sefer bu sefer bu sefer", "bu sefer"),
        ("evet evet evet", "evet"),
    ]
    for inp, expected in cases:
        words = [Y(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(inp.split())]
        res = eng(clean_repeats=True).fuse(words, [S(A, 0, len(words) * 0.5 + 1)])
        assert len(res) == 1
        assert res[0].text == expected


def test_punctuate_sentences():
    cases = [
        ("xin chào bác sĩ", "Xin chào bác sĩ."),
        ("bác sĩ có khỏe không", "Bác sĩ có khỏe không?"),
        ("hôm nay thế nào", "Hôm nay thế nào?"),
        ("doktor bey iyi misiniz", "Doktor bey iyi misiniz?"),
        ("yarın gelecek misin", "Yarın gelecek misin?"),
        ("ilaçları düzenli kullanın", "İlaçları düzenli kullanın."),
    ]
    for inp, expected in cases:
        words = [Y(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(inp.split())]
        res = eng(punctuate=True).fuse(words, [S(A, 0, len(words) * 0.5 + 1)])
        assert len(res) == 1
        assert res[0].text == expected


def test_clean_words_phonetic_fixes():
    cases = [
        ("thường thì bấm mất ba ngày", "thường thì tầm mất ba ngày"),
        ("ví dụ nhận những cái triệu chứng", "ví dụ như những triệu chứng"),
        ("những cái trường hợp nhiễm cúm", "những trường hợp nhiễm cúm"),
        ("và hoở thì rất có", "và hoặc thì rất có"),
    ]
    for inp, expected in cases:
        words = [Y(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(inp.split())]
        res = eng(clean_repeats=True).fuse(words, [S(A, 0, len(words) * 0.5 + 1)])
        assert len(res) == 1
        assert res[0].text == expected


def test_rich_punctuation_formatting():
    # cúm a -> cúm A, virgül yerleşimi
    inp = "nhiễm cúm a tuy nhiên chúng ta chẳng hạn"
    words = [Y(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(inp.split())]
    res = eng(punctuate=True).fuse(words, [S(A, 0, len(words) * 0.5 + 1)])
    assert res[0].text == "Nhiễm cúm A, tuy nhiên chúng ta, chẳng hạn."


def test_split_protects_compound_words():
    # "có thể" bileşik kelimesi tam kesim sınırında ise bölünmemeli
    words = [
        Y("chúng", 0.0, 5.0),
        Y("ta", 5.0, 10.0),
        Y("có", 10.0, 14.9),
        Y("thể", 15.0, 19.9),
        Y("làm", 20.0, 24.0),
        Y("được", 24.0, 27.0),
    ]
    res = eng(max_utterance=15.0).fuse(words, [S(A, 0, 30)])
    # "có thể" bölünmemeli, aynı parçada kalmalı
    co_the_parts = [u.text for u in res if "có thể" in u.text]
    assert len(co_the_parts) >= 1


def test_clean_words_oral_fillers_and_southern_dialect():
    cases = [
        ("tuy á là em đã tìm cách á trị rồi", "tuy là em đã tìm cách trị rồi"),
        ("như vậy á", "như vậy"),
        ("dầu gọi nào", "dầu gội nào"),
        ("lệ mái tóc", "lại mái tóc"),
        ("bị gãy gụng", "bị gãy rụng"),
        ("rụng rụng tóc", "rụng tóc"),
        ("cung cung cấp dinh dưỡng", "cung cấp dinh dưỡng"),
        ("làm đẹp làm đẹp tóc", "làm đẹp tóc"),
        ("để a phục hồi", "để phục hồi"),
        ("và à phục hồi", "và phục hồi"),
    ]
    for inp, expected in cases:
        words = [Y(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(inp.split())]
        res = eng(clean_repeats=True).fuse(words, [S(A, 0, len(words) * 0.5 + 1)])
        assert len(res) == 1
        assert res[0].text == expected


def test_dialogue_roles_identification():
    # Katılımcı soru sorar -> Sunucu cevap verir/yönlendirir
    p1 = "Hay rất là rụng tóc rất là nhiều thì em muốn hỏi bác sĩ là nguyên nhân do đâu?"
    host = "Cảm ơn câu hỏi của em chúng tôi xin tiếp tục mời quý khán giả tại phim trường nêu câu hỏi ạ."
    p2 = "Hiện nay thì chị em phụ nữ làm đẹp tóc nhiều nên bị hư tổn, bác sĩ cho biết cách phục hồi?"
    
    ws1 = [Y(w, i * 0.3, i * 0.3 + 0.25) for i, w in enumerate(p1.split())]
    t_h = ws1[-1].end + 0.5
    ws_h = [Y(w, t_h + i * 0.3, t_h + i * 0.3 + 0.25) for i, w in enumerate(host.split())]
    t_p2 = ws_h[-1].end + 0.5
    ws2 = [Y(w, t_p2 + i * 0.3, t_p2 + i * 0.3 + 0.25) for i, w in enumerate(p2.split())]
    
    all_words = ws1 + ws_h + ws2
    segs = [S(A, ws1[0].start, ws1[-1].end), S(B, ws_h[0].start, ws_h[-1].end), S(A, ws2[0].start, ws2[-1].end)]
    
    res = eng(detect_roles=True, punctuate=True, clean_repeats=True).fuse(all_words, segs)
    assert len(res) == 3
    assert res[0].speaker == "[Katılımcı 1 - Kadın]"
    assert res[1].speaker == "[Sunucu - Erkek]"
    assert res[2].speaker == "[Katılımcı 2 - Kadın]"


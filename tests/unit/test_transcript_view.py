"""Temiz transkript görünümü: bloklar, etiketler, temizlik, noktalama; ses perdesi."""
import numpy as np

from viet_asr.application.transcript_view import as_text, blocks, clean_tokens, speaker_labels
from viet_asr.domain import (
    LanguageDecision, LanguageSource, SpeakerInfo, SpeakerSegment, Transcript, Utterance,
)
from viet_asr.infrastructure.voice import PitchProfiler, frame_f0, gender_of

A, B, C = "speaker_00", "speaker_01", "speaker_02"


def tr(utts, lang="vi", infos=()):
    return Transcript("j", 60.0, utts, "m", LanguageDecision(lang, 1.0, LanguageSource.REQUEST),
                      speakers=tuple(infos))


def test_temizlik_dolgu_ve_kekeleme_ikileme_korunur():
    toks = "thì thì á em muốn hỏi từ từ ờ các các các bước".split()
    assert [toks[i] for i in clean_tokens(toks)] == "thì em muốn hỏi từ từ các bước".split()
    v = "thay đổi à thời tiết vậy à".split()
    assert [v[i] for i in clean_tokens(v)] == "thay đổi thời tiết vậy à".split()
    t = "yavaş yavaş ıı bu bu ilacı alın".split()
    assert [t[i] for i in clean_tokens(t)] == "yavaş yavaş bu ilacı alın".split()


def test_bloklar_konusmaci_degisiminde_yeni_satir_ve_zaman():
    utts = [Utterance(A, 0.0, 6.0, "xin chào quý vị hôm nay chúng ta nói về rụng tóc", breaks=(4,)),
            Utterance(A, 6.2, 9.0, "mời chị đặt câu hỏi"),
            Utterance(B, 9.5, 14.0, "thưa bác sĩ em bị rụng tóc thì nguyên nhân do đâu ạ"),
            Utterance(A, 75.0, 80.0, "cảm ơn câu hỏi của chị")]
    infos = [SpeakerInfo(A, 118.0, "erkek", 20), SpeakerInfo(B, 215.0, "kadın", 8)]
    bs = blocks(tr(utts, infos=infos), "studyo")
    assert [(b.label, round(b.start), round(b.end)) for b in bs] == [
        ("Sunucu - Erkek", 0, 9), ("Katılımcı 1 - Kadın", 10, 14), ("Sunucu - Erkek", 75, 80)]
    assert bs[0].text == "Xin chào quý vị. Hôm nay chúng ta nói về rụng tóc mời chị đặt câu hỏi."
    assert bs[1].text.endswith("do đâu ạ?")
    out = as_text(bs)
    assert out.splitlines()[0].startswith("[00:00 - 00:09] [Sunucu - Erkek]: Xin chào")
    assert out.splitlines()[2].startswith("[01:15 - 01:20]")


def test_etiketler_cinsiyet_yalnizca_olcumden():
    utts = [Utterance(A, 0, 1, "a"), Utterance(B, 1, 2, "b"), Utterance(C, 2, 3, "c"), Utterance(A, 3, 4, "d")]
    lab = speaker_labels(utts, [SpeakerInfo(B, 170.0, None, 9)], "genel")
    assert (lab[A], lab[B], lab[C]) == ("Konuşmacı 1", "Konuşmacı 2", "Konuşmacı 3")  # belirsiz: yazılmaz
    st = speaker_labels(utts, [], "studyo")
    assert (st[A], st[B], st[C]) == ("Sunucu", "Katılımcı 1", "Katılımcı 2")
    assert speaker_labels(utts, [], "yok")[B] == B
    assert speaker_labels([Utterance(A, 0, 1, "a")], [], "studyo")[A] == "Konuşmacı 1"  # tek kişi


def test_turkce_soru_ve_buyuk_harf():
    utts = [Utterance(A, 0, 3, "ilacı her gün almam gerekiyor mu", breaks=()),
            Utterance(A, 4, 6, "iki hafta kullanın")]
    bs = blocks(tr(utts, "tr"), "genel")
    assert bs[0].text == "İlacı her gün almam gerekiyor mu? İki hafta kullanın."


def test_duzeltilmis_noktalanmis_metne_dokunulmaz_ve_kapatilabilir():
    utts = [Utterance(A, 0, 3, "Merhaba, nasılsınız?", edited=True)]
    assert blocks(tr(utts, "tr"))[0].text == "Merhaba, nasılsınız?"
    raw = blocks(tr([Utterance(A, 0, 3, "thì thì xin chào")]), clean=False, punctuate=False)
    assert raw[0].text == "thì thì xin chào"


def voice(f0, sec=3.0, sr=16000):
    t = np.arange(int(sec * sr)) / sr
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))  # harmonikli ses
    return (0.1 * sig * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_ses_perdesi_ve_cinsiyet():
    assert abs(np.median(frame_f0(voice(110))) - 110) < 3
    assert abs(np.median(frame_f0(voice(220))) - 220) < 5
    assert (gender_of(120), gender_of(210), gender_of(170), gender_of(None)) == ("erkek", "kadın", None, None)
    audio = np.concatenate([voice(115), voice(225), np.zeros(8000, np.float32)])
    segs = [SpeakerSegment(A, 0, 3), SpeakerSegment(B, 3, 6), SpeakerSegment("BELIRSIZ", 6, 6.5)]
    got = {i.speaker: (i.gender, round(i.f0 or 0)) for i in PitchProfiler().profile(audio, segs)}
    assert got[A][0] == "erkek" and got[B][0] == "kadın" and "BELIRSIZ" not in got
    short = PitchProfiler().profile(voice(115, 1.0), [SpeakerSegment(A, 0, 1)])
    assert short[0].f0 is None and short[0].gender is None              # 2 sn'den az ötümlü ses


def test_turkce_soru_eki_ayri_sozcuk_olmali():
    from viet_asr.application.transcript_view import _is_question

    for soru in ("hasta ateşi var mı", "gelecek misin", "bunu yapabilir miyiz", "evde miydi", "geldi mi"):
        assert _is_question(soru.split(), "tr") is True, soru
    # "mi/mu/mü" ile BAŞLAYAN sıradan sözcükler soru değildir (eskiden hepsi "?" alıyordu)
    for duz in ("ben çok mutlu", "müzik dinliyorum", "bu mimar iyi", "mümkün değil", "misafir geldi",
                "mutfakta çalışıyor"):
        assert _is_question(duz.split(), "tr") is False, duz


def test_vietnamca_khong_sao_soru_degil():
    from viet_asr.application.transcript_view import _is_question

    assert _is_question("dạ không sao".split(), "vi") is False
    assert _is_question("bác sĩ có nhà không".split(), "vi") is True

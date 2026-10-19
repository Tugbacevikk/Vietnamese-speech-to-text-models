"""Temiz transkript: okunacak/paylaşılacak metin görünümü.

Kayıttaki satırlar (ölçüm, düzeltme, RTTM, entegrasyon için) DEĞİŞMEZ; bu modül onlardan bir
görünüm üretir:

    [00:00 - 00:14] [Sunucu - Erkek]: Xin chào quý vị. Hôm nay chúng ta nói về rụng tóc.
    [00:14 - 00:31] [Katılımcı 1 - Kadın]: Thưa bác sĩ, em bị rụng tóc nhiều thì nguyên nhân do đâu?

- Blok: aynı konuşmacının art arda gelen satırları tek blokta; konuşmacı değişince yeni satır.
- Etiket: "genel" -> Konuşmacı 1, 2...; "studyo" -> en çok söz alan Sunucu, diğerleri
  ilk konuştukları sırayla Katılımcı 1, 2...; "yok" -> speaker_00. Cinsiyet METİNDEN ya da
  rolden tahmin edilmez; yalnızca ses perdesi ölçümü kesinse eklenir (infrastructure/voice.py).
- Temizlik: dolgu sesleri (á, ờ, ừm; ıı, ee...) ve kekeleme tekrarları ("thì thì") atılır.
  Anlamlı ikilemeler ("từ từ", "yavaş yavaş") korunur: yalnızca bilinen bağlaç/ek tekrarı tek
  kelimeye iner; üç ve fazlası her kelimede teke iner.
- Noktalama: cümle sınırı = konuşmadaki duraklama (>= 0,6 sn, birleştirme sırasında ölçülür);
  cümle büyük harfle başlar, soru sözcüğüyle biterse "?", değilse ".".

Bilinçli olarak YAPILMAYAN: belli bir kayda göre kelime değiştirme ("con" -> "câu" gibi).
Bağlama göre düzeltme dil modeli ister; kurala dökülünce başka kayıtlarda doğru metni bozuyor
(15 Ekim ölçümü: "cách thủ công" -> "cách khắc phục công").
"""
from __future__ import annotations

import re

from dataclasses import dataclass

from viet_asr.domain import BELIRSIZ, SpeakerInfo, Transcript, Utterance

LABEL_MODES = ("genel", "studyo", "yok")

FILLERS = {"á", "ơ", "ờ", "ờm", "ừm", "ưm", "ớ", "ıı", "ııı", "ı", "ee", "eee", "eh", "hmm", "hım", "ım", "mm"}
# ikisi art arda gelince kekeleme sayılanlar (anlamlı ikileme olmayan bağlaç, edat, sık sözcük)
STUTTERS = {
    "thì", "với", "các", "cái", "là", "và", "của", "được", "những", "cho", "trong", "để", "có",
    "bị", "rồi", "còn", "mà", "này", "đó", "em", "anh", "chị", "tôi", "đặc", "người", "rụng",
    "cung", "cấp", "làm", "đẹp", "nhuộm", "phục", "hồi",
    "ve", "de", "da", "ile", "için", "ama", "fakat", "ki", "bu", "bir", "şu", "o", "ise", "ben",
    "sen", "yani",
}
VI_QUESTION_END = {"không", "chưa", "hả", "sao", "nhỉ", "gì", "đâu", "nào", "ai", "chăng", "chứ", "hông"}
VI_QUESTION_PHRASES = ("thế nào", "bao nhiêu", "tại sao", "vì sao", "làm sao", "phải không", "do đâu",
                       "được không", "có được", "là gì")
TR_QUESTION_WORDS = {"nasıl", "neden", "niçin", "niye", "kim", "nerede", "nereden", "nereye", "kaç",
                     "hangi", "hangisi"}
# Türkçe soru eki AYRI bir sözcüktür: mı/mi/mu/mü ve kişi ekli biçimleri (misin, mıyız, muydu...).
# Başı "mi/mu/mü" ile başlayan her sözcük değil ("mutlu", "müzik", "mimar", "mümkün", "misafir" soru değil).
_TR_Q_RE = re.compile(r"^m[ıiuü](?:y[ıiuü]m|s[ıiuü]n(?:[ıiuü]z)?|y[ıiuü]z|d[ıiuü]r|yd[ıiuü])?$")


@dataclass(frozen=True)
class Block:
    speaker: str
    label: str
    start: float
    end: float
    text: str


def clean_tokens(tokens: list[str]) -> list[int]:
    """Korunacak kelimelerin sırası: dolgu sesleri atılır, kekeleme tekrarı teke iner."""
    keep = [i for i, t in enumerate(tokens) if t.lower().strip(".,?!") not in FILLERS
            and not (t.lower() == "à" and i < len(tokens) - 1)]  # sondaki "à" soru eki, aradaki dolgu
    out: list[int] = []
    i = 0
    while i < len(keep):
        j = i
        key = tokens[keep[i]].lower().strip(".,?!")
        while j < len(keep) and tokens[keep[j]].lower().strip(".,?!") == key:
            j += 1
        n = j - i
        out.extend(keep[i:j] if n == 1 or (n == 2 and key not in STUTTERS) else keep[i:i + 1])
        i = j
    return out


def _cap(s: str) -> str:
    if not s:
        return s
    first = {"i": "İ", "ı": "I"}.get(s[0], s[0].upper())
    return first + s[1:]


def _is_question(words: list[str], lang: str | None) -> bool:
    low = [w.lower().strip(".,?!") for w in words]
    if not low:
        return False
    text, last = " ".join(low[-6:]), low[-1]   # soru kalıbı cümlenin SONUNDA aranır
    if lang in (None, "vi"):
        if low[-2:] == ["không", "sao"]:   # "không sao" = sorun değil (bildirim)
            return False
        if last in VI_QUESTION_END or any(p in text for p in VI_QUESTION_PHRASES):
            return True
        if last == "ạ" and len(low) > 1 and (low[-2] in VI_QUESTION_END or "do đâu" in text):
            return True
    if lang in (None, "tr"):
        tail = low[-2:] if len(low) > 1 else low
        if any(_TR_Q_RE.match(w) for w in tail):
            return True
        if any(w in TR_QUESTION_WORDS for w in low[:3]) and not low[-1].endswith(("dır", "dir", "dur", "dür")):
            return True
    return False


def _sentences(words: list[str], lang: str | None) -> str:
    if not words:
        return ""
    if words[-1][-1:] in ".?!":  # elle düzeltilmiş ve noktalanmış metin: dokunma
        return _cap(" ".join(words))
    from viet_asr.application.punctuator import capitalize_proper_nouns
    text = capitalize_proper_nouns(_cap(" ".join(words)), lang=lang or "vi")
    return text + ("?" if _is_question(words, lang) else ".")


def speaker_labels(utterances: list[Utterance], infos: tuple[SpeakerInfo, ...] | list[SpeakerInfo],
                   mode: str = "genel") -> dict[str, str]:
    order: list[str] = []
    turns: dict[str, int] = {}
    prev = None
    for u in utterances:
        if u.speaker == BELIRSIZ:
            continue
        if u.speaker not in order:
            order.append(u.speaker)
        if u.speaker != prev:
            turns[u.speaker] = turns.get(u.speaker, 0) + 1
        prev = u.speaker
    labels = {BELIRSIZ: "Belirsiz"}
    if mode == "yok":
        return {s: s for s in order} | labels
    gender = {i.speaker: {"erkek": "Erkek", "kadın": "Kadın"}.get(i.gender or "") for i in infos}
    if mode == "studyo" and len(order) >= 2:
        host = max(order, key=lambda s: (turns[s], -order.index(s)))  # eşitlikte ilk konuşan
        n = 0
        for s in order:
            if s == host:
                labels[s] = "Sunucu"
            else:
                n += 1
                labels[s] = f"Katılımcı {n}"
    else:
        for n, s in enumerate(order, 1):
            labels[s] = f"Konuşmacı {n}"
    for s in order:
        if gender.get(s):
            labels[s] += f" - {gender[s]}"
    return labels


def blocks(t: Transcript, mode: str = "genel", clean: bool = True, punctuate: bool = True,
           sentence_gap: float = 0.6) -> list[Block]:
    labels = speaker_labels(t.utterances, t.speakers, mode)
    lang = t.language.code if t.language else None
    groups: list[list[Utterance]] = []
    for u in t.utterances:
        if groups and groups[-1][-1].speaker == u.speaker:
            groups[-1].append(u)
        else:
            groups.append([u])
    out = []
    for g in groups:
        tokens: list[str] = []
        cuts: set[int] = set()
        prev_end = None
        for u in g:
            ws = u.text.split()
            if prev_end is not None and u.start - prev_end >= sentence_gap:
                cuts.add(len(tokens))
            cuts.update(len(tokens) + b for b in u.breaks)
            tokens += ws
            prev_end = u.end
        keep = clean_tokens(tokens) if clean else list(range(len(tokens)))
        if punctuate:
            sents, cur = [], []
            for k in keep:
                if cur and k in cuts:
                    sents.append(cur)
                    cur = []
                cur.append(tokens[k])
            if cur:
                sents.append(cur)
            text = " ".join(_sentences(s, lang) for s in sents)
        else:
            text = " ".join(tokens[k] for k in keep)
        if text:
            out.append(Block(g[0].speaker, labels.get(g[0].speaker, g[0].speaker),
                             g[0].start, g[-1].end, text))
    return out


def clock(sec: float) -> str:
    s = int(round(sec))
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def as_text(bs: list[Block]) -> str:
    return "\n".join(f"[{clock(b.start)} - {clock(b.end)}] [{b.label}]: {b.text}" for b in bs) + "\n"

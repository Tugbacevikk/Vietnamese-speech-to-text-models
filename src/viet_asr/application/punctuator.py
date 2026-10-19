"""Vietnamca ve Türkçe noktalama ve büyük harf iyileştirmesi (Punctuation & Capitalization).

İnternetsiz (çevrimdışı) çalışır; akustik duraklamaları (Word zamanlaması),
söylem bağlaçlarını (discourse markers), soru kalıplarını ve özel isim / unvan
büyük harflerini birleştirerek ASR çıktısını yüksek okunabilirlikte biçimlendirir.
"""
from __future__ import annotations

import re
from typing import Sequence

from viet_asr.domain import Word

# Soru ekleri ve sözcükleri (Vietnamca)
VI_QUESTION_END = {
    "không", "hả", "sao", "chưa", "nhỉ", "thế", "gì", "đâu", "nào", "ai",
    "chăng", "chứ", "hông"
}
VI_QUESTION_PHRASES = (
    "thế nào", "bao nhiêu", "tại sao", "vì sao", "như thế nào", "làm sao",
    "phải không", "do đâu", "được không", "có được", "là gì"
)

# Soru ekleri ve sözcükleri (Türkçe)
TR_QUESTIONS = {
    "mi", "mı", "mu", "mü", "misin", "mısın", "musun", "müsün",
    "misiniz", "mısınız", "musunuz", "müsünüz",
    "miyim", "mıyım", "muyum", "müyüm",
    "midir", "mıdır", "mudur", "müdür",
    "mıyız", "miyiz", "muyuz", "müyüz",
    "miydi", "mıydı", "muydu", "müydü",
    "nasıl", "nasılsın", "nasılsınız", "kim", "nerede", "nereden", "nereye", "kaç", "hangi", "hangisi"
}
TR_QUESTION_PHRASES = (
    "ne düşünüyorsun", "ne dersin", "ne yapıyorsun", "ne demek", "neden", "niçin", "niye",
    "nasıl yani", "sen nasılsın", "siz nasılsınız", "ne olarak"
)

# Cümle başı veya ara sözcük olarak virgül alan Vietnamca söylem kalıpları
VI_DISCOURSE_MARKERS = {
    "thưa bác sĩ": "Thưa bác sĩ,",
    "thưa bác si": "Thưa bác sĩ,",
    "thưa cô": "Thưa cô,",
    "thưa thầy": "Thưa thầy,",
    "dạ vâng": "Dạ vâng,",
    "dạ": "Dạ,",
    "vâng": "Vâng,",
    "tuy nhiên": "Tuy nhiên,",
    "ngoài ra": "Ngoài ra,",
    "hơn nữa": "Hơn nữa,",
    "do đó": "Do đó,",
    "vì vậy": "Vì vậy,",
    "đặc biệt": "Đặc biệt,",
    "ví dụ": "Ví dụ,",
    "thực ra": "Thực ra,",
    "thật ra": "Thật ra,",
    "nói chung": "Nói chung,",
    "tóm lại": "Tóm lại,",
    "trước hết": "Trước hết,",
    "sau đó": "Sau đó,",
}

# Vietnamca özel isimler, yer adları ve kurumlar
VI_PROPER_NOUNS = {
    "việt nam": "Việt Nam",
    "hà nội": "Hà Nội",
    "sài gòn": "Sài Gòn",
    "đà nẵng": "Đà Nẵng",
    "hải phòng": "Hải Phòng",
    "cần thơ": "Cần Thơ",
    "huế": "Huế",
    "tp hcm": "TP.HCM",
    "tp.hcm": "TP.HCM",
    "tphcm": "TP.HCM",
    "bạch mai": "Bạch Mai",
    "chợ rẫy": "Chợ Rẫy",
    "việt đức": "Việt Đức",
    "bộ y tế": "Bộ Y tế",
    "trung quốc": "Trung Quốc",
    "hàn quốc": "Hàn Quốc",
    "nhật bản": "Nhật Bản",
    "thái lan": "Thái Lan",
    "singapore": "Singapore",
    "hoa kỳ": "Hoa Kỳ",
    "mỹ": "Mỹ",
    "pháp": "Pháp",
    "đức": "Đức",
    "anh": "Anh",
    "nga": "Nga",
}

# Cümle içi bölünmemesi ve arasına virgül girmemesi gereken bileşik kelimeler
VI_COMPOUNDS = {
    ("bác", "sĩ"), ("có", "thể"), ("chúng", "tôi"), ("chúng", "ta"),
    ("không", "sao"), ("y", "tế"), ("bệnh", "viện"), ("điều", "trị"),
    ("sức", "khỏe"), ("phát", "triển"), ("thời", "gian"), ("nguyên", "nhân"),
    ("rụng", "tóc"), ("phục", "hồi"), ("chăm", "sóc"), ("sử", "dụng"),
}


def capitalize_initial(s: str) -> str:
    """İlk harfi büyük harf yapar (Türkçe İ/I ve Unicode NFC uyumlu)."""
    if not s:
        return s
    first = s[0]
    if first == "i":
        return "İ" + s[1:]
    if first == "ı":
        return "I" + s[1:]
    return first.upper() + s[1:]


def capitalize_proper_nouns(text: str, lang: str = "vi") -> str:
    """Vietnamca veya Türkçe bilinen özel isimleri ve yer adlarını büyük harfe çevirir."""
    if not text or lang not in ("vi", None):
        return text

    res = text
    for lower_name, proper_name in VI_PROPER_NOUNS.items():
        # Kelime sınırlarına duyarlı değiştirme
        pattern = r"(?<!\w)" + re.escape(lower_name) + r"(?!\w)"
        res = re.sub(pattern, proper_name, res, flags=re.IGNORECASE)
    return res


def is_question_sentence(raw_text: str, last_word: str = "", lang: str | None = None) -> bool:
    """Metnin soru cümlesi olup olmadığını tespit eder."""
    lower_raw = raw_text.lower().strip()
    clean_last = last_word.lower().strip(".,?!;:") if last_word else (lower_raw.split()[-1] if lower_raw else "")

    if lang in ("vi", None):
        if any(p in lower_raw for p in VI_QUESTION_PHRASES):
            return True
        if clean_last in VI_QUESTION_END:
            return True
        if clean_last == "ạ" and any(qw in lower_raw for qw in (
            "nguyên nhân do đâu", "như thế nào", "thực phẩm nào", "phục hồi tóc",
            "thưa bác sĩ", "gì", "nào", "đâu", "sao", "có thể", "làm sao", "do đâu"
        )):
            return True

    if lang in ("tr", None):
        if any(p in lower_raw for p in TR_QUESTION_PHRASES):
            return True
        if clean_last in TR_QUESTIONS:
            return True
        if (clean_last.startswith(("mı", "mi", "mu", "mü")) and len(clean_last) <= 8 and
                clean_last not in ("millet", "minik", "minder", "miktar", "milli")):
            return True

    return False


def punctuate_words(ws: Sequence[Word], lang: str | None = None,
                    min_pause_comma: float = 0.30,
                    max_pause_comma: float = 0.60) -> str:
    """Word nesnelerindeki ses duraklama süreleri ve dil kurallarıyla noktalanmış metin üretir.

    - 0.30s - 0.60s arası duraklamalar doğal nefes / cümle içi parça arası virgül (,) üretir.
    - Söylem bağlaçları ("Thưa bác sĩ,", "Dạ,", "Vâng,") otomatik virgüllenir.
    - Özel isimler ve yer adları ("Việt Nam", "Hà Nội") büyük harfe çevrilir.
    - Soru cümleleri '?', bildirim cümleleri '.' ile biter.
    """
    if not ws:
        return ""

    tokens: list[str] = [w.text for w in ws]
    n = len(ws)

    # 1. Akustik duraklamalara göre virgül ekleme
    for i in range(1, n):
        pause = ws[i].start - ws[i - 1].end
        if min_pause_comma <= pause < max_pause_comma:
            w_prev = ws[i - 1].text.lower().strip(".,?!;:")
            w_curr = ws[i].text.lower().strip(".,?!;:")
            if (w_prev, w_curr) not in VI_COMPOUNDS and not tokens[i - 1].endswith((",", ".", "?", "!")):
                tokens[i - 1] += ","

    raw = " ".join(tokens).strip()
    if not raw:
        return ""

    # 2. Cümle başı söylem bağlaçlarına virgül ekleme (henüz virgülü yoksa)
    lower_raw = raw.lower()
    for marker, formatted in VI_DISCOURSE_MARKERS.items():
        if lower_raw.startswith(marker):
            marker_len = len(marker)
            # Kelime sınırını kontrol et
            if len(raw) == marker_len or raw[marker_len] in (" ", ","):
                rem = raw[marker_len:].lstrip(" ,")
                raw = formatted + (" " + rem if rem else "")
                break

    # 3. Cümle başı büyük harf
    res = capitalize_initial(raw)

    # 4. Özel isimler ve yer adları
    res = capitalize_proper_nouns(res, lang=lang)

    # 5. Cümle sonu noktalama
    last_w = ws[-1].text.lower().strip(".,?!;:")
    if res[-1] not in ".?!,":
        res += "?" if is_question_sentence(res, last_w, lang=lang) else "."
    elif res[-1] == ",":
        res = res[:-1] + ("?" if is_question_sentence(res, last_w, lang=lang) else ".")

    return res


def punctuate_text(text: str, lang: str | None = None) -> str:
    """Yalnızca metin verildiğinde (zaman damgası olmadan) çevrimdışı noktalama ve büyük harf uygular."""
    raw = text.strip()
    if not raw:
        return ""

    # Varsa sonundaki fazla boşluk ve noktalama işaretlerini ayıkla
    lower_raw = raw.lower()

    # 1. Söylem bağlaçları
    for marker, formatted in VI_DISCOURSE_MARKERS.items():
        if lower_raw.startswith(marker):
            marker_len = len(marker)
            if len(raw) == marker_len or raw[marker_len] in (" ", ","):
                rem = raw[marker_len:].lstrip(" ,")
                raw = formatted + (" " + rem if rem else "")
                break

    # 2. Cümle başı büyük harf
    res = capitalize_initial(raw)

    # 3. Özel isimler
    res = capitalize_proper_nouns(res, lang=lang)

    # 4. Cümle sonu noktalama
    words = res.split()
    last_w = words[-1].lower().strip(".,?!;:") if words else ""
    if res[-1] not in ".?!,":
        res += "?" if is_question_sentence(res, last_w, lang=lang) else "."
    elif res[-1] == ",":
        res = res[:-1] + ("?" if is_question_sentence(res, last_w, lang=lang) else ".")

    return res

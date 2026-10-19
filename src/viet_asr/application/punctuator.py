"""Vietnamca özel isim büyük harfi (yalnızca görüntü katmanı: transcript_view; veritabanına yazılmaz).

Vietnamca heceler ayrı yazıldığı için tek heceli ülke adları günlük kelimelerle aynıdır:
"phương pháp", "liệu pháp" (tedavi), "thẩm mỹ" (estetik), "anh" (abi / sen). 626 VietMed
referansında (211 bin kelime) tek heceli adlar koşulsuz büyütülünce 585 dönüşümün yaklaşık 470'i
yanlıştı ("pháp" 340, "anh" 92, "mỹ" çoğu "thẩm mỹ"). Bu yüzden:

  - çok heceli adlar ("việt nam", "bộ y tế", "bạch mai"...) her yerde büyütülür (doğruluk %100);
  - tek heceli ülke adları ("mỹ", "pháp", "đức", "nga", "anh") yalnızca öncesinde yer / dil /
    uyruk bildiren bir sözcük varsa ve sonrasında bileşik oluşturan bir sözcük yoksa büyütülür:
    "ở mỹ", "người đức", "tiếng anh" -> evet; "thẩm mỹ", "phương pháp", "bên anh" -> hayır.

Bilerek tutucu: büyütülmeyen bir özel isim okunurluğu azaltır, yanlış büyütülen bir sıradan kelime
metni bozar. Kapsam dışı kalan örnekler eklenerek genişletilebilir (tests/unit/test_punctuator.py).
"""
from __future__ import annotations

import re

# Çok heceli: bağlama bakmadan büyütülür
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
    "đức phật": "Đức Phật",
}

# Tek heceli ülke adları: yalnızca öncesinde bunlardan biri varsa
VI_COUNTRIES = {"mỹ": "Mỹ", "pháp": "Pháp", "đức": "Đức", "nga": "Nga", "anh": "Anh"}
_PLACE_BEFORE = ("ở", "tại", "sang", "qua", "từ", "nước", "người", "tiếng", "đến", "tới")
# "anh" zamir olarak çok yaygın: yalnızca ülke / dil bildiren sözcüklerden sonra
_ANH_BEFORE = ("nước", "tiếng")
# Bunlardan biri hemen ardından geliyorsa ülke adı değil, bileşik sözcüktür ("mỹ phẩm", "pháp luật")
_COMPOUND_AFTER = ("phẩm", "thuật", "nhân", "học", "viện", "luật", "lý", "y", "sư", "đình", "tính", "tin",
                   "hóa", "quan", "cảm", "ngữ", "danh", "nghệ", "miều", "lệnh", "trị", "định")

_MULTI = [(re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)", re.IGNORECASE), v)
          for k, v in sorted(VI_PROPER_NOUNS.items(), key=lambda kv: -len(kv[0]))]
_AFTER = r"(?!\s+(?:" + "|".join(_COMPOUND_AFTER) + r")(?!\w))(?!\w)"
_COUNTRY = re.compile(r"(?<!\w)(" + "|".join(_PLACE_BEFORE) + r")(\s+)(mỹ|pháp|đức|nga)" + _AFTER, re.IGNORECASE)
_ANH = re.compile(r"(?<!\w)(" + "|".join(_ANH_BEFORE) + r")(\s+)(anh)" + _AFTER, re.IGNORECASE)


def capitalize_proper_nouns(text: str, lang: str | None = "vi") -> str:
    """Vietnamca metinde bilinen özel isimleri büyütür; başka dillerde metne dokunmaz."""
    if not text or lang not in ("vi", None):
        return text
    res = text
    for pat, proper in _MULTI:
        res = pat.sub(proper, res)
    res = _COUNTRY.sub(lambda m: m.group(1) + m.group(2) + VI_COUNTRIES[m.group(3).lower()], res)
    return _ANH.sub(lambda m: m.group(1) + m.group(2) + VI_COUNTRIES[m.group(3).lower()], res)

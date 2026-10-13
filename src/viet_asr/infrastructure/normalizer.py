"""Dile özel metin düzeltme.

Vietnamca (Zipformer): çıktı BÜYÜK HARF ve noktalamasız gelir -> NFC + küçük harf.
  NFC önemli: aynı aksanlı harfin iki farklı kodlanışı tek biçime iner, yoksa
  WER yapay olarak artar.
Türkçe (Whisper): çıktı büyük/küçük harf ve noktalama içerir. Sistem çıktısında
  (normalizer_for) iki dil aynı biçimde olsun diye küçük harfe çevrilir ve noktalama
  silinir (13 Ekim kararı: noktalama gerekmiyor, entegre sistem tek biçim alır).
  Küçük harf Türkçe kuralıyla (I -> ı, İ -> i); Python'un lower() bunu yanlış yapar.
  Kesme işareti kelime içindeyse kalır ("istanbul'da"), ondalık ayırıcı da ("3,5").
"""
from __future__ import annotations

import re
import unicodedata

# kelime içi kesme ('da) ve rakam arası nokta/virgül (3,5) korunur, diğer işaretler silinir
_PUNCT = re.compile(r"(?<=\w)['’](?=\w)|(?<=\d)[.,](?=\d)|([^\w\s])")


def _strip(text: str) -> str:
    return _PUNCT.sub(lambda m: " " if m.group(1) else m.group(0).replace("’", "'"), text)


def _clean(text: str, strip_punct: bool) -> str:
    if strip_punct:
        text = _strip(text)
    return " ".join(text.split())


class ViNormalizer:
    def __init__(self, strip_punct: bool = False):
        self.strip_punct = strip_punct

    def normalize(self, text: str) -> str:
        return _clean(unicodedata.normalize("NFC", text).lower(), self.strip_punct)


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


class TrNormalizer:
    def __init__(self, lowercase: bool = False, strip_punct: bool = False):
        self.lowercase, self.strip_punct = lowercase, strip_punct

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = turkish_lower(text)
        return _clean(text, self.strip_punct)


def normalizer_for(lang: str):
    """Sistem çıktısının biçimi: iki dilde de küçük harf, noktalamasız."""
    if lang == "vi":
        return ViNormalizer(strip_punct=True)
    return TrNormalizer(lowercase=True, strip_punct=True)

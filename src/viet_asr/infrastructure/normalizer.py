"""Dile özel metin düzeltme.

Vietnamca (Zipformer): çıktı BÜYÜK HARF ve noktalamasız gelir -> NFC + küçük harf.
  NFC önemli: aynı aksanlı harfin iki farklı kodlanışı tek biçime iner, yoksa
  WER yapay olarak artar.
Türkçe (Whisper): çıktı zaten büyük/küçük harf ve noktalama içerir -> korunur,
  yalnızca NFC ve boşluk temizliği. Küçük harfe çevirmek gerekirse Türkçe
  kuralı kullanılır (I -> ı, İ -> i); Python'un lower() bunu yanlış yapar.
"""
from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]")


def _clean(text: str, strip_punct: bool) -> str:
    if strip_punct:
        text = _PUNCT.sub(" ", text)
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


NORMALIZERS = {"vi": ViNormalizer, "tr": TrNormalizer}


def normalizer_for(lang: str):
    return NORMALIZERS.get(lang, TrNormalizer)()

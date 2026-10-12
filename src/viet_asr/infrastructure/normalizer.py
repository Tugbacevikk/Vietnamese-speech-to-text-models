"""Vietnamca metin düzeltme.

Zipformer çıktısı BÜYÜK HARF ve noktalamasızdır. Burada:
  - Unicode NFC (aynı harfin iki farklı kodlanışı tek biçime iner;
    Vietnamca aksanlarında bu çok önemli, yoksa WER yapay olarak artar)
  - küçük harf
  - fazla boşluk temizliği
Noktalama ve büyük harf geri getirme ayrı bir adım olarak sonra eklenecek.
"""
from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]")


class ViNormalizer:
    def __init__(self, strip_punct: bool = False):
        self.strip_punct = strip_punct

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text).lower()
        if self.strip_punct:
            text = _PUNCT.sub(" ", text)
        return " ".join(text.split())

"""Kalite takibi: kişilerin arayüzde yaptığı düzeltmelerden yaklaşık hata oranı.

Bir iş, en az bir satırı düzeltilmiş ya da silinmişse "elden geçirilmiş" sayılır; o işin bütün
satırları (dokunulmayanlar doğru kabul edilir) hesaba girer. Her satır için modelin yazdığı ile
son hali kelime kelime karşılaştırılır (düzenleme mesafesi: değişen + eklenen + silinen kelime).
Silinen satırın bütün kelimeleri hatadır (model uydurmuş).

    düzeltme oranı = düzeltilen kelime / modelin yazdığı kelime

Bu, kelime hata oranının (WER) yaklaşığıdır: kişinin gözünden kaçan hata sayılmaz (gerçek oran
biraz daha yüksektir), büyük/küçük harf ve noktalama sayılmaz. Hafta, dil ve model bazında
izlenir: oran yükselirse ses kalitesi ya da model sorunu, ince ayardan sonra düşmesi beklenir.
"""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from viet_asr.domain import ReviewLine

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def words(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.casefold()).split()


def edit_distance(a: list[str], b: list[str]) -> int:
    """Kelime düzeyinde Levenshtein (değiştirme, ekleme, silme birer hata)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def week_of(dt) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-H{w:02d}"


def quality_report(lines: Iterable[ReviewLine]) -> list[dict]:
    """(hafta, dil, model) başına satır; en yeni hafta önce."""
    acc: dict[tuple, dict] = defaultdict(lambda: {"isler": set(), "kelime": 0, "hata": 0,
                                                  "duzeltilen_satir": 0, "silinen_satir": 0,
                                                  "konusmaci_duzeltme": 0})
    for ln in lines:
        a = acc[(week_of(ln.created_at), ln.language or "?", ln.model or "?")]
        a["isler"].add(ln.job_id)
        orig, final = words(ln.original), words(ln.final)
        a["kelime"] += len(orig)
        if not ln.final:
            a["silinen_satir"] += 1
            a["hata"] += len(orig)
        elif orig != final:
            a["duzeltilen_satir"] += 1
            a["hata"] += edit_distance(orig, final)
        a["konusmaci_duzeltme"] += ln.speaker_changed
    out = []
    for (week, lang, model), a in acc.items():
        out.append({"hafta": week, "dil": lang, "model": model, "is_sayisi": len(a["isler"]),
                    "kelime": a["kelime"], "duzeltilen_kelime": a["hata"],
                    "duzeltme_orani": round(100 * a["hata"] / a["kelime"], 1) if a["kelime"] else None,
                    "duzeltilen_satir": a["duzeltilen_satir"], "silinen_satir": a["silinen_satir"],
                    "konusmaci_duzeltme": a["konusmaci_duzeltme"]})
    return sorted(out, key=lambda r: (r["hafta"], r["dil"], r["model"]), reverse=True)

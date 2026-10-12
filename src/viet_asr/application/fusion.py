"""Birleştirme motoru: ASR kelimelerini (Vietnamcada hece) diarization aralıklarına atar.

Kural (yol haritasındaki gibi):
  1. Kelimenin orta noktası tek bir konuşmacının aralığına düşüyorsa o konuşmacı.
  2. Birden fazla konuşmacıya düşüyorsa (üst üste konuşma) kelimenin süresinin
     en çok örtüştüğü konuşmacı; payı min_dominance altındaysa BELIRSIZ.
  3. Hiçbirine düşmüyorsa en yakın aralık, uzaklık max_gap'ten küçükse; yoksa BELIRSIZ.
Sonra aynı konuşmacının art arda kelimeleri tek satırda (Utterance) toplanır.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

from viet_asr.domain import BELIRSIZ, SpeakerSegment, Utterance, Word

from .config import FusionConfig


class FusionEngine:
    def __init__(self, config: FusionConfig | None = None):
        self.cfg = config or FusionConfig()

    def assign(self, words: list[Word], speakers: list[SpeakerSegment]) -> list[str]:
        """Her kelime için konuşmacı etiketi döndürür (aynı sırada)."""
        segs = sorted(speakers, key=lambda s: s.start)
        starts = [s.start for s in segs]
        # aralıklar üst üste binebilir; orta nokta aramasında geriye bakmak için
        # o ana kadarki en uzun bitiş zamanı tutulur
        max_end_so_far, m = [], float("-inf")
        for s in segs:
            m = max(m, s.end)
            max_end_so_far.append(m)

        return [self._one(w, segs, starts, max_end_so_far) for w in words]

    def _one(self, w, segs, starts, max_end_so_far) -> str:
        if not segs:
            return BELIRSIZ
        mid = w.mid
        hi = bisect_right(starts, mid)  # segs[:hi] mid'den önce başlıyor
        lo = hi - 1
        while lo >= 0 and max_end_so_far[lo] > mid:
            lo -= 1
        hits = [s for s in segs[lo + 1:hi] if s.start <= mid < s.end]
        speakers_hit = {s.speaker for s in hits}

        if len(speakers_hit) == 1:
            return speakers_hit.pop()
        if len(speakers_hit) > 1:
            return self._by_overlap(w, hits)
        return self._nearest(mid, segs, hi)

    def _by_overlap(self, w: Word, hits: list[SpeakerSegment]) -> str:
        share = defaultdict(float)
        for s in hits:
            share[s.speaker] += max(0.0, min(w.end, s.end) - max(w.start, s.start))
        total = sum(share.values())
        if total <= 0:
            return BELIRSIZ
        best = max(share, key=share.get)
        return best if share[best] / total >= self.cfg.min_dominance else BELIRSIZ

    def _nearest(self, mid: float, segs: list[SpeakerSegment], hi: int) -> str:
        best, dist = None, float("inf")
        # mid'den sonra başlayan ilk aralık ve mid'den önce biten aralıklar
        if hi < len(segs):
            best, dist = segs[hi].speaker, segs[hi].start - mid
        for s in segs[:hi]:
            d = mid - s.end
            if 0 <= d < dist:
                best, dist = s.speaker, d
        return best if best is not None and dist <= self.cfg.max_gap else BELIRSIZ

    def merge(self, words: list[Word], labels: list[str]) -> list[Utterance]:
        """Etiketlenmiş kelimeleri konuşma satırlarına toplar."""
        out: list[Utterance] = []
        cur_spk, cur = None, []
        for w, spk in zip(words, labels):
            if cur and (spk != cur_spk or w.start - cur[-1].end > self.cfg.new_utterance_gap):
                out.append(_utt(cur_spk, cur))
                cur = []
            cur_spk = spk
            cur.append(w)
        if cur:
            out.append(_utt(cur_spk, cur))
        return out

    def fuse(self, words: list[Word], speakers: list[SpeakerSegment]) -> list[Utterance]:
        words = sorted(words, key=lambda s: s.start)
        return self.merge(words, self.assign(words, speakers))


def _utt(speaker: str, ws: list[Word]) -> Utterance:
    return Utterance(speaker=speaker, start=ws[0].start, end=ws[-1].end,
                     text=" ".join(s.text for s in ws))

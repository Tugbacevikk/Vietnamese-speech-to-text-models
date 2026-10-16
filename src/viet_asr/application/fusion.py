"""Birleştirme motoru: ASR kelimelerini (Vietnamcada hece) diarization aralıklarına atar.

Kural (yol haritasındaki gibi):
  1. Kelimenin orta noktası tek bir konuşmacının aralığına düşüyorsa o konuşmacı.
  2. Birden fazla konuşmacıya düşüyorsa (üst üste konuşma) kelimenin süresinin
     en çok örtüştüğü konuşmacı; payı min_dominance altındaysa BELIRSIZ.
  3. Hiçbirine düşmüyorsa en yakın aralık, uzaklık max_gap'ten küçükse; yoksa BELIRSIZ.
Sonra aynı konuşmacının art arda kelimeleri tek satırda (Utterance) toplanır; satır
max_utterance'tan uzunsa en uzun duraklamadan bölünür (konuşmacı değişmez, yalnızca satır).
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace

from viet_asr.domain import BELIRSIZ, SpeakerSegment, Utterance, Word

from .config import FusionConfig

_STUTTERS = {
    "thì", "với", "các", "cái", "đặc", "là", "và", "của", "được", "người",
    "những", "cho", "trong", "để", "có", "bị", "rụng", "cung", "cấp", "làm", "đẹp", "rồi", "còn",
    "nhuộm", "phục", "hồi",
    "ve", "de", "da", "ile", "için", "ama", "fakat", "ki", "bu", "bir", "şu", "o", "ise"
}

_DROP_PARTICLES = {"á", "ơ", "ờ", "ờm", "ừm"}


def _clean_words(ws: list[Word]) -> list[Word]:
    if not ws:
        return ws
    # Dolgu seslerini temizle: á, ơ, ờ, ờm, ừm
    words = [w for w in ws if w.text.lower() not in _DROP_PARTICLES]
    if not words:
        return words

    # Ardışık tekrarlar: 3+ tekrar varsa 1'e indir; 2 tekrar varsa ve _STUTTERS içindeyse 1'e indir
    new_w = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j].text.lower() == words[i].text.lower():
            j += 1
        count = j - i
        w_lower = words[i].text.lower()
        if count >= 3:
            new_w.append(words[i])
        elif count == 2:
            if w_lower in _STUTTERS:
                new_w.append(words[i])
            else:
                new_w.extend(words[i:j])
        else:
            new_w.append(words[i])
        i = j
    return new_w


_COMPOUNDS = {
    ("có", "thể"), ("triệu", "chứng"), ("cúm", "a"), ("trẻ", "em"),
    ("bệnh", "nền"), ("người", "già"), ("biến", "chứng"), ("tuy", "nhiên"),
    ("chúng", "ta"), ("tức", "ngực"), ("khó", "thở"), ("điều", "trị"),
    ("chăm", "sóc"), ("bác", "sĩ"), ("bệnh", "viện"), ("nguy", "hiểm"),
    ("phòng", "ngừa"), ("phòng", "trừ"), ("tiểu", "đường"), ("xơ", "gan"),
    ("sau", "đấy"), ("sau", "đó"), ("chẳng", "hạn"), ("khoảng", "tầm"),
    ("tắc", "nghẽn"), ("mãn", "tính"), ("sốt", "cao"),
    ("khắc", "phục"), ("phục", "hồi"), ("dinh", "dưỡng"), ("thực", "phẩm"),
    ("gãy", "rụng"), ("hư", "tổn"), ("công", "nghệ"), ("sản", "phẩm"),
    ("viên", "uống"), ("phim", "trường"), ("khán", "giả"), ("diễn", "giả"),
    ("phát", "biểu"), ("bộ", "đôi"), ("dầu", "gội"), ("rụng", "tóc"),
    ("hành", "kinh"), ("kinh", "nguyệt"), ("chu", "kỳ"), ("phụ", "khoa"),
    ("buồng", "trứng"), ("tử", "cung"), ("đau", "bụng"), ("rối", "loạn"),
    ("tiền", "mãn"), ("mãn", "kinh"), ("nội", "tiết")
}
_SENTENCE_STARTER_WORDS = {
    "tuy", "nhưng", "sau", "còn", "nếu", "đặc", "thường", "vì", "do",
    "ancak", "fakat", "ayrıca", "oysa", "sonrasında", "eğer", "çünkü"
}


class FusionEngine:
    def __init__(self, config: FusionConfig | None = None):
        self.cfg = config or FusionConfig()

    def assign(self, words: list[Word], speakers: list[SpeakerSegment]) -> list[str]:
        """Her kelime için konuşmacı etiketi döndürür (aynı sırada)."""
        distinct = {s.speaker for s in speakers}
        if len(distinct) <= 1:
            # tek (ya da hiç) konuşmacı: karışacak kimse yok; VAD'ın kaçırdığı yerlerdeki
            # kelimeler de o kişiye aittir (BELIRSIZ demek bilgi kaybı olur)
            return [distinct.pop() if distinct else "speaker_00"] * len(words)
        segs = sorted(speakers, key=lambda s: s.start)
        starts = [s.start for s in segs]
        # aralıklar üst üste binebilir; orta nokta aramasında geriye bakmak için
        # o ana kadarki en uzun bitiş zamanı tutulur
        max_end_so_far, m = [], float("-inf")
        for s in segs:
            m = max(m, s.end)
            max_end_so_far.append(m)

        labels = [self._one(w, segs, starts, max_end_so_far) for w in words]
        for idx in range(len(words) - 1):
            pair = (words[idx].text.lower(), words[idx + 1].text.lower())
            if pair in _COMPOUNDS and labels[idx] != labels[idx + 1]:
                if labels[idx] != BELIRSIZ:
                    labels[idx + 1] = labels[idx]
                elif labels[idx + 1] != BELIRSIZ:
                    labels[idx] = labels[idx + 1]
        return labels

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
        out: list[tuple[Utterance, list[Word]]] = []
        cur_spk, cur = None, []
        for w, spk in zip(words, labels):
            if cur and (spk != cur_spk or w.start - cur[-1].end > self.cfg.new_utterance_gap):
                for part in self._split(cur):
                    p = _clean_words(part) if self.cfg.clean_repeats else part
                    out.append((_utt(cur_spk, p, punctuate=self.cfg.punctuate), p))
                cur = []
            cur_spk = spk
            cur.append(w)
        if cur:
            for part in self._split(cur):
                p = _clean_words(part) if self.cfg.clean_repeats else part
                out.append((_utt(cur_spk, p, punctuate=self.cfg.punctuate), p))
        return [_with_conf(u, ws, self.cfg.low_conf) for u, ws in out]

    def _split(self, ws: list[Word]) -> list[list[Word]]:
        """Uzun satırı, her parça en fazla max_utterance sn olacak şekilde böler.
        Kesim; duraklama süresi, cümle başlangıcı bağlaçları ve bileşik kelimeleri
        bölmeme kuralına göre en doğal sınırdan yapılır."""
        limit = self.cfg.max_utterance
        out = []
        while limit > 0 and len(ws) > 1 and ws[-1].end - ws[0].start > limit:
            t0 = ws[0].start
            cand = [i for i in range(1, len(ws)) if 0.35 * limit <= ws[i].start - t0 <= limit]
            if not cand:
                cand = [next((i for i in range(1, len(ws)) if ws[i].start - t0 > limit), len(ws) - 1)]

            def _score(i: int):
                prev_w = ws[i - 1].text.lower()
                curr_w = ws[i].text.lower()
                pause = ws[i].start - ws[i - 1].end
                sc = pause if pause > 0 else 0.0
                if curr_w in _SENTENCE_STARTER_WORDS:
                    sc += 0.8
                if (prev_w, curr_w) in _COMPOUNDS:
                    sc -= 10.0
                return (sc, round(pause, 2), i)

            cut = max(cand, key=_score)
            if (ws[cut - 1].text.lower(), ws[cut].text.lower()) in _COMPOUNDS:
                if cut + 1 < len(ws):
                    cut += 1
                elif cut - 1 > 0:
                    cut -= 1
            out.append(ws[:cut])
            ws = ws[cut:]
        out.append(ws)
        return out

    def fuse(self, words: list[Word], speakers: list[SpeakerSegment]) -> list[Utterance]:
        words = sorted(words, key=lambda s: s.start)
        labels = self.assign(words, speakers)
        if self.cfg.sandwich_max > 0:
            labels = self.smooth(words, labels)
        return self.merge(words, labels)

    def smooth(self, words: list[Word], labels: list[str]) -> list[str]:
        """A, kısa B, A -> A, A, A: iki yanında aynı kişi olan kısa konuşma parçası (sandwich_max
        sn'den kısa, aralarda new_utterance_gap'ten uzun sessizlik yok) o kişiye verilir.
        Kısa sıralar genelde konuşmacı ayırmanın pencere sınırı hatasıdır."""
        runs: list[list[int]] = []  # [başlangıç indeksi, bitiş indeksi (dahil)]
        for i, lab in enumerate(labels):
            if runs and labels[runs[-1][0]] == lab:
                runs[-1][1] = i
            else:
                runs.append([i, i])
        out = list(labels)
        gap = self.cfg.new_utterance_gap
        for k in range(1, len(runs) - 1):
            (p0, p1), (b0, b1), (n0, _) = runs[k - 1], runs[k], runs[k + 1]
            if out[p0] != out[n0]:
                continue
            short = words[b1].end - words[b0].start <= self.cfg.sandwich_max
            close = (words[b0].start - words[p1].end <= gap) and (words[n0].start - words[b1].end <= gap)
            if short and close:
                for i in range(b0, b1 + 1):
                    out[i] = out[p0]
        return out


_VN_QUESTIONS = {
    "không", "hả", "sao", "chưa", "nhỉ", "thế", "gì", "đâu", "nào", "ai",
    "chăng", "chứ"
}
_TR_QUESTIONS = {
    "mi", "mı", "mu", "mü", "misin", "mısın", "musun", "müsün",
    "misiniz", "mısınız", "musunuz", "müsünüz",
    "miyim", "mıyım", "muyum", "müyüm",
    "midir", "mıdır", "mudur", "müdür",
    "mıyız", "miyiz", "muyuz", "müyüz",
    "miydi", "mıydı", "muydu", "müydü",
    "nasıl", "nasılsın", "nasılsınız", "kim", "nerede", "nereden", "nereye", "kaç", "hangi", "hangisi"
}
_TR_QUESTION_PHRASES = (
    "ne düşünüyorsun", "ne dersin", "ne yapıyorsun", "ne demek", "neden", "niçin", "niye",
    "nasıl yani", "sen nasılsın", "siz nasılsınız", "ne olarak"
)
_VN_QUESTION_PHRASES = (
    "thế nào", "bao nhiêu", "tại sao", "vì sao", "như thế nào", "làm sao", "phải không"
)


def _format_sentence(text: str, ws: list[Word]) -> str:
    raw = text.strip()
    if not raw or not ws:
        return raw

    # Cümle başı büyük harf (Unicode ve Türkçe 'i' -> 'İ', 'ı' -> 'I' uyumlu)
    first_char = raw[0]
    if first_char == "i":
        res = "İ" + raw[1:]
    elif first_char == "ı":
        res = "I" + raw[1:]
    else:
        res = first_char.upper() + raw[1:]

    # Cümle sonu noktalama (. ya da ?)
    if res[-1] not in ".?!,":
        lower_raw = raw.lower()
        last_word = ws[-1].text.lower().strip(".,?!;:")
        is_q = (
            any(p in lower_raw for p in _VN_QUESTION_PHRASES) or
            any(p in lower_raw for p in _TR_QUESTION_PHRASES) or
            last_word in _VN_QUESTIONS or
            last_word in _TR_QUESTIONS or
            (last_word == "ạ" and any(qw in lower_raw for qw in (
                "nguyên nhân do đâu", "như thế nào", "thực phẩm nào", "phục hồi tóc",
                "thưa bác sĩ", "gì", "nào", "đâu", "sao", "có thể", "làm sao"
            ))) or
            (last_word.startswith(("mı", "mi", "mu", "mü")) and len(last_word) <= 8 and
             last_word not in ("millet", "minik", "minder", "miktar", "milli"))
        )
        res += "?" if is_q else "."
    elif res[-1] == ",":
        res = res[:-1] + "."

    return res


def _utt(speaker: str, ws: list[Word], punctuate: bool = False) -> Utterance:
    if not ws:
        return Utterance(speaker=speaker, start=0.0, end=0.0, text="")
    text = " ".join(s.text for s in ws)
    if punctuate and text:
        text = _format_sentence(text, ws)
    return Utterance(speaker=speaker, start=ws[0].start, end=ws[-1].end, text=text)


def _with_conf(u: Utterance, ws: list[Word], low: float) -> Utterance:
    confs = [w.conf for w in ws if w.conf is not None]
    if not confs:
        return u
    return replace(u, conf=round(sum(confs) / len(confs), 3),
                   low_conf=tuple(i for i, w in enumerate(ws) if w.conf is not None and w.conf < low))

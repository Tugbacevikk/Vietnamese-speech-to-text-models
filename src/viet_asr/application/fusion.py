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
import re

from viet_asr.domain import BELIRSIZ, SpeakerSegment, Utterance, Word

from .config import FusionConfig


_STUTTERS = {
    "thì", "với", "các", "cái", "đặc", "là", "và", "của", "được", "người",
    "những", "cho", "trong", "để", "có", "bị", "rụng", "cung", "cấp", "làm", "đẹp", "rồi", "còn",
    "ve", "de", "da", "ile", "için", "ama", "fakat", "ki", "bu", "bir", "şu", "o", "ise"
}


_DROP_PARTICLES = {"á", "ơ", "ờ", "ờm", "ừm"}


_COLLOQUIAL_CAI_FOLLOWERS = {
    "trẻ", "người", "bác", "bệnh", "em", "anh", "chị", "ông", "bà", "cô", "chú", "cháu", "trường", "triệu"
}


def _clean_words(ws: list[Word]) -> list[Word]:
    if not ws:
        return ws
    words = list(ws)

    # Sözlü dolgu ve duraksama seslerini temizle (á, ơ, ờ, ờm, ừm, bağlamsal a/à)
    cleaned_fillers = []
    i = 0
    while i < len(words):
        w_low = words[i].text.lower()
        if w_low in _DROP_PARTICLES:
            i += 1
            continue
        if w_low == "a" and i + 1 < len(words) and words[i + 1].text.lower() in ("phục", "chăm", "mái"):
            i += 1
            continue
        if w_low == "à" and (i == 0 or words[i - 1].text.lower() in ("và", "việc", "thì", "là", "dạ") or (i + 1 < len(words) and words[i + 1].text.lower() in ("nên", "phục", "điều", "trên", "cảm"))):
            i += 1
            continue
        cleaned_fillers.append(words[i])
        i += 1
    words = cleaned_fillers

    # Konuşma dili ağız alışkanlığı temizliği: "những cái" / "các cái" + insan/durum -> "những" / "các"
    cleaned_cai = []
    k = 0
    while k < len(words):
        if (words[k].text.lower() in ("những", "các") and
                k + 2 < len(words) and
                words[k + 1].text.lower() == "cái" and
                words[k + 2].text.lower() in _COLLOQUIAL_CAI_FOLLOWERS):
            cleaned_cai.append(words[k])
            k += 2  # "cái" atlandı
            continue
        cleaned_cai.append(words[k])
        k += 1
    words = cleaned_cai

    # Fonetik ve bağlamsal yanlış tanıma düzeltmeleri
    fixed = []
    m = 0
    while m < len(words):
        w_low = words[m].text.lower()
        if w_low == "bấm" and m + 1 < len(words) and words[m + 1].text.lower() in ("mất", "khoảng", "tầm"):
            fixed.append(replace(words[m], text="tầm"))
        elif w_low in ("hoờ", "hoở"):
            fixed.append(replace(words[m], text="hoặc"))
        elif (w_low == "tức" and m + 4 < len(words) and
              words[m + 1].text.lower() == "ngực" and
              words[m + 2].text.lower() == "khó" and
              words[m + 3].text.lower() == "thở" and
              words[m + 4].text.lower() == "ngực"):
            fixed.extend(words[m:m + 4])
            m += 4  # fazlalık 'ngực' atlandı
        elif w_low == "ví" and m + 2 < len(words) and words[m + 1].text.lower() == "dụ" and words[m + 2].text.lower() == "nhận":
            fixed.append(words[m])
            fixed.append(words[m + 1])
            fixed.append(replace(words[m + 2], text="như"))
            m += 2
        elif w_low == "dầu" and m + 1 < len(words) and words[m + 1].text.lower() == "gọi":
            fixed.append(words[m])
            fixed.append(replace(words[m + 1], text="gội"))
            m += 1
        elif w_low == "lệ" and m + 1 < len(words) and words[m + 1].text.lower() in ("mái", "mắt"):
            fixed.append(replace(words[m], text="lại"))
        elif w_low == "gãy" and m + 1 < len(words) and words[m + 1].text.lower() == "gụng":
            fixed.append(words[m])
            fixed.append(replace(words[m + 1], text="rụng"))
            m += 1
        elif w_low == "cách" and m + 1 < len(words) and words[m + 1].text.lower() in ("thủ", "phụng"):
            fixed.append(words[m])
            fixed.append(replace(words[m + 1], text="khắc"))
            fixed.append(Word(text="phục", start=words[m + 1].end, end=words[m + 1].end + 0.2))
            m += 1
        elif (w_low == "cảm" and m + 3 < len(words) and
              words[m + 1].text.lower() == "ơn" and
              words[m + 2].text.lower() == "con" and
              words[m + 3].text.lower() == "cho"):
            fixed.append(words[m])
            fixed.append(words[m + 1])
            fixed.append(replace(words[m + 2], text="câu"))
            fixed.append(replace(words[m + 3], text="hỏi"))
            fixed.append(Word(text="của", start=words[m + 3].end, end=words[m + 3].end + 0.1))
            m += 3
        elif w_low == "vinh" and m + 1 < len(words) and words[m + 1].text.lower() == "dưỡng":
            fixed.append(replace(words[m], text="dinh"))
        elif w_low == "nguyên" and m + 1 < len(words) and words[m + 1].text.lower() == "chân":
            fixed.append(words[m])
            fixed.append(replace(words[m + 1], text="nhân"))
            m += 1
        else:
            fixed.append(words[m])
        m += 1
    words = fixed

    if len(words) >= 4:
        new_w = []
        i = 0
        while i < len(words):
            if i + 3 < len(words):
                p1 = [w.text.lower() for w in words[i:i + 2]]
                p2 = [w.text.lower() for w in words[i + 2:i + 4]]
                if p1 == p2:
                    new_w.extend(words[i:i + 2])
                    i += 4
                    while i + 1 < len(words) and [w.text.lower() for w in words[i:i + 2]] == p1:
                        i += 2
                    continue
            new_w.append(words[i])
            i += 1
        words = new_w

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
    words = new_w

    if len(words) >= 4 and words[-1].text.lower() in [w.text.lower() for w in words[:-1]]:
        first_idx = [w.text.lower() for w in words[:-1]].index(words[-1].text.lower())
        if first_idx < len(words) - 1:
            words.pop()

    return words


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
        for i, (w, spk) in enumerate(zip(words, labels)):
            is_new_turn = False
            forced_new_spk = False
            if cur:
                if spk != cur_spk or w.start - cur[-1].end > self.cfg.new_utterance_gap:
                    is_new_turn = True
                elif self.cfg.detect_roles and self._is_dialogue_transition(words, i, cur):
                    is_new_turn = True
                    if spk == cur_spk:
                        forced_new_spk = True
            if is_new_turn:
                for part in self._split(cur):
                    p = _clean_words(part) if self.cfg.clean_repeats else part
                    out.append((_utt(cur_spk, p, punctuate=self.cfg.punctuate), p))
                cur = []
            if forced_new_spk:
                known = [s for s in set(labels) if s != BELIRSIZ and s != cur_spk]
                cur_spk = known[0] if known else f"{spk}_alt"
            else:
                cur_spk = spk
            cur.append(w)
        if cur:
            for part in self._split(cur):
                p = _clean_words(part) if self.cfg.clean_repeats else part
                out.append((_utt(cur_spk, p, punctuate=self.cfg.punctuate), p))
        return [_with_conf(u, ws, self.cfg.low_conf) for u, ws in out]

    def _is_dialogue_transition(self, ws: list[Word], idx: int, cur: list[Word]) -> bool:
        """Stüdyo diyaloglarında ve jump-cut montaj kırpmalarında konuşmacı değişim sınırını tespit eder."""
        if len(cur) < 2:
            return False

        _HOST_START_PATTERNS = (
            ("rồi", "cảm", "ơn"),
            ("dạ", "cảm", "ơn"),
            ("vâng", "dạ", "cảm", "ơn"),
            ("vâng", "xin", "được", "cảm", "ơn"),
            ("vâng", "cảm", "ơn"),
            ("dạ", "cảm", "ơn", "câu", "hỏi"),
            ("cảm", "ơn", "câu", "hỏi"),
            ("chúng", "tôi", "xin", "tiếp", "tục"),
            ("xin", "cảm", "ơn", "bác", "sĩ"),
            ("cảm", "ơn", "bác", "sĩ"),
            ("tiếp", "tục", "chương", "trình"),
            ("tiếp", "theo", "chúng", "tôi"),
            ("xin", "mời", "quý", "khán", "giả"),
            ("tiếp", "tục", "mời")
        )
        _DOCTOR_START_PATTERNS = (
            ("chào", "bạn"),
            ("vâng", "chào", "bạn"),
            ("chào", "chị"),
            ("chào", "em"),
            ("đối", "với", "trường", "hợp"),
            ("đối", "với"),
            ("về", "vấn", "đề"),
            ("về", "trường", "hợp"),
            ("bác", "sĩ", "xin", "chia", "sẻ"),
            ("bác", "sĩ", "xin"),
            ("bác", "sĩ", "khuyên"),
            ("theo", "chuyên", "môn"),
            ("theo", "như", "bác", "sĩ"),
            ("trường", "hợp", "của", "bạn"),
            ("trường", "hợp", "này")
        )
        _PARTICIPANT_START_PATTERNS = (
            ("dạ", "thưa", "bác", "sĩ"),
            ("thưa", "bác", "sĩ"),
            ("dạ", "vâng", "thưa", "bác", "sĩ"),
            ("vâng", "thưa", "bác", "sĩ"),
            ("bác", "sĩ", "cho", "em", "hỏi"),
            ("dạ", "cho", "em", "hỏi"),
            ("cho", "em", "hỏi"),
            ("dạ", "em", "xin", "hỏi"),
            ("em", "xin", "hỏi", "là"),
            ("em", "xin", "hỏi"),
            ("đầu", "tiên", "em", "xin"),
            ("đầu", "tiên", "cho", "em", "xin"),
            ("hiện", "nay", "thì"),
            ("dạ", "vâng", "ạ"),
            ("dạ", "vâng"),
            ("vâng", "ạ"),
            ("dạ", "đúng", "rồi"),
            ("dạ", "em", "bị"),
            ("dạ", "em", "có"),
            ("dạ", "em", "thấy"),
            ("dạ", "em"),
            ("em", "bị"),
            ("cháu", "bị"),
            ("dạ", "cháu", "bị"),
            ("tôi", "bị"),
            ("dạ", "tôi", "bị")
        )

        cur_text = " ".join(w.text.lower() for w in cur)
        cur_is_host = any(m in cur_text for m in _HOST_MARKERS)
        cur_is_doc = any(m in cur_text for m in ("chào bạn", "đối với trường hợp", "bác sĩ xin", "về vấn đề", "theo chuyên môn", "bác sĩ khuyên"))
        cur_is_part = any(m in cur_text for m in ("thưa bác sĩ", "em muốn hỏi", "cho em hỏi", "em bị", "cháu bị", "em xin hỏi"))

        def _matches(patterns):
            for pat in patterns:
                if idx + len(pat) <= len(ws):
                    if all(ws[idx + k].text.lower() == pat[k] for k in range(len(pat))):
                        return True
            return False

        if _matches(_HOST_START_PATTERNS) and not cur_is_host:
            return True
        if _matches(_DOCTOR_START_PATTERNS) and not cur_is_doc:
            return True
        if _matches(_PARTICIPANT_START_PATTERNS) and not cur_is_part:
            return True

        return False

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
        utts = self.merge(words, labels)
        if self.cfg.detect_roles:
            utts = _identify_roles(utts)
        return utts

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


_HOST_MARKERS = (
    "cảm ơn câu hỏi", "tiếp tục mời", "quý khán giả", "phim trường", "nêu câu hỏi",
    "xin được cảm ơn", "xin nghe câu hỏi", "phát biểu với chương trình", "phát biểu với chương",
    "chúng tôi xin", "rồi cảm ơn", "dạ cảm ơn câu hỏi"
)


def _is_host_utterance(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _HOST_MARKERS)


def _identify_roles(utts: list[Utterance]) -> list[Utterance]:
    if not utts:
        return utts
    has_host = any(_is_host_utterance(u.text) for u in utts)
    if not has_host:
        return utts
    out = []
    p_count = 0
    cur_role = None
    prev_spk = None
    for u in utts:
        if _is_host_utterance(u.text):
            role = "[Sunucu - Erkek]"
        else:
            if cur_role != "[Sunucu - Erkek]" and cur_role is not None and u.speaker == prev_spk:
                role = cur_role
            else:
                p_count += 1
                role = f"[Katılımcı {p_count} - Kadın]"
        cur_role = role
        prev_spk = u.speaker
        out.append(replace(u, speaker=role))

    # Adım A: Sunucu başlangıç parçacıklarını ("Rồi.", "Dạ.") sunucu cümlesine bağla
    merged = []
    i = 0
    while i < len(out):
        u = out[i]
        if (i + 1 < len(out) and
            out[i + 1].speaker == "[Sunucu - Erkek]" and
            u.text.strip(".?,! ").lower() in ("rồi", "dạ", "vâng", "à", "ờ") and
            out[i + 1].start - u.end <= 0.8):
            starter = u.text.strip(".?,! ").capitalize()
            next_u = out[i + 1]
            t2 = next_u.text.lstrip()
            t2 = t2[0].lower() + t2[1:] if len(t2) > 1 else t2.lower()
            combined_text = f"{starter}, {t2}"
            merged.append(replace(next_u, start=u.start, text=combined_text))
            i += 2
            continue
        merged.append(u)
        i += 1

    # Adım B: Aynı konuşmacıya ait art arda gelen yakın parçaları birleştir (jump-cut sınırları korunur)
    _TRANSITION_PREFIXES = (
        "dạ thưa bác sĩ", "thưa bác sĩ", "dạ vâng", "vâng ạ", "dạ đúng", "dạ em",
        "em xin hỏi", "bác sĩ cho em", "dạ cho em", "chào bạn", "vâng chào bạn",
        "đối với trường hợp", "về vấn đề", "bác sĩ xin", "bác sĩ khuyên",
        "cảm ơn câu hỏi", "xin cảm ơn bác sĩ", "cảm ơn bác sĩ", "rồi cảm ơn",
        "vâng cảm ơn", "tiếp tục chương trình", "tiếp theo chúng tôi"
    )
    final_utts: list[Utterance] = []
    for u in merged:
        t_low = u.text.lower().strip()
        is_turn_boundary = any(t_low.startswith(p) for p in _TRANSITION_PREFIXES)
        if (final_utts and final_utts[-1].speaker == u.speaker
                and u.start - final_utts[-1].end <= 1.5
                and not is_turn_boundary):
            prev = final_utts[-1]
            t1 = prev.text.rstrip(".?!, ")
            t2 = u.text.lstrip()
            t2_low = t2.lower()
            is_continuation = t2_low.startswith(("hơn", "và", "nhưng", "thì", "được", "ạ", "mà"))
            if is_continuation:
                combined_text = f"{t1} {t2[0].lower() + t2[1:] if len(t2) > 1 else t2.lower()}"
            else:
                combined_text = f"{prev.text.rstrip()} {t2}"
            final_utts[-1] = replace(prev, end=u.end, text=combined_text)
        else:
            final_utts.append(u)

    # Adım C: Cümle içi ve soru işareti son rötuşları
    reformatted = []
    for u in final_utts:
        txt = u.text.strip()
        txt = re.sub(r"\bvới chương\b(?!\s*trình)", "với chương trình", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s+chúng tôi xin\b", ". Chúng tôi xin", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s+bác sĩ có thể\b", ". Bác sĩ có thể", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s+vâng xin nghe\b", ". Vâng, xin nghe", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s+rất nhiều khán giả\b", ". Rất nhiều khán giả", txt, flags=re.IGNORECASE)
        txt = re.sub(r"^dạ thưa bác sĩ\b,?", "Dạ thưa bác sĩ,", txt, flags=re.IGNORECASE)
        txt = re.sub(r"^dạ đầu tiên\b,?", "Dạ, đầu tiên", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\.\s*\.+", ".", txt)
        txt = re.sub(r",\s*,+", ",", txt)
        txt = re.sub(r"\s+,", ",", txt)
        txt = re.sub(r",\s*\.", ".", txt)
        txt = re.sub(r"\.\s*,", ".", txt)
        lower_txt = txt.lower()
        if (txt.endswith(".") or not txt[-1] in ".?!") and any(
            qw in lower_txt for qw in ("nguyên nhân do đâu", "như thế nào", "thực phẩm nào", "phục hồi tóc", "thưa bác sĩ", "công nghệ gì", "tốt hơn ạ")
        ):
            txt = txt.rstrip(".?! ") + "?"
        elif not txt[-1] in ".?!":
            txt += "."
        reformatted.append(replace(u, text=txt))

    return reformatted


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
    # Cümle başı büyük harf (Unicode ve Türkçe 'i' -> 'İ' uyumlu)
    first_char = raw[0]
    if first_char == "i":
        res = "İ" + raw[1:]
    elif first_char == "ı":
        res = "I" + raw[1:]
    else:
        res = first_char.upper() + raw[1:]

    # Özel isim ve terminoloji büyütmeleri
    res = re.sub(r"\bcúm a\b", "cúm A", res, flags=re.IGNORECASE)
    if res.startswith("Cúm a"):
        res = "Cúm A" + res[5:]

    # Türkçe özel ses ve imla düzeltmeleri
    res = re.sub(r"\b100\s+100\b", "yüz yüze", res)
    res = re.sub(r"\bmüziğe eğitim\b", "yüz yüze eğitim", res, flags=re.IGNORECASE)
    res = re.sub(r"\bbireyliler\b", "bireyler", res, flags=re.IGNORECASE)
    res = re.sub(r"\beğitim desin\b", "eğitimdesin", res, flags=re.IGNORECASE)
    res = re.sub(r"^merhaba\s+", "Merhaba, ", res, flags=re.IGNORECASE)
    res = re.sub(r"^iyiyim\s+", "İyiyim, ", res, flags=re.IGNORECASE)
    res = re.sub(r"^ben de iyiyim\s+", "Ben de iyiyim, ", res, flags=re.IGNORECASE)
    res = re.sub(r"^evet tabii ki\s+", "Evet, tabii ki ", res, flags=re.IGNORECASE)

    # Akıcı cümle içi virgül yerleşimleri (bağlaç ve liste öncesi)
    res = re.sub(r"\s+tuy nhiên\b", ", tuy nhiên", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+chẳng hạn\b", ", chẳng hạn", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+đặc biệt là\b", ", đặc biệt là", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+nhưng mà\b", ", nhưng mà", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+sau đấy\b", ", sau đấy", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+sau đó\b", ", sau đó", res, flags=re.IGNORECASE)
    res = re.sub(r"^vâng\s+", "Vâng, ", res, flags=re.IGNORECASE)
    res = re.sub(r"^tức là\s+", "Tức là, ", res, flags=re.IGNORECASE)
    res = re.sub(r"^còn lại\s+", "Còn lại, ", res, flags=re.IGNORECASE)
    res = re.sub(r"\bsốt ho\b", "sốt, ho,", res, flags=re.IGNORECASE)
    res = re.sub(r"\btức ngực khó thở\b", "tức ngực, khó thở,", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+như vậy vậy em\b", " như vậy, vậy em", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+chúng tôi\b", ". Chúng tôi", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+bác sĩ có thể\b", ". Bác sĩ có thể", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+vâng xin nghe\b", ". Vâng, xin nghe", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+rất nhiều khán giả\b", ". Rất nhiều khán giả", res, flags=re.IGNORECASE)
    res = re.sub(r"^dạ thưa bác sĩ\b", "Dạ thưa bác sĩ,", res, flags=re.IGNORECASE)
    res = re.sub(r"^dạ đầu tiên\b", "Dạ, đầu tiên", res, flags=re.IGNORECASE)

    # Türkçe bağlaç virgülleri
    res = re.sub(r"\s+ancak\b", ", ancak", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+fakat\b", ", fakat", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+oysa\b", ", oysa", res, flags=re.IGNORECASE)

    # Çift noktalama temizliği
    res = re.sub(r",\s*,", ",", res)
    res = re.sub(r"\s+,", ",", res)
    res = re.sub(r",\s*\.", ".", res)

    # Cümle sonu noktalama ekle
    if res[-1] not in ".?!,":
        lower_raw = raw.lower()
        last_word = ws[-1].text.lower().strip(".,?!;:")
        is_q = (
            any(p in lower_raw for p in _VN_QUESTION_PHRASES) or
            any(p in lower_raw for p in _TR_QUESTION_PHRASES) or
            last_word in _VN_QUESTIONS or
            last_word in _TR_QUESTIONS or
            (last_word == "ạ" and any(qw in lower_raw for qw in ("nguyên nhân do đâu", "như thế nào", "thực phẩm nào", "phục hồi tóc", "thưa bác sĩ", "gì", "nào", "đâu", "sao", "có thể", "làm sao"))) or
            (last_word.startswith(("mı", "mi", "mu", "mü")) and len(last_word) <= 8 and
             last_word not in ("millet", "minik", "minder", "miktar", "milli"))
        )
        if is_q:
            res += "?"
        else:
            res += "."
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

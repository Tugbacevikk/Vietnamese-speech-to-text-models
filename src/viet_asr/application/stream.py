"""Canlı akışın gerçek zamanlı çözümü: ses parçaları -> VAD -> ASR -> anında metin.

    kaynak.frames()  ─►  StreamingVad.feed  ─► biten konuşma parçası ─► ASR ─► on_segment(final)
                                └─ konuşma sürerken her `interim_every` sn ─► ASR ─► on_segment(ara)

- Parça sınırı konuşmadaki duraklama (cümle ortasından kesmez); uzun konuşmada en geç VAD'ın
  max_speech süresinde (10 sn) parça kapanır: gecikmenin üst sınırı.
- Sesin tamamına bakan motorlarda (Zipformer, segmentation="full") parçalar BİTİŞİK: VAD
  yalnızca kesim yerini seçer, aradaki ses atılmaz. Dosya hattındaki gibi: VAD müzik altındaki ya
  da kısık konuşmayı kaçırıyor. 16 Ekim, 2 VietMed kaydı RTP ile: yalnızca VAD parçaları
  çözülünce WER %13,8 / %21,0 (çoğu silinen kelime), aynı kaydın dosyadan çözümü %4,5 / %14,0.
- Ara sonuç (final=False) yalnızca hızlı motorlarda (Zipformer): Whisper'da her ara çözüm
  saniyeler sürer, canlı akışı geride bırakır.
- Konuşmacı ayırma burada yok: telefon santralleri her tarafı ayrı RTP akışı olarak verir
  (taraf = akış). Tek akışta birden çok kişi varsa kesin sonuç görüşme bitince, kaydedilen ses
  normal hattan (konuşmacı ayırma + temiz transkript) geçirilerek alınır (interface/streams.py).
"""
from __future__ import annotations

import logging
import threading
import time
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from viet_asr.domain import StreamSegment, VadSegment, Word

from .ports import AsrEngine, AudioStream, StreamingVad

log = logging.getLogger(__name__)
SR = 16000


@dataclass
class StreamSummary:
    duration: float = 0.0
    segments: int = 0
    words: int = 0
    max_latency: float = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, 95))


class StreamTranscriber:
    def __init__(self, vad_factory: Callable[[], StreamingVad], low_conf: float = 0.5,
                 interim_every: float = 1.5, min_interim: float = 1.0, max_chunk: float = 8.0,
                 min_chunk: float = 5.0):
        self.vad_factory = vad_factory
        self.low_conf = low_conf
        self.interim_every, self.min_interim = interim_every, min_interim
        # bitişik modda VAD kesim vermezse (duraklamasız uzun konuşma) en sessiz yerden zorla kes.
        # 4 VietMed kaydında çevrimdışı benzetim: 15 sn -> WER %11,0, kesin metin en geç 7,1 sn
        # sonra; 8 sn -> WER %10,8, en geç 3,0 sn (kesim parçanın son %40'ında aranır).
        self.max_chunk = max_chunk
        # bitişik modda parça en az bu kadar (dosya hattındaki EnergyChunker min_len): kısa parçada
        # bağlam az, dolgu sesleri daha çok yazılıyor. Bu arada ekranda ara sonuç görünür.
        self.min_chunk = min_chunk

    def run(self, source: AudioStream, asr: AsrEngine, on_segment: Callable[[StreamSegment], None],
            record: Callable[[np.ndarray], None] | None = None, interim: bool = True,
            stop: threading.Event | None = None) -> StreamSummary:
        contiguous = getattr(asr, "segmentation", "vad") == "full"
        try:  # bitişik modda VAD'ın zorla kesmesi (kelime ortası) yerine en sessiz yerden keseriz
            vad = self.vad_factory(max_speech=self.max_chunk * 2) if contiguous else self.vad_factory()
        except TypeError:
            vad = self.vad_factory()
        summary = StreamSummary()
        arrived: list[tuple[int, float]] = []  # (o ana kadar gelen örnek, duvar saati)
        total, index, last_interim = 0, 0, 0.0

        def emit(seg: VadSegment, final: bool) -> None:
            nonlocal index
            words = asr.transcribe([seg]) if len(seg.samples) else []
            end_sample = int(round(seg.end * SR))
            k = bisect_left(arrived, (end_sample, 0.0))
            t_end = arrived[k][1] if k < len(arrived) else time.monotonic()
            latency = max(0.0, time.monotonic() - t_end)
            out = _segment(index, seg, words, final, latency, self.low_conf)
            if final:
                index += 1
                summary.segments += 1
                summary.words += len(words)
                summary.latencies.append(latency)
                summary.max_latency = max(summary.max_latency, latency)
            if out.text or final:
                on_segment(out)

        buf: list[np.ndarray] = []   # bitişik mod: son kesimden bu yana gelen ses
        buf_start = 0                # buf'un akıştaki ilk örneği

        def take(cut: int) -> VadSegment | None:
            """buf'tan [buf_start, cut) aralığını parça yapıp çıkarır."""
            nonlocal buf, buf_start
            if cut <= buf_start:
                return None
            a = np.concatenate(buf) if buf else np.zeros(0, np.float32)
            n = cut - buf_start
            seg = VadSegment(start=buf_start / SR, samples=a[:n])
            buf, buf_start = ([a[n:]] if len(a) > n else []), cut
            return seg

        try:
            for frame in source.frames():
                if stop is not None and stop.is_set():
                    break
                frame = np.asarray(frame, dtype=np.float32)
                if record is not None:
                    record(frame)
                total += len(frame)
                arrived.append((total, time.monotonic()))
                if len(arrived) > 4096:
                    del arrived[:2048]
                if contiguous:
                    buf.append(frame)
                for seg in vad.feed(frame):
                    if contiguous:
                        cut = min(total, int(round(seg.end * SR)))
                        if cut - buf_start < self.min_chunk * SR:
                            continue  # kısa: sonraki duraklamada birlikte kesilir
                        seg = take(cut)
                        if seg is None:
                            continue
                    emit(seg, True)
                    last_interim = total / SR
                if contiguous and total - buf_start > self.max_chunk * SR:
                    emit(take(buf_start + _quietest_cut(np.concatenate(buf))), True)
                    last_interim = total / SR
                if interim and total / SR - last_interim >= self.interim_every:
                    cur = vad.ongoing()
                    if contiguous and cur is not None and buf:
                        cur = VadSegment(start=buf_start / SR, samples=np.concatenate(buf))
                    if cur is not None and cur.end - cur.start >= self.min_interim:
                        emit(cur, False)
                    last_interim = total / SR
            tail = vad.flush()
            if contiguous:
                seg = take(total)
                if seg is not None and seg.end - seg.start >= 0.3:
                    emit(seg, True)
            else:
                for seg in tail:
                    emit(seg, True)
        finally:
            source.close()
        summary.duration = total / SR
        return summary


def _quietest_cut(a: np.ndarray, frame: float = 0.1) -> int:
    """Zorla kesim: parçanın son %40'ındaki en sessiz 100 ms'nin ortası (kelime ortasından
    kesmez; sona yakın kestiği için kesin metin geç kalmaz)."""
    n = int(frame * SR)
    k = len(a) // n
    if k < 2:
        return len(a)
    rms = np.sqrt((a[:k * n].reshape(k, n) ** 2).mean(axis=1))
    lo = int(k * 0.6)
    i = lo + int(np.argmin(rms[lo:]))
    return i * n + n // 2


def _segment(index: int, seg: VadSegment, words: list[Word], final: bool, latency: float,
             low: float) -> StreamSegment:
    confs = [w.conf for w in words if w.conf is not None]
    return StreamSegment(
        index=index, start=round(words[0].start if words else seg.start, 3),
        end=round(max([w.end for w in words] + [seg.end]) if words else seg.end, 3),
        text=" ".join(w.text for w in words), final=final,
        conf=round(sum(confs) / len(confs), 3) if confs else None,
        low_conf=tuple(i for i, w in enumerate(words) if w.conf is not None and w.conf < low),
        latency=round(latency, 3))

"""Silero VAD (sherpa-onnx üzerinden): sessizlikleri atıp konuşma parçalarını bulur."""
from __future__ import annotations

import numpy as np
import sherpa_onnx

from viet_asr.application.config import VadConfig
from viet_asr.domain import VadSegment


class SileroVad:
    def __init__(self, cfg: VadConfig):
        self.cfg = cfg
        mc = sherpa_onnx.VadModelConfig()
        mc.silero_vad.model = cfg.model
        mc.silero_vad.threshold = cfg.threshold
        mc.silero_vad.min_silence_duration = cfg.min_silence_duration
        mc.silero_vad.min_speech_duration = cfg.min_speech_duration
        mc.silero_vad.max_speech_duration = cfg.max_speech_duration
        mc.sample_rate = cfg.sample_rate
        mc.num_threads = 1
        self._model_cfg = mc

    def split(self, audio: np.ndarray) -> list[VadSegment]:
        sr = self.cfg.sample_rate
        # her çağrıda yeni dedektör: içerdeki durum (state) işler arasında taşınmasın
        vad = sherpa_onnx.VoiceActivityDetector(
            self._model_cfg, buffer_size_in_seconds=len(audio) / sr + 5)
        w = self._model_cfg.silero_vad.window_size
        out: list[VadSegment] = []

        def drain():
            while not vad.empty():
                seg = vad.front
                # front.start bir örnek (sample) indeksidir -> saniyeye çevir
                out.append(VadSegment(start=seg.start / sr,
                                      samples=np.asarray(seg.samples, dtype=np.float32),
                                      sample_rate=sr))
                vad.pop()

        for i in range(0, len(audio), w):
            vad.accept_waveform(audio[i:i + w])
            drain()
        vad.flush()
        drain()
        return self._pad(out, audio)

    def _pad(self, segs: list[VadSegment], audio: np.ndarray) -> list[VadSegment]:
        """Parçaları iki yandan genişletir; iki parça arasındaki boşluk ortadan paylaşılır."""
        p = int(self.cfg.pad * self.cfg.sample_rate)
        if p <= 0 or not segs:
            return segs
        sr = self.cfg.sample_rate
        bounds = [(round(s.start * sr), round(s.start * sr) + len(s.samples)) for s in segs]
        out = []
        for i, (a, b) in enumerate(bounds):
            lo = 0 if i == 0 else (bounds[i - 1][1] + a) // 2
            hi = len(audio) if i == len(bounds) - 1 else (b + bounds[i + 1][0]) // 2
            a2, b2 = max(lo, a - p), min(hi, b + p)
            out.append(VadSegment(start=a2 / sr, samples=audio[a2:b2], sample_rate=sr))
        return out


class SileroStreamingVad:
    """Canlı akış için VAD (StreamingVad portu): parça parça beslenir, biten konuşmayı verir.

    Dosya VAD'ından farkları:
      - max_speech_duration akış ayarından (varsayılan 10 sn): uzun konuşmada da en geç bu
        kadar sürede metin çıksın (gecikme üst sınırı).
      - Kenar payı (pad) son `history` sn'lik geçmişten eklenir; parça, konuşma bittikten
        min_silence sonra tamamlandığı için sağ pay için gereken ses elde var.
    """

    def __init__(self, cfg: VadConfig, max_speech: float = 10.0, history: float = 30.0):
        self.cfg = cfg
        mc = sherpa_onnx.VadModelConfig()
        mc.silero_vad.model = cfg.model
        mc.silero_vad.threshold = cfg.threshold
        mc.silero_vad.min_silence_duration = cfg.min_silence_duration
        mc.silero_vad.min_speech_duration = cfg.min_speech_duration
        mc.silero_vad.max_speech_duration = max_speech
        mc.sample_rate = cfg.sample_rate
        mc.num_threads = 1
        self.vad = sherpa_onnx.VoiceActivityDetector(mc, buffer_size_in_seconds=max_speech + 30)
        self.w = mc.silero_vad.window_size
        self.sr = cfg.sample_rate
        self.hist = np.zeros(int(history * self.sr), np.float32)
        self.total = 0            # şimdiye kadar gelen örnek sayısı
        self._carry = np.zeros(0, np.float32)
        self._last_end = 0        # son verilen parçanın bitişi (örnek): pay buna taşmasın

    def _remember(self, x: np.ndarray) -> None:
        """Halka tampon: örnek i, hist[i % N] konumunda (kopyalama yok)."""
        N, n = len(self.hist), len(x)
        if n >= N:
            x, self.total = x[-N:], self.total + n - N
            n = N
        i = self.total % N
        k = min(n, N - i)
        self.hist[i:i + k] = x[:k]
        self.hist[:n - k] = x[k:]
        self.total += n

    def _slice(self, a: int, b: int) -> np.ndarray:
        N = len(self.hist)
        a, b = max(a, self.total - N, 0), min(b, self.total)
        if b <= a:
            return np.zeros(0, np.float32)
        idx = np.arange(a, b) % N
        return self.hist[idx]

    def _drain(self) -> list[VadSegment]:
        out = []
        p = int(self.cfg.pad * self.sr)
        while not self.vad.empty():
            seg = self.vad.front
            a, b = seg.start, seg.start + len(seg.samples)
            a2, b2 = max(self._last_end, a - p), min(self.total, b + p)
            samples = self._slice(a2, b2)
            if len(samples) < b2 - a2:  # geçmiş yetmedi (çok uzun parça): VAD'ın verdiği kalsın
                a2, samples = a, np.asarray(seg.samples, dtype=np.float32)
            out.append(VadSegment(start=a2 / self.sr, samples=samples, sample_rate=self.sr))
            self._last_end = max(self._last_end, b2)
            self.vad.pop()
        return out

    def feed(self, samples: np.ndarray) -> list[VadSegment]:
        x = np.concatenate([self._carry, np.asarray(samples, dtype=np.float32)])
        n = len(x) // self.w * self.w
        self._carry = x[n:]
        out = []
        for i in range(0, n, self.w):
            chunk = x[i:i + self.w]
            self._remember(chunk)
            self.vad.accept_waveform(chunk)
            out += self._drain()
        return out

    def ongoing(self) -> VadSegment | None:
        if not self.vad.is_speech_detected():
            return None
        cs = self.vad.current_segment
        if cs is None or not len(cs.samples):
            return None
        a = max(self._last_end, cs.start - int(self.cfg.pad * self.sr))
        return VadSegment(start=a / self.sr, samples=self._slice(a, self.total), sample_rate=self.sr)

    def flush(self) -> list[VadSegment]:
        if len(self._carry):
            self._remember(self._carry)
            self.vad.accept_waveform(self._carry)
            self._carry = np.zeros(0, np.float32)
        self.vad.flush()
        return self._drain()

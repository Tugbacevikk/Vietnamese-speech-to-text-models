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
        return out

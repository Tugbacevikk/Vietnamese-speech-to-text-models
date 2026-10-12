"""Sesin TAMAMINI kapsayan parçalama: kesimler en sessiz anlara konur.

Neden? VAD (Silero) müzik altındaki ya da gürültülü konuşmayı "konuşma değil"
diye atıyor; atılan kısım ASR'a hiç ulaşmıyor. VietMed'den 50 dosyada (77 dk,
12 Ekim) Zipformer-68M:
    VAD eşik 0,5      WER %20,8  (eksik hece %13,1)
    VAD eşik 0,2      WER %15,6  (eksik %7,3)
    VAD yok, 15 sn    WER %10,8  (eksik %1,9; sessizliğe uydurma yalnızca +%1,1)
Zipformer ve Omnilingual sessizlik/müzikte metin uydurmuyor, bu yüzden onlara
sesin tamamı verilir. Whisper uydurduğu için ona VAD parçaları verilmeye devam eder.

Yöntem: her parça en fazla max_len sn; kesim noktası [min_len, max_len] aralığındaki
en düşük enerjili 100 ms'lik pencerenin ortası. Tamamen sessiz parçalar atlanır.
"""
from __future__ import annotations

import numpy as np

from viet_asr.domain import VadSegment

SR = 16000


class EnergyChunker:
    def __init__(self, max_len: float = 15.0, min_len: float = 5.0, frame: float = 0.1,
                 silence_rms: float = 1e-3):
        self.max_len, self.min_len, self.frame, self.silence_rms = max_len, min_len, frame, silence_rms

    def split(self, audio: np.ndarray) -> list[VadSegment]:
        n, f = len(audio), int(self.frame * SR)
        if n == 0:
            return []
        frames = n // f
        energy = (np.sqrt(np.mean(audio[: frames * f].reshape(frames, f) ** 2, axis=1))
                  if frames else np.zeros(0))
        out, start = [], 0
        while start < n:
            if n - start <= self.max_len * SR:
                end = n
            else:
                lo = (start + int(self.min_len * SR)) // f
                hi = min(frames, (start + int(self.max_len * SR)) // f)
                k = lo + int(np.argmin(energy[lo:hi])) if hi > lo else hi
                end = k * f + f // 2
            chunk = audio[start:end]
            if float(np.sqrt(np.mean(chunk ** 2))) >= self.silence_rms:
                out.append(VadSegment(start=start / SR, samples=chunk, sample_rate=SR))
            start = end
        return out

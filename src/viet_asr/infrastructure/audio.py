"""Ses dosyasını 16 kHz mono float32 diziye çevirir."""
from __future__ import annotations

import numpy as np
import soundfile as sf

TARGET_SR = 16000


class SoundfileLoader:
    def load(self, path: str) -> np.ndarray:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)  # stereo -> mono
        if sr != TARGET_SR:
            import soxr  # yalnızca gerekirse yüklenir

            audio = soxr.resample(audio, sr, TARGET_SR).astype(np.float32)
        return np.ascontiguousarray(audio)

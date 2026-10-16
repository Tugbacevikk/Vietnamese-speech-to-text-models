"""Konuşmacının ses perdesi (temel frekans, F0) ve buradan kaba cinsiyet tahmini.

Neden metinden değil sesten: "sunucu erkektir" gibi bir varsayım yanlış etiket üretir; ses
perdesi ölçülebilir bir özelliktir. Yetişkin erkek konuşmada F0 çoğunlukla 85-155 Hz, kadında
165-255 Hz. Arası (ve çocuk sesi, falsetto, telefon kaydı) belirsiz sayılır, etikete yazılmaz.

Yöntem: 40 ms çerçevelerde normalize öz-ilinti (autocorrelation), 70-350 Hz arasında en
yüksek tepe. Sessiz ve sessiz-harf çerçeveleri (enerji ya da tepe düşük) atılır; konuşmacının
F0'ı geçerli çerçevelerin ortancası (oktav hatalarına dayanıklı). Model gerekmez, CPU'da
dakikalık seste ~0,1 sn.
"""
from __future__ import annotations

import numpy as np

from viet_asr.domain import BELIRSIZ, SpeakerInfo, SpeakerSegment

SR = 16000
FRAME, HOP = 0.040, 0.010
FMIN, FMAX = 70.0, 350.0
VOICED_PEAK = 0.45      # normalize öz-ilinti tepesi bunun altındaysa çerçeve sessiz/gürültü
MALE_MAX, FEMALE_MIN = 155.0, 185.0
MIN_VOICED_SEC = 2.0    # bundan az ötümlü ses varsa tahmin yapılmaz


def frame_f0(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Her ötümlü çerçevenin F0'ı (Hz); ötümsüz çerçeveler atılmış dizi."""
    n, h = int(FRAME * sr), int(HOP * sr)
    if len(audio) < n:
        return np.empty(0)
    idx = np.arange(0, len(audio) - n + 1, h)
    frames = np.stack([audio[i:i + n] for i in idx]).astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    loud = rms > max(1e-4, 0.25 * np.median(rms[rms > 0])) if np.any(rms > 0) else rms > 1
    frames = frames[loud]
    if not len(frames):
        return np.empty(0)
    frames *= np.hanning(n)
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(frames, size, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), size, axis=1)[:, :n]
    ac /= ac[:, :1] + 1e-12
    lo, hi = int(sr / FMAX), int(sr / FMIN)
    seg = ac[:, lo:hi + 1]
    k = seg.argmax(axis=1)
    peak = seg[np.arange(len(seg)), k]
    lag = (k + lo).astype(np.float64)
    # parabolik ara değer: lag çözünürlüğünden (1/16000 sn) daha ince F0
    inner = (k > 0) & (k < seg.shape[1] - 1)
    a, b, c = (seg[inner, k[inner] - 1], seg[inner, k[inner]], seg[inner, k[inner] + 1])
    den = a - 2 * b + c
    lag[inner] += np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    ok = peak >= VOICED_PEAK
    return sr / lag[ok]


def gender_of(f0: float | None) -> str | None:
    if f0 is None:
        return None
    if f0 <= MALE_MAX:
        return "erkek"
    if f0 >= FEMALE_MIN:
        return "kadın"
    return None


class PitchProfiler:
    """Konuşmacı başına ortanca F0 ve kaba cinsiyet (SpeakerProfiler portu)."""

    def __init__(self, sample_rate: int = SR):
        self.sr = sample_rate

    def profile(self, audio: np.ndarray, speakers: list[SpeakerSegment]) -> list[SpeakerInfo]:
        parts: dict[str, list[np.ndarray]] = {}
        for s in speakers:
            if s.speaker == BELIRSIZ:
                continue
            a, b = int(s.start * self.sr), int(s.end * self.sr)
            if b > a:
                parts.setdefault(s.speaker, []).append(audio[a:b])
        out = []
        for spk in sorted(parts):
            f0s = np.concatenate([frame_f0(p, self.sr) for p in parts[spk]] or [np.empty(0)])
            voiced = len(f0s) * HOP
            f0 = float(np.median(f0s)) if voiced >= MIN_VOICED_SEC else None
            out.append(SpeakerInfo(spk, round(f0, 1) if f0 else None, gender_of(f0), round(voiced, 1)))
        return out

import numpy as np

from viet_asr.infrastructure.sherpa_asr import band_limit


def test_band_limit_keeps_length_and_cuts_highs():
    t = np.arange(16000) / 16000
    x = (np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    y = band_limit(x, 16000, 8000)
    assert len(y) == len(x) and y.dtype == np.float32
    spec = np.abs(np.fft.rfft(y))
    assert spec[500] > 100 and spec[6000] < 5   # 6 kHz kesildi, 500 Hz kaldı


def test_band_limit_off():
    x = np.ones(100, np.float32)
    assert band_limit(x, 16000, 0) is x

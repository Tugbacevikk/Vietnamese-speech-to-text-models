"""Rust birleştirme (viet_asr_native) Python başvuru sürümüyle birebir aynı mı?

Modül kurulu değilse atlanır: pip install ./native
"""
import random

import pytest

from viet_asr.application.config import FusionConfig
from viet_asr.application.fusion import FusionEngine
from viet_asr.domain import SpeakerSegment, Word

native = pytest.importorskip("viet_asr_native")


def random_case(rng: random.Random):
    n_spk = rng.randint(2, 4)
    segs, t = [], 0.0
    for _ in range(rng.randint(1, 25)):
        t += rng.choice([0.0, 0.0, rng.uniform(0, 1.5)])
        dur = rng.choice([0.5, 1.0, rng.uniform(0.1, 4.0)])
        start = max(0.0, t - rng.choice([0.0, 0.0, rng.uniform(0, 1.0)]))  # üst üste binme de olsun
        segs.append(SpeakerSegment(f"speaker_{rng.randrange(n_spk):02d}", round(start, 2), round(start + dur, 2)))
        t = start + dur
    words, t = [], 0.0
    for _ in range(rng.randint(0, 60)):
        t += rng.uniform(0, 0.6)
        d = rng.choice([0.0, 0.2, rng.uniform(0, 0.8)])
        words.append(Word("x", round(t, 2), round(t + d, 2)))
    return words, segs


def test_rastgele_girdilerde_python_ile_ayni():
    from viet_asr.infrastructure.native_fusion import NativeFusionEngine

    rng = random.Random(7)
    for cfg in (FusionConfig(), FusionConfig(min_dominance=0.5, max_gap=0.0), FusionConfig(max_gap=3.0)):
        py, rs = FusionEngine(cfg), NativeFusionEngine(cfg)
        for _ in range(3000):
            words, segs = random_case(rng)
            assert rs.assign(words, segs) == py.assign(words, segs), (words, segs)


def test_tek_konusmaci_ve_bos():
    from viet_asr.infrastructure.native_fusion import NativeFusionEngine

    e = NativeFusionEngine()
    w = [Word("a", 0, 1)]
    assert e.assign(w, []) == ["speaker_00"]
    assert e.assign(w, [SpeakerSegment("speaker_03", 5, 6)]) == ["speaker_03"]
    assert e.assign([], [SpeakerSegment("speaker_00", 0, 1), SpeakerSegment("speaker_01", 1, 2)]) == []
    with pytest.raises(ValueError):
        native.assign_labels([0.0], [], [], [], [], 0.6, 0.5)


def test_enerji_bolucu_python_ile_ayni():
    import numpy as np

    from viet_asr.infrastructure.chunker import energy_cuts

    rng = np.random.default_rng(3)
    for case in range(300):
        n = int(rng.integers(0, 16000 * 70))
        a = (rng.standard_normal(n) * rng.choice([0.0005, 0.01, 0.2])).astype(np.float32)
        for _ in range(int(rng.integers(0, 6))):                      # sessiz bölgeler (tam sıfır)
            s = int(rng.integers(0, max(1, n)))
            a[s:s + int(rng.integers(0, 16000 * 3))] = 0
        args = (16000, float(rng.choice([10.0, 15.0, 20.0])), 5.0, 0.1, 1e-3)
        assert [tuple(c) for c in native.energy_cuts(a, *args)] == energy_cuts(a, *args), case


def test_enerji_bolucu_secimi():
    import numpy as np

    from viet_asr.infrastructure.chunker import EnergyChunker

    a = np.full(16000 * 40, 0.1, dtype=np.float32)
    py, rs = EnergyChunker(native=False), EnergyChunker(native=True)
    assert rs.native and [(s.start, len(s.samples)) for s in py.split(a)] == \
        [(s.start, len(s.samples)) for s in rs.split(a)]

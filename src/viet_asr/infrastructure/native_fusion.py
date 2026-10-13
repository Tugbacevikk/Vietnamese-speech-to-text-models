"""Birleştirmenin Rust sürümü (native/ klasörü, viet_asr_native modülü).

Rust'a geçişin ilk parçası: kural application/fusion.py'deki Python sürümüyle birebir
aynıdır (başvuru kaynağı o); tests/unit/test_native.py rastgele girdilerle ikisini
karşılaştırır. Modül kurulu değilse ya da VIET_ASR_NATIVE=0 ise Python sürümü çalışır,
yani Rust derleyicisi olmayan bilgisayarda da sistem aynen çalışır.

    pip install ./native        # Rust (cargo) gerekir; bir kez
"""
from __future__ import annotations

import os

from viet_asr.application.config import FusionConfig
from viet_asr.application.fusion import FusionEngine
from viet_asr.domain import BELIRSIZ, SpeakerSegment, Word

try:
    import viet_asr_native as _native
except ImportError:  # pragma: no cover - kurulum durumuna bağlı
    _native = None


def native_available() -> bool:
    return _native is not None and os.environ.get("VIET_ASR_NATIVE", "1") != "0"


class NativeFusionEngine(FusionEngine):
    """Etiketleme (assign) Rust'ta; satırlara toplama (merge) Python'da (ucuz)."""

    def __init__(self, config: FusionConfig | None = None):
        if _native is None:
            raise ImportError("viet_asr_native kurulu değil: pip install ./native")
        super().__init__(config)

    def assign(self, words: list[Word], speakers: list[SpeakerSegment]) -> list[str]:
        distinct = {s.speaker for s in speakers}
        if len(distinct) <= 1:  # Python sürümüyle aynı kısa yol
            return super().assign(words, speakers)
        names = sorted(distinct)
        idx = {n: i for i, n in enumerate(names)}
        labels = _native.assign_labels(
            [w.start for w in words], [w.end for w in words],
            [s.start for s in speakers], [s.end for s in speakers],
            [idx[s.speaker] for s in speakers],
            self.cfg.min_dominance, self.cfg.max_gap)
        return [names[k] if k >= 0 else BELIRSIZ for k in labels]


def build_fusion(config: FusionConfig) -> FusionEngine:
    return NativeFusionEngine(config) if native_available() else FusionEngine(config)

"""Konuşmacı ayrıştırma: VAD parçaları -> kısa pencereler -> ses imzası -> kümeleme.

Gerçek ses (13 Ekim, VietMed'den 56 diyalog, CAM++ imzası, eşik 0,70): konuşmacı
sayısı 47/56 doğru, cümlelerin %91,7'si doğru kişide; sayı istekte verilirse %89,9.
Aşağıdaki sentetik ölçümler ilk ayardan (ERes2Net, eşik 0,65) kalmadır.

Neden pyannote segmentation yerine bu? (12 Ekim ölçümü, 14 sentetik diyalog,
1-3 konuşmacı, 1,8-2,8 sn'lik konuşma sıraları):
  - pyannote seg-3.0 + kümeleme: kısa sıralarla konuşan iki Vietnamca sesi
    "tam 2 kişi" diye zorlansa bile ayıramadı (cümleleri zamana göre böldü);
    otomatik sayıda 14'te en fazla 11 doğru. Sorun konuşmacı değişimini bulan
    adımda: ses imzası modeli aynı cümleleri rahatça ayırıyor.
  - bu yöntem (3D-Speaker imzası, 1,5 sn pencere, 0,75 sn kayma, eşik 0,65),
    3 farklı rastgele karışımda:
      duraksamalı sıralar:   42/42 doğru sayı, cümlelerin %100'ü doğru kişide
      duraksamasız sıralar:  41/42 doğru sayı, cümlelerin %99,4-100'ü
    hız RTF ~0,04 (pyannote yolu ~0,15).
Sınırları: iki kişinin aynı anda konuştuğu yerleri ayırmaz; konuşmacı değişimini
~0,75 sn çözünürlükle yakalar. Ölçüm sentetik seslerle: gerçek kayıtlarla
(benchmarks/konusmaci_testi.py) doğrulanmalı.

Konuşmacı sayısı biliniyorsa (num_speakers) küme ağacı tam o sayıda kesilir;
bilinmiyorsa uzaklık eşiğiyle kendisi bulur.
"""
from __future__ import annotations

import numpy as np
import sherpa_onnx
from scipy.cluster.hierarchy import fcluster, linkage

from viet_asr.application.config import DiarizerConfig
from viet_asr.domain import SpeakerSegment, VadSegment

SR = 16000


class EmbeddingClusterDiarizer:
    def __init__(self, cfg: DiarizerConfig):
        self.cfg = cfg
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=cfg.embedding_model, num_threads=cfg.num_threads))

    # --- adımlar ---
    def _windows(self, segments: list[VadSegment]) -> list[tuple[float, float, np.ndarray]]:
        """Uzun parçaları kayan pencerelere böl; her pencerenin 'kendi' aralığı
        komşu pencerelerle örtüşmenin ortasından kesilir (çakışmasız)."""
        w, h = int(self.cfg.window * SR), int(self.cfg.hop * SR)
        out = []
        for seg in segments:
            n = len(seg.samples)
            if n < int(self.cfg.min_window * SR):
                continue  # çok kısa: imzası güvenilmez, kelimeler en yakın konuşmacıya gider
            if n <= w:
                out.append((seg.start, seg.end, seg.samples))
                continue
            starts = list(range(0, n - w + 1, h))
            if starts[-1] + w < n:
                starts.append(n - w)
            for k, s in enumerate(starts):
                own0 = seg.start + (s if k == 0 else (s + starts[k - 1] + w) / 2) / SR
                own1 = seg.start + (s + w if k == len(starts) - 1 else (s + w + starts[k + 1]) / 2) / SR
                out.append((own0, own1, seg.samples[s:s + w]))
        return out

    def _embed(self, x: np.ndarray) -> np.ndarray:
        st = self.extractor.create_stream()
        st.accept_waveform(SR, x)
        st.input_finished()
        e = np.asarray(self.extractor.compute(st), dtype=np.float64)
        return e / (np.linalg.norm(e) + 1e-9)

    def _cluster(self, E: np.ndarray, durations: np.ndarray, k: int | None) -> np.ndarray:
        if len(E) == 1 or k == 1:
            return np.zeros(len(E), dtype=int)
        Z = linkage(E, method="average", metric="cosine")
        if k and k > 1:
            lab = fcluster(Z, t=min(k, len(E)), criterion="maxclust") - 1
        else:
            lab = fcluster(Z, t=self.cfg.cluster_threshold, criterion="distance") - 1
        # toplam süresi çok kısa kümeleri (gürültü, öksürük) en yakın büyük kümeye kat
        tot = np.bincount(lab, weights=durations)
        if k and k > 1:
            big = [int(c) for c in np.argsort(-tot)[:k]]  # sayı verildiyse en büyük k küme kalır
        else:
            need = max(self.cfg.min_speaker_duration, self.cfg.min_speaker_share * durations.sum())
            big = [c for c in range(len(tot)) if tot[c] >= need]
        if not big:
            return np.zeros(len(E), dtype=int)
        cents = {c: E[lab == c].mean(axis=0) for c in big}
        for c in set(lab) - set(big):
            idx = lab == c
            m = E[idx].mean(axis=0)
            lab[idx] = max(big, key=lambda b: float(m @ cents[b]) / (np.linalg.norm(cents[b]) + 1e-9))
        return lab

    def embed(self, segments: list[VadSegment]):
        """Pencereler ve imzaları (ayarlama betikleri bunu bir kez hesaplayıp tekrar kullanır)."""
        wins = self._windows(segments)
        E = np.stack([self._embed(x) for _, _, x in wins]) if wins else np.zeros((0, 1))
        return [(a, b) for a, b, _ in wins], E

    def label(self, spans: list[tuple[float, float]], E: np.ndarray,
              num_speakers: int | None = None) -> list[SpeakerSegment]:
        if not spans:
            return []
        dur = np.array([b - a for a, b in spans])
        k = num_speakers if num_speakers and num_speakers > 0 else (
            self.cfg.num_speakers if self.cfg.num_speakers > 0 else None)
        lab = self._cluster(E, dur, k)
        # komşu aynı etiketli pencereleri birleştir, etiketleri konuşma sırasına göre adlandır
        names: dict[int, str] = {}
        out: list[SpeakerSegment] = []
        for (a, b), l in zip(spans, lab):
            name = names.setdefault(int(l), f"speaker_{len(names):02d}")
            if out and out[-1].speaker == name and a - out[-1].end < 1e-3:
                out[-1] = SpeakerSegment(name, out[-1].start, b)
            else:
                out.append(SpeakerSegment(name, a, b))
        return out

    def diarize(self, audio: np.ndarray, segments: list[VadSegment] | None = None,
                num_speakers: int | None = None) -> list[SpeakerSegment]:
        if not segments:
            return []
        spans, E = self.embed(segments)
        return self.label(spans, E, num_speakers)

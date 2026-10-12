"""Uygulama katmanının dış dünyadan beklediği arayüzler (portlar).

Altyapı katmanı bunları gerçekler (sherpa-onnx, PostgreSQL, webhook...).
Uygulama katmanı yalnızca bu arayüzleri bilir; böylece model değişince
sadece altyapıdaki bir sınıf değişir.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from viet_asr.domain import (
    AudioJob, JobRecord, JobStatus, LanguageDecision, SpeakerSegment, Transcript, VadSegment, Word,
)


class Vad(Protocol):
    def split(self, audio: np.ndarray) -> list[VadSegment]: ...


class AsrEngine(Protocol):
    name: str
    segmentation: str  # "full" (sesin tamamı) ya da "vad" (yalnızca konuşma parçaları)

    def transcribe(self, segments: list[VadSegment]) -> list[Word]: ...


# dil kodu -> o dilin ASR motoru. Gerçeklemesi motoru ilk kullanımda yükleyebilir.
AsrEngines = Mapping[str, AsrEngine]


class LanguageDetector(Protocol):
    def probabilities(self, audio: np.ndarray) -> dict[str, float]:
        """Modelin bildiği TÜM diller için olasılık (toplamı 1). Kısıtlama uygulamada yapılır."""
        ...


class Diarizer(Protocol):
    def diarize(self, audio: np.ndarray, segments: list[VadSegment] | None = None,
                num_speakers: int | None = None) -> list[SpeakerSegment]:
        """num_speakers verilirse tam o kadar konuşmacı; None ise sesten bulunur."""
        ...


class TextNormalizer(Protocol):
    def normalize(self, text: str) -> str: ...


class AudioLoader(Protocol):
    def load(self, path: str) -> np.ndarray:
        """16 kHz mono float32 döndürür."""
        ...


class ResultRepository(Protocol):
    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None: ...

    def save(self, transcript: Transcript) -> None:
        """Sonucu yazar ve işi DONE yapar; ikisi birlikte olur ya da hiçbiri olmaz."""
        ...


class JobStore(ResultRepository, Protocol):
    """İş kayıtlarının tamamı. Gerçeklemesi (PostgreSQL) sonuç ile bildirim
    olayını (outbox) aynı işlemde yazar: sonuç kaydedildiyse bildirim kaybolmaz."""

    def create(self, job: AudioJob) -> None: ...

    def get(self, job_id: str) -> JobRecord | None: ...

    def set_language(self, job_id: str, decision: LanguageDecision) -> None: ...

    def get_transcript(self, job_id: str) -> Transcript | None: ...


class JobQueue(Protocol):
    """İşi doğru kuyruğa koyar: dil belliyse o dilin ASR kuyruğu, değilse dil tespiti."""

    def enqueue_detection(self, job_id: str) -> None: ...

    def enqueue_transcription(self, job_id: str, language: str) -> None: ...


class Notifier(Protocol):
    def notify(self, job: AudioJob, transcript: Transcript) -> None: ...

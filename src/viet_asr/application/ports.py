"""Uygulama katmanının dış dünyadan beklediği arayüzler (portlar).

Altyapı katmanı bunları gerçekler (sherpa-onnx, PostgreSQL, webhook...).
Uygulama katmanı yalnızca bu arayüzleri bilir; böylece model değişince
sadece altyapıdaki bir sınıf değişir.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from viet_asr.domain import AudioJob, JobStatus, SpeakerSegment, Transcript, VadSegment, Word


class Vad(Protocol):
    def split(self, audio: np.ndarray) -> list[VadSegment]: ...


class AsrEngine(Protocol):
    name: str

    def transcribe(self, segments: list[VadSegment]) -> list[Word]: ...


# dil kodu -> o dilin ASR motoru. Gerçeklemesi motoru ilk kullanımda yükleyebilir.
AsrEngines = Mapping[str, AsrEngine]


class LanguageDetector(Protocol):
    def probabilities(self, audio: np.ndarray) -> dict[str, float]:
        """Modelin bildiği TÜM diller için olasılık (toplamı 1). Kısıtlama uygulamada yapılır."""
        ...


class Diarizer(Protocol):
    def diarize(self, audio: np.ndarray) -> list[SpeakerSegment]: ...


class TextNormalizer(Protocol):
    def normalize(self, text: str) -> str: ...


class AudioLoader(Protocol):
    def load(self, path: str) -> np.ndarray:
        """16 kHz mono float32 döndürür."""
        ...


class ResultRepository(Protocol):
    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None: ...

    def save(self, transcript: Transcript) -> None: ...


class Notifier(Protocol):
    def notify(self, job: AudioJob, transcript: Transcript) -> None: ...

"""Çekirdek veri tipleri. Hiçbir dış kütüphaneye bağlı değildir.

Zaman birimi her yerde saniyedir (float), ses dosyasının başından itibaren.
Vietnamcada metin birimi hecedir: boşlukla ayrılan her parça bir hecedir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

BELIRSIZ = "BELIRSIZ"  # konuşmacısı güvenle belirlenemeyen hece


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class AudioJob:
    """Bir ses dosyasını işleme isteği."""

    job_id: str
    audio_path: str
    callback_url: str | None = None


@dataclass(frozen=True)
class VadSegment:
    """VAD'ın bulduğu konuşma parçası. samples 16 kHz mono float32."""

    start: float
    samples: np.ndarray = field(repr=False)
    sample_rate: int = 16000

    @property
    def end(self) -> float:
        return self.start + len(self.samples) / self.sample_rate


@dataclass(frozen=True)
class Syllable:
    """ASR çıktısındaki tek hece, mutlak zamanlarla."""

    text: str
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True)
class SpeakerSegment:
    """Diarization çıktısı: [start, end) aralığında konuşan."""

    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class Utterance:
    """Aynı konuşmacının art arda gelen heceleri (son çıktının bir satırı)."""

    speaker: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    job_id: str
    duration: float
    utterances: list[Utterance]
    model: str

    @property
    def text(self) -> str:
        return " ".join(u.text for u in self.utterances)

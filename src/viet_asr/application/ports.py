"""Uygulama katmanının dış dünyadan beklediği arayüzler (portlar).

Altyapı katmanı bunları gerçekler (sherpa-onnx, PostgreSQL, webhook...).
Uygulama katmanı yalnızca bu arayüzleri bilir; böylece model değişince
sadece altyapıdaki bir sınıf değişir.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import datetime
from typing import Protocol

import numpy as np

from viet_asr.domain import (
    AudioJob, Correction, JobRecord, JobStatus, LanguageDecision, ReviewLine, SpeakerInfo, SpeakerSegment,
    Transcript,
    Utterance, VadSegment, Word,
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


class AudioStream(Protocol):
    """Canlı ses kaynağı (RTP, ffmpeg ile URL/SDP/dosya): 16 kHz mono float32 parçalar."""

    def frames(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class StreamingVad(Protocol):
    """Parça parça beslenen VAD. Zamanlar akışın başından itibaren sn."""

    def feed(self, samples: np.ndarray) -> list[VadSegment]:
        """Biten konuşma parçaları (varsa)."""
        ...

    def ongoing(self) -> VadSegment | None:
        """Henüz bitmemiş konuşma (ara sonuç için); konuşma yoksa None."""
        ...

    def flush(self) -> list[VadSegment]: ...


class SpeakerProfiler(Protocol):
    def profile(self, audio: np.ndarray, speakers: list[SpeakerSegment]) -> list[SpeakerInfo]:
        """Konuşmacı başına sesten ölçülenler (ses perdesi, kaba cinsiyet)."""
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

    def set_progress(self, job_id: str, stage: str, progress: float) -> None:
        """İşlenirken aşama ve yaklaşık ilerleme (0..1)."""
        ...

    def save(self, transcript: Transcript) -> None:
        """Sonucu yazar ve işi DONE yapar; ikisi birlikte olur ya da hiçbiri olmaz."""
        ...


class JobStore(ResultRepository, Protocol):
    """İş kayıtlarının tamamı. Gerçeklemesi (PostgreSQL) sonuç ile bildirim
    olayını (outbox) aynı işlemde yazar: sonuç kaydedildiyse bildirim kaybolmaz."""

    def create(self, job: AudioJob) -> str:
        """Kaydeder, geçerli job_id'yi döndürür. external_id daha önce kullanıldıysa
        yeni iş açılmaz; var olan işin kimliği döner."""
        ...

    def get(self, job_id: str) -> JobRecord | None: ...

    def set_language(self, job_id: str, decision: LanguageDecision) -> None: ...

    def get_transcript(self, job_id: str) -> Transcript | None: ...

    def recent_jobs(self, limit: int = 20) -> list[JobRecord]:
        """En yeni işler (arayüzdeki geçmiş listesi)."""
        ...

    def update_utterance(self, job_id: str, idx: int, speaker: str | None = None,
                         text: str | None = None) -> Utterance | None:
        """Biten işin bir satırını düzeltir; modelin ilk hali saklanır. Yoksa None."""
        ...

    def reopen(self, job_id: str, num_speakers: int | None) -> JobRecord | None:
        """Bitmiş işi yeniden analiz için QUEUED yapar; olmazsa None."""
        ...

    def delete_utterance(self, job_id: str, idx: int) -> bool:
        """Satırı sonuçtan kaldırır (kayıtta kalır). Olmazsa False."""
        ...

    def corrections(self, limit: int = 10_000, consented_only: bool = False) -> list[Correction]:
        """Düzeltilmiş satırlar (en yenisi önce): ince ayar verisi.
        consented_only: yalnızca veri setine aktarma onayı verilmiş işlerden."""
        ...

    def audit(self, actor: str, action: str, job_id: str | None = None, detail: str | None = None,
              ip: str | None = None) -> None:
        """Erişim kaydı: kim, hangi işte, ne yaptı."""
        ...

    def queue_position(self, job_id: str) -> int | None:
        """Kuyruktaki işin önünde bekleyen iş sayısı; kuyrukta değilse None."""
        ...

    def review_lines(self, since: datetime) -> list[ReviewLine]:
        """since'ten sonra açılmış ve kişinin elden geçirdiği işlerin tüm satırları (kalite takibi)."""
        ...

    def set_consent(self, job_id: str, by: str, use: str) -> bool:
        """Bitmiş işe veri seti onayı (kim, test|egitim) yazar. Bitmemişse False."""
        ...

    def add_event(self, event: str, url: str, payload: dict, job_id: str | None = None) -> None:
        """İşten bağımsız bildirim (canlı oturum sonu 'live.ended'); outbox'tan imzalı gider."""
        ...

    def delete_job(self, job_id: str) -> str | None:
        """İşi, sonucunu ve bekleyen bildirimlerini siler; ses dosyasının yolunu döndürür
        (silinecekse çağıran siler). İş yoksa None."""
        ...

    def fail(self, job_id: str, error: str) -> None:
        """Son karar: iş FAILED ve (callback varsa) 'job.failed' bildirimi tek işlemde.
        set_status(FAILED) yalnızca kaydeder; tekrar denenecek ara hatalar bildirim üretmez."""
        ...


class JobQueue(Protocol):
    """İşi doğru kuyruğa koyar: dil belliyse o dilin ASR kuyruğu, değilse dil tespiti."""

    def enqueue_detection(self, job_id: str) -> None: ...

    def enqueue_transcription(self, job_id: str, language: str) -> None: ...


class Notifier(Protocol):
    def notify(self, job: AudioJob, transcript: Transcript) -> None: ...


class SpeechDetector(Protocol):
    """512 örneklik (16 kHz'de 32 ms) çerçevede konuşma var mı? (canlı altyazı)"""

    def __call__(self, frame: np.ndarray) -> bool: ...

    def reset(self) -> None: ...


class LiveRecorder(Protocol):
    """Canlı oturumun sesini ve kesinleşen metnini anında diske yazar (çökmeye dayanıklı)."""

    name: str

    @property
    def audio_path(self) -> str: ...

    def write_audio(self, samples: np.ndarray) -> None: ...

    def write_final(self, item: dict) -> None: ...

    def close(self) -> None: ...

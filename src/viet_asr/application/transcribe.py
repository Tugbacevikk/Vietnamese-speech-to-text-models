"""Ana kullanım senaryosu: ses -> konuşmacılı metin."""
from __future__ import annotations

import logging
import time

from viet_asr.domain import AudioJob, JobStatus, Transcript

from .fusion import FusionEngine
from .ports import AsrEngine, AudioLoader, Diarizer, Notifier, ResultRepository, Vad

log = logging.getLogger(__name__)


class TranscribeAndDiarize:
    def __init__(self, loader: AudioLoader, vad: Vad, asr: AsrEngine, diarizer: Diarizer,
                 fusion: FusionEngine, repo: ResultRepository, notifier: Notifier | None = None):
        self.loader, self.vad, self.asr, self.diarizer = loader, vad, asr, diarizer
        self.fusion, self.repo, self.notifier = fusion, repo, notifier

    def run(self, job: AudioJob) -> Transcript:
        self.repo.set_status(job.job_id, JobStatus.RUNNING)
        try:
            transcript = self._process(job)
        except Exception as e:
            self.repo.set_status(job.job_id, JobStatus.FAILED, error=f"{type(e).__name__}: {e}")
            raise
        self.repo.save(transcript)
        self.repo.set_status(job.job_id, JobStatus.DONE)
        if self.notifier is not None:
            self.notifier.notify(job, transcript)
        return transcript

    def _process(self, job: AudioJob) -> Transcript:
        t0 = time.perf_counter()
        audio = self.loader.load(job.audio_path)
        duration = len(audio) / 16000

        segments = self.vad.split(audio)
        syllables = self.asr.transcribe(segments) if segments else []
        speakers = self.diarizer.diarize(audio) if segments else []
        utterances = self.fusion.fuse(syllables, speakers)

        dt = time.perf_counter() - t0
        log.info("job=%s sure=%.1fs rtf=%.3f parca=%d hece=%d satir=%d", job.job_id, duration,
                 dt / max(duration, 1e-9), len(segments), len(syllables), len(utterances))
        return Transcript(job_id=job.job_id, duration=duration, utterances=utterances,
                          model=self.asr.name)

"""Ana kullanım senaryosu: ses -> dil kararı -> konuşmacılı metin."""
from __future__ import annotations

import logging
import time

from viet_asr.domain import AudioJob, JobStatus, LanguageDecision, Transcript, UnsupportedLanguageError

from .fusion import FusionEngine
from .language import LanguageResolver
from .ports import AsrEngines, AudioLoader, Diarizer, Notifier, ResultRepository, Vad

log = logging.getLogger(__name__)


class TranscribeAndDiarize:
    def __init__(self, loader: AudioLoader, vad: Vad, language: LanguageResolver,
                 engines: AsrEngines, diarizer: Diarizer, fusion: FusionEngine,
                 repo: ResultRepository, notifier: Notifier | None = None):
        self.loader, self.vad, self.language, self.engines = loader, vad, language, engines
        self.diarizer, self.fusion, self.repo, self.notifier = diarizer, fusion, repo, notifier

    def run(self, job: AudioJob, decision: LanguageDecision | None = None) -> Transcript:
        """decision verilirse (ör. dil tespiti ayrı kuyrukta yapıldıysa) tekrar tespit edilmez."""
        self.repo.set_status(job.job_id, JobStatus.RUNNING)
        try:
            transcript = self._process(job, decision)
        except UnsupportedLanguageError as e:
            self.repo.set_status(job.job_id, JobStatus.REJECTED, error=str(e))
            raise
        except Exception as e:
            self.repo.set_status(job.job_id, JobStatus.FAILED, error=f"{type(e).__name__}: {e}")
            raise
        self.repo.save(transcript)  # sonucu yazar VE işi DONE yapar (tek işlemde)
        if self.notifier is not None:
            self.notifier.notify(job, transcript)
        return transcript

    def _process(self, job: AudioJob, decision: LanguageDecision | None) -> Transcript:
        t0 = time.perf_counter()
        audio = self.loader.load(job.audio_path)
        duration = len(audio) / 16000

        segments = self.vad.split(audio)
        lang = decision or self.language.resolve(job.language, segments)
        asr = self.engines[lang.code]
        words = asr.transcribe(segments) if segments else []
        speakers = self.diarizer.diarize(audio) if segments else []
        utterances = self.fusion.fuse(words, speakers)

        dt = time.perf_counter() - t0
        log.info("job=%s dil=%s(%s %.2f%s) sure=%.1fs rtf=%.3f parca=%d kelime=%d satir=%d",
                 job.job_id, lang.code, lang.source.value, lang.confidence,
                 " BELIRSIZ" if lang.uncertain else "", duration, dt / max(duration, 1e-9),
                 len(segments), len(words), len(utterances))
        return Transcript(job_id=job.job_id, duration=duration, utterances=utterances,
                          model=asr.name, language=lang)

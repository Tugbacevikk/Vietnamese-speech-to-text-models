"""Bellekte iş deposu: testler ve tek dosyalık CLI kullanımı için (kalıcı değil)."""
from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timezone

from viet_asr.domain import AudioJob, JobRecord, JobStatus, LanguageDecision, Transcript


def _now():
    return datetime.now(timezone.utc)


class MemoryJobStore:
    def __init__(self):
        self.jobs: dict[str, JobRecord] = {}
        self.results: dict[str, Transcript] = {}
        self.events: list[tuple[str, str]] = []  # (job_id, olay) — outbox yerine
        self._lock = threading.Lock()

    def create(self, job: AudioJob) -> None:
        with self._lock:
            if job.job_id in self.jobs:
                raise KeyError(f"iş zaten var: {job.job_id}")
            self.jobs[job.job_id] = JobRecord(job, JobStatus.QUEUED, created_at=_now(), updated_at=_now())

    def _update(self, job_id: str, **kw) -> JobRecord | None:
        with self._lock:
            r = self.jobs.get(job_id)
            if r is None:
                # CLI'da iş önceden açılmadan da çalışılabilsin
                r = JobRecord(AudioJob(job_id, ""), JobStatus.QUEUED, created_at=_now())
            r = replace(r, updated_at=_now(), **kw)
            self.jobs[job_id] = r
            return r

    def get(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        r = self._update(job_id, status=status, error=error)
        if r.job.callback_url and status in (JobStatus.FAILED, JobStatus.REJECTED):
            self.events.append((job_id, f"job.{status.value}"))

    def set_language(self, job_id: str, decision: LanguageDecision) -> None:
        self._update(job_id, decision=decision)

    def save(self, t: Transcript) -> None:
        self.results[t.job_id] = t
        r = self._update(t.job_id, status=JobStatus.DONE, decision=t.language, error=None)
        if r.job.callback_url:
            self.events.append((t.job_id, "job.done"))

    def get_transcript(self, job_id: str) -> Transcript | None:
        return self.results.get(job_id)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.jobs.values():
            out[r.status.value] = out.get(r.status.value, 0) + 1
        return out

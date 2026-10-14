"""Bellekte iş deposu: testler ve tek dosyalık CLI kullanımı için (kalıcı değil)."""
from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timezone

from viet_asr.domain import AudioJob, Correction, JobRecord, JobStatus, LanguageDecision, Transcript


def _now():
    return datetime.now(timezone.utc)


class MemoryJobStore:
    def __init__(self):
        self.jobs: dict[str, JobRecord] = {}
        self.results: dict[str, Transcript] = {}
        self.events: list[tuple[str, str]] = []  # (job_id, olay) — outbox yerine
        self._lock = threading.Lock()
        self._originals: dict = {}
        self._edits: dict = {}

    def create(self, job: AudioJob) -> str:
        with self._lock:
            if job.job_id in self.jobs:
                raise KeyError(f"iş zaten var: {job.job_id}")
            if job.external_id:
                for r in self.jobs.values():
                    if r.job.external_id == job.external_id:
                        return r.job.job_id
            self.jobs[job.job_id] = JobRecord(job, JobStatus.QUEUED, created_at=_now(), updated_at=_now())
            return job.job_id

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
        if r.job.callback_url and status == JobStatus.REJECTED:
            self.events.append((job_id, "job.rejected"))

    def fail(self, job_id: str, error: str) -> None:
        r = self._update(job_id, status=JobStatus.FAILED, error=error)
        if r.job.callback_url:
            self.events.append((job_id, "job.failed"))

    def set_language(self, job_id: str, decision: LanguageDecision) -> None:
        self._update(job_id, decision=decision)

    def save(self, t: Transcript) -> None:
        self.results[t.job_id] = t
        r = self._update(t.job_id, status=JobStatus.DONE, decision=t.language, error=None)
        if r.job.callback_url:
            self.events.append((t.job_id, "job.done"))

    def get_transcript(self, job_id: str) -> Transcript | None:
        return self.results.get(job_id)

    def update_utterance(self, job_id, idx, speaker=None, text=None):
        with self._lock:
            t, r = self.results.get(job_id), self.jobs.get(job_id)
            if t is None or r is None or r.status != JobStatus.DONE or not 0 <= idx < len(t.utterances):
                return None
            old = t.utterances[idx]
            new = replace(old, speaker=speaker if speaker is not None else old.speaker,
                          text=text if text is not None else old.text, edited=True)
            utts = list(t.utterances)
            utts[idx] = new
            self.results[job_id] = replace(t, utterances=utts)
            orig = self._originals.setdefault((job_id, idx), (old.speaker, old.text))
            self._edits[(job_id, idx)] = (orig, _now())
            return new

    def corrections(self, limit=10_000):
        out = []
        for (jid, idx), ((osp, otx), at) in self._edits.items():
            u, r = self.results[jid].utterances[idx], self.jobs[jid]
            out.append(Correction(jid, r.job.audio_path, r.decision.code if r.decision else None, idx,
                                  u.start, u.end, u.speaker, u.text, osp, otx, at))
        return sorted(out, key=lambda c: c.edited_at, reverse=True)[:limit]

    def recent_jobs(self, limit: int = 20) -> list[JobRecord]:
        return sorted(self.jobs.values(), key=lambda r: r.created_at, reverse=True)[:limit]

    def delete_job(self, job_id: str) -> str | None:
        with self._lock:
            r = self.jobs.pop(job_id, None)
            self.results.pop(job_id, None)
        return r.job.audio_path if r else None

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.jobs.values():
            out[r.status.value] = out.get(r.status.value, 0) + 1
        return out

    # --- bakım ve ölçüm ---
    _DONE = (JobStatus.DONE, JobStatus.FAILED, JobStatus.REJECTED)

    def finished_with_audio_before(self, before):
        return [(j, r.job.audio_path) for j, r in self.jobs.items()
                if r.status in self._DONE and r.updated_at < before and r.job.audio_path]

    def clear_audio(self, job_id):
        r = self.jobs[job_id]
        self.jobs[job_id] = replace(r, job=replace(r.job, audio_path=""))

    def delete_finished_before(self, before):
        old = [j for j, r in self.jobs.items() if r.status in self._DONE and r.updated_at < before]
        for j in old:
            self.jobs.pop(j)
            self.results.pop(j, None)
        return len(old)

    def stale_running(self, before):
        return [j for j, r in self.jobs.items() if r.status == JobStatus.RUNNING and r.updated_at < before]

    def recent_done(self, limit=500):
        done = sorted((r for r in self.jobs.values() if r.status == JobStatus.DONE),
                      key=lambda r: r.updated_at, reverse=True)[:limit]
        return [(self.results[r.job.job_id].duration if r.job.job_id in self.results else 0.0,
                 (r.updated_at - r.created_at).total_seconds()) for r in done]

"""SQLite iş deposu + outbox: tek bilgisayarda (yerel mod) PostgreSQL yerine.

PostgresJobStore ile aynı arayüz ve aynı kural: sonuç ile "gönderilecek bildirim"
satırı aynı işlemde yazılır. Kurulum gerektirmez; tek dosya (jobs.db).
Aynı anda tek yazar olduğu için birden çok süreç değil, tek süreç + iş parçacıkları
için uygundur.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from viet_asr.application.serialization import transcript_to_dict
from viet_asr.domain import (
    AudioJob, JobRecord, JobStatus, LanguageDecision, LanguageSource, Transcript, Utterance,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, status TEXT NOT NULL, audio_path TEXT NOT NULL,
    requested_lang TEXT, callback_url TEXT, num_speakers INTEGER,
    lang_code TEXT, lang_confidence REAL, lang_source TEXT, lang_uncertain INTEGER,
    model TEXT, duration REAL, error TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status);
CREATE TABLE IF NOT EXISTS utterances (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, idx INTEGER NOT NULL,
    speaker TEXT NOT NULL, start_s REAL NOT NULL, end_s REAL NOT NULL, text TEXT NOT NULL,
    PRIMARY KEY (job_id, idx)
);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    event TEXT NOT NULL, url TEXT NOT NULL, payload TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT NOT NULL,
    sent_at TEXT, last_error TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS outbox_pending_idx ON outbox (next_attempt_at) WHERE sent_at IS NULL;
"""

# ret kesindir: anında bildirim. FAILED'in bildirimi yalnızca fail() ile (son deneme).
_TERMINAL_EVENTS = {JobStatus.REJECTED: "job.rejected"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="milliseconds")


class SqliteJobStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None,
                                     timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()

    def close(self) -> None:
        self._conn.close()

    def migrate(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    # --- yazma ---
    def create(self, job: AudioJob) -> None:
        now = _iso(_now())
        with self._tx() as c:
            c.execute("INSERT INTO jobs (id, status, audio_path, requested_lang, callback_url, "
                      "num_speakers, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                      (job.job_id, JobStatus.QUEUED.value, job.audio_path, job.language,
                       job.callback_url, job.num_speakers, now, now))

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        with self._tx() as c:
            c.execute("UPDATE jobs SET status=?, error=?, updated_at=? WHERE id=?",
                      (status.value, error, _iso(_now()), job_id))
            row = c.execute("SELECT callback_url FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row and row["callback_url"] and status in _TERMINAL_EVENTS:
                self._outbox(c, job_id, _TERMINAL_EVENTS[status], row["callback_url"],
                             {"job_id": job_id, "status": status.value, "error": error})

    def fail(self, job_id: str, error: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE jobs SET status=?, error=?, updated_at=? WHERE id=?",
                      (JobStatus.FAILED.value, error, _iso(_now()), job_id))
            row = c.execute("SELECT callback_url FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row and row["callback_url"]:
                self._outbox(c, job_id, "job.failed", row["callback_url"],
                             {"job_id": job_id, "status": JobStatus.FAILED.value, "error": error})

    def set_language(self, job_id: str, d: LanguageDecision) -> None:
        with self._tx() as c:
            c.execute("UPDATE jobs SET lang_code=?, lang_confidence=?, lang_source=?, "
                      "lang_uncertain=?, updated_at=? WHERE id=?",
                      (d.code, d.confidence, d.source.value, int(d.uncertain), _iso(_now()), job_id))

    def save(self, t: Transcript) -> None:
        d = t.language
        with self._tx() as c:
            cur = c.execute(
                "UPDATE jobs SET status=?, model=?, duration=?, lang_code=?, lang_confidence=?, "
                "lang_source=?, lang_uncertain=?, error=NULL, updated_at=? WHERE id=?",
                (JobStatus.DONE.value, t.model, t.duration, d.code, d.confidence, d.source.value,
                 int(d.uncertain), _iso(_now()), t.job_id))
            if cur.rowcount == 0:
                raise KeyError(f"iş yok: {t.job_id}")
            c.execute("DELETE FROM utterances WHERE job_id=?", (t.job_id,))
            c.executemany("INSERT INTO utterances (job_id, idx, speaker, start_s, end_s, text) "
                          "VALUES (?,?,?,?,?,?)",
                          [(t.job_id, i, u.speaker, u.start, u.end, u.text)
                           for i, u in enumerate(t.utterances)])
            row = c.execute("SELECT callback_url FROM jobs WHERE id=?", (t.job_id,)).fetchone()
            if row["callback_url"]:
                self._outbox(c, t.job_id, "job.done", row["callback_url"],
                             {"status": JobStatus.DONE.value, **transcript_to_dict(t)})

    @staticmethod
    def _outbox(c, job_id, event, url, payload) -> None:
        now = _iso(_now())
        c.execute("INSERT INTO outbox (job_id, event, url, payload, next_attempt_at, created_at) "
                  "VALUES (?,?,?,?,?,?)", (job_id, event, url, json.dumps(payload, ensure_ascii=False),
                                           now, now))

    # --- okuma ---
    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _record(r) if r else None

    def get_transcript(self, job_id: str) -> Transcript | None:
        with self._lock:
            j = self._conn.execute("SELECT * FROM jobs WHERE id=? AND status=?",
                                   (job_id, JobStatus.DONE.value)).fetchone()
            if j is None:
                return None
            rows = self._conn.execute("SELECT * FROM utterances WHERE job_id=? ORDER BY idx",
                                      (job_id,)).fetchall()
        return Transcript(job_id=job_id, duration=j["duration"] or 0.0, model=j["model"] or "",
                          language=_decision(j),
                          utterances=[Utterance(r["speaker"], r["start_s"], r["end_s"], r["text"])
                                      for r in rows])

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT status, count(*) AS n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    def unfinished(self) -> list[JobRecord]:
        """Süreç kapanırken yarım kalan işler (yeniden başlatınca tekrar kuyruğa alınır)."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs WHERE status IN (?, ?) ORDER BY created_at",
                                      (JobStatus.QUEUED.value, JobStatus.RUNNING.value)).fetchall()
        return [_record(r) for r in rows]

    # --- bakım ve ölçüm ---
    _DONE = ("done", "failed", "rejected")

    def finished_with_audio_before(self, before: datetime) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute("SELECT id, audio_path FROM jobs WHERE status IN (?,?,?) "
                                      "AND updated_at < ? AND audio_path <> ''",
                                      (*self._DONE, _iso(before))).fetchall()
        return [(r["id"], r["audio_path"]) for r in rows]

    def clear_audio(self, job_id: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE jobs SET audio_path='' WHERE id=?", (job_id,))

    def delete_finished_before(self, before: datetime) -> int:
        with self._tx() as c:
            return c.execute("DELETE FROM jobs WHERE status IN (?,?,?) AND updated_at < ?",
                             (*self._DONE, _iso(before))).rowcount

    def stale_running(self, before: datetime) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT id FROM jobs WHERE status=? AND updated_at < ?",
                                      (JobStatus.RUNNING.value, _iso(before))).fetchall()
        return [r["id"] for r in rows]

    def recent_done(self, limit: int = 500) -> list[tuple[float, float]]:
        with self._lock:
            rows = self._conn.execute("SELECT duration, created_at, updated_at FROM jobs WHERE status=? "
                                      "ORDER BY updated_at DESC LIMIT ?",
                                      (JobStatus.DONE.value, limit)).fetchall()
        return [(float(r["duration"] or 0),
                 (datetime.fromisoformat(r["updated_at"]) - datetime.fromisoformat(r["created_at"])
                  ).total_seconds()) for r in rows]

    # --- outbox gönderici için ---
    def claim_outbox(self, limit: int, max_attempts: int, lease: timedelta) -> list[dict]:
        """Gönderilecek satırları alır ve 'lease' süresince başkasına görünmez yapar."""
        now = _now()
        with self._tx() as c:
            rows = c.execute("SELECT * FROM outbox WHERE sent_at IS NULL AND next_attempt_at <= ? "
                             "AND attempts < ? ORDER BY id LIMIT ?",
                             (_iso(now), max_attempts, limit)).fetchall()
            for r in rows:
                c.execute("UPDATE outbox SET next_attempt_at=? WHERE id=?", (_iso(now + lease), r["id"]))
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def mark_sent(self, outbox_id: int) -> None:
        with self._tx() as c:
            c.execute("UPDATE outbox SET sent_at=?, attempts=attempts+1, last_error=NULL WHERE id=?",
                      (_iso(_now()), outbox_id))

    def mark_failed(self, outbox_id: int, error: str, retry_in: timedelta) -> None:
        with self._tx() as c:
            c.execute("UPDATE outbox SET attempts=attempts+1, last_error=?, next_attempt_at=? WHERE id=?",
                      (error[:500], _iso(_now() + retry_in), outbox_id))

    def outbox_rows(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute("SELECT * FROM outbox ORDER BY id").fetchall()]


def _decision(r) -> LanguageDecision | None:
    if not r["lang_code"]:
        return None
    return LanguageDecision(r["lang_code"], float(r["lang_confidence"] or 0.0),
                            LanguageSource(r["lang_source"]), bool(r["lang_uncertain"]))


def _record(r) -> JobRecord:
    job = AudioJob(r["id"], r["audio_path"], r["requested_lang"], r["callback_url"], r["num_speakers"])
    return JobRecord(job=job, status=JobStatus(r["status"]), decision=_decision(r), error=r["error"],
                     created_at=datetime.fromisoformat(r["created_at"]),
                     updated_at=datetime.fromisoformat(r["updated_at"]))

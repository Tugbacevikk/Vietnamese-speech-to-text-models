"""PostgreSQL iş deposu + outbox (giden kutusu).

Outbox neden? İş bitince müşteriye webhook gönderilmeli. Sonucu veritabanına
yazıp hemen ardından HTTP isteği atarsak, ikisinin arasında süreç çökerse
bildirim kaybolur (ya da sonuç yazılmadan bildirim gider). Bunun yerine sonuç
ve "gönderilecek bildirim" satırı AYNI işlemde (transaction) yazılır; ayrı bir
gönderici (dispatcher) outbox'taki satırları okuyup gönderir, başarısızsa
artan aralıklarla tekrar dener. Sonuç: bildirim en az bir kez gider.
"""
from __future__ import annotations

from datetime import timedelta

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from viet_asr.application.serialization import transcript_to_dict
from viet_asr.domain import (
    AudioJob, JobRecord, JobStatus, LanguageDecision, LanguageSource, Transcript, Utterance,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    audio_path      TEXT NOT NULL,
    requested_lang  TEXT,
    callback_url    TEXT,
    lang_code       TEXT,
    lang_confidence REAL,
    lang_source     TEXT,
    lang_uncertain  BOOLEAN,
    model           TEXT,
    duration        REAL,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status);

CREATE TABLE IF NOT EXISTS utterances (
    job_id   TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx      INTEGER NOT NULL,
    speaker  TEXT NOT NULL,
    start_s  REAL NOT NULL,
    end_s    REAL NOT NULL,
    text     TEXT NOT NULL,
    PRIMARY KEY (job_id, idx)
);

CREATE TABLE IF NOT EXISTS outbox (
    id              BIGSERIAL PRIMARY KEY,
    job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    event           TEXT NOT NULL,
    url             TEXT NOT NULL,
    payload         JSONB NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sent_at         TIMESTAMPTZ,
    last_error      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS outbox_pending_idx ON outbox (next_attempt_at) WHERE sent_at IS NULL;
"""

_TERMINAL_EVENTS = {JobStatus.FAILED: "job.failed", JobStatus.REJECTED: "job.rejected"}


class PostgresJobStore:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 5):
        self.pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size,
                                   kwargs={"row_factory": dict_row}, open=True)

    def close(self) -> None:
        self.pool.close()

    def migrate(self) -> None:
        with self.pool.connection() as c:
            c.execute(SCHEMA)

    # --- yazma ---
    def create(self, job: AudioJob) -> None:
        with self.pool.connection() as c:
            c.execute("INSERT INTO jobs (id, status, audio_path, requested_lang, callback_url) "
                      "VALUES (%s, %s, %s, %s, %s)",
                      (job.job_id, JobStatus.QUEUED.value, job.audio_path, job.language,
                       job.callback_url))

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        with self.pool.connection() as c, c.transaction():
            row = c.execute("UPDATE jobs SET status=%s, error=%s, updated_at=now() WHERE id=%s "
                            "RETURNING callback_url", (status.value, error, job_id)).fetchone()
            if row and row["callback_url"] and status in _TERMINAL_EVENTS:
                payload = {"job_id": job_id, "status": status.value, "error": error}
                self._outbox(c, job_id, _TERMINAL_EVENTS[status], row["callback_url"], payload)

    def set_language(self, job_id: str, d: LanguageDecision) -> None:
        with self.pool.connection() as c:
            c.execute("UPDATE jobs SET lang_code=%s, lang_confidence=%s, lang_source=%s, "
                      "lang_uncertain=%s, updated_at=now() WHERE id=%s",
                      (d.code, d.confidence, d.source.value, d.uncertain, job_id))

    def save(self, t: Transcript) -> None:
        d = t.language
        with self.pool.connection() as c, c.transaction():
            row = c.execute(
                "UPDATE jobs SET status=%s, model=%s, duration=%s, lang_code=%s, lang_confidence=%s, "
                "lang_source=%s, lang_uncertain=%s, error=NULL, updated_at=now() WHERE id=%s "
                "RETURNING callback_url",
                (JobStatus.DONE.value, t.model, t.duration, d.code, d.confidence, d.source.value,
                 d.uncertain, t.job_id)).fetchone()
            if row is None:
                raise KeyError(f"iş yok: {t.job_id}")
            c.execute("DELETE FROM utterances WHERE job_id=%s", (t.job_id,))  # tekrar işlenirse
            with c.cursor() as cur:
                cur.executemany(
                    "INSERT INTO utterances (job_id, idx, speaker, start_s, end_s, text) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [(t.job_id, i, u.speaker, u.start, u.end, u.text)
                     for i, u in enumerate(t.utterances)])
            if row["callback_url"]:
                payload = {"status": JobStatus.DONE.value, **transcript_to_dict(t)}
                self._outbox(c, t.job_id, "job.done", row["callback_url"], payload)

    @staticmethod
    def _outbox(c, job_id: str, event: str, url: str, payload: dict) -> None:
        c.execute("INSERT INTO outbox (job_id, event, url, payload) VALUES (%s, %s, %s, %s)",
                  (job_id, event, url, Jsonb(payload)))

    # --- okuma ---
    def get(self, job_id: str) -> JobRecord | None:
        with self.pool.connection() as c:
            r = c.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
        return _record(r) if r else None

    def get_transcript(self, job_id: str) -> Transcript | None:
        with self.pool.connection() as c:
            j = c.execute("SELECT * FROM jobs WHERE id=%s AND status=%s",
                          (job_id, JobStatus.DONE.value)).fetchone()
            if j is None:
                return None
            rows = c.execute("SELECT * FROM utterances WHERE job_id=%s ORDER BY idx",
                             (job_id,)).fetchall()
        return Transcript(
            job_id=job_id, duration=j["duration"] or 0.0, model=j["model"] or "",
            language=_decision(j),
            utterances=[Utterance(r["speaker"], r["start_s"], r["end_s"], r["text"]) for r in rows])

    def counts(self) -> dict[str, int]:
        with self.pool.connection() as c:
            rows = c.execute("SELECT status, count(*) AS n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    # --- outbox gönderici için ---
    def claim_outbox(self, conn, limit: int, max_attempts: int) -> list[dict]:
        """Gönderilecek satırları kilitler. SKIP LOCKED: birden çok gönderici çakışmaz."""
        return conn.execute(
            "SELECT * FROM outbox WHERE sent_at IS NULL AND next_attempt_at <= now() "
            "AND attempts < %s ORDER BY id LIMIT %s FOR UPDATE SKIP LOCKED",
            (max_attempts, limit)).fetchall()

    @staticmethod
    def mark_sent(conn, outbox_id: int) -> None:
        conn.execute("UPDATE outbox SET sent_at=now(), attempts=attempts+1, last_error=NULL "
                     "WHERE id=%s", (outbox_id,))

    @staticmethod
    def mark_failed(conn, outbox_id: int, error: str, retry_in: timedelta) -> None:
        conn.execute("UPDATE outbox SET attempts=attempts+1, last_error=%s, "
                     "next_attempt_at=now() + %s WHERE id=%s", (error[:500], retry_in, outbox_id))


def _decision(r: dict) -> LanguageDecision | None:
    if not r.get("lang_code"):
        return None
    return LanguageDecision(r["lang_code"], float(r["lang_confidence"] or 0.0),
                            LanguageSource(r["lang_source"]), bool(r["lang_uncertain"]))


def _record(r: dict) -> JobRecord:
    job = AudioJob(r["id"], r["audio_path"], r["requested_lang"], r["callback_url"])
    return JobRecord(job=job, status=JobStatus(r["status"]), decision=_decision(r),
                     error=r["error"], created_at=r["created_at"], updated_at=r["updated_at"])


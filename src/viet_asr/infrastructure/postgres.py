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
    num_speakers    INTEGER,
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
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS num_speakers INTEGER;

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

# set_status ile anında bildirim üreten durum: ret kesindir, tekrar denenmez.
# FAILED tekrar denenebilir; bildirimi yalnızca fail() (son deneme) üretir.
_TERMINAL_EVENTS = {JobStatus.REJECTED: "job.rejected"}


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
            c.execute("INSERT INTO jobs (id, status, audio_path, requested_lang, callback_url, "
                      "num_speakers) VALUES (%s, %s, %s, %s, %s, %s)",
                      (job.job_id, JobStatus.QUEUED.value, job.audio_path, job.language,
                       job.callback_url, job.num_speakers))

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        with self.pool.connection() as c, c.transaction():
            row = c.execute("UPDATE jobs SET status=%s, error=%s, updated_at=now() WHERE id=%s "
                            "RETURNING callback_url", (status.value, error, job_id)).fetchone()
            if row and row["callback_url"] and status in _TERMINAL_EVENTS:
                payload = {"job_id": job_id, "status": status.value, "error": error}
                self._outbox(c, job_id, _TERMINAL_EVENTS[status], row["callback_url"], payload)

    def fail(self, job_id: str, error: str) -> None:
        with self.pool.connection() as c, c.transaction():
            row = c.execute("UPDATE jobs SET status=%s, error=%s, updated_at=now() WHERE id=%s "
                            "RETURNING callback_url", (JobStatus.FAILED.value, error, job_id)).fetchone()
            if row and row["callback_url"]:
                self._outbox(c, job_id, "job.failed", row["callback_url"],
                             {"job_id": job_id, "status": JobStatus.FAILED.value, "error": error})

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
    def claim_outbox(self, limit: int, max_attempts: int, lease: timedelta) -> list[dict]:
        """Gönderilecek satırları alır ve 'lease' süresince başka göndericilere görünmez yapar.
        Kilit HTTP isteği boyunca TUTULMAZ (yavaş bir alıcı veritabanını bekletmesin);
        gönderici çökerse satır lease bitince yeniden görünür. SKIP LOCKED: çakışma yok."""
        with self.pool.connection() as c:
            return c.execute(
                "UPDATE outbox SET next_attempt_at = now() + %s WHERE id IN ("
                " SELECT id FROM outbox WHERE sent_at IS NULL AND next_attempt_at <= now()"
                " AND attempts < %s ORDER BY id LIMIT %s FOR UPDATE SKIP LOCKED) RETURNING *",
                (lease, max_attempts, limit)).fetchall()

    def mark_sent(self, outbox_id: int) -> None:
        with self.pool.connection() as c:
            c.execute("UPDATE outbox SET sent_at=now(), attempts=attempts+1, last_error=NULL "
                      "WHERE id=%s", (outbox_id,))

    def mark_failed(self, outbox_id: int, error: str, retry_in: timedelta) -> None:
        with self.pool.connection() as c:
            c.execute("UPDATE outbox SET attempts=attempts+1, last_error=%s, "
                      "next_attempt_at=now() + %s WHERE id=%s", (error[:500], retry_in, outbox_id))

    # --- bakım ve ölçüm ---
    _DONE = ("done", "failed", "rejected")

    def finished_with_audio_before(self, before) -> list[tuple[str, str]]:
        with self.pool.connection() as c:
            rows = c.execute("SELECT id, audio_path FROM jobs WHERE status = ANY(%s) "
                             "AND updated_at < %s AND audio_path <> ''",
                             (list(self._DONE), before)).fetchall()
        return [(r["id"], r["audio_path"]) for r in rows]

    def clear_audio(self, job_id: str) -> None:
        with self.pool.connection() as c:
            c.execute("UPDATE jobs SET audio_path='' WHERE id=%s", (job_id,))

    def delete_finished_before(self, before) -> int:
        with self.pool.connection() as c:
            return c.execute("DELETE FROM jobs WHERE status = ANY(%s) AND updated_at < %s",
                             (list(self._DONE), before)).rowcount

    def stale_running(self, before) -> list[str]:
        with self.pool.connection() as c:
            rows = c.execute("SELECT id FROM jobs WHERE status=%s AND updated_at < %s",
                             (JobStatus.RUNNING.value, before)).fetchall()
        return [r["id"] for r in rows]

    def recent_done(self, limit: int = 500) -> list[tuple[float, float]]:
        """(ses süresi, istekten bitişe geçen süre) — son biten işler."""
        with self.pool.connection() as c:
            rows = c.execute("SELECT duration, EXTRACT(EPOCH FROM updated_at - created_at) AS lat "
                             "FROM jobs WHERE status=%s ORDER BY updated_at DESC LIMIT %s",
                             (JobStatus.DONE.value, limit)).fetchall()
        return [(float(r["duration"] or 0), float(r["lat"])) for r in rows]

    def unfinished(self) -> list[JobRecord]:
        with self.pool.connection() as c:
            rows = c.execute("SELECT * FROM jobs WHERE status IN (%s, %s) ORDER BY created_at",
                             (JobStatus.QUEUED.value, JobStatus.RUNNING.value)).fetchall()
        return [_record(r) for r in rows]


def _decision(r: dict) -> LanguageDecision | None:
    if not r.get("lang_code"):
        return None
    return LanguageDecision(r["lang_code"], float(r["lang_confidence"] or 0.0),
                            LanguageSource(r["lang_source"]), bool(r["lang_uncertain"]))


def _record(r: dict) -> JobRecord:
    job = AudioJob(r["id"], r["audio_path"], r["requested_lang"], r["callback_url"],
                   r.get("num_speakers"))
    return JobRecord(job=job, status=JobStatus(r["status"]), decision=_decision(r),
                     error=r["error"], created_at=r["created_at"], updated_at=r["updated_at"])


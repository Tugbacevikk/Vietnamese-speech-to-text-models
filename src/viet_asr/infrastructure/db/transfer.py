"""Veritabanından veritabanına taşıma (SQLite -> PostgreSQL): işler, satırlar, bildirim kuyruğu.

İki tarafın şeması önce Alembic ile en son sürüme getirilir; sonra satırlar SQLAlchemy
tablolarıyla kopyalanır. Böylece tür dönüşümleri (JSON -> JSONB, saat dilimli zaman, mantıksal
değer) modeldeki tanımla yapılır, el ile SQL yok.

Tekrar çalıştırılabilir: hedefte zaten olan iş (aynı id) atlanır, satırları da onunla gelir;
bildirimler (outbox) kendi kimliği yerine (iş, olay, oluşturulma zamanı) ile eşlenir ve hedefin
sayacından yeni kimlik alır (hedefte önceden bildirim varsa kimlikler çakışmaz).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import create_engine, func, select

from viet_asr.infrastructure.db.models import JobRow, OutboxRow, UtteranceRow
from viet_asr.infrastructure.db.store import SqlAlchemyJobStore

log = logging.getLogger(__name__)
BATCH = 500


@dataclass
class TransferReport:
    source: dict[str, int] = field(default_factory=dict)   # kaynaktaki satır sayıları
    copied: dict[str, int] = field(default_factory=dict)   # bu çalıştırmada kopyalanan
    skipped: dict[str, int] = field(default_factory=dict)  # hedefte zaten vardı
    target: dict[str, int] = field(default_factory=dict)   # sonunda hedefte olan

    @property
    def complete(self) -> bool:
        """Kaynaktaki her satır hedefte var mı (hedefte fazlası olabilir: önceden kullanılmışsa)."""
        return all(self.target.get(t, 0) >= n for t, n in self.source.items())


def _safe(url: str) -> str:
    """Günlüğe parolasız adres."""
    from sqlalchemy.engine import make_url

    return make_url(url).render_as_string(hide_password=True)


def _count(conn, table) -> int:
    return conn.execute(select(func.count()).select_from(table)).scalar_one()


def transfer(source_url: str, target_url: str, dry_run: bool = False) -> TransferReport:
    """source -> target. dry_run: yalnızca sayar, hedefe yazmaz (şemasına da dokunmaz)."""
    if source_url == target_url:
        raise ValueError("kaynak ve hedef aynı veritabanı")
    SqlAlchemyJobStore(source_url).migrate()  # eski sürüm SQLite'ı da yeni şemaya getirir
    src = create_engine(source_url)
    rep = TransferReport()
    jobs, utts, outbox = JobRow.__table__, UtteranceRow.__table__, OutboxRow.__table__
    with src.connect() as s:
        for t in (jobs, utts, outbox):
            rep.source[t.name] = _count(s, t)
    if dry_run:
        return rep

    SqlAlchemyJobStore(target_url).migrate()
    dst = create_engine(target_url)
    log.info("taşınıyor: %s -> %s", _safe(source_url), _safe(target_url))
    with src.connect() as s, dst.begin() as d:
        have = set(d.execute(select(jobs.c.id)).scalars())
        new_ids: list[str] = []
        rows = []
        for r in s.execute(select(jobs).order_by(jobs.c.created_at)).mappings():
            if r["id"] in have:
                continue
            rows.append(dict(r))
            new_ids.append(r["id"])
            if len(rows) >= BATCH:
                d.execute(jobs.insert(), rows)
                rows = []
        if rows:
            d.execute(jobs.insert(), rows)
        rep.copied["jobs"], rep.skipped["jobs"] = len(new_ids), rep.source["jobs"] - len(new_ids)

        # satırlar: yalnızca bu çalıştırmada eklenen işlerin (var olan işin satırlarına dokunulmaz)
        n_utt, fresh = 0, set(new_ids)
        rows = []
        for r in s.execute(select(utts).order_by(utts.c.job_id, utts.c.idx)).mappings():
            if r["job_id"] not in fresh:
                continue
            rows.append(dict(r))
            if len(rows) >= BATCH:
                d.execute(utts.insert(), rows)
                n_utt += len(rows)
                rows = []
        if rows:
            d.execute(utts.insert(), rows)
            n_utt += len(rows)
        rep.copied["utterances"] = n_utt
        rep.skipped["utterances"] = rep.source["utterances"] - n_utt

        # bildirimler: (iş, olay, zaman) ile eşlenir; kimlik hedefin sayacından
        seen = {tuple(k) for k in d.execute(select(outbox.c.job_id, outbox.c.event, outbox.c.created_at))}
        cols = [c for c in outbox.c if c.name != "id"]
        rows = []
        for r in s.execute(select(*cols).order_by(outbox.c.id)).mappings():
            if (r["job_id"], r["event"], r["created_at"]) in seen:
                continue
            rows.append(dict(r))
        for i in range(0, len(rows), BATCH):
            d.execute(outbox.insert(), rows[i:i + BATCH])
        n_out = len(rows)
        rep.copied["outbox"], rep.skipped["outbox"] = n_out, rep.source["outbox"] - n_out

    with dst.connect() as d:
        for t in (jobs, utts, outbox):
            rep.target[t.name] = _count(d, t)
    src.dispose()
    dst.dispose()
    return rep

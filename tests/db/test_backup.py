"""Veritabanı yedeği: al, doğrula, geri yükle, eskileri sil; bakımın günde bir yedek alması.

PostgreSQL testleri TEST_DATABASE_URL ve pg_dump ister (yoksa atlanır). Docker yolu, gerçek
pg_dump'ı çağıran sahte bir 'docker' komutuyla denenir (konteyner içindeki gibi -h/-p verilmez).
"""
import os
import shutil
import stat
import sys
import time
import uuid
from datetime import datetime

import pytest

from viet_asr.domain import AudioJob, LanguageDecision, LanguageSource, Transcript, Utterance
from viet_asr.infrastructure.backup import BackupError, DatabaseBackup
from viet_asr.infrastructure.db import SqlAlchemyJobStore

PG = os.environ.get("TEST_DATABASE_URL")


def fill(store, job_id="j1"):
    store.create(AudioJob(job_id, "/veri/a.wav", "vi"))
    store.save(Transcript(job_id=job_id, duration=3.0, model="m",
                          language=LanguageDecision("vi", 1.0, LanguageSource.REQUEST),
                          utterances=[Utterance("speaker_00", 0.0, 1.0, "xin chào"),
                                      Utterance("speaker_01", 1.0, 2.0, "chào")]))
    store.update_utterance(job_id, 1, text="chào bác sĩ")  # ince ayar verisi


def roundtrip(store, b: DatabaseBackup):
    fill(store)
    path = b.run()
    assert path.exists() and not list(b.dir.glob("*.part"))
    store.delete_job("j1")                                 # felaket: iş silindi
    fill(store, "sonradan")                                # yedekten sonra gelen iş
    b.restore(path)
    t = store.get_transcript("j1")
    assert [u.text for u in t.utterances] == ["xin chào", "chào bác sĩ"] and t.utterances[1].edited
    assert store.get("sonradan") is None                   # tablo yedektekiyle değişti
    assert store.corrections()[0].text == "chào bác sĩ"


def test_sqlite_yedek_ve_geri_yukleme(tmp_path):
    url = f"sqlite:///{tmp_path / 'jobs.db'}"
    store = SqlAlchemyJobStore(url)
    store.migrate()
    b = DatabaseBackup(url, tmp_path / "yedek", docker_container="")
    roundtrip(store, b)


@pytest.fixture
def pg_db():
    if not PG or not shutil.which("pg_dump"):
        pytest.skip("TEST_DATABASE_URL ya da pg_dump yok")
    import psycopg
    from sqlalchemy.engine import make_url

    name = "yedek_" + uuid.uuid4().hex[:8]
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute(f"CREATE DATABASE {name}")
    url = make_url(PG).set(database=name).render_as_string(hide_password=False)
    yield url
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute(f"DROP DATABASE {name} WITH (FORCE)")


def test_postgresql_yedek_dogrulama_ve_geri_yukleme(tmp_path, pg_db):
    store = SqlAlchemyJobStore(pg_db)
    store.migrate()
    b = DatabaseBackup(store.url, tmp_path / "yedek", docker_container="")
    roundtrip(store, b)
    bad = tmp_path / "yedek" / "yedek_bozuk.dump"
    bad.write_bytes(b"PGDMP bozuk")
    with pytest.raises(BackupError):
        b.verify(bad)


def test_docker_icindeki_pg_dump_kullanilir(tmp_path, pg_db, monkeypatch):
    """PostgreSQL Docker'da, Windows'ta pg_dump yok: yedek 'docker exec' ile alınır."""
    from sqlalchemy.engine import make_url

    u = make_url(pg_db)
    calls = tmp_path / "cagrilar.txt"
    shim = tmp_path / "bin" / "docker"
    shim.parent.mkdir()
    # sahte docker: inspect -> çalışıyor; exec -> konteynerin içindeki gibi (yerel bağlantı)
    shim.write_text(f"""#!{sys.executable}
import os, subprocess, sys
a = sys.argv[1:]
open({str(calls)!r}, "a").write(" ".join(a) + "\\n")
if a[0] == "inspect":
    print("true"); sys.exit(0)
assert a[:2] == ["exec", "-i"] and a[2:4] == ["-e", "PGPASSWORD"] and a[4] == "asr-pg"
assert "-h" not in a and "-p" not in a, "konteyner içinde dış port verilmemeli"
env = {{**os.environ, "PGHOST": {u.host!r}, "PGPORT": "{u.port or 5432}"}}
sys.exit(subprocess.run(a[5:], env=env).returncode)
""")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{shim.parent}{os.pathsep}{os.environ['PATH']}")
    store = SqlAlchemyJobStore(pg_db)
    store.migrate()
    b = DatabaseBackup(store.url, tmp_path / "yedek", docker_container="asr-pg", docker="docker")
    roundtrip(store, b)
    log = calls.read_text()
    assert "exec -i -e PGPASSWORD asr-pg pg_dump" in log and "pg_restore --clean" in log
    # parola komut satırında değil (ortam değişkeniyle gider): adres ya da PGPASSWORD=... yok
    assert "@" not in log and f"PGPASSWORD={u.password}" not in log


def test_eski_yedekler_silinir_en_yeniler_kalir(tmp_path):
    d = tmp_path / "yedek"
    d.mkdir()
    now = time.time()
    for i, age in enumerate([40, 30, 20, 10, 1]):              # gün
        p = d / f"yedek_2026-10-{i + 10:02d}_000000.dump"
        p.write_text("x")
        os.utime(p, (now - age * 86400,) * 2)
    (d / "yedek_yarim.dump.part").write_text("x")
    b = DatabaseBackup("sqlite:///x.db", d, keep_days=14, keep_min=3)
    assert b.prune(now) == 2                                   # 40 ve 30 günlük
    assert len(b.backups()) == 3 and b.age_hours(now) == pytest.approx(24, abs=0.1)
    assert b.due(24, now) and not b.due(48, now) and not b.due(0, now)
    assert DatabaseBackup("sqlite:///x.db", d, keep_days=1, keep_min=3).prune(now) == 0  # en az 3


def test_bakim_gunde_bir_yedek_alir_hatada_bekler(tmp_path, monkeypatch):
    from viet_asr.interface.composition import build_maintenance
    from viet_asr.interface.settings import Settings

    url = f"sqlite:///{tmp_path / 'up' / 'jobs.db'}"
    store = SqlAlchemyJobStore(url)
    store.migrate()
    s = Settings(data_dir=str(tmp_path / "up"), recordings_dir=str(tmp_path / "rec"),
                 backup_docker_container="")
    m = build_maintenance(store, s)
    assert m.run_once()["yedek"] == 1 and len(list((tmp_path / "yedek").iterdir())) == 1
    assert m.run_once()["yedek"] == 0                          # 24 saat dolmadı
    assert build_maintenance(store, Settings(data_dir=str(tmp_path / "up"), backup_every_hours=0)
                             ).run_once().get("yedek") is None  # kapalı

    tried = []
    real_run = DatabaseBackup.run

    def boom(self, now: datetime | None = None):
        tried.append(1)
        raise BackupError("disk dolu")

    monkeypatch.setattr(DatabaseBackup, "run", boom)
    for p in (tmp_path / "yedek").iterdir():
        p.unlink()                                             # yedek yok: süresi geldi
    m = build_maintenance(store, s)
    assert m.run_once()["yedek"] == 0 and m.run_once()["yedek"] == 0
    assert len(tried) == 1                                     # 1 saat dolmadan yeniden denenmez
    monkeypatch.setattr(DatabaseBackup, "run", real_run)

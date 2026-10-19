"""Veritabanı yedeği: PostgreSQL (pg_dump) ya da SQLite (çevrimiçi kopya); eski yedekleri siler.

Yedekte ne var: işler, konuşma satırları, arayüzdeki düzeltmeler (ince ayar verisi), onaylar,
bildirim kuyruğu. Ses dosyaları yok (saklama süresi dolunca zaten silinirler).

PostgreSQL için pg_dump nerede?
  1. BACKUP_DOCKER_CONTAINER (varsayılan asr-pg) çalışıyorsa konteynerin içindeki pg_dump:
     sunucuyla aynı sürüm, Windows'a ayrıca kurulum gerekmez.
  2. PG_DUMP ayarı, PATH ya da "C:\\Program Files\\PostgreSQL\\<sürüm>\\bin".
Parola komut satırına yazılmaz (PGPASSWORD ortam değişkeniyle verilir).

Biçim: pg_dump "custom" (-Fc, sıkıştırılmış). Yazım önce .part dosyasına, bitince ad değişir:
yarım kalan yedek tam yedek sanılmaz. Ardından arşiv okunarak doğrulanır (pg_restore --list).
Geri yükleme: restore(dosya) -> pg_restore --clean --if-exists (hedefteki tablolar yedektekiyle
değişir).
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.engine import make_url

log = logging.getLogger(__name__)
PREFIX = "yedek_"


class BackupError(RuntimeError):
    pass


@dataclass
class _Pg:
    user: str
    password: str
    host: str
    port: int
    database: str


def _pg(url: str) -> _Pg:
    u = make_url(url)
    return _Pg(u.username or "postgres", u.password or "", u.host or "localhost", u.port or 5432,
               u.database or "postgres")


def _windows_pg_bin(name: str) -> str | None:
    found = sorted(glob.glob(rf"C:\Program Files\PostgreSQL\*\bin\{name}.exe"))
    return found[-1] if found else None


class DatabaseBackup:
    def __init__(self, url: str, directory: str | Path, keep_days: float = 14, keep_min: int = 3,
                 docker_container: str = "asr-pg", pg_dump: str = "", docker: str = "docker"):
        self.url, self.dir = url, Path(directory)
        self.keep_days, self.keep_min = keep_days, keep_min
        self.container, self.pg_dump_path, self.docker = docker_container, pg_dump, docker
        self.sqlite = url.startswith("sqlite")

    # --- araç seçimi ---
    def _container_running(self) -> bool:
        if not self.container or shutil.which(self.docker) is None:
            return False
        try:
            r = subprocess.run([self.docker, "inspect", "-f", "{{.State.Running}}", self.container],
                               capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return r.returncode == 0 and r.stdout.strip() == "true"

    def _tool(self, name: str) -> list[str] | None:
        """[komut...] -> pg_dump/pg_restore'u çalıştıracak önek; None = bulunamadı."""
        if self._container_running():
            return [self.docker, "exec", "-i", "-e", "PGPASSWORD", self.container, name]
        explicit = None
        if self.pg_dump_path:  # PG_DUMP=...\bin\pg_dump.exe -> aynı klasördeki pg_restore.exe
            p = Path(self.pg_dump_path)
            explicit = str(p.with_name(name + p.suffix))
        for cand in (explicit, shutil.which(name), _windows_pg_bin(name)):
            if cand and Path(cand).is_file():
                return [cand]
        return None

    def _pg_args(self, inside_docker: bool) -> tuple[list[str], dict]:
        p = _pg(self.url)
        env = {**os.environ, "PGPASSWORD": p.password}
        if inside_docker:  # konteynerin içinden: kendi yerel bağlantısı (port eşlemesi dışarıda)
            return ["-U", p.user, "-d", p.database], env
        return ["-h", p.host, "-p", str(p.port), "-U", p.user, "-d", p.database], env

    # --- yedek ---
    def run(self, now: datetime | None = None) -> Path:
        """Yedek alır, doğrular, eskileri siler. -> yedek dosyası."""
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
        final = self.dir / f"{PREFIX}{stamp}.{'sqlite' if self.sqlite else 'dump'}"
        part = final.with_name(final.name + ".part")
        t0 = time.perf_counter()
        try:
            if self.sqlite:
                self._sqlite_copy(part)
            else:
                self._pg_dump(part)
            if part.stat().st_size == 0:
                raise BackupError("yedek dosyası boş")
            self.verify(part)
            part.replace(final)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        removed = self.prune()
        log.info("veritabanı yedeği: %s (%.1f MB, %.1f sn)%s", final, final.stat().st_size / 1e6,
                 time.perf_counter() - t0, f", {removed} eski yedek silindi" if removed else "")
        return final

    def _sqlite_copy(self, out: Path) -> None:
        path = self.url.split("///", 1)[1]
        if not Path(path).is_file():
            raise BackupError(f"veritabanı dosyası yok: {path}")
        src = sqlite3.connect(path)
        dst = sqlite3.connect(out)
        try:
            src.backup(dst)  # çalışırken de tutarlı kopya
        finally:
            dst.close()
            src.close()

    def _pg_dump(self, out: Path) -> None:
        cmd = self._tool("pg_dump")
        if cmd is None:
            raise BackupError("pg_dump bulunamadı: PostgreSQL Docker'daysa BACKUP_DOCKER_CONTAINER "
                              "(konteyner adı), kurulum dosyasıyla kurulduysa PG_DUMP=<...\\bin\\pg_dump.exe>")
        args, env = self._pg_args(inside_docker=cmd[0] == self.docker)
        with out.open("wb") as f:
            r = subprocess.run([*cmd, "-Fc", "--no-owner", *args], stdout=f,
                               stderr=subprocess.PIPE, env=env, timeout=3600)
        if r.returncode != 0:
            raise BackupError(f"pg_dump başarısız: {r.stderr.decode(errors='replace').strip()[:500]}")

    def verify(self, path: Path) -> None:
        """Arşiv okunabiliyor ve tabloları içeriyor mu (bozuk yedek fark edilsin)."""
        if self.sqlite:
            c = sqlite3.connect(path)
            try:
                if c.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise BackupError("SQLite yedeği bozuk")
                tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            finally:
                c.close()
        else:
            cmd = self._tool("pg_restore")
            if cmd is None:
                return  # pg_restore yok: boyut kontrolüyle yetinilir
            with path.open("rb") as f:
                r = subprocess.run([*cmd, "--list"], stdin=f, capture_output=True, timeout=600)
            if r.returncode != 0:
                raise BackupError(f"yedek okunamadı: {r.stderr.decode(errors='replace')[:300]}")
            listing = r.stdout.decode(errors="replace")
            tables = {t for t in ("jobs", "utterances", "outbox") if f"TABLE DATA public {t} " in listing}
        missing = {"jobs", "utterances", "outbox"} - tables
        if missing:
            raise BackupError(f"yedekte tablo eksik: {sorted(missing)}")

    # --- saklama ---
    def backups(self) -> list[Path]:
        return sorted(p for p in self.dir.glob(f"{PREFIX}*") if not p.name.endswith(".part"))

    def prune(self, now: float | None = None) -> int:
        """keep_days'ten eski yedekleri siler; en yeni keep_min yedek her zaman kalır."""
        if self.keep_days <= 0:
            return 0
        now = now or time.time()
        items = self.backups()
        old = [p for p in items[:-self.keep_min] if now - p.stat().st_mtime > self.keep_days * 86400] \
            if len(items) > self.keep_min else []
        for p in old:
            p.unlink(missing_ok=True)
        return len(old)

    def age_hours(self, now: float | None = None) -> float | None:
        """En yeni yedeğin yaşı (saat); yedek yoksa None."""
        items = self.backups()
        if not items:
            return None
        return ((now or time.time()) - max(p.stat().st_mtime for p in items)) / 3600

    def due(self, every_hours: float, now: float | None = None) -> bool:
        if every_hours <= 0:
            return False
        age = self.age_hours(now)
        return age is None or age >= every_hours

    # --- geri yükleme ---
    def restore(self, path: str | Path) -> None:
        """Yedeği veritabanına geri yükler: yedekteki tablolar hedeftekilerin yerine geçer."""
        path = Path(path)
        if not path.is_file():
            raise BackupError(f"yedek yok: {path}")
        if self.sqlite:
            target = self.url.split("///", 1)[1]
            src, dst = sqlite3.connect(path), sqlite3.connect(target)
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
            return
        cmd = self._tool("pg_restore")
        if cmd is None:
            raise BackupError("pg_restore bulunamadı (pg_dump ile aynı yerde olmalı)")
        args, env = self._pg_args(inside_docker=cmd[0] == self.docker)
        with path.open("rb") as f:
            r = subprocess.run([*cmd, "--clean", "--if-exists", "--no-owner", "--single-transaction",
                                *args], stdin=f, capture_output=True, env=env, timeout=3600)
        if r.returncode != 0:
            raise BackupError(f"geri yükleme başarısız: {r.stderr.decode(errors='replace')[:500]}")

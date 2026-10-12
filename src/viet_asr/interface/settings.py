"""Ortam değişkenlerinden ayarlar (Docker'da docker-compose.yml verir)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _list(v: str) -> list[str]:
    return [x.strip().lower() for x in v.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.environ.get(
        "DATABASE_URL", "postgresql://asr:asr@localhost:5432/asr"))
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    models_dir: str = field(default_factory=lambda: os.environ.get("MODELS_DIR", "models"))
    data_dir: str = field(default_factory=lambda: os.environ.get("DATA_DIR", "data/uploads"))
    languages: list[str] = field(default_factory=lambda: _list(os.environ.get("LANGUAGES", "vi,tr")))
    vi_model: str | None = field(default_factory=lambda: os.environ.get("VI_MODEL") or None)
    tr_model: str | None = field(default_factory=lambda: os.environ.get("TR_MODEL") or None)
    threads: int = field(default_factory=lambda: int(os.environ.get("ONNX_THREADS", "2")))
    # 0: konuşmacı sayısı sesten bulunur (istekte num_speakers verilirse o kullanılır)
    speakers: int = field(default_factory=lambda: int(os.environ.get("NUM_SPEAKERS", "0")))
    diarizer: str = field(default_factory=lambda: os.environ.get("DIARIZER", "cluster"))
    hotwords: str | None = field(default_factory=lambda: os.environ.get("VI_HOTWORDS") or None)
    task_time_limit: int = field(default_factory=lambda: int(os.environ.get("TASK_TIME_LIMIT", "1800")))
    max_audio_minutes: float = field(default_factory=lambda: float(os.environ.get("MAX_AUDIO_MINUTES", "60")))
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("MAX_UPLOAD_MB", "100")))
    # sunucudaki bir dosya yolunu doğrudan kabul et (toplu iş için; yalnızca güvenilir ağda aç)
    allow_local_paths: bool = field(default_factory=lambda: os.environ.get(
        "ALLOW_LOCAL_PATHS", "0") == "1")
    # worker açılırken önceden yüklenecek diller (ilk işte gecikme olmasın)
    warmup: list[str] = field(default_factory=lambda: _list(os.environ.get("WARMUP_LANGS", "")))
    # dil tespiti worker'ı: Whisper-small'ı da baştan yükle (yoksa ilk zor seste ~5-15 sn gecikme)
    warmup_lid: bool = field(default_factory=lambda: os.environ.get("WARMUP_LID", "0") == "1")

    @property
    def concurrency(self) -> int:
        """CONCURRENCY=auto (varsayılan): kullanılabilir çekirdek / ONNX_THREADS."""
        v = os.environ.get("CONCURRENCY", "auto")
        if v != "auto":
            return max(1, int(v))
        return max(1, available_cpus() // max(1, self.threads))

    @property
    def models(self) -> dict[str, str]:
        return {k: v for k, v in (("vi", self.vi_model), ("tr", self.tr_model)) if v}


def available_cpus() -> int:
    """Bu sürecin gerçekten kullanabileceği çekirdek sayısı.

    os.cpu_count() Docker'da ana makinenin TÜM çekirdeklerini döndürür; konteynere
    sınır konduysa (cpus: 4) cgroup'tan okunur, yoksa sürecin izinli çekirdekleri.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Windows / macOS
        return os.cpu_count() or 1

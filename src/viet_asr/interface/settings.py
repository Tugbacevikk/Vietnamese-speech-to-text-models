"""Ortam değişkenlerinden ayarlar (Docker'da docker-compose.yml verir)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    speakers: int = field(default_factory=lambda: int(os.environ.get("NUM_SPEAKERS", "2")))
    hotwords: str | None = field(default_factory=lambda: os.environ.get("VI_HOTWORDS") or None)
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("MAX_UPLOAD_MB", "100")))
    # sunucudaki bir dosya yolunu doğrudan kabul et (toplu iş için; yalnızca güvenilir ağda aç)
    allow_local_paths: bool = field(default_factory=lambda: os.environ.get(
        "ALLOW_LOCAL_PATHS", "0") == "1")
    # worker açılırken önceden yüklenecek diller (ilk işte gecikme olmasın)
    warmup: list[str] = field(default_factory=lambda: _list(os.environ.get("WARMUP_LANGS", "")))
    # dil tespiti worker'ı: Whisper-small'ı da baştan yükle (yoksa ilk zor seste ~5-15 sn gecikme)
    warmup_lid: bool = field(default_factory=lambda: os.environ.get("WARMUP_LID", "0") == "1")

    @property
    def models(self) -> dict[str, str]:
        return {k: v for k, v in (("vi", self.vi_model), ("tr", self.tr_model)) if v}

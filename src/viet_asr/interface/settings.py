"""Ortam değişkenlerinden ayarlar.

Sıra: gerçek ortam değişkeni > proje kökündeki .env dosyası > buradaki varsayılan.
Windows'ta baslat.bat ilk çalıştırmada .env.example'dan .env oluşturur.
"""
from __future__ import annotations

import os

try:  # .env varsa yükle (python-dotenv yoksa sessizce geç)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass
from dataclasses import dataclass, field
from pathlib import Path


def _list(v: str) -> list[str]:
    return [x.strip().lower() for x in v.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # development: API anahtarı ve http/özel ağ callback'i serbest; production: zorunlu/kısıtlı
    app_env: str = field(default_factory=lambda: os.environ.get("APP_ENV", "development").lower())
    api_key: str = field(default_factory=lambda: os.environ.get("API_KEY", "").strip())
    webhook_secret: str = field(default_factory=lambda: os.environ.get("WEBHOOK_SECRET", "").strip())
    # boş değilse callback yalnızca bu adreslere gidebilir (iç ağ adresleri de yazılabilir)
    webhook_allowed_hosts: list[str] = field(default_factory=lambda: _list(
        os.environ.get("WEBHOOK_ALLOWED_HOSTS", "")))
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

    # saklama: biten işlerin ses dosyası N saat, iş kaydı ve sonucu N gün sonra silinir (0 = silme)
    audio_retention_hours: int = field(default_factory=lambda: int(os.environ.get(
        "AUDIO_RETENTION_HOURS", "24")))
    result_retention_days: int = field(default_factory=lambda: int(os.environ.get(
        "RESULT_RETENTION_DAYS", "30")))
    # RUNNING'de bundan uzun kalan iş askıda sayılır; MAX_JOB_ATTEMPTS denemeden sonra FAILED
    processing_stale_sec: int = field(default_factory=lambda: int(os.environ.get(
        "PROCESSING_STALE_SEC", "1800")))
    max_job_attempts: int = field(default_factory=lambda: int(os.environ.get("MAX_JOB_ATTEMPTS", "3")))
    # modeller yüklenirken MANIFEST.json'daki sha256 ile karşılaştır
    verify_models: bool = field(default_factory=lambda: os.environ.get("VERIFY_MODELS", "1") == "1")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

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

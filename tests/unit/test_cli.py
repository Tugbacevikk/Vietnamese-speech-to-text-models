"""Komut satırı: ayarlar ortam değişkeninden (sunucuyla aynı kaynak) gelir, argüman ezer."""
from viet_asr.interface import cli


class _Dur(Exception):
    pass


def yakala(monkeypatch, argv, env=None):
    """main() hat kurulmadan hemen önce durdurulur; hat ayarlarını (Settings) döndürür."""
    for k in ("WHISPER_DEVICE", "WHISPER_COMPUTE_TYPE", "WHISPER_BEAM_SIZE", "WHISPER_THREADS", "TR_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    got = {}

    def sahte(s, workers=1):
        got["s"] = s
        raise _Dur

    monkeypatch.setattr(cli, "config_from_settings", sahte)
    try:
        cli.main(argv)
    except _Dur:
        pass
    return got["s"]


def test_ortam_degiskenleri_okunur(monkeypatch):
    s = yakala(monkeypatch, ["a.wav"], {"WHISPER_DEVICE": "cpu", "WHISPER_COMPUTE_TYPE": "int8",
                                        "WHISPER_BEAM_SIZE": "1", "TR_MODEL": "faster-whisper-small"})
    assert (s.whisper_device, s.whisper_compute_type, s.whisper_beam_size) == ("cpu", "int8", 1)
    assert s.models["tr"] == "faster-whisper-small"


def test_arguman_ortam_degiskenini_ezer(monkeypatch):
    s = yakala(monkeypatch, ["a.wav", "--whisper-device", "cpu", "--whisper-beam-size", "2",
                             "--whisper-threads", "4", "--tr-model", "omnilingual-300m"],
               {"WHISPER_DEVICE": "cuda", "WHISPER_BEAM_SIZE": "5"})
    assert (s.whisper_device, s.whisper_beam_size, s.whisper_threads) == ("cpu", 2, 4)
    assert s.models["tr"] == "omnilingual-300m"


def test_hicbiri_verilmezse_varsayilan(monkeypatch):
    s = yakala(monkeypatch, ["a.wav"])
    assert (s.whisper_device, s.whisper_beam_size) == ("auto", 5)

"""İnce ayar döngüsü birim testleri."""
import json
from pathlib import Path

from scripts.ince_ayar_dongusu import check_leakage_with_test_sets, run_cycle


def test_leak_check_clean(tmp_path):
    train_tsv = tmp_path / "train.tsv"
    train_tsv.write_text("dosya\tmetin\tkayit\tilk_metin\tkonusmaci\n"
                         "audio/1.wav\tX\trec1\tX\tspk1\n", encoding="utf-8")
    test_tsv = tmp_path / "test.tsv"
    test_tsv.write_text("dosya\tmetin\tkayit\tilk_metin\tkonusmaci\n"
                        "audio/2.wav\tY\trec2\tY\tspk2\n", encoding="utf-8")

    leaks = check_leakage_with_test_sets(train_tsv, [test_tsv])
    assert leaks == []


def test_leak_check_detects_same_recording(tmp_path):
    train_tsv = tmp_path / "train.tsv"
    train_tsv.write_text("dosya\tmetin\tkayit\tilk_metin\tkonusmaci\n"
                         "audio/1.wav\tX\trec_common\tX\tspk1\n", encoding="utf-8")
    test_tsv = tmp_path / "test.tsv"
    test_tsv.write_text("dosya\tmetin\tkayit\tilk_metin\tkonusmaci\n"
                        "audio/2.wav\tY\trec_common\tY\tspk2\n", encoding="utf-8")

    leaks = check_leakage_with_test_sets(train_tsv, [test_tsv])
    assert len(leaks) >= 1
    assert "rec_common" in leaks[0]


def test_run_cycle_threshold_not_met(tmp_path):
    db_file = tmp_path / "test.db"
    work_dir = tmp_path / "ft_work"
    status = run_cycle(esik=50, zorla=False, kuru_calisma=True,
                       db_url=f"sqlite:///{db_file}", work_dir=work_dir)
    assert status["sonuc"] == "esik_bekleniyor"
    assert status["esik_asildi"] is False
    assert (work_dir / "dongu_raporu.json").is_file()

"""İnce ayar döngüsü birim testleri."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ince_ayar_dongusu import check_leakage_with_test_sets, run, threshold_status


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


def test_esik_hem_sayi_hem_sure_ister():
    class C:
        def __init__(self, s, e): self.start, self.end = s, e
    corrs = [C(0, 10)] * 3
    assert threshold_status(corrs, esik=3, esik_sn=30)["esik_asildi"] is True
    assert threshold_status(corrs, esik=3, esik_sn=31)["esik_asildi"] is False      # süre yetmiyor
    assert threshold_status(corrs, esik=4, esik_sn=10)["esik_asildi"] is False      # sayı yetmiyor


def test_durum_hicbir_sey_yazmaz_ve_esik_bekler(tmp_path):
    work = tmp_path / "ft"
    db = f"sqlite:///{tmp_path / 'test.db'}"
    st = run("durum", 50, 3600.0, False, db, "vi", work, [])
    assert st["onayli_duzeltme_sayisi"] == 0 and not work.exists()        # durum: yan etki yok
    st = run("hazirla", 50, 3600.0, False, db, "vi", work, [])
    assert st["sonuc"] == "esik_bekleniyor" and not (work / "veri").exists()


def test_terfi_ve_sahte_olcum_yok():
    """Betik model terfisi yapmaz; Kapı 2 gerçek ölçüm olan finetune/kapi2.py'dir."""
    src = (Path(__file__).resolve().parents[2] / "scripts" / "ince_ayar_dongusu.py").read_text(encoding="utf-8")
    assert "mock_" not in src and "copytree" not in src and "gate2(" not in src

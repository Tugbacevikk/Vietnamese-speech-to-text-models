"""İnce ayar için veri eşiği izleyicisi ve veri hazırlığı.

Arayüzde düzeltilen ve "veri setine ekle" onayı verilen satırlar yeterince birikince (varsayılan
500 satır ve 1 saat ses) eğitime hazır olduklarını söyler ve verileri hazırlar. EĞİTİM VE MODEL
TERFİSİ BURADA YAPILMAZ: eğitim WSL2 + GPU ister (finetune/egit.py), model kararı ise gerçek bir
ölçüm olan Kapı 2'dir (finetune/kapi2.py). Bu betik o ikisini taklit etmez, nasıl çalıştırılacağını yazar.
(Eşikler sezgiseldir, bir ölçüme dayanmaz; birkaç saatlik veriyle denemek için --esik ve --esik-saat.)

    python scripts/ince_ayar_dongusu.py durum                 # yalnızca sayar, hiçbir şey yazmaz
    python scripts/ince_ayar_dongusu.py hazirla               # eşik doluysa veriyi dışa aktarır
    python scripts/ince_ayar_dongusu.py hazirla --zorla --test-tsv test/tek/referans.tsv

Sızıntı: eğitim verisi, verdiğiniz test TSV'leriyle (--test-tsv, birden çok kez) karşılaştırılır:
aynı ses ya da aynı kayıt iki yerde ise durur. Test TSV'si verilmezse bu denetim YAPILMAZ ve
rapor bunu açıkça yazar (finetune/veri_hazirla.py bölme sırasında zaten sızıntıyı engeller).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "finetune"))

from ortak import audio_hash, leak_check  # noqa: E402

from viet_asr.infrastructure.db import SqlAlchemyJobStore  # noqa: E402
from viet_asr.interface.settings import Settings  # noqa: E402

EN_KISA, EN_UZUN = 1.0, 20.0   # icefall sınırı (scripts/duzeltmeleri_disa_aktar.py ile aynı)


def approved_corrections(store, dil: str, min_dur: float = EN_KISA, max_dur: float = EN_UZUN) -> list:
    """Veri setine aktarma onayı verilmiş, eğitime uygun uzunlukta düzeltilmiş satırlar."""
    return [c for c in store.corrections(consented_only=True)
            if (not dil or c.language == dil) and min_dur <= c.end - c.start <= max_dur]


def threshold_status(corrs: list, esik: int, esik_sn: float) -> dict:
    total = sum(c.end - c.start for c in corrs)
    return {"onayli_duzeltme_sayisi": len(corrs), "toplam_sure_sn": round(total, 1),
            "esik_sayi": esik, "esik_sure_sn": esik_sn,
            "esik_asildi": len(corrs) >= esik and total >= esik_sn}


def _rows(tsv: Path) -> list[tuple[str, str]]:
    out = []
    with tsv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            p = Path(row["dosya"])
            if not p.is_absolute() and not p.is_file():
                p = tsv.parent / p
            out.append((audio_hash(p) if p.is_file() else row["dosya"], row.get("kayit", "")))
    return out


def check_leakage_with_test_sets(train_tsv: Path, test_tsvs: list[Path]) -> list[str]:
    """Eğitim TSV'si ile test TSV'leri arasında aynı ses / aynı kayıt var mı (sorun listesi)."""
    splits = {"train": _rows(train_tsv)}
    for t in test_tsvs:
        splits[t.stem if t.stem not in splits else f"{t.parent.name}_{t.stem}"] = _rows(t)
    return leak_check(splits)


NEXT_STEPS = """Sonraki adımlar (GPU'lu makinede, finetune/README.md):
  1. python finetune/veri_hazirla.py --vietmed --train {tsv} --cikti ~/ft/veri
  2. python finetune/egit.py --icefall ~/ft/icefall --veri ~/ft/veri --model ~/ft/temel_model --exp-dir ~/ft/exp
  3. bash finetune/disari_aktar.sh ~/ft ~/ft/exp 10 3 models/vi-ince-ayar-<tarih>
  4. python finetune/kapi2.py --yeni models/vi-ince-ayar-<tarih> --tibbi <test.tsv> --genel <genel.tsv>
  5. Kapı 2 GEÇERSE .env: VI_MODEL=models/vi-ince-ayar-<tarih>   (geçmezse model kullanılmaz)"""


def run(komut: str, esik: int, esik_sn: float, zorla: bool, db_url: str | None, dil: str,
        work_dir: Path, test_tsvs: list[Path]) -> dict:
    store = SqlAlchemyJobStore(db_url or Settings().local_db)
    store.migrate()
    corrs = approved_corrections(store, dil)
    status = {"tarih": datetime.now(timezone.utc).isoformat(), "dil": dil, **threshold_status(corrs, esik, esik_sn)}

    print(f"Dil {dil}: {status['onayli_duzeltme_sayisi']} onaylı satır (eşik {esik}), "
          f"{status['toplam_sure_sn'] / 60:.1f} dk ses (eşik {esik_sn / 60:.0f} dk) -> "
          f"{'EŞİK AŞILDI' if status['esik_asildi'] else 'eşik henüz aşılmadı'}")
    if komut == "durum":
        return status                                   # yalnızca okur
    if not status["esik_asildi"] and not zorla:
        status["sonuc"] = "esik_bekleniyor"
        print("Veri hazırlanmadı (--zorla ile eşik beklenmeden hazırlanır).")
        return status

    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "veri"
    cmd = [sys.executable, str(ROOT / "scripts" / "duzeltmeleri_disa_aktar.py"), "--cikti", str(out), "--dil", dil]
    if db_url:
        cmd += ["--db", db_url]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        msg = (res.stderr or res.stdout).strip()
        status.update(sonuc="veri_yok" if "0 satır yazıldı" in msg else "disa_aktarma_hatasi", hata=msg[-500:])
        print("Eğitime uygun satır çıkmadı:" if status["sonuc"] == "veri_yok" else "Dışa aktarma başarısız:", msg[-300:])
        return status
    tsv = out / "liste.tsv"
    status["veri_tsv"] = str(tsv)
    print(res.stdout.strip().splitlines()[-1] if res.stdout.strip() else f"Veri: {tsv}")

    if test_tsvs:
        leaks = check_leakage_with_test_sets(tsv, test_tsvs)
        status["sizinti_denetimi"] = "yapildi"
        status["sizintilar"] = leaks
        if leaks:
            status["sonuc"] = "sizinti_var"
            print(f"SIZINTI: {len(leaks)} sorun; eğitimden önce düzeltin:")
            for x in leaks[:5]:
                print("  -", x)
            return status
        print("Sızıntı denetimi: temiz.")
    else:
        status["sizinti_denetimi"] = "yapilmadi (test TSV'si verilmedi)"
        print("UYARI: sızıntı denetimi yapılmadı (--test-tsv verilmedi).")

    status["sonuc"] = "veri_hazir"
    status["sonraki_adimlar"] = NEXT_STEPS.format(tsv=tsv)
    print("\n" + status["sonraki_adimlar"])
    (work_dir / "dongu_raporu.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    with (work_dir / "tarihce.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(status, ensure_ascii=False) + "\n")
    return status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="komut", required=True)
    for name, hlp in (("durum", "sayar, hiçbir şey yazmaz"), ("hazirla", "eşik doluysa eğitim verisini dışa aktarır")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--esik", type=int, default=500, help="gerekli onaylı satır (varsayılan 500)")
        s.add_argument("--esik-saat", type=float, default=1.0, help="gerekli toplam ses, saat (varsayılan 1)")
        s.add_argument("--dil", default="vi")
        s.add_argument("--db", default=None)
        if name == "hazirla":
            s.add_argument("--zorla", action="store_true", help="eşik dolmasa da hazırla")
            s.add_argument("--klasor", type=Path, default=Path("data/ft_dongu"))
            s.add_argument("--test-tsv", type=Path, action="append", default=[], help="sızıntı denetimi için test TSV'si")
    a = ap.parse_args(argv)
    res = run(a.komut, a.esik, a.esik_saat * 3600.0, getattr(a, "zorla", False), a.db, a.dil,
              getattr(a, "klasor", Path("data/ft_dongu")), getattr(a, "test_tsv", []))
    return 0 if res.get("sonuc") in (None, "veri_hazir") else 1


if __name__ == "__main__":
    raise SystemExit(main())

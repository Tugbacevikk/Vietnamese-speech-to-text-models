"""Vietnamca ve Türkçe Çevrimdışı Noktalama ve Büyük Harf Doğruluk & Hız Ölçümü.

Kullanım:
    python benchmarks/noktalama_testi.py
    python benchmarks/noktalama_testi.py --tsv data/noktalama_referans.tsv
    python benchmarks/noktalama_testi.py --ornek-olustur

Ölçülenler:
- Noktalama F1 skoru: Virgül (,), Nokta (.), Soru işareti (?) için Doğruluk (Precision),
  Duyarlılık (Recall) ve F1 skoru.
- Büyük harf doğruluğu: Cümle başı ve özel isimlerin büyük harfe dönüştürülme oranı.
- Hız ve Gecikme: Cümle başına milisaniye gecikme, saniyede işlenen cümle/kelime sayısı.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viet_asr.application.punctuator import punctuate_text, punctuate_words  # noqa: E402
from viet_asr.domain import Word  # noqa: E402

# Doğruluk ve hız ölçümü için örnek tıbbi ve genel diyalog referans kümesi
ORNEK_REFERANS_KUMESI = [
    # (ham_girdi, dogru_referans)
    ("thưa bác sĩ em bị đau đầu và sốt cao từ tối qua",
     "Thưa bác sĩ, em bị đau đầu ve sốt cao từ tối qua."),
    ("bác sĩ cho em hỏi bệnh này có lây không",
     "Bác sĩ cho em hỏi bệnh này có lây không?"),
    ("dạ em hiểu rồi em sẽ uống thuốc theo đơn",
     "Dạ, em hiểu rồi em sẽ uống thuốc theo đơn."),
    ("vâng tôi đã đi xét nghiệm máu ở hà nội",
     "Vâng, tôi đã đi xét nghiệm máu ở Hà Nội."),
    ("tuy nhiên triệu chứng đau bụng vẫn chưa thuyên giảm",
     "Tuy nhiên, triệu chứng đau bụng vẫn chưa thuyên giảm."),
    ("nguyên nhân gây ra hiện tượng này là gì",
     "Nguyên nhân gây ra hiện tượng này là gì?"),
    ("ngoài ra người bệnh cần tái khám sau hai tuần",
     "Ngoài ra, người bệnh cần tái khám sau hai tuần."),
    ("bệnh viện bạch mai là cơ sở y tế hàng đầu việt nam",
     "Bệnh viện Bạch Mai là cơ sở y tế hàng đầu Việt Nam."),
    ("em cần kiêng những thực phẩm nào",
     "Em cần kiêng những thực phẩm nào?"),
    ("đặc biệt là bệnh nhân có tiền sử tim mạch",
     "Đặc biệt, là bệnh nhân có tiền sử tim mạch."),
    ("chi phí phẫu thuật khoảng bao nhiêu",
     "Chi phí phẫu thuật khoảng bao nhiêu?"),
    ("thuốc này nên uống trước hay sau bữa ăn",
     "Thuốc này nên uống trước hay sau bữa ăn?"),
    ("bác sĩ có thể giải thích thêm về kết quả chụp ct không",
     "Bác sĩ có thể giải thích thêm về kết quả chụp CT không?"),
    ("hôm nay tôi cảm thấy sức khỏe đã tốt hơn nhiều",
     "Hôm nay tôi cảm thấy sức khỏe đã tốt hơn nhiều."),
    ("do đó việc điều trị cần được tuân thủ nghiêm ngặt",
     "Do đó, việc điều trị cần được tuân thủ nghiêm ngặt."),
    ("ví dụ như các loại rau xanh và trái cây tươi",
     "Ví dụ, như các loại rau xanh ve trái cây tươi."),
    ("thực ra tôi đã từng mắc bệnh này cách đây ba năm",
     "Thực ra, tôi đã từng mắc bệnh này cách đây ba năm."),
    ("ngày mai bác sĩ có làm việc tại phòng khám không",
     "Ngày mai bác sĩ có làm việc tại phòng khám không?"),
    ("tôi sinh sống tại đà nẵng và làm việc tại tp hcm",
     "Tôi sinh sống tại Đà Nẵng và làm việc tại TP.HCM."),
    ("bộ y tế khuyến cáo người dân đeo khẩu trang tại nơi công cộng",
     "Bộ Y tế khuyến cáo người dân đeo khẩu trang tại nơi công cộng."),
]


def _extract_punctuation_tokens(text: str) -> list[tuple[str, str]]:
    """Metni (kelime_koku, noktalama) ikililerine ayırır.
    Örnek: "Thưa bác sĩ, em sốt." -> [('thưa', ''), ('bác', ''), ('sĩ', ','), ('em', ''), ('sốt', '.')]
    """
    tokens = text.strip().split()
    res = []
    for t in tokens:
        m = re.match(r"^([^\W\d_]+|[0-9]+)([\.,\?!;:~]*)$", t, flags=re.UNICODE)
        if m:
            word, punct = m.group(1).lower(), m.group(2)
            res.append((word, punct))
        else:
            cleaned = re.sub(r"[\.,\?!;:~]", "", t).lower()
            punct = "".join(ch for ch in t if ch in ".,?!;:~")
            res.append((cleaned, punct))
    return res


def evaluate_punctuation(pairs: list[tuple[str, str]], lang: str = "vi") -> dict:
    """Noktalama ve büyük harf doğruluğunu ölçer."""
    punct_types = [",", ".", "?"]
    stats = {p: {"tp": 0, "fp": 0, "fn": 0} for p in punct_types}

    total_words = 0
    correct_case_words = 0

    t0 = time.perf_counter()
    predicted_sentences = []
    for raw_inp, ref in pairs:
        pred = punctuate_text(raw_inp, lang=lang)
        predicted_sentences.append(pred)
    elapsed_sec = time.perf_counter() - t0

    for (_, ref), pred in zip(pairs, predicted_sentences):
        ref_tokens = ref.split()
        pred_tokens = pred.split()

        ref_pairs = _extract_punctuation_tokens(ref)
        pred_pairs = _extract_punctuation_tokens(pred)

        # Kelime bazında büyük/küçük harf doğruluğu
        for r_tok, p_tok in zip(ref_tokens, pred_tokens):
            total_words += 1
            r_clean = re.sub(r"[^\w]", "", r_tok)
            p_clean = re.sub(r"[^\w]", "", p_tok)
            if r_clean == p_clean:
                correct_case_words += 1

        # Noktalama işaretleri karşılaştırması
        min_len = min(len(ref_pairs), len(pred_pairs))
        for i in range(min_len):
            r_w, r_p = ref_pairs[i]
            p_w, p_p = pred_pairs[i]

            for p_type in punct_types:
                in_ref = p_type in r_p
                in_pred = p_type in p_p
                if in_ref and in_pred:
                    stats[p_type]["tp"] += 1
                elif not in_ref and in_pred:
                    stats[p_type]["fp"] += 1
                elif in_ref and not in_pred:
                    stats[p_type]["fn"] += 1

    # Metrikleri hesapla
    punct_metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    for p_type, counts in stats.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        punct_metrics[p_type] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec * 100, 1),
            "recall": round(rec * 100, 1),
            "f1": round(f1 * 100, 1),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    ov_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    ov_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    ov_f1 = (2 * ov_prec * ov_rec) / (ov_prec + ov_rec) if (ov_prec + ov_rec) > 0 else 0.0

    case_acc = (correct_case_words / total_words * 100) if total_words > 0 else 100.0

    n_sents = len(pairs)
    latency_ms_per_sent = (elapsed_sec / n_sents * 1000) if n_sents > 0 else 0.0
    sents_per_sec = (n_sents / elapsed_sec) if elapsed_sec > 0 else 0.0
    words_per_sec = (total_words / elapsed_sec) if elapsed_sec > 0 else 0.0

    return {
        "ornek_sayisi": n_sents,
        "kelime_sayisi": total_words,
        "gecen_sure_ms": round(elapsed_sec * 1000, 2),
        "gecikme_ms_cumle": round(latency_ms_per_sent, 3),
        "hiz_cumle_sn": round(sents_per_sec, 1),
        "hiz_kelime_sn": round(words_per_sec, 1),
        "buyuk_harf_dogruluk": round(case_acc, 1),
        "genel_noktalama": {
            "precision": round(ov_prec * 100, 1),
            "recall": round(ov_rec * 100, 1),
            "f1": round(ov_f1 * 100, 1),
        },
        "isaretler": punct_metrics,
    }


def ornek_dosya_olustur(hedef: Path) -> None:
    """Örnek TSV dosyası oluşturur."""
    hedef.parent.mkdir(parents=True, exist_ok=True)
    with hedef.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ham_metin", "dogru_metin"])
        for raw, ref in ORNEK_REFERANS_KUMESI:
            w.writerow([raw, ref])
    print(f"Ornek noktalama referans sablonu olusturuldu: {hedef}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tsv", type=Path, default=None, help="ham_metin ve dogru_metin iceren TSV dosyasi")
    parser.add_argument("--cikti", type=Path, default=Path("sonuc_noktalama.json"), help="rapor JSON dosyasi")
    parser.add_argument("--ornek-olustur", action="store_true", help="ornek TSV dosyasi olustur ve cik")
    parser.add_argument("--dil", default="vi", help="dil kodu (vi / tr)")
    args = parser.parse_args(argv)

    if args.ornek_olustur:
        target = Path("data/ornek_noktalama/noktalama_referans_sablon.tsv")
        ornek_dosya_olustur(target)
        return 0

    pairs: list[tuple[str, str]] = []
    if args.tsv and args.tsv.exists():
        with args.tsv.open(encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    pairs.append((row[0].strip(), row[1].strip()))
        print(f"TSV'den {len(pairs)} kayit okundu: {args.tsv}")
    else:
        pairs = ORNEK_REFERANS_KUMESI
        print(f"Yerlesik {len(pairs)} ornek diyalog kumeli test seti kullaniliyor.")

    report = evaluate_punctuation(pairs, lang=args.dil)

    # Windows cp1254 uyumlu temiz konsol ciktisi
    print("\n" + "=" * 60)
    print("NOKTALAMA VE BUYUK HARF DOGRULUK & HIZ RAPORU")
    print("=" * 60)
    print(f"Ornek Sayisi         : {report['ornek_sayisi']}")
    print(f"Toplam Kelime        : {report['kelime_sayisi']}")
    print(f"Buyuk Harf Dogrulugu : %{report['buyuk_harf_dogruluk']}")
    print("-" * 60)
    print("NOKTALAMA F1 SKORLARI:")
    gen = report["genel_noktalama"]
    print(f"  Genel Noktalama    : Precision: %{gen['precision']} | Recall: %{gen['recall']} | F1: %{gen['f1']}")
    for punct, m in report["isaretler"].items():
        name = "Virgul (,)" if punct == "," else ("Nokta (.)" if punct == "." else "Soru Isareti (?)")
        print(f"  {name:18} : Precision: %{m['precision']:<5} | Recall: %{m['recall']:<5} | F1: %{m['f1']}")
    print("-" * 60)
    print("HIZ VE GECIKME:")
    print(f"  Gecikme (Ortalama) : {report['gecikme_ms_cumle']:.3f} ms/cumle")
    print(f"  Hiz                : {report['hiz_cumle_sn']:.1f} cumle/sn ({report['hiz_kelime_sn']:.1f} kelime/sn)")
    print(f"  Toplam Islem Suresi: {report['gecen_sure_ms']:.2f} ms")
    print("=" * 60)

    with args.cikti.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nAyrintili rapor JSON olarak kaydedildi: {args.cikti}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

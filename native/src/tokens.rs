//! Model parçalarını (token) kelimelere birleştirme.
//!
//! `viet_asr.infrastructure.sherpa_asr.tokens_to_words` ile BİREBİR aynı kural (başvuru o).
//! Dile özgü yazım düzeltmesi (normalizer) Python'da kalır: burada yalnızca gruplama,
//! mutlak zaman, bitiş zamanı ve log olasılık ortalaması hesaplanır.
//!
//! İki biçim:
//!   - BPE (Zipformer): "▁BÁC", "▁S", "Ĩ"   -> "▁" ya da boşlukla başlayan parça yeni kelime
//!   - harf (Omnilingual): "m","e"," ","d"  -> tek başına ayırıcı yeni kelime başlatır

/// Ham kelime: normalizer'dan önceki metin, mutlak başlangıç/bitiş ve
/// parçaların log olasılık ortalaması (verilmediyse None).
#[derive(Debug, Clone, PartialEq)]
pub struct RawWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub mean_log_prob: Option<f64>,
}

fn is_sep(c: char) -> bool {
    c == ' ' || c == '\u{2581}'
}

/// `log_probs` uzunluğu `tokens` ile aynı değilse yok sayılır (Python'daki gibi).
/// `tokens` ile `timestamps` farklı uzunluktaysa kısa olan kadar okunur (Python `zip`).
pub fn group(
    tokens: &[String],
    timestamps: &[f64],
    seg_start: f64,
    seg_end: f64,
    max_duration: f64,
    log_probs: Option<&[f64]>,
) -> Vec<RawWord> {
    let lps = log_probs.filter(|l| !l.is_empty() && l.len() == tokens.len());
    // (metin, başlangıç, log olasılık toplamı, sayısı)
    let mut words: Vec<(String, f64, f64, usize)> = Vec::new();
    let mut new_word = true;
    for (n, (tok, &t)) in tokens.iter().zip(timestamps).enumerate() {
        if tok.trim_matches(is_sep).is_empty() {
            new_word = true;
            continue;
        }
        let (lp_sum, lp_n) = match lps {
            Some(l) => (l[n], 1),
            None => (0.0, 0),
        };
        let starts_word = tok.starts_with(is_sep);
        match words.last_mut() {
            Some(w) if !(starts_word || new_word) => {
                w.0.push_str(tok);
                w.2 += lp_sum;
                w.3 += lp_n;
            }
            _ => {
                words.push((tok.trim_start_matches(is_sep).to_string(), seg_start + t, lp_sum, lp_n));
                new_word = false;
            }
        }
    }
    let starts: Vec<f64> = words.iter().map(|w| w.1).collect();
    words
        .into_iter()
        .enumerate()
        .map(|(i, (text, start, sum, n))| {
            let nxt = starts.get(i + 1).copied().unwrap_or(seg_end);
            RawWord {
                text,
                start,
                end: start.max(nxt.min(start + max_duration)),
                mean_log_prob: if n > 0 { Some(sum / n as f64) } else { None },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn bpe_parcalari_birlesir() {
        let w = group(&s(&["▁BÁC", "▁S", "Ĩ"]), &[0.0, 0.5, 0.7], 10.0, 11.5, 1.0, None);
        assert_eq!(w.len(), 2);
        assert_eq!(w[1].text, "SĨ");
        assert_eq!((w[0].start, w[0].end), (10.0, 10.5));
        assert_eq!((w[1].start, w[1].end), (10.5, 11.5));
        assert_eq!(w[0].mean_log_prob, None);
    }

    #[test]
    fn harf_bicimi_ve_ayirici() {
        let w = group(&s(&["m", "e", " ", "d"]), &[0.0, 0.1, 0.2, 0.3], 0.0, 1.0, 1.0, None);
        let t: Vec<&str> = w.iter().map(|x| x.text.as_str()).collect();
        assert_eq!(t, ["me", "d"]);
    }

    #[test]
    fn log_olasilik_ortalamasi_ve_uzunluk_uyusmazligi() {
        let w = group(&s(&["▁A", "B"]), &[0.0, 0.1], 0.0, 1.0, 1.0, Some(&[-0.2, -0.4]));
        assert!((w[0].mean_log_prob.unwrap() + 0.3).abs() < 1e-12);
        let w = group(&s(&["▁A", "B"]), &[0.0, 0.1], 0.0, 1.0, 1.0, Some(&[-0.2]));
        assert_eq!(w[0].mean_log_prob, None);
    }

    #[test]
    fn bitis_en_fazla_max_duration() {
        let w = group(&s(&["▁A", "▁B"]), &[0.0, 3.0], 0.0, 5.0, 1.0, None);
        assert_eq!(w[0].end, 1.0);
        assert!(group(&[], &[], 0.0, 1.0, 1.0, None).is_empty());
    }
}

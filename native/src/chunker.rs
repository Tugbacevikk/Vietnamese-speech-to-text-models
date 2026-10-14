//! Enerji bölücü: sesin tamamını en sessiz anlardan, en fazla `max_len` sn'lik parçalara böler.
//!
//! `viet_asr.infrastructure.chunker.energy_cuts` ile BİREBİR aynı kural (başvuru o).
//! Enerji f64 ile hesaplanır. Dönen: sessiz olmayan parçaların [başlangıç, bitiş) örnek indeksi.

pub fn energy_cuts(
    audio: &[f32],
    sr: usize,
    max_len: f64,
    min_len: f64,
    frame: f64,
    silence_rms: f64,
) -> Vec<(usize, usize)> {
    let n = audio.len();
    let f = (frame * sr as f64) as usize;
    if n == 0 || f == 0 {
        return Vec::new();
    }
    let frames = n / f;
    let energy: Vec<f64> = (0..frames)
        .map(|i| rms(&audio[i * f..(i + 1) * f]))
        .collect();
    let max_samples = max_len * sr as f64;
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < n {
        let end = if (n - start) as f64 <= max_samples {
            n
        } else {
            let lo = (start + (min_len * sr as f64) as usize) / f;
            let hi = frames.min((start + max_samples as usize) / f);
            let k = if hi > lo {
                // numpy argmin: eşitlikte ilk en küçük
                let mut best = lo;
                for i in lo + 1..hi {
                    if energy[i] < energy[best] {
                        best = i;
                    }
                }
                best
            } else {
                hi
            };
            k * f + f / 2
        };
        if rms(&audio[start..end]) >= silence_rms {
            out.push((start, end));
        }
        start = end;
    }
    out
}

fn rms(x: &[f32]) -> f64 {
    if x.is_empty() {
        return f64::NAN; // numpy: boş dizinin ortalaması nan (karşılaştırma False)
    }
    let s: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    (s / x.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kisa_ses_tek_parca_sessiz_atilir() {
        let a = vec![0.1f32; 16000 * 3];
        assert_eq!(energy_cuts(&a, 16000, 15.0, 5.0, 0.1, 1e-3), vec![(0, 48000)]);
        let z = vec![0.0f32; 16000 * 3];
        assert!(energy_cuts(&z, 16000, 15.0, 5.0, 0.1, 1e-3).is_empty());
    }

    #[test]
    fn uzun_ses_en_sessiz_yerden_kesilir() {
        let mut a = vec![0.1f32; 16000 * 30];
        for v in &mut a[16000 * 9..16000 * 9 + 1600] {
            *v = 0.0; // 9. saniyede 100 ms sessizlik
        }
        let c = energy_cuts(&a, 16000, 15.0, 5.0, 0.1, 1e-3);
        assert_eq!(c[0], (0, 16000 * 9 + 800));
        assert!(c.iter().all(|(s, e)| e - s <= 16000 * 15));
    }
}

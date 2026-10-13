//! Birleştirme: ASR kelimelerini konuşmacı aralıklarına atar.
//!
//! `viet_asr.application.fusion.FusionEngine.assign` ile BİREBİR aynı kurallar (eşitlik
//! durumlarında bile); Python sürümü başvuru kaynağıdır, `tests/unit/test_native.py`
//! rastgele girdilerle iki sonucu karşılaştırır. Dış bağımlılık yok: düz diziler girer,
//! düz dizi çıkar (Python'dan da, ileride Rust servisinden de aynı çağrı).

/// Konuşmacısı güvenle belirlenemeyen kelime.
pub const UNKNOWN: i64 = -1;

#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub speaker: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub min_dominance: f64,
    pub max_gap: f64,
}

/// Her kelime için konuşmacı numarası (`UNKNOWN` = belirsiz). `words`: (başlangıç, bitiş).
/// Tek ya da hiç konuşmacı yoksa kısa yol çağıran tarafta (Python ile aynı): burada yok.
pub fn assign(words: &[(f64, f64)], segments: &[Segment], p: Params) -> Vec<i64> {
    let mut segs = segments.to_vec();
    // Python'daki sorted(..., key=start) kararlı sıralama: eşit başlangıçta giriş sırası korunur
    segs.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    let starts: Vec<f64> = segs.iter().map(|s| s.start).collect();
    let mut max_end = Vec::with_capacity(segs.len());
    let mut m = f64::NEG_INFINITY;
    for s in &segs {
        m = m.max(s.end);
        max_end.push(m);
    }
    words.iter().map(|&(ws, we)| one(ws, we, &segs, &starts, &max_end, p)).collect()
}

fn one(ws: f64, we: f64, segs: &[Segment], starts: &[f64], max_end: &[f64], p: Params) -> i64 {
    if segs.is_empty() {
        return UNKNOWN;
    }
    let mid = (ws + we) / 2.0;
    let hi = starts.partition_point(|&s| s <= mid); // bisect_right
    let mut lo = hi as isize - 1;
    while lo >= 0 && max_end[lo as usize] > mid {
        lo -= 1;
    }
    let hits: Vec<&Segment> = segs[(lo + 1) as usize..hi]
        .iter()
        .filter(|s| s.start <= mid && mid < s.end)
        .collect();
    let mut distinct: Vec<i64> = Vec::new();
    for h in &hits {
        if !distinct.contains(&h.speaker) {
            distinct.push(h.speaker);
        }
    }
    match distinct.len() {
        1 => distinct[0],
        0 => nearest(mid, segs, hi, p),
        _ => by_overlap(ws, we, &hits, p),
    }
}

fn by_overlap(ws: f64, we: f64, hits: &[&Segment], p: Params) -> i64 {
    // ekleme sırasını koruyan sözlük (Python dict gibi): eşitlikte ilk gelen kazanır
    let mut share: Vec<(i64, f64)> = Vec::new();
    for s in hits {
        let ov = (we.min(s.end) - ws.max(s.start)).max(0.0);
        match share.iter_mut().find(|(k, _)| *k == s.speaker) {
            Some(e) => e.1 += ov,
            None => share.push((s.speaker, ov)),
        }
    }
    let total: f64 = share.iter().map(|(_, v)| v).sum();
    if total <= 0.0 {
        return UNKNOWN;
    }
    let mut best = share[0];
    for &e in &share[1..] {
        if e.1 > best.1 {
            best = e;
        }
    }
    if best.1 / total >= p.min_dominance { best.0 } else { UNKNOWN }
}

fn nearest(mid: f64, segs: &[Segment], hi: usize, p: Params) -> i64 {
    let (mut best, mut dist) = (None, f64::INFINITY);
    if hi < segs.len() {
        best = Some(segs[hi].speaker);
        dist = segs[hi].start - mid;
    }
    for s in &segs[..hi] {
        let d = mid - s.end;
        if 0.0 <= d && d < dist {
            best = Some(s.speaker);
            dist = d;
        }
    }
    match best {
        Some(b) if dist <= p.max_gap => b,
        _ => UNKNOWN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: Params = Params { min_dominance: 0.6, max_gap: 0.5 };

    fn seg(start: f64, end: f64, speaker: i64) -> Segment {
        Segment { start, end, speaker }
    }

    #[test]
    fn orta_nokta_tek_konusmaci() {
        let s = [seg(0.0, 2.0, 0), seg(2.0, 4.0, 1)];
        assert_eq!(assign(&[(0.5, 1.0), (2.5, 3.0)], &s, P), vec![0, 1]);
    }

    #[test]
    fn ust_uste_baskin_ya_da_belirsiz() {
        let s = [seg(0.0, 3.0, 0), seg(1.0, 2.0, 1)];
        assert_eq!(assign(&[(1.2, 1.8)], &s, P), vec![UNKNOWN]); // pay eşit: %50 < %60
        assert_eq!(assign(&[(0.0, 1.95)], &s, P), vec![0]);
    }

    #[test]
    fn bosluk_en_yakin_ya_da_belirsiz() {
        let s = [seg(0.0, 1.0, 0), seg(3.0, 4.0, 1)];
        assert_eq!(assign(&[(1.1, 1.3), (2.7, 2.9), (1.8, 2.0)], &s, P), vec![0, 1, UNKNOWN]);
    }
}

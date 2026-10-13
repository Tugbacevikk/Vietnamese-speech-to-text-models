//! viet_asr Rust çekirdeği. Saf algoritmalar `fusion` gibi modüllerde, Python bağı
//! yalnızca `python` özelliği açıkken derlenir (Rust servisi PyO3'süz kullanabilsin).

pub mod fusion;

#[cfg(feature = "python")]
mod py {
    use super::fusion;
    use pyo3::prelude::*;

    /// assign_labels(word_starts, word_ends, seg_starts, seg_ends, seg_speakers,
    ///               min_dominance, max_gap) -> list[int]   (-1 = belirsiz)
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn assign_labels(
        py: Python<'_>,
        word_starts: Vec<f64>,
        word_ends: Vec<f64>,
        seg_starts: Vec<f64>,
        seg_ends: Vec<f64>,
        seg_speakers: Vec<i64>,
        min_dominance: f64,
        max_gap: f64,
    ) -> PyResult<Vec<i64>> {
        if word_starts.len() != word_ends.len()
            || seg_starts.len() != seg_ends.len()
            || seg_starts.len() != seg_speakers.len()
        {
            return Err(pyo3::exceptions::PyValueError::new_err("dizi uzunlukları uyuşmuyor"));
        }
        let words: Vec<(f64, f64)> = word_starts.into_iter().zip(word_ends).collect();
        let segs: Vec<fusion::Segment> = (0..seg_starts.len())
            .map(|i| fusion::Segment { start: seg_starts[i], end: seg_ends[i], speaker: seg_speakers[i] })
            .collect();
        let p = fusion::Params { min_dominance, max_gap };
        // hesap sırasında Python kilidi bırakılır: diğer iş parçacıkları beklemesin
        Ok(py.allow_threads(|| fusion::assign(&words, &segs, p)))
    }

    #[pymodule]
    fn viet_asr_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(assign_labels, m)?)?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}

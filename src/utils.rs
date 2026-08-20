#![allow(non_snake_case)]
use ndarray::{s, Array1};

/// Computes the sample variance of a 1D slice. Returns 0.0 for fewer than two
/// elements (where sample variance is undefined) instead of producing NaN/inf.
pub fn variance(slice: &Array1<f64>) -> f64 {
    if slice.len() < 2 {
        return 0.0;
    }
    let mean = slice.mean().unwrap_or(0.0);
    let count = slice.len() as f64;
    slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (count - 1.0)
}

/// Applies the Box-Cox transform. `lambda == 0` is the natural log; otherwise
/// `(x^lambda - 1) / lambda`. Requires strictly positive input.
pub fn box_cox(data: &Array1<f64>, lambda: f64) -> Array1<f64> {
    if lambda == 0.0 {
        data.mapv(f64::ln)
    } else {
        data.mapv(|x| (x.powf(lambda) - 1.0) / lambda)
    }
}

/// Inverts the Box-Cox transform (see [`box_cox`]).
pub fn inv_box_cox(data: &Array1<f64>, lambda: f64) -> Array1<f64> {
    if lambda == 0.0 {
        data.mapv(f64::exp)
    } else {
        data.mapv(|y| {
            let base = lambda * y + 1.0;
            if base <= 0.0 {
                0.0
            } else {
                base.powf(1.0 / lambda)
            }
        })
    }
}

/// Computes difference of order `d` on time series data.
pub fn difference(data: &Array1<f64>, order: usize) -> Array1<f64> {
    let mut current = data.clone();
    for _ in 0..order {
        if current.len() <= 1 {
            return Array1::zeros(0);
        }
        let diff = &current.slice(s![1..]) - &current.slice(s![..-1]);
        current = diff;
    }
    current
}

pub mod serde_array1 {
    use ndarray::Array1;
    use serde::{Deserialize, Deserializer, Serializer};

    /// Serializes an `Array1<f64>` into a flat JSON array `[0.5, -0.2]`
    pub fn serialize<S>(arr: &Array1<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(arr.len()))?;
        for elem in arr.iter() {
            seq.serialize_element(elem)?;
        }
        seq.end()
    }

    /// Deserializes a flat JSON array `[0.5, -0.2]` back into an `Array1<f64>`
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Array1<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec = Vec::<f64>::deserialize(deserializer)?;
        Ok(Array1::from_vec(vec))
    }
}

pub mod serde_opt_array1 {
    use ndarray::Array1;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(opt: &Option<Array1<f64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match opt {
            Some(arr) => super::serde_array1::serialize(arr, serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Array1<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt_vec = Option::<Vec<f64>>::deserialize(deserializer)?;
        Ok(opt_vec.map(Array1::from_vec))
    }
}

pub mod serde_array2 {
    use ndarray::Array2;
    use serde::{Deserialize, Deserializer, Serializer};

    /// Serializes an `Array2<f64>` into a nested JSON array `[[1.0, 2.0], [3.0, 4.0]]`
    pub fn serialize<S>(matrix: &Array2<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeSeq;

        let mut outer_seq = serializer.serialize_seq(Some(matrix.nrows()))?;
        for row in matrix.rows() {
            let row_slice = row.as_slice().unwrap_or(&[]);
            outer_seq.serialize_element(row_slice)?;
        }
        outer_seq.end()
    }

    /// Deserializes a nested JSON array `[[1.0, 2.0], [3.0, 4.0]]` into an `Array2<f64>`
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Array2<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec_2d = Vec::<Vec<f64>>::deserialize(deserializer)?;

        let nrows = vec_2d.len();
        if nrows == 0 {
            return Ok(Array2::zeros((0, 0)));
        }

        let ncols = vec_2d[0].len();

        // Ensure all rows have equal column dimensions
        if vec_2d.iter().any(|row| row.len() != ncols) {
            return Err(serde::de::Error::custom(
                "Inconsistent column length in 2D array matrix",
            ));
        }

        let flat: Vec<f64> = vec_2d.into_iter().flatten().collect();
        Array2::from_shape_vec((nrows, ncols), flat)
            .map_err(|e| serde::de::Error::custom(e.to_string()))
    }
}

pub mod serde_opt_array2 {
    use ndarray::Array2;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(opt: &Option<Array2<f64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match opt {
            Some(matrix) => super::serde_array2::serialize(matrix, serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Array2<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt_vec = Option::<Vec<Vec<f64>>>::deserialize(deserializer)?;
        match opt_vec {
            Some(vec_2d) => {
                let nrows = vec_2d.len();
                if nrows == 0 {
                    return Ok(Some(Array2::zeros((0, 0))));
                }
                let ncols = vec_2d[0].len();
                if vec_2d.iter().any(|row| row.len() != ncols) {
                    return Err(serde::de::Error::custom(
                        "Inconsistent column length in 2D array matrix",
                    ));
                }
                let flat: Vec<f64> = vec_2d.into_iter().flatten().collect();
                Array2::from_shape_vec((nrows, ncols), flat)
                    .map(Some)
                    .map_err(|e| serde::de::Error::custom(e.to_string()))
            }
            None => Ok(None),
        }
    }
}

/// Reintegrates a differenced forecast back to the original series scale.
///
/// `forecast_diff` holds forecasts of the `d`-th difference of the series. Undoing
/// one level of differencing is a cumulative sum anchored on the last value of the
/// `(d-1)`-th difference of the observed `history`; the routine then recurses down
/// to level 0. Anchoring on the correct per-level tail (rather than always on the
/// last raw observation) is what makes `d > 1` integration correct.
pub fn integrate_forecast(
    forecast_diff: &Array1<f64>,
    history: &Array1<f64>,
    d: usize,
) -> Array1<f64> {
    if d == 0 {
        return forecast_diff.clone();
    }

    // Anchor = last value of the (d-1)-th difference of the observed history.
    let hist_lower = difference(history, d - 1);
    let anchor = *hist_lower.last().unwrap_or(&0.0);

    let mut cum = anchor;
    let mut lifted = Vec::with_capacity(forecast_diff.len());
    for &val in forecast_diff.iter() {
        cum += val;
        lifted.push(cum);
    }

    // `lifted` is now the forecast of the (d-1)-th difference; recurse.
    integrate_forecast(&Array1::from(lifted), history, d - 1)
}

/// Reintegrates a seasonally-differenced forecast back to the pre-seasonal-diff scale.
///
/// `forecast_sdiff` holds forecasts of `(1 - B^m)^D z`, and `z_hist` is the observed
/// series at the pre-seasonal-difference level (i.e. after any non-seasonal
/// differencing). Undoing one seasonal level is `x_t = w_t + x_{t-m}`, anchored on
/// the tail of the `(D-1)`-th seasonal difference of `z_hist`; the routine recurses
/// down to level 0. Returns the forecast on the `z` scale.
pub fn seasonal_integrate_forecast(
    forecast_sdiff: &Array1<f64>,
    z_hist: &Array1<f64>,
    m: usize,
    D: usize,
) -> Array1<f64> {
    if D == 0 || m <= 1 {
        return forecast_sdiff.clone();
    }

    // History at the (D-1)-th seasonal-difference level provides the lag-m anchors.
    let base = seasonal_difference(z_hist, m, D - 1);
    if base.len() < m {
        // Not enough history to seasonally integrate; fall back to the raw forecast.
        return forecast_sdiff.clone();
    }

    let mut extended = base.to_vec();
    let start = extended.len();
    for &w in forecast_sdiff.iter() {
        let idx = extended.len();
        let prev = extended[idx - m];
        extended.push(w + prev);
    }
    let lifted = Array1::from(extended[start..].to_vec());

    seasonal_integrate_forecast(&lifted, z_hist, m, D - 1)
}

pub fn seasonal_difference(data: &Array1<f64>, m: usize, D: usize) -> Array1<f64> {
    let mut current = data.clone();
    for _ in 0..D {
        if current.len() <= m {
            return Array1::zeros(0);
        }
        let m_neg = -(m as isize);
        let diff = &current.slice(s![m..]) - &current.slice(s![..m_neg]);
        current = diff;
    }
    current
}

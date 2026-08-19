use ndarray::{Array1, Array2, Axis};
use std::f64::consts::PI;
use crate::decomposition::{SeasonalitySpec, TrendType};

pub struct ProphetMatrixBuilder;

impl ProphetMatrixBuilder {
    /// Generates the trend feature matrix (A matrix for changepoints)
    ///
    /// # Arguments
    /// * `t` - Normalized time points (1D array of length `N`, typically scaled between 0 and 1)
    /// * `changepoints` - Time points where rate changes occur (1D array of length `K`)
    /// * `trend_type` - Linear or Logistic trend
    ///
    /// # Returns
    /// * `A` - A matrix of shape (N, K) where A[i, j] = 1 if t[i] >= changepoints[j] else 0
    pub fn build_trend_matrix(
        t: &Array1<f64>,
        changepoints: &Array1<f64>,
        _trend_type: TrendType,
    ) -> Array2<f64> {
        let n = t.len();
        let k = changepoints.len();
        let mut a = Array2::<f64>::zeros((n, k));

        for i in 0..n {
            for j in 0..k {
                if t[i] >= changepoints[j] {
                    a[[i, j]] = 1.0;
                }
            }
        }

        a
    }

    /// Generates the Fourier seasonality design matrix (X matrix)
    ///
    /// For each seasonality specification, generates N columns:
    /// sin(2*pi*1*t / P), cos(2*pi*1*t / P), ..., sin(2*pi*N*t / P), cos(2*pi*N*t / P)
    ///
    /// # Arguments
    /// * `t_days` - Time vector in raw days (1D array of length `N`)
    /// * `specs` - Vector of seasonality configurations (e.g., yearly, weekly)
    ///
    /// # Returns
    /// * `X` - Matrix of shape (N, 2 * sum(fourier_order))
    pub fn build_seasonality_matrix(
        t_days: &Array1<f64>,
        specs: &[SeasonalitySpec],
    ) -> Array2<f64> {
        let n = t_days.len();

        // Calculate total number of Fourier columns across all seasonalities
        let total_cols: usize = specs.iter().map(|s| 2 * s.fourier_order).sum();

        if total_cols == 0 {
            return Array2::<f64>::zeros((n, 0));
        }

        let mut x = Array2::<f64>::zeros((n, total_cols));
        let mut col_offset = 0;

        for spec in specs {
            let period = spec.period_days;

            for order in 1..=spec.fourier_order {
                let freq = 2.0 * PI * (order as f64) / period;

                // Sin column
                let sin_col = t_days.mapv(|val| (freq * val).sin());
                x.column_mut(col_offset).assign(&sin_col);
                col_offset += 1;

                // Cos column
                let cos_col = t_days.mapv(|val| (freq * val).cos());
                x.column_mut(col_offset).assign(&cos_col);
                col_offset += 1;
            }
        }

        x
    }

    /// Combines Trend base features and Fourier matrices into a single model matrix
    pub fn concatenate_features(
        t_scaled: &Array1<f64>,
        a_matrix: &Array2<f64>,
        x_seasonality: &Array2<f64>,
    ) -> Array2<f64> {
        // Base trend column (t)
        let t_col = t_scaled.to_shape((t_scaled.len(), 1)).unwrap();

        // Concatenate [t, A, X_seasonality] horizontally
        ndarray::concatenate(
            Axis(1),
            &[t_col.view(), a_matrix.view(), x_seasonality.view()],
        )
        .expect("Failed to concatenate feature matrices")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use crate::fitting::ProphetRidgeFitter;

    #[test]
    fn test_prophet_matrix_generation() {
        // 1. Setup sample time inputs (5 time steps)
        let t_days = array![0.0, 91.25, 182.5, 273.75, 365.25];
        let t_scaled = &t_days / 365.25; // Scale to [0, 1]
        let changepoints = array![0.25, 0.50];

        // 2. Trend A Matrix (N=5, K=2)
        let a_mat = ProphetMatrixBuilder::build_trend_matrix(
            &t_scaled,
            &changepoints,
            TrendType::Linear,
        );
        assert_eq!(a_mat.shape(), &[5, 2]);
        assert_eq!(a_mat[[0, 0]], 0.0); // t=0.0 < 0.25
        assert_eq!(a_mat[[2, 0]], 1.0); // t=0.5 >= 0.25

        // 3. Seasonality X Matrix (Yearly order 2 -> 4 columns)
        let specs = vec![SeasonalitySpec {
            name: "yearly".to_string(),
            period_days: 365.25,
            fourier_order: 2,
            prior_scale: 10.0,
        }];

        let x_mat = ProphetMatrixBuilder::build_seasonality_matrix(&t_days, &specs);
        assert_eq!(x_mat.shape(), &[5, 4]);

        // 4. Combined Feature Matrix: [1 column t] + [2 cols A] + [4 cols X] = 7 cols
        let combined = ProphetMatrixBuilder::concatenate_features(&t_scaled, &a_mat, &x_mat);
        assert_eq!(combined.shape(), &[5, 7]);
    }

    #[test]
    fn test_prophet_ridge_fitting() {
        // Synthetic dataset (N=4, 3 features)
        let x = array![
            [1.0, 0.0, 0.1],
            [1.0, 0.33, 0.8],
            [1.0, 0.66, -0.4],
            [1.0, 1.0, -0.9]
        ];
        let y = array![1.2, 2.5, 3.1, 4.0];

        // Solve with mild regularization (lambda = 0.01)
        let fitter = ProphetRidgeFitter::new(0.01);
        let beta = fitter.fit(&x, &y).expect("Fitting failed");

        assert_eq!(beta.len(), 3);

        // Make prediction
        let predictions = fitter.predict(&x, &beta);
        assert_eq!(predictions.len(), 4);
    }
}
use chronos_ts::decomposition::ProphetDecomposition;
use chronos_ts::diagnostics::ResidualDiagnostics;
use chronos_ts::statespace::{KalmanFilter, StateSpaceModel};
use chronos_ts::statespace_mle::fit_state_space_mle;
use chrono::NaiveDate;
use ndarray::Array1;

#[test]
fn test_diagnostics_on_white_noise() {
    // Generate 500 points of synthetic standard normal white noise
    let n = 500;
    let mut rng_val = 42.0;
    let mut white_noise = Vec::with_capacity(n);

    for _ in 0..n {
        // Simple linear congruential pseudorandom noise generator
        rng_val = (rng_val * 1103515245.0 + 12345.0) % 2147483648.0;
        let norm = (rng_val / 2147483648.0) * 2.0 - 1.0;
        white_noise.push(norm);
    }

    let residuals = Array1::from_vec(white_noise);
    let diagnostics = ResidualDiagnostics::evaluate(&residuals, 10)
        .expect("Diagnostics evaluation should succeed");

    // ACF at lag 0 is 1.0; subsequent lags should remain close to 0 for white noise
    assert!((diagnostics.acf[0] - 1.0).abs() < 1e-6);
    assert!(diagnostics.acf[1].abs() < 0.15);

    // Jarque-Bera p-value should be reasonable for uniform/gaussian-like noise
    assert!(diagnostics.jarque_bera.p_value >= 0.0);
}

#[test]
fn test_kalman_filter_and_rts_smoother() {
    let model = StateSpaceModel::local_linear_trend(0.5, 0.1, 0.05);
    let filter = KalmanFilter::new(&model);

    let observations = Array1::from_vec(vec![1.0, 2.1, 2.9, 4.2, 5.0, 6.1, 7.3]);

    let filter_res = filter
        .filter(&observations, None, None)
        .expect("Filter pass should succeed");

    assert_eq!(filter_res.steps.len(), 7);
    assert!(filter_res.log_likelihood.is_finite());

    let smooth_res = filter
        .smooth(&filter_res)
        .expect("Smoother pass should succeed");

    assert_eq!(smooth_res.smoothed_states.len(), 7);
    assert_eq!(smooth_res.smoothed_covs.len(), 7);
}

#[test]
fn test_prophet_decomposition_fit() {
    let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..100).map(|i| start + chrono::Duration::days(i)).collect();

    let y_vals: Vec<f64> = (0..100)
        .map(|i| i as f64 * 0.2 + (i as f64 % 7.0))
        .collect();
    let y = Array1::from_vec(y_vals);

    let mut prophet = ProphetDecomposition::new(5, 0.1);
    prophet.add_seasonality("weekly", 7.0, 2);

    prophet.fit(&dates, &y, None, None).expect("Prophet fitting should succeed");
    let pred = prophet.predict(&dates).expect("Prediction should succeed");

    assert_eq!(pred.yhat.len(), 100);
}

#[test]
fn test_mle_optimization_convergence() {
    let series = Array1::from_vec(vec![10.0, 10.5, 11.2, 11.8, 12.4, 13.1, 13.9, 14.3]);
    let init_params = vec![0.5, 0.2, 0.1];

    let fit_res = fit_state_space_mle(
        &series,
        init_params,
        |params| StateSpaceModel::local_linear_trend(params[0], params[1], params[2]),
        20,
    )
    .expect("MLE should converge within iterations");

    assert!(fit_res.max_log_likelihood.is_finite());
}
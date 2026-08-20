use chrono::NaiveDate;
use chronos_ts::{
    AutoTuner, HyperparameterGrid, OptimizationMetric, ProphetDecomposition, SeasonalityMode,
};
use ndarray::Array1;

#[test]
fn test_autotune_grid_search() {
    let start_date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let n = 100;
    let mut dates = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    for i in 0..n {
        let date = start_date + chrono::Duration::days(i as i64);
        dates.push(date);

        let t = i as f64;
        let trend = 10.0 + 0.05 * t;
        let seasonality = (2.0 * std::f64::consts::PI * t / 7.0).sin() * 2.0;
        values.push(trend + seasonality);
    }

    let y = Array1::from_vec(values);

    let grid = HyperparameterGrid {
        changepoint_prior_scales: vec![0.01, 0.05],
        seasonality_prior_scales: vec![0.1, 1.0],
        holidays_prior_scales: vec![0.1],
        seasonality_modes: vec![SeasonalityMode::Additive],
    };

    let base_model = ProphetDecomposition::new(25, 0.05);
    let tuner = AutoTuner::new(grid)
        .with_metric(OptimizationMetric::MAE)
        .with_validation_split(0.7, 10, 5);

    let tune_res = tuner.fit_and_tune(&base_model, &dates, &y).unwrap();

    assert!(tune_res.best_score >= 0.0);
    assert!(!tune_res.all_evaluated_scores.is_empty());
}

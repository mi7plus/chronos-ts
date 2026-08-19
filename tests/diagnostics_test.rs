use chrono::NaiveDate;
use chronos_ts::{CrossValidationEvaluator, ProphetDecomposition};
use ndarray::Array1;

#[test]
fn test_horizon_degradation_metrics() {
    let start_date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let n = 120;
    let mut dates = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    for i in 0..n {
        let date = start_date + chrono::Duration::days(i as i64);
        dates.push(date);
        let t = i as f64;
        values.push(10.0 + 0.1 * t + (t * 0.2).sin());
    }

    let y = Array1::from_vec(values);
    let model = ProphetDecomposition::new(15, 0.05);

    // Initial train: 60 days, forecast horizon: 10 days, step: 10 days
    let evaluator = CrossValidationEvaluator::new(10, 60, 10);
    let report = evaluator.evaluate(&model, &dates, &y).unwrap();

    assert!(report.total_folds > 0);
    assert_eq!(report.horizon_metrics.len(), 10);

    // Verify horizon indexing from 1 to 10
    assert_eq!(report.horizon_metrics[0].horizon_step, 1);
    assert_eq!(report.horizon_metrics[9].horizon_step, 10);

    // Metric assertions
    assert!(report.overall_mae >= 0.0);
    assert!(report.overall_rmse >= report.overall_mae);
}
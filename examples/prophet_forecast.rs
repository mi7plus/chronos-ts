//! Fits a Prophet-style decomposition (trend + weekly seasonality) and prints an
//! out-of-sample forecast with uncertainty intervals.
//!
//! Run with: `cargo run --example prophet_forecast`

use chrono::{Duration, NaiveDate};
use chronos_ts::{ProphetDecomposition, SeasonalityMode};
use ndarray::Array1;

fn main() {
    let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let train_days = 90;
    let horizon = 14;

    let all_dates: Vec<NaiveDate> = (0..train_days + horizon)
        .map(|i| start + Duration::days(i as i64))
        .collect();

    // Ground truth: linear trend + weekly seasonality.
    let signal = |i: usize| {
        let t = i as f64;
        10.0 + 0.1 * t + 2.0 * (2.0 * std::f64::consts::PI * t / 7.0).sin()
    };
    let train_y = Array1::from_shape_fn(train_days, signal);

    let mut model = ProphetDecomposition::new(10, 0.05);
    model.seasonality_mode = SeasonalityMode::Additive;
    model.add_seasonality("weekly", 7.0, 3);
    model
        .fit(&all_dates[..train_days], &train_y, None, None)
        .expect("prophet fit failed");

    let future_dates = &all_dates[train_days..];
    let pred = model
        .predict_with_intervals(future_dates, 0.9, 500)
        .expect("prophet prediction failed");

    let lower = pred.yhat_lower.expect("intervals requested");
    let upper = pred.yhat_upper.expect("intervals requested");

    println!(
        "{:>12}  {:>9}  {:>9}  {:>20}",
        "date", "actual", "yhat", "90% interval"
    );
    for (k, date) in future_dates.iter().enumerate() {
        let actual = signal(train_days + k);
        println!(
            "{:>12}  {:>9.3}  {:>9.3}  [{:>7.3}, {:>7.3}]",
            date, actual, pred.yhat[k], lower[k], upper[k]
        );
    }
}

//! Fits an ARIMA model with automatic order selection and prints a forecast with
//! prediction intervals.
//!
//! Run with: `cargo run --example arima_forecast`

use chronos_ts::arima::{auto_arima, AutoArimaOptions, EstimationMethod};
use ndarray::Array1;

fn main() {
    // Synthetic AR(1)-like series around a mean of 20.
    let mut value = 20.0;
    let series: Array1<f64> = Array1::from_shape_fn(120, |i| {
        let shock = ((i as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5;
        value = 20.0 + 0.7 * (value - 20.0) + shock;
        value
    });

    let opts = AutoArimaOptions {
        max_p: 3,
        max_q: 3,
        estimation: EstimationMethod::Css, // try EstimationMethod::Mle for exact ML
        ..Default::default()
    };

    let model = auto_arima(&series, opts).expect("auto_arima failed");
    println!("Selected order: {:?}", model.order);
    println!("Intercept (mean/drift): {:.4}", model.intercept);
    println!("sigma^2: {:.4}, AIC: {:.2}", model.sigma2, model.aic());

    let horizon = 8;
    let forecast = model.forecast_with_intervals(&series, horizon);
    println!(
        "\n{:>4}  {:>10}  {:>20}",
        "step", "forecast", "95% interval"
    );
    for h in 0..horizon {
        println!(
            "{:>4}  {:>10.3}  [{:>8.3}, {:>8.3}]",
            h + 1,
            forecast.mean[h],
            forecast.lower_95[h],
            forecast.upper_95[h]
        );
    }
}

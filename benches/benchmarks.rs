use chrono::{Duration, NaiveDate};
use chronos_ts::arima::{auto_arima, AutoArimaOptions};
use chronos_ts::{ProphetDecomposition, SeasonalityMode};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;

fn synthetic_series(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| (i as f64 * 0.1).sin() + (i as f64 * 0.05))
}

fn benchmark_auto_arima(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto_arima");
    for &n in &[100usize, 1000, 5000] {
        let series = synthetic_series(n);
        let opts = AutoArimaOptions {
            max_p: 3,
            max_d: 1,
            max_q: 3,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(n), &series, |b, s| {
            b.iter(|| {
                let _ = auto_arima(s, opts);
            })
        });
    }
    group.finish();
}

fn benchmark_prophet(c: &mut Criterion) {
    let n = 730; // two years of daily data
    let start = NaiveDate::from_ymd_opt(2022, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..n).map(|i| start + Duration::days(i as i64)).collect();
    let y = Array1::from_shape_fn(n, |i| {
        let t = i as f64;
        50.0 + 0.05 * t
            + 5.0 * (2.0 * std::f64::consts::PI * t / 365.25).sin()
            + 2.0 * (2.0 * std::f64::consts::PI * t / 7.0).sin()
    });

    c.bench_function("prophet_fit_730_days", |b| {
        b.iter(|| {
            let mut model = ProphetDecomposition::new(25, 0.05);
            model.seasonality_mode = SeasonalityMode::Additive;
            model.add_seasonality("yearly", 365.25, 6);
            model.add_seasonality("weekly", 7.0, 3);
            let _ = model.fit(&dates, &y, None, None);
        })
    });
}

criterion_group!(benches, benchmark_auto_arima, benchmark_prophet);
criterion_main!(benches);

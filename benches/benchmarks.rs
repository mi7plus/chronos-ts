use criterion::{criterion_group, criterion_main, Criterion};
use chronos_ts::arima::{auto_arima, AutoArimaOptions};
use ndarray::Array1;

fn benchmark_auto_arima(c: &mut Criterion) {
    // Generate sample synthetic series
    let series: Array1<f64> = Array1::from_shape_fn(100, |i| (i as f64 * 0.1).sin() + (i as f64 * 0.05));
    let opts = AutoArimaOptions {
        max_p: 2,
        max_d: 1,
        max_q: 2,
        ..Default::default()
    };

    c.bench_function("auto_arima_fit_100_obs", |b| {
        b.iter(|| {
            let _ = auto_arima(&series, opts);
        })
    });
}

criterion_group!(benches, benchmark_auto_arima);
criterion_main!(benches);
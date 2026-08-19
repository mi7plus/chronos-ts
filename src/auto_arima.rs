use crate::arima::{SarimaModel, SarimaOrder};
use crate::errors::{ChronosError, Result};
use crate::stat_tests::{estimate_d, estimate_D};
use ndarray::Array1;

#[derive(Debug, Clone, Copy)]
pub enum InformationCriterion {
    Aic,
    Aicc,
    Bic,
}

pub struct AutoArimaOptions {
    pub max_p: usize,
    pub max_q: usize,
    pub max_P: usize,
    pub max_Q: usize,
    pub max_d: usize,
    pub max_D: usize,
    pub m: usize,
    pub criterion: InformationCriterion,
    pub stepwise: bool,
}

impl Default for AutoArimaOptions {
    fn default() -> Self {
        Self {
            max_p: 5,
            max_q: 5,
            max_P: 2,
            max_Q: 2,
            max_d: 2,
            max_D: 1,
            m: 1,
            criterion: InformationCriterion::Aicc,
            stepwise: true,
        }
    }
}

/// Runs Hyndman-Khandakar Auto-ARIMA selection
pub fn auto_arima_search(data: &Array1<f64>, opts: AutoArimaOptions) -> Result<SarimaModel> {
    let n = data.len();

    // 1. Estimate differencing orders automatically via Unit-Root / Stationarity tests
    let d = estimate_d(data, opts.max_d, 0.05);
    let D = estimate_D(data, opts.m, opts.max_D);

    let get_ic = |model: &SarimaModel| match opts.criterion {
        InformationCriterion::Aic => model.aic(),
        InformationCriterion::Aicc => model.aicc(n),
        InformationCriterion::Bic => model.bic(n),
    };

    if !opts.stepwise {
        // Grid Search baseline
        let mut best_model: Option<SarimaModel> = None;
        let mut best_score = f64::INFINITY;

        for p in 0..=opts.max_p {
            for q in 0..=opts.max_q {
                for P in 0..=opts.max_P {
                    for Q in 0..=opts.max_Q {
                        let order = SarimaOrder { p, d, q, P, D, Q, m: opts.m };
                        if let Ok(model) = SarimaModel::fit(data, order) {
                            let score = get_ic(&model);
                            if score < best_score {
                                best_score = score;
                                best_model = Some(model);
                            }
                        }
                    }
                }
            }
        }
        return best_model.ok_or_else(|| ChronosError::ConvergenceFailure("Failed to fit grid models".into()));
    }

    // 2. Stepwise Heuristic Optimization
    let seed_orders = vec![
        SarimaOrder { p: 2, d, q: 2, P: 1, D, Q: 1, m: opts.m },
        SarimaOrder { p: 0, d, q: 0, P: 0, D, Q: 0, m: opts.m },
        SarimaOrder { p: 1, d, q: 0, P: 1, D, Q: 0, m: opts.m },
        SarimaOrder { p: 0, d, q: 1, P: 0, D, Q: 1, m: opts.m },
    ];

    let mut best_model: Option<SarimaModel> = None;
    let mut best_score = f64::INFINITY;

    for order in seed_orders {
        if let Ok(model) = SarimaModel::fit(data, order) {
            let score = get_ic(&model);
            if score < best_score {
                best_score = score;
                best_model = Some(model);
            }
        }
    }

    let mut current_model = best_model.ok_or_else(|| ChronosError::ConvergenceFailure("Failed initial seed models".into()))?;

    // Local stepwise search around current best
    let mut improved = true;
    while improved {
        improved = false;
        let curr_o = current_model.order;

        let mut candidate_orders = Vec::new();

        // Variations of p, q, P, Q (+1 / -1 steps)
        if curr_o.p < opts.max_p { candidate_orders.push(SarimaOrder { p: curr_o.p + 1, ..curr_o }); }
        if curr_o.p > 0 { candidate_orders.push(SarimaOrder { p: curr_o.p - 1, ..curr_o }); }
        if curr_o.q < opts.max_q { candidate_orders.push(SarimaOrder { q: curr_o.q + 1, ..curr_o }); }
        if curr_o.q > 0 { candidate_orders.push(SarimaOrder { q: curr_o.q - 1, ..curr_o }); }

        for cand in candidate_orders {
            if let Ok(model) = SarimaModel::fit(data, cand) {
                let score = get_ic(&model);
                if score < best_score {
                    best_score = score;
                    current_model = model;
                    improved = true;
                    break;
                }
            }
        }
    }

    Ok(current_model)
}
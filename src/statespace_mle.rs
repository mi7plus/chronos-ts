use crate::errors::{ChronosError, Result};
use crate::statespace::{KalmanFilter, StateSpaceModel};
use argmin::core::{CostFunction, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::Array1;

pub struct StateSpaceLikelihoodCost<'a, F>
where
    F: Fn(&[f64]) -> StateSpaceModel,
{
    observations: &'a Array1<f64>,
    model_builder: &'a F,
}

impl<'a, F> StateSpaceLikelihoodCost<'a, F>
where
    F: Fn(&[f64]) -> StateSpaceModel,
{
    pub fn new(observations: &'a Array1<f64>, model_builder: &'a F) -> Self {
        Self {
            observations,
            model_builder,
        }
    }
}

impl<'a, F> CostFunction for StateSpaceLikelihoodCost<'a, F>
where
    F: Fn(&[f64]) -> StateSpaceModel + Send + Sync,
{
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let positive_params: Vec<f64> = param.iter().map(|&p| p.exp()).collect();
        let model = (self.model_builder)(&positive_params);
        let filter = KalmanFilter::new(&model);

        match filter.filter(self.observations, None, None) {
            Ok(res) => {
                if res.log_likelihood.is_finite() {
                    Ok(-res.log_likelihood)
                } else {
                    Ok(f64::INFINITY)
                }
            }
            Err(_) => Ok(f64::INFINITY),
        }
    }
}

// Numerical gradient via central finite differences
impl<'a, F> Gradient for StateSpaceLikelihoodCost<'a, F>
where
    F: Fn(&[f64]) -> StateSpaceModel + Send + Sync,
{
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(
        &self,
        param: &Self::Param,
    ) -> std::result::Result<Self::Gradient, argmin::core::Error> {
        let h = 1e-5;
        let mut grad = vec![0.0; param.len()];
        let mut param_plus = param.clone();
        let mut param_minus = param.clone();

        for i in 0..param.len() {
            param_plus[i] = param[i] + h;
            param_minus[i] = param[i] - h;

            let cost_plus = self.cost(&param_plus)?;
            let cost_minus = self.cost(&param_minus)?;

            grad[i] = (cost_plus - cost_minus) / (2.0 * h);

            param_plus[i] = param[i];
            param_minus[i] = param[i];
        }

        Ok(grad)
    }
}

pub struct MleFitResult {
    pub fitted_parameters: Vec<f64>,
    pub max_log_likelihood: f64,
    pub optimized_model: StateSpaceModel,
}

pub fn fit_state_space_mle<F>(
    observations: &Array1<f64>,
    initial_params: Vec<f64>,
    model_builder: F,
    max_iters: u64,
) -> Result<MleFitResult>
where
    F: Fn(&[f64]) -> StateSpaceModel + Send + Sync,
{
    let init_unbounded = initial_params
        .iter()
        .map(|&p| (p.max(1e-6)).ln())
        .collect::<Vec<_>>();

    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 7);

    let cost_fn = StateSpaceLikelihoodCost::new(observations, &model_builder);

    let res = Executor::new(cost_fn, solver)
        .configure(|state| state.param(init_unbounded).max_iters(max_iters))
        .run();

    match res {
        Ok(exec_state) => {
            let best_log_params = exec_state
                .state
                .get_best_param()
                .ok_or_else(|| ChronosError::OptimizationError("No parameters returned".into()))?;

            let fitted_params = best_log_params.iter().map(|&p| p.exp()).collect::<Vec<_>>();

            let optimized_model = model_builder(&fitted_params);
            let max_log_likelihood = -exec_state.state.get_best_cost();

            Ok(MleFitResult {
                fitted_parameters: fitted_params,
                max_log_likelihood,
                optimized_model,
            })
        }
        Err(e) => Err(ChronosError::OptimizationError(e.to_string())),
    }
}

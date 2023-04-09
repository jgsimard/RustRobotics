use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, OMatrix, OVector, RealField, U1,
};

use crate::localization::bayesian_filter::GaussianBayesianFilter;
use crate::models::measurement::MeasurementModel;
use crate::models::motion::MotionModel;
use crate::utils::state::GaussianState;

/// S : State Size, Z: Observation Size, U: Input Size
pub struct UnscentedKalmanFilter<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, Z>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Z, S>
        + Allocator<T, S, U>
        + Allocator<T, U, U>
        + Allocator<T, S, Z>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
{
    q: OMatrix<T, S, S>,
    r: OMatrix<T, Z, Z>,
    gamma: T,
    observation_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
    motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
    mw: Vec<T>,
    cw: Vec<T>,
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> UnscentedKalmanFilter<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, Z>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Z, S>
        + Allocator<T, S, U>
        + Allocator<T, U, U>
        + Allocator<T, S, Z>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
{
    pub fn new(
        q: OMatrix<T, S, S>,
        r: OMatrix<T, Z, Z>,
        observation_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
        motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> UnscentedKalmanFilter<T, S, Z, U> {
        let dim = q.shape_generic().0.value();
        let (mw, cw, gamma) =
            UnscentedKalmanFilter::<T, S, Z, U>::sigma_weights(dim, alpha, beta, kappa);
        UnscentedKalmanFilter {
            q,
            r,
            observation_model,
            motion_model,
            gamma,
            mw,
            cw,
        }
    }

    fn sigma_weights(dim: usize, alpha: T, beta: T, kappa: T) -> (Vec<T>, Vec<T>, T) {
        let n = T::from_usize(dim).unwrap();
        let lambda = alpha.powi(2) * (n + kappa) - n;

        let v = T::one() / ((T::one() + T::one()) * (n + lambda));
        let mut mw = vec![v; 2 * dim + 1];
        let mut cw = vec![v; 2 * dim + 1];

        // special cases
        let v = lambda / (n + lambda);
        mw[0] = v;
        cw[0] = v + T::one() - alpha.powi(2) + beta;

        let gamma = (n + lambda).sqrt();
        (mw, cw, gamma)
    }

    pub fn generate_sigma_points(&self, state: &GaussianState<T, S>) -> Vec<OVector<T, S>> {
        let dim = self.q.shape_generic().0.value();
        // use cholesky to compute the matrix square root  // cholesky(A) = L * L^T
        let sigma = state.cov.clone().cholesky().expect("unable to sqrt").l() * self.gamma;
        // let mut sigma_points = vec![state.x; 2 * S::USIZE + 1];
        // for i in 0..S::USIZE {
        //     let sigma_column = sigma.column(i);
        //     sigma_points[i + 1] += sigma_column;
        //     sigma_points[i + 1 + S::USIZE] -= sigma_column;
        // }
        let mut sigma_points = Vec::with_capacity(2 * dim + 1);
        sigma_points.push(state.x.clone());
        for i in 0..dim {
            let sigma_column = sigma.column(i);
            sigma_points.push(&state.x + sigma_column);
            sigma_points.push(&state.x - sigma_column);
        }
        sigma_points
    }
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> GaussianBayesianFilter<T, S, Z, U>
    for UnscentedKalmanFilter<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, Z>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Z, S>
        + Allocator<T, S, U>
        + Allocator<T, U, U>
        + Allocator<T, S, Z>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
{
    fn estimate(
        &self,
        state: &GaussianState<T, S>,
        u: &OVector<T, U>,
        z: &OVector<T, Z>,
        dt: T,
    ) -> GaussianState<T, S> {
        let dim_s = self.q.shape_generic().0;
        let dim_z = self.r.shape_generic().0;
        // predict
        let sigma_points = self.generate_sigma_points(state);
        let sp_xpred: Vec<OVector<T, S>> = sigma_points
            .iter()
            .map(|x| self.motion_model.prediction(x, u, dt))
            .collect();

        let mean_xpred: OVector<T, S> = sp_xpred
            .iter()
            .zip(self.mw.iter())
            .map(|(x, w)| x * *w)
            .fold(OMatrix::zeros_generic(dim_s, U1), |a, b| a + b);

        let cov_xpred = sp_xpred
            .iter()
            .map(|x| x - &mean_xpred)
            .zip(self.cw.iter())
            .map(|(dx, cw)| &dx * dx.transpose() * *cw)
            .fold(OMatrix::zeros_generic(dim_s, dim_s), |a, b| a + b)
            + &self.q;

        let prediction = GaussianState {
            x: mean_xpred.clone(),
            cov: cov_xpred.clone(),
        };

        // update
        let sp_xpred = self.generate_sigma_points(&prediction);
        let sp_z: Vec<OVector<T, Z>> = sp_xpred
            .iter()
            .map(|x| self.observation_model.prediction(x, None))
            .collect();

        let mean_z: OVector<T, Z> = sp_z
            .iter()
            .zip(self.mw.iter())
            .map(|(x, w)| x * *w)
            .fold(OMatrix::zeros_generic(dim_z, U1), |a, b| a + b);

        let cov_z = sp_z
            .iter()
            .map(|x| x - &mean_z)
            .zip(self.cw.iter())
            .map(|(dx, cw)| &dx * dx.transpose() * *cw)
            .fold(OMatrix::zeros_generic(dim_z, dim_z), |a, b| a + b)
            + &self.r;

        let s = sp_xpred
            .iter()
            .zip(sp_z.iter().zip(self.cw.iter()))
            .map(|(x_pred, (z_point, cw))| {
                (x_pred - &mean_xpred) * (z_point - &mean_z).transpose() * *cw
            })
            .fold(OMatrix::zeros_generic(dim_s, dim_z), |a, b| a + b);

        let y = z - mean_z;
        let kalman_gain = s * cov_z.clone().try_inverse().unwrap();

        let x_est = mean_xpred + &kalman_gain * y;
        let cov_est = cov_xpred - &kalman_gain * cov_z * kalman_gain.transpose();
        GaussianState {
            x: x_est,
            cov: cov_est,
        }
    }
}

// Unscented kalman filter (UKF) localization sample
// author: Jean-Gabriel Simard (@jgsimard)

#![allow(non_snake_case)]
use nalgebra::{RealField, SMatrix, SVector, Scalar};

use crate::utils::state::GaussianStateStatic;

/// S : State Size, Z: Observation Size, U: Input Size
pub struct UnscentedKalmanFilter<
    T: RealField + Scalar + Copy,
    const S: usize,
    const Z: usize,
    const U: usize,
> {
    pub Q: SMatrix<T, S, S>,
    pub R: SMatrix<T, Z, Z>,
    pub gamma: T,
    pub mw: Vec<T>,
    pub cw: Vec<T>,
}

pub trait UnscentedKalmanfilterModel<
    T: RealField + Scalar + Copy,
    const S: usize,
    const Z: usize,
    const U: usize,
>
{
    fn motion_model(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SVector<T, S>;
    fn observation_model(&self, x: &SVector<T, S>) -> SVector<T, Z>;
}

impl<T: RealField + Scalar + Copy, const S: usize, const Z: usize, const U: usize>
    UnscentedKalmanFilter<T, S, Z, U>
{
    pub fn new(
        Q: SMatrix<T, S, S>,
        R: SMatrix<T, Z, Z>,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> UnscentedKalmanFilter<T, S, Z, U> {
        let (mw, cw, gamma) =
            UnscentedKalmanFilter::<T, S, Z, U>::sigma_weights(alpha, beta, kappa);
        UnscentedKalmanFilter {
            Q,
            R,
            gamma,
            mw,
            cw,
        }
    }
    fn sigma_weights(alpha: T, beta: T, kappa: T) -> (Vec<T>, Vec<T>, T) {
        let n = T::from_usize(S).unwrap();
        let lambda = alpha.powi(2) * (n + kappa) - n;

        let v = T::one() / ((T::one() + T::one()) * (n + lambda));
        let mut mw = vec![v; 2 * S + 1];
        let mut cw = vec![v; 2 * S + 1];

        // special cases
        let v = lambda / (n + lambda);
        mw[0] = v;
        cw[0] = v + T::one() - alpha.powi(2) + beta;

        let gamma = (n + lambda).sqrt();
        (mw, cw, gamma)
    }

    fn generate_sigma_points(&self, state: &GaussianStateStatic<T, S>) -> Vec<SVector<T, S>> {
        // use cholesky to compute the matrix square root  // cholesky(A) = L * L^T
        let sigma = state.P.cholesky().expect("unable to sqrt").l() * self.gamma;
        let mut sigma_points = Vec::new();
        sigma_points.push(state.x);
        for i in 0..S {
            let sigma_column = sigma.column(i);
            sigma_points.push(state.x + sigma_column);
            sigma_points.push(state.x - sigma_column);
        }
        sigma_points
    }

    pub fn estimate(
        &self,
        model: &impl UnscentedKalmanfilterModel<T, S, Z, U>,
        state: &GaussianStateStatic<T, S>,
        u: &SVector<T, U>,
        z: &SVector<T, Z>,
        dt: T,
    ) -> GaussianStateStatic<T, S> {
        // predict
        let sigma_points = self.generate_sigma_points(state);
        let sp_xpred: Vec<SVector<T, S>> = sigma_points
            .iter()
            .map(|x| model.motion_model(x, u, dt))
            .collect();

        let mean_xpred: SVector<T, S> = sp_xpred
            .iter()
            .zip(self.mw.iter())
            .map(|(x, w)| x * *w)
            .sum();

        let cov_xpred = sp_xpred
            .iter()
            .map(|x| x - mean_xpred)
            .zip(self.cw.iter())
            .map(|(dx, cw)| dx * dx.transpose() * *cw)
            .sum::<SMatrix<T, S, S>>()
            + self.Q;

        let prediction = GaussianStateStatic {
            x: mean_xpred,
            P: cov_xpred,
        };

        // update
        let sp_xpred = self.generate_sigma_points(&prediction);
        let sp_z: Vec<SVector<T, Z>> = sp_xpred
            .iter()
            .map(|x| model.observation_model(x))
            .collect();

        let mean_z: SVector<T, Z> = sp_z.iter().zip(self.mw.iter()).map(|(x, w)| x * *w).sum();

        let cov_z = sp_z
            .iter()
            .map(|x| x - mean_z)
            .zip(self.cw.iter())
            .map(|(dx, cw)| dx * dx.transpose() * *cw)
            .sum::<SMatrix<T, Z, Z>>()
            + self.R;

        let s = sp_xpred
            .iter()
            .zip(sp_z.iter().zip(self.cw.iter()))
            .map(|(x_pred, (z_point, cw))| {
                (x_pred - mean_xpred) * (z_point - mean_z).transpose() * *cw
            })
            .sum::<SMatrix<T, S, Z>>();

        let y = z - mean_z;
        let kalman_gain = s * cov_z.try_inverse().unwrap();

        let x_est = mean_xpred + kalman_gain * y;
        let p_est = cov_xpred - kalman_gain * cov_z * kalman_gain.transpose();
        GaussianStateStatic { x: x_est, P: p_est }
    }
}

/// S : State Size, Z: Observation Size, U: Input Size
pub struct UnscentedKalmanFilterArray<
    T: RealField + Scalar + Copy,
    const S: usize,
    const Z: usize,
    const U: usize,
> where
    [(); 2 * S + 1]: Sized,
{
    pub Q: SMatrix<T, S, S>,
    pub R: SMatrix<T, Z, Z>,
    pub gamma: T,
    pub mw: SVector<T, { 2 * S + 1 }>,
    pub cw: SVector<T, { 2 * S + 1 }>,
}

impl<T: RealField + Scalar + Copy, const S: usize, const Z: usize, const U: usize>
    UnscentedKalmanFilterArray<T, S, Z, U>
where
    [(); 2 * S + 1]: Sized,
{
    pub fn new(
        Q: SMatrix<T, S, S>,
        R: SMatrix<T, Z, Z>,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> UnscentedKalmanFilterArray<T, S, Z, U> {
        let (mw, cw, gamma) =
            UnscentedKalmanFilterArray::<T, S, Z, U>::sigma_weights(alpha, beta, kappa);
        UnscentedKalmanFilterArray {
            Q,
            R,
            gamma,
            mw,
            cw,
        }
    }
    fn sigma_weights(
        alpha: T,
        beta: T,
        kappa: T,
    ) -> (SVector<T, { 2 * S + 1 }>, SVector<T, { 2 * S + 1 }>, T) {
        let n = T::from_usize(S).unwrap();
        let lambda = alpha.powi(2) * (n + kappa) - n;

        let v = T::one() / ((T::one() + T::one()) * (n + lambda));
        let mut mw = SVector::<T, { 2 * S + 1 }>::from_element(v);
        let mut cw = SVector::<T, { 2 * S + 1 }>::from_element(v);

        // special cases
        let v = lambda / (n + lambda);
        mw[0] = v;
        cw[0] = v + T::one() - alpha.powi(2) + beta;

        let gamma = (n + lambda).sqrt();
        (mw, cw, gamma)
    }

    fn generate_sigma_points(&self, state: &GaussianStateStatic<T, S>) -> Vec<SVector<T, S>> {
        // use cholesky to compute the matrix square root
        let sigma = state.P.cholesky().expect("unable to sqrt").l() * self.gamma;
        let mut sigma_points = Vec::new();
        sigma_points.push(state.x);
        for i in 0..S {
            let sigma_column = sigma.column(i);
            sigma_points.push(state.x + sigma_column);
            sigma_points.push(state.x - sigma_column);
        }
        sigma_points
    }

    pub fn estimate(
        &self,
        model: &impl UnscentedKalmanfilterModel<T, S, Z, U>,
        state: &GaussianStateStatic<T, S>,
        u: &SVector<T, U>,
        z: &SVector<T, Z>,
        dt: T,
    ) -> GaussianStateStatic<T, S> {
        // predict
        let sigma_points = self.generate_sigma_points(state);
        let sp_xpred: Vec<SVector<T, S>> = sigma_points
            .iter()
            .map(|x| model.motion_model(x, u, dt))
            .collect();

        let mean_xpred: SVector<T, S> = sp_xpred
            .iter()
            .zip(self.mw.iter())
            .map(|(x, w)| x * *w)
            .sum();

        let cov_xpred = sp_xpred
            .iter()
            .map(|x| x - mean_xpred)
            .zip(self.cw.iter())
            .map(|(dx, cw)| dx * dx.transpose() * *cw)
            .sum::<SMatrix<T, S, S>>()
            + self.Q;

        let prediction = GaussianStateStatic {
            x: mean_xpred,
            P: cov_xpred,
        };

        // update
        let sp_xpred = self.generate_sigma_points(&prediction);
        let sp_z: Vec<SVector<T, Z>> = sp_xpred
            .iter()
            .map(|x| model.observation_model(x))
            .collect();

        let mean_z: SVector<T, Z> = sp_z.iter().zip(self.mw.iter()).map(|(x, w)| x * *w).sum();

        let cov_z = sp_z
            .iter()
            .map(|x| x - mean_z)
            .zip(self.cw.iter())
            .map(|(dx, cw)| dx * dx.transpose() * *cw)
            .sum::<SMatrix<T, Z, Z>>()
            + self.R;

        let s = sp_xpred
            .iter()
            .zip(sp_z.iter().zip(self.cw.iter()))
            .map(|(x_pred, (z_point, cw))| {
                (x_pred - mean_xpred) * (z_point - mean_z).transpose() * *cw
            })
            .sum::<SMatrix<T, S, Z>>();

        let y = z - mean_z;
        let kalman_gain = s * cov_z.try_inverse().unwrap();

        let x_est = mean_xpred + kalman_gain * y;
        let p_est = cov_xpred - kalman_gain * cov_z * kalman_gain.transpose();
        GaussianStateStatic { x: x_est, P: p_est }
    }
}

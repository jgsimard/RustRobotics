// Unscented kalman filter (UKF) localization sample
// author: Jean-Gabriel Simard (@jgsimard)

#![allow(non_snake_case)]
use nalgebra::{RealField, SMatrix, SVector, Scalar};
// use std::mem::MaybeUninit;

use crate::utils::state::GaussianStateStatic;

/// S : State Size, Z: Observation Size, U: Input Size
pub trait UnscentedKalmanFilterStatic<
    T: RealField + Scalar + Copy,
    const S: usize,
    const Z: usize,
    const U: usize,
>
{
    fn gamma(&self) -> T;
    fn mw(&self) -> &Vec<T>;
    fn cw(&self) -> &Vec<T>;

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

    // fn sigma_weights_array(alpha: T, beta: T, kappa: T) -> ([T; 2 * S + 1], [T; 2 * S + 1], T) {
    //     let n = T::from_usize(S).unwrap();
    //     let lambda = alpha.powi(2) * (n.clone() + kappa) - n.clone();

    //     let v = T::one() / ((T::one() + T::one()) * (n.clone() + lambda.clone()));
    //     let mut mw = [v; 2 * S + 1];
    //     let mut cw = [v; 2 * S + 1];

    //     // special cases
    //     let v = lambda.clone() / (n.clone() + lambda.clone());
    //     mw[0] = v.clone();
    //     cw[0] = v.clone() + T::one() - alpha.powi(2) + beta;

    //     let gamma = (n.clone() + lambda.clone()).sqrt();
    //     (mw, cw, gamma)
    // }

    fn generate_sigma_points(&self, state: &GaussianStateStatic<T, S>) -> Vec<SVector<T, S>> {
        // use cholesky to compute the matrix square root
        let sigma = state.P.cholesky().expect("unable to sqrt").l() * self.gamma();
        let mut sigma_points = Vec::new();
        sigma_points.push(state.x);
        for i in 0..S {
            let sigma_column = sigma.column(i);
            sigma_points.push(state.x + sigma_column);
            sigma_points.push(state.x - sigma_column);
        }
        sigma_points
    }

    // fn generate_sigma_points_array(
    //     &self,
    //     state: &GaussianStateStatic<T, S>,
    // ) -> [SVector<T, S>; 2 * S + 1]
    // where
    //     [SVector<T, S>; 2 * S + 1]: Sized,
    // {
    //     // use cholesky to compute the matrix square root
    //     let sigma = state.P.clone().cholesky().expect("unable to sqrt").l() * self.gamma();
    //     let mut sigma_points: [SVector<T, S>; 2 * S + 1] =
    //         unsafe { MaybeUninit::uninit().assume_init() };
    //     sigma_points[0] = state.x.clone();
    //     for i in 0..S {
    //         let sigma_column = sigma.column(i);
    //         sigma_points[i + 1] = &state.x + &sigma_column;
    //         sigma_points[i + 1 + S] = &state.x - &sigma_column;
    //     }
    //     sigma_points
    // }

    fn estimate(
        &self,
        state: &GaussianStateStatic<T, S>,
        u: &SVector<T, U>,
        z: &SVector<T, Z>,
        dt: T,
    ) -> GaussianStateStatic<T, S> {
        // predict
        let sigma_points = self.generate_sigma_points(state);
        let sp_xpred: Vec<SVector<T, S>> = sigma_points
            .iter()
            .map(|x| self.motion_model(x, u, dt))
            .collect();

        let mean_xpred: SVector<T, S> = sp_xpred
            .iter()
            .zip(self.mw().iter())
            .map(|(x, w)| x * *w)
            .sum();

        let cov_xpred = sp_xpred
            .iter()
            .map(|x| x - mean_xpred)
            .zip(self.cw().iter())
            .map(|(dx, cw)| dx * dx.transpose() * *cw)
            .sum::<SMatrix<T, S, S>>()
            + self.q();

        let prediction = GaussianStateStatic {
            x: mean_xpred,
            P: cov_xpred,
        };

        // update
        let sp_xpred = self.generate_sigma_points(&prediction);
        let sp_z: Vec<SVector<T, Z>> = sp_xpred.iter().map(|x| self.observation_model(x)).collect();

        let mean_z: SVector<T, Z> = sp_z.iter().zip(self.mw().iter()).map(|(x, w)| x * *w).sum();

        let cov_z = sp_z
            .iter()
            .map(|x| x - mean_z)
            .zip(self.cw().iter())
            .map(|(dx, cw)| dx * dx.transpose() * *cw)
            .sum::<SMatrix<T, Z, Z>>()
            + self.r();

        let s = sp_xpred
            .iter()
            .zip(sp_z.iter().zip(self.cw().iter()))
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

    fn motion_model(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SVector<T, S>;
    fn observation_model(&self, x: &SVector<T, S>) -> SVector<T, Z>;

    fn r(&self) -> SMatrix<T, Z, Z>;
    fn q(&self) -> SMatrix<T, S, S>;
}

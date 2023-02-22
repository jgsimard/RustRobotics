// Extended kalman filter (EKF) localization sample
// author: Atsushi Sakai (@Atsushi_twi)
//         Jean-Gabriel Simard (@jgsimard)
#![allow(non_snake_case)]
use nalgebra::{RealField, SMatrix, SVector};

use crate::utils::state::GaussianStateStatic;

pub trait ExtendedKalmanFilterModel<T: RealField, const S: usize, const Z: usize, const U: usize> {
    fn motion_model(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SVector<T, S>;
    fn observation_model(&self, x: &SVector<T, S>) -> SVector<T, Z>;

    /// Jacobian of Motion Model
    fn jacobian_motion_model(
        &self,
        x: &SVector<T, S>,
        u: &SVector<T, U>,
        dt: T,
    ) -> SMatrix<T, S, S>;

    /// Jacobian of Observation Model
    fn jacobian_observation_model(&self) -> SMatrix<T, Z, S>;
}

pub struct ExtendedKalmanFilter<T: RealField, const S: usize, const Z: usize, const U: usize> {
    pub Q: SMatrix<T, S, S>,
    pub R: SMatrix<T, Z, Z>,
}

/// S : State Size, Z: Observation Size, U: Input Size
impl<T: RealField, const S: usize, const Z: usize, const U: usize>
    ExtendedKalmanFilter<T, S, Z, U>
{
    pub fn predict(
        &self,
        model: &impl ExtendedKalmanFilterModel<T, S, Z, U>,
        estimate: &GaussianStateStatic<T, S>,
        u: &SVector<T, U>,
        dt: T,
    ) -> GaussianStateStatic<T, S> {
        let x_pred = model.motion_model(&estimate.x, u, dt.clone());
        let j_f = model.jacobian_motion_model(&x_pred, u, dt);
        let p_pred = &j_f * &estimate.P * j_f.transpose() + &self.Q;
        GaussianStateStatic {
            x: x_pred,
            P: p_pred,
        }
    }

    pub fn update(
        &self,
        model: &impl ExtendedKalmanFilterModel<T, S, Z, U>,
        prediction: &GaussianStateStatic<T, S>,
        z: &SVector<T, Z>,
    ) -> GaussianStateStatic<T, S> {
        let j_h = model.jacobian_observation_model();
        let z_pred = model.observation_model(&prediction.x);
        let y = z - z_pred;
        let s = &j_h * &prediction.P * j_h.transpose() + &self.R;
        let kalman_gain = &prediction.P * j_h.transpose() * s.try_inverse().unwrap();
        let x_est = &prediction.x + &kalman_gain * y;
        let p_est = (SMatrix::<T, S, S>::identity() - kalman_gain * j_h) * &prediction.P;
        GaussianStateStatic { x: x_est, P: p_est }
    }
}

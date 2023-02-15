// Extended kalman filter (EKF) localization sample
// author: Atsushi Sakai (@Atsushi_twi)
//         Jean-Gabriel Simard (@jgsimard)
#![allow(non_snake_case)]
use nalgebra::{
    allocator::Allocator, DMatrix, DVector, DefaultAllocator, Dim, OMatrix, OVector, RealField,
    SMatrix, SVector,
};

use crate::utils::{GaussianState, GaussianStateDynamic, GaussianStateStatic};

/// S : State Size, Z: Observation Size, U: Input Size
pub trait ExtendedKalmanFilterStatic<T: RealField, const S: usize, const Z: usize, const U: usize> {
    fn predict(
        &self,
        estimate: &GaussianStateStatic<T, S>,
        u: &SVector<T, U>,
        dt: T,
    ) -> GaussianStateStatic<T, S> {
        let x_pred = self.motion_model(&estimate.x, u, dt.clone());
        let j_f = self.jacobian_motion_model(&x_pred, u, dt);
        let p_pred = &j_f * &estimate.P * j_f.transpose() + self.q();
        GaussianStateStatic {
            x: x_pred,
            P: p_pred,
        }
    }

    fn update(
        &self,
        gaussian_state_pred: &GaussianStateStatic<T, S>,
        z: &SVector<T, Z>,
    ) -> GaussianStateStatic<T, S> {
        let x_pred = &gaussian_state_pred.x;
        let p_pred = &gaussian_state_pred.P;
        let j_h = self.jacobian_observation_model();
        let z_pred = self.observation_model(x_pred);
        let y = z - z_pred;
        let s = &j_h * p_pred * j_h.transpose() + self.r();
        let kalman_gain = p_pred * j_h.transpose() * s.try_inverse().unwrap();
        let x_est = x_pred + &kalman_gain * y;
        let p_est = (SMatrix::<T, S, S>::identity() - kalman_gain * j_h) * p_pred;
        GaussianStateStatic { x: x_est, P: p_est }
    }

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
    fn r(&self) -> SMatrix<T, Z, Z>;
    fn q(&self) -> SMatrix<T, S, S>;
}

pub trait ExtendedKalmanFilterDynamic<T: RealField> {
    fn predict(
        &self,
        estimate: &GaussianStateDynamic<T>,
        u: &DVector<T>,
        dt: T,
    ) -> GaussianStateDynamic<T> {
        let x_pred = self.motion_model(&estimate.x, u, dt.clone());
        let j_f = self.jacobian_motion_model(&x_pred, u, dt);
        let p_pred = &j_f * &estimate.P * j_f.transpose() + self.q();
        GaussianStateDynamic {
            x: x_pred,
            P: p_pred,
        }
    }

    fn update(
        &self,
        prediction: &GaussianStateDynamic<T>,
        z: &DVector<T>,
    ) -> GaussianStateDynamic<T> {
        let j_h = self.jacobian_observation_model();
        let z_pred = self.observation_model(&prediction.x);
        let y = z - z_pred;
        let s = &j_h * &prediction.P * j_h.transpose() + self.r();
        let kalman_gain = &prediction.P * j_h.transpose() * s.try_inverse().unwrap();
        let x_est = &prediction.x + &kalman_gain * y;
        let shape = prediction.P.shape();
        let p_est = (DMatrix::<T>::identity(shape.0, shape.1) - kalman_gain * j_h) * &prediction.P;
        GaussianStateDynamic { x: x_est, P: p_est }
    }

    fn motion_model(&self, x: &DVector<T>, u: &DVector<T>, dt: T) -> DVector<T>;
    fn observation_model(&self, x: &DVector<T>) -> DVector<T>;

    /// Jacobian of Motion Model
    fn jacobian_motion_model(&self, x: &DVector<T>, u: &DVector<T>, dt: T) -> DMatrix<T>;

    /// Jacobian of Observation Model
    fn jacobian_observation_model(&self) -> DMatrix<T>;

    fn r(&self) -> DMatrix<T>;
    fn q(&self) -> DMatrix<T>;
}

pub trait ExtendedKalmanFilter<T: RealField, S: Dim, Z: Dim, U: Dim> {
    fn predict(
        &self,
        gaussian_state_est: &GaussianState<T, S>,
        u: &OVector<T, U>,
        dt: T,
    ) -> GaussianState<T, S>
    where
        DefaultAllocator:
            Allocator<T, U> + Allocator<T, S> + Allocator<T, S, S> + Allocator<T, S, U>,
    {
        let x_pred = self.motion_model(&gaussian_state_est.x, u, dt.clone());
        let j_f = self.jacobian_motion_model(&x_pred, u, dt);
        let p_pred = &j_f * &gaussian_state_est.P * j_f.transpose() + self.q();
        GaussianState {
            x: x_pred,
            P: p_pred,
        }
    }

    fn update(&self, estimate: &GaussianState<T, S>, z: &OVector<T, Z>) -> GaussianState<T, S>
    where
        DefaultAllocator: Allocator<T, Z>
            + Allocator<T, S>
            + Allocator<T, S, S>
            + Allocator<T, Z, Z>
            + Allocator<T, Z, S>
            + Allocator<T, S, Z>,
    {
        let j_h = self.jacobian_observation_model();
        let z_pred = self.observation_model(&estimate.x);
        let y = z - z_pred;
        let s = &j_h * &estimate.P * j_h.transpose() + self.r();
        let kalman_gain = &estimate.P * j_h.transpose() * s.try_inverse().unwrap();
        let x_est = &estimate.x + &kalman_gain * y;
        let ps = &estimate.P.shape_generic();
        let p_est = (OMatrix::identity_generic(ps.0, ps.1) - kalman_gain * j_h) * &estimate.P;
        GaussianState { x: x_est, P: p_est }
    }

    fn motion_model(&self, x: &OVector<T, S>, u: &OVector<T, U>, dt: T) -> OVector<T, S>
    where
        DefaultAllocator: Allocator<T, U> + Allocator<T, S>;

    fn observation_model(&self, x: &OVector<T, S>) -> OVector<T, Z>
    where
        DefaultAllocator: Allocator<T, S> + Allocator<T, Z>;

    fn r(&self) -> OMatrix<T, Z, Z>
    where
        DefaultAllocator: Allocator<T, Z, Z>;

    fn q(&self) -> OMatrix<T, S, S>
    where
        DefaultAllocator: Allocator<T, S, S>;

    /// Jacobian of Motion Model
    fn jacobian_motion_model(
        &self,
        x: &OVector<T, S>,
        u: &OVector<T, U>,
        dt: T,
    ) -> OMatrix<T, S, S>
    where
        DefaultAllocator: Allocator<T, S> + Allocator<T, U> + Allocator<T, S, S>;

    /// Jacobian of Observation Model
    fn jacobian_observation_model(&self) -> OMatrix<T, Z, S>
    where
        DefaultAllocator: Allocator<T, Z, S>;
}

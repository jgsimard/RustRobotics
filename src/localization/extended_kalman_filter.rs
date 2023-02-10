// Extended kalman filter (EKF) localization sample
// author: Atsushi Sakai (@Atsushi_twi)
//         Jean-Gabriel Simard (@jgsimard)
#![allow(non_snake_case)]

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::utils::{GaussianStateDynamic, GaussianStateStatic};

pub trait ExtendedKalmanFilterStatic<
    const STATE_SIZE: usize,
    const OBSERVATION_SIZE: usize,
    const INPUT_SIZE: usize,
    T: nalgebra::RealField,
>
{
    fn motion_model(
        &self,
        x: &SVector<T, STATE_SIZE>,
        u: &SVector<T, INPUT_SIZE>,
        dt: T,
    ) -> SVector<T, STATE_SIZE> {
        self.f(x, dt.clone()) * x + self.b(x, dt) * u
    }

    fn observation_model(&self, x: &SVector<T, STATE_SIZE>) -> SVector<T, OBSERVATION_SIZE> {
        self.h() * x
    }

    fn predict(
        &self,
        kalman_state_est: &GaussianStateStatic<T, STATE_SIZE>,
        u: &SVector<T, INPUT_SIZE>,
        dt: T,
    ) -> GaussianStateStatic<T, STATE_SIZE> {
        let x_pred = self.motion_model(&kalman_state_est.x, u, dt.clone());
        let j_f = self.jacob_f(&x_pred, u, dt);
        let p_pred = &j_f * &kalman_state_est.P * j_f.transpose() + self.q();
        GaussianStateStatic {
            x: x_pred,
            P: p_pred,
        }
    }

    fn update(
        &self,
        kalman_state_pred: &GaussianStateStatic<T, STATE_SIZE>,
        z: &SVector<T, OBSERVATION_SIZE>,
    ) -> GaussianStateStatic<T, STATE_SIZE> {
        let x_pred = &kalman_state_pred.x;
        let p_pred = &kalman_state_pred.P;
        let j_h = self.jacob_h();
        let z_pred = self.observation_model(x_pred);
        let y = z - z_pred;
        let s = &j_h * p_pred * j_h.transpose() + self.r();
        let kalman_gain = p_pred * j_h.transpose() * s.try_inverse().unwrap();
        let x_est = x_pred + &kalman_gain * y;
        let p_est = (SMatrix::<T, STATE_SIZE, STATE_SIZE>::identity() - kalman_gain * j_h) * p_pred;
        GaussianStateStatic { x: x_est, P: p_est }
    }

    fn estimation(
        &self,
        kalman_state_est: &GaussianStateStatic<T, STATE_SIZE>,
        z: &SVector<T, OBSERVATION_SIZE>,
        u: &SVector<T, INPUT_SIZE>,
        dt: T,
    ) -> GaussianStateStatic<T, STATE_SIZE> {
        let kalman_state_pred = self.predict(kalman_state_est, u, dt);
        self.update(&kalman_state_pred, z)
    }

    fn f(&self, x: &SVector<T, STATE_SIZE>, dt: T) -> SMatrix<T, STATE_SIZE, STATE_SIZE>;
    fn b(&self, x: &SVector<T, STATE_SIZE>, dt: T) -> SMatrix<T, STATE_SIZE, INPUT_SIZE>;
    fn h(&self) -> SMatrix<T, OBSERVATION_SIZE, STATE_SIZE>;
    fn r(&self) -> SMatrix<T, OBSERVATION_SIZE, OBSERVATION_SIZE>;
    fn q(&self) -> SMatrix<T, STATE_SIZE, STATE_SIZE>;

    /// Jacobian of Motion Model
    fn jacob_f(
        &self,
        x: &SVector<T, STATE_SIZE>,
        u: &SVector<T, INPUT_SIZE>,
        dt: T,
    ) -> SMatrix<T, STATE_SIZE, STATE_SIZE>;

    /// Jacobian of Observation Model
    fn jacob_h(&self) -> SMatrix<T, OBSERVATION_SIZE, STATE_SIZE>;
}

pub trait ExtendedKalmanFilterDynamic<T: nalgebra::RealField> {
    fn motion_model(&self, x: &DVector<T>, u: &DVector<T>, dt: T) -> DVector<T> {
        self.f(x, dt.clone()) * x + self.b(x, dt) * u
    }

    fn observation_model(&self, x: &DVector<T>) -> DVector<T> {
        self.h() * x
    }

    fn predict(
        &self,
        kalman_state_est: &GaussianStateDynamic<T>,
        u: &DVector<T>,
        dt: T,
    ) -> GaussianStateDynamic<T> {
        let x_pred = self.motion_model(&kalman_state_est.x, u, dt.clone());
        let j_f = self.jacob_f(&x_pred, u, dt);
        let p_pred = &j_f * &kalman_state_est.P * j_f.transpose() + self.q();
        GaussianStateDynamic {
            x: x_pred,
            P: p_pred,
        }
    }

    fn update(
        &self,
        kalman_state_pred: &GaussianStateDynamic<T>,
        z: &DVector<T>,
    ) -> GaussianStateDynamic<T> {
        let x_pred = &kalman_state_pred.x;
        let p_pred = &kalman_state_pred.P;
        let j_h = self.jacob_h();
        let z_pred = self.observation_model(x_pred);
        let y = z - z_pred;
        let s = j_h.clone() * p_pred * j_h.transpose() + self.r();
        let kalman_gain = p_pred * j_h.transpose() * s.try_inverse().unwrap();
        let x_est = x_pred + &kalman_gain * y;
        let shape = kalman_state_pred.P.shape();
        let p_est = (DMatrix::<T>::identity(shape.0, shape.1) - kalman_gain * j_h) * p_pred;
        GaussianStateDynamic { x: x_est, P: p_est }
    }

    fn estimation(
        &self,
        kalman_state_est: &GaussianStateDynamic<T>,
        z: &DVector<T>,
        u: &DVector<T>,
        dt: T,
    ) -> GaussianStateDynamic<T> {
        let kalman_state_pred = self.predict(kalman_state_est, u, dt);
        self.update(&kalman_state_pred, z)
    }

    fn f(&self, x: &DVector<T>, dt: T) -> DMatrix<T>;
    fn b(&self, x: &DVector<T>, dt: T) -> DMatrix<T>;
    fn h(&self) -> DMatrix<T>;
    fn r(&self) -> DMatrix<T>;
    fn q(&self) -> DMatrix<T>;

    /// Jacobian of Motion Model
    fn jacob_f(&self, x: &DVector<T>, u: &DVector<T>, dt: T) -> DMatrix<T>;

    /// Jacobian of Observation Model
    fn jacob_h(&self) -> DMatrix<T>;
}

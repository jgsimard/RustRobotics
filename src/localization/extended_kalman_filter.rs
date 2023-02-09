// Extended kalman filter (EKF) localization sample
// author: Atsushi Sakai (@Atsushi_twi)
//         Jean-Gabriel Simard (@jgsimard)

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

pub trait ExtendedKalmanFilterStatic<
    const STATE_SIZE: usize,
    const OBSERVATION_SIZE: usize,
    const INPUT_SIZE: usize,
    T: nalgebra::RealField + nalgebra::Scalar,
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
        x_est: &SVector<T, STATE_SIZE>,
        p_est: &SMatrix<T, STATE_SIZE, STATE_SIZE>,
        u: &SVector<T, INPUT_SIZE>,
        dt: T,
    ) -> (SVector<T, STATE_SIZE>, SMatrix<T, STATE_SIZE, STATE_SIZE>) {
        let x_pred = self.motion_model(x_est, u, dt.clone());
        let j_f = self.jacob_f(&x_pred, u, dt);
        let p_pred = j_f.clone() * p_est * j_f.transpose() + self.q();
        (x_pred, p_pred)
    }

    fn update(
        &self,
        x_pred: &SVector<T, STATE_SIZE>,
        p_pred: &SMatrix<T, STATE_SIZE, STATE_SIZE>,
        z: &SVector<T, OBSERVATION_SIZE>,
    ) -> (SVector<T, STATE_SIZE>, SMatrix<T, STATE_SIZE, STATE_SIZE>) {
        let j_h = self.jacob_h();
        let z_pred = self.observation_model(x_pred);
        let y = z - z_pred;
        let s = j_h.clone() * p_pred * j_h.transpose() + self.r();
        let kalman_gain = p_pred * j_h.transpose() * s.try_inverse().unwrap();
        let new_x_est = x_pred + kalman_gain.clone() * y;
        let new_p_est =
            (SMatrix::<T, STATE_SIZE, STATE_SIZE>::identity() - kalman_gain * j_h) * p_pred;

        (new_x_est, new_p_est)
    }

    fn estimation(
        &self,
        x_est: &SVector<T, STATE_SIZE>,
        p_est: &SMatrix<T, STATE_SIZE, STATE_SIZE>,
        z: &SVector<T, OBSERVATION_SIZE>,
        u: &SVector<T, INPUT_SIZE>,
        dt: T,
    ) -> (SVector<T, STATE_SIZE>, SMatrix<T, STATE_SIZE, STATE_SIZE>) {
        let (x_pred, p_pred) = self.predict(x_est, p_est, u, dt);
        self.update(&x_pred, &p_pred, z)
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

pub trait ExtendedKalmanFilterDynamic<T: nalgebra::RealField + nalgebra::Scalar> {
    fn motion_model(&self, x: &DVector<T>, u: &DVector<T>, dt: T) -> DVector<T> {
        self.f(x, dt.clone()) * x + self.b(x, dt) * u
    }

    fn observation_model(&self, x: &DVector<T>) -> DVector<T> {
        self.h() * x
    }

    fn predict(
        &self,
        x_est: &DVector<T>,
        p_est: &DMatrix<T>,
        u: &DVector<T>,
        dt: T,
    ) -> (DVector<T>, DMatrix<T>) {
        let x_pred = self.motion_model(x_est, u, dt.clone());
        let j_f = self.jacob_f(&x_pred, u, dt);
        let p_pred = j_f.clone() * p_est * j_f.transpose() + self.q();
        (x_pred, p_pred)
    }

    fn update(
        &self,
        x_pred: &DVector<T>,
        p_pred: &DMatrix<T>,
        z: &DVector<T>,
    ) -> (DVector<T>, DMatrix<T>) {
        let j_h = self.jacob_h();
        let z_pred = self.observation_model(x_pred);
        let y = z - z_pred;
        let s = j_h.clone() * p_pred * j_h.transpose() + self.r();
        let kalman_gain = p_pred * j_h.transpose() * s.try_inverse().unwrap();
        let new_x_est = x_pred + kalman_gain.clone() * y;
        let shape = p_pred.shape();
        let new_p_est = (DMatrix::<T>::identity(shape.0, shape.1) - kalman_gain * j_h) * p_pred;

        (new_x_est, new_p_est)
    }

    fn estimation(
        &self,
        x_est: &DVector<T>,
        p_est: &DMatrix<T>,
        z: &DVector<T>,
        u: &DVector<T>,
        dt: T,
    ) -> (DVector<T>, DMatrix<T>) {
        let (x_pred, p_pred) = self.predict(x_est, p_est, u, dt);
        self.update(&x_pred, &p_pred, z)
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

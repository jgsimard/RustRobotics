// Extended kalman filter (EKF) localization sample
// author: Atsushi Sakai (@Atsushi_twi)
//         Jean-Gabriel Simard (@jgsimard)

use crate::utils::{Matrix, Vector};

pub trait ExtendedKalmanFilter<
    const STATE_SIZE: usize,
    const OBSERVATION_SIZE: usize,
    const INPUT_SIZE: usize,
>
{
    fn motion_model(
        &self,
        x: Vector<STATE_SIZE>,
        u: Vector<INPUT_SIZE>,
        dt: f32,
    ) -> Vector<STATE_SIZE> {
        self.f(x, dt) * x + self.b(x, dt) * u
    }

    fn observation_model(&self, x: Vector<STATE_SIZE>) -> Vector<OBSERVATION_SIZE> {
        self.h() * x
    }

    fn predict(
        &self,
        x_est: Vector<STATE_SIZE>,
        p_est: Matrix<STATE_SIZE, STATE_SIZE>,
        u: Vector<INPUT_SIZE>,
        dt: f32,
    ) -> (Vector<STATE_SIZE>, Matrix<STATE_SIZE, STATE_SIZE>) {
        let x_pred = self.motion_model(x_est, u, dt);
        let j_f = self.jacob_f(x_pred, u, dt);
        let p_pred = j_f * p_est * j_f.transpose() + self.q();
        (x_pred, p_pred)
    }

    fn update(
        &self,
        x_pred: Vector<STATE_SIZE>,
        p_pred: Matrix<STATE_SIZE, STATE_SIZE>,
        z: Vector<OBSERVATION_SIZE>,
    ) -> (Vector<STATE_SIZE>, Matrix<STATE_SIZE, STATE_SIZE>) {
        let j_h = self.jacob_h();
        let z_pred = self.observation_model(x_pred);
        let y = z - z_pred;
        let s = j_h * p_pred * j_h.transpose() + self.r();
        let k = p_pred * j_h.transpose() * s.try_inverse().unwrap();
        let new_x_est = x_pred + k * y;
        let new_p_est = (Matrix::<STATE_SIZE, STATE_SIZE>::identity() - k * j_h) * p_pred;

        (new_x_est, new_p_est)
    }

    fn estimation(
        &self,
        x_est: Vector<STATE_SIZE>,
        p_est: Matrix<STATE_SIZE, STATE_SIZE>,
        z: Vector<OBSERVATION_SIZE>,
        u: Vector<INPUT_SIZE>,
        dt: f32,
    ) -> (Vector<STATE_SIZE>, Matrix<STATE_SIZE, STATE_SIZE>) {
        let (x_pred, p_pred) = self.predict(x_est, p_est, u, dt);
        self.update(x_pred, p_pred, z)
    }

    fn f(&self, x: Vector<STATE_SIZE>, dt: f32) -> Matrix<STATE_SIZE, STATE_SIZE>;
    fn b(&self, x: Vector<STATE_SIZE>, dt: f32) -> Matrix<STATE_SIZE, INPUT_SIZE>;
    fn h(&self) -> Matrix<OBSERVATION_SIZE, STATE_SIZE>;
    fn r(&self) -> Matrix<OBSERVATION_SIZE, OBSERVATION_SIZE>;
    fn q(&self) -> Matrix<STATE_SIZE, STATE_SIZE>;

    /// Jacobian of Motion Model
    fn jacob_f(
        &self,
        x: Vector<STATE_SIZE>,
        _u: Vector<INPUT_SIZE>,
        dt: f32,
    ) -> Matrix<STATE_SIZE, STATE_SIZE>;

    /// Jacobian of Observation Model
    fn jacob_h(&self) -> Matrix<OBSERVATION_SIZE, STATE_SIZE>;
}

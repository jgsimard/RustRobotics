// Extended kalman filter (EKF) localization sample
// author: Atsushi Sakai (@Atsushi_twi)
//         Jean-Gabriel Simard (@jgsimard)
#![allow(non_snake_case)]
use nalgebra::{RealField, SMatrix, SVector};

use crate::utils::state::GaussianStateStatic;

/// S : State Size, Z: Observation Size, U: Input Size
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

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ExtendedKalmanFilter<T: RealField, const S: usize, const Z: usize, const U: usize> {
    Q: SMatrix<T, S, S>,
    R: SMatrix<T, Z, Z>,
}

impl<T: RealField, const S: usize, const Z: usize, const U: usize>
    ExtendedKalmanFilter<T, S, Z, U>
{
    pub fn new(Q: SMatrix<T, S, S>, R: SMatrix<T, Z, Z>) -> ExtendedKalmanFilter<T, S, Z, U> {
        ExtendedKalmanFilter { Q, R }
    }

    pub fn predict(
        &self,
        model: &impl ExtendedKalmanFilterModel<T, S, Z, U>,
        estimate: &GaussianStateStatic<T, S>,
        u: &SVector<T, U>,
        dt: T,
    ) -> GaussianStateStatic<T, S> {
        let j_f = model.jacobian_motion_model(&estimate.x, u, dt.clone());
        let x_pred = model.motion_model(&estimate.x, u, dt);
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

#[cfg(test)]
mod tests {
    extern crate test;
    use crate::localization::extended_kalman_filter::{
        ExtendedKalmanFilter, ExtendedKalmanFilterModel,
    };
    use crate::utils::deg2rad;
    use crate::utils::state::GaussianStateStatic as GaussianState;
    use nalgebra::{Matrix2x4, Matrix4, Vector2, Vector4};
    use test::{black_box, Bencher};

    struct SimpleProblem {}

    impl ExtendedKalmanFilterModel<f32, 4, 2, 2> for SimpleProblem {
        fn motion_model(&self, x: &Vector4<f32>, u: &Vector2<f32>, dt: f32) -> Vector4<f32> {
            let yaw = x[2];
            let v = x[3];
            Vector4::new(
                x.x + yaw.cos() * v * dt,
                x.y + yaw.sin() * v * dt,
                yaw + u.y * dt,
                u.x,
            )
        }

        fn observation_model(&self, x: &Vector4<f32>) -> Vector2<f32> {
            x.xy()
        }

        #[allow(clippy::deprecated_cfg_attr)]
        fn jacobian_motion_model(
            &self,
            x: &Vector4<f32>,
            u: &Vector2<f32>,
            dt: f32,
        ) -> Matrix4<f32> {
            let yaw = x[2];
            let v = u[0];
            #[cfg_attr(rustfmt, rustfmt_skip)]
            Matrix4::<f32>::new(
                1., 0., -dt * v * (yaw).sin(), dt * (yaw).cos(),
                0., 1., dt * v * (yaw).cos(), dt * (yaw).sin(),
                0., 0., 1., 0.,
                0., 0., 0., 0.,
            )
        }

        #[allow(clippy::deprecated_cfg_attr)]
        fn jacobian_observation_model(&self) -> Matrix2x4<f32> {
            #[cfg_attr(rustfmt, rustfmt_skip)]
            Matrix2x4::<f32>::new(
                1., 0., 0., 0.,
                0., 1., 0., 0.
            )
        }
    }

    #[bench]
    fn ekf(b: &mut Bencher) {
        // setup ukf
        let q = Matrix4::<f32>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
        let r = nalgebra::Matrix2::identity();
        let ekf = ExtendedKalmanFilter::<f32, 4, 2, 2>::new(q, r);

        let simple_problem = SimpleProblem {};

        let dt = 0.1;
        let u: Vector2<f32> = Default::default();
        let kalman_state = GaussianState {
            x: Vector4::<f32>::new(0., 0., 0., 0.),
            P: Matrix4::<f32>::identity(),
        };
        let z: Vector2<f32> = Default::default();

        b.iter(|| {
            black_box(ekf.predict(&simple_problem, &kalman_state, &u, dt));
            black_box(ekf.update(&simple_problem, &kalman_state, &z));
        });
    }
}

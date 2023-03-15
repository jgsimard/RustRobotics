#![allow(non_snake_case)]
use nalgebra::{RealField, SMatrix, SVector};
use rustc_hash::FxHashMap;

use crate::models::measurement::MeasurementModel;
use crate::models::motion::MotionModel;
use crate::utils::state::GaussianStateStatic;

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ExtendedKalmanFilter<T: RealField, const S: usize, const Z: usize, const U: usize> {
    R: SMatrix<T, S, S>,
    Q: SMatrix<T, Z, Z>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
    motion_model: Box<dyn MotionModel<T, S, Z, U>>,
}

impl<T: RealField, const S: usize, const Z: usize, const U: usize>
    ExtendedKalmanFilter<T, S, Z, U>
{
    pub fn new(
        R: SMatrix<T, S, S>,
        Q: SMatrix<T, Z, Z>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
        motion_model: Box<dyn MotionModel<T, S, Z, U>>,
    ) -> ExtendedKalmanFilter<T, S, Z, U> {
        ExtendedKalmanFilter {
            R,
            Q,
            measurement_model,
            motion_model,
        }
    }

    pub fn estimate(
        &self,
        // model: &impl ExtendedKalmanFilterModel<T, S, Z, U>,
        estimate: &GaussianStateStatic<T, S>,
        u: &SVector<T, U>,
        z: &SVector<T, Z>,
        dt: T,
    ) -> GaussianStateStatic<T, S> {
        // predict
        let G = self
            .motion_model
            .jacobian_wrt_state(&estimate.x, u, dt.clone());
        let x_pred = self.motion_model.prediction(&estimate.x, u, dt);
        let p_pred = &G * &estimate.P * G.transpose() + &self.R;

        // update
        let H = self.measurement_model.jacobian(&x_pred, None);
        let z_pred = self.measurement_model.prediction(&x_pred, None);

        let s = &H * &p_pred * H.transpose() + &self.Q;
        let kalman_gain = &p_pred * H.transpose() * s.try_inverse().unwrap();
        let x_est = &x_pred + &kalman_gain * (z - z_pred);
        let p_est = (SMatrix::<T, S, S>::identity() - kalman_gain * H) * &p_pred;
        GaussianStateStatic { x: x_est, P: p_est }
    }
}

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ExtendedKalmanFilterKnownCorrespondences<
    T: RealField,
    const S: usize,
    const Z: usize,
    const U: usize,
> {
    R: SMatrix<T, S, S>,
    Q: SMatrix<T, Z, Z>,
    landmarks: FxHashMap<u32, SVector<T, S>>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
    motion_model: Box<dyn MotionModel<T, S, Z, U>>,
    fixed_noise: bool,
}

impl<T: RealField, const S: usize, const Z: usize, const U: usize>
    ExtendedKalmanFilterKnownCorrespondences<T, S, Z, U>
{
    pub fn new(
        R: SMatrix<T, S, S>,
        Q: SMatrix<T, Z, Z>,
        landmarks: FxHashMap<u32, SVector<T, S>>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
        motion_model: Box<dyn MotionModel<T, S, Z, U>>,
        fixed_noise: bool,
    ) -> ExtendedKalmanFilterKnownCorrespondences<T, S, Z, U> {
        ExtendedKalmanFilterKnownCorrespondences {
            Q,
            R,
            landmarks,
            measurement_model,
            motion_model,
            fixed_noise,
        }
    }

    pub fn estimate(
        &self,
        estimate: &GaussianStateStatic<T, S>,
        control: Option<SVector<T, U>>,
        measurements: Option<Vec<(u32, SVector<T, Z>)>>,
        dt: T,
    ) -> GaussianStateStatic<T, S> {
        let mut x_out = estimate.x.clone();
        let mut p_out = estimate.P.clone();
        // predict
        if let Some(u) = control {
            let G = self
                .motion_model
                .jacobian_wrt_state(&estimate.x, &u, dt.clone());

            let x_est = self.motion_model.prediction(&estimate.x, &u, dt.clone());
            let p_est = if self.fixed_noise {
                // fixed version
                &G * &estimate.P * G.transpose() + &self.R
            } else {
                // adaptive version
                let V = self.motion_model.jacobian_wrt_input(&estimate.x, &u, dt);
                let M = self.motion_model.cov_noise_control_space(&u);
                &G * &estimate.P * G.transpose() + &V * M * V.transpose()
            };
            x_out = x_est;
            p_out = p_est;
        }

        // update / correction step
        if let Some(measurements) = measurements {
            for (id, z) in measurements
                .iter()
                .filter(|(id, _)| self.landmarks.contains_key(id))
            {
                let landmark = self.landmarks.get(id);
                let z_pred = self.measurement_model.prediction(&x_out, landmark);
                let H = self.measurement_model.jacobian(&x_out, landmark);
                let s = &H * &p_out * H.transpose() + &self.Q;
                let kalman_gain = &p_out * H.transpose() * s.try_inverse().unwrap();
                x_out += &kalman_gain * (z - z_pred);
                p_out = (SMatrix::<T, S, S>::identity() - kalman_gain * H) * &p_out
            }
        }

        GaussianStateStatic { x: x_out, P: p_out }
    }
}

#[cfg(test)]
mod tests {
    use crate::localization::extended_kalman_filter::ExtendedKalmanFilter;
    use crate::models::measurement::SimpleProblemMeasurementModel;
    use crate::models::motion::SimpleProblemMotionModel;
    use crate::utils::deg2rad;
    use crate::utils::state::GaussianStateStatic as GaussianState;
    use nalgebra::{Matrix4, Vector2, Vector4};

    #[test]
    fn ekf_runs() {
        // setup ukf
        let q = Matrix4::<f64>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
        let r = nalgebra::Matrix2::identity();
        let motion_model = Box::new(SimpleProblemMotionModel {});
        let measurement_model = Box::new(SimpleProblemMeasurementModel {});
        let ekf = ExtendedKalmanFilter::<f64, 4, 2, 2>::new(q, r, measurement_model, motion_model);

        let dt = 0.1;
        let u: Vector2<f64> = Default::default();
        let kalman_state = GaussianState {
            x: Vector4::<f64>::new(0., 0., 0., 0.),
            P: Matrix4::<f64>::identity(),
        };
        let z: Vector2<f64> = Default::default();

        ekf.estimate(&kalman_state, &u, &z, dt);
    }
}

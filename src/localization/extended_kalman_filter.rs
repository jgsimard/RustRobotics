use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, OMatrix, OVector, RealField};
use rustc_hash::FxHashMap;

use crate::models::measurement::MeasurementModel;
use crate::models::motion::MotionModel;
use crate::utils::state::GaussianState;

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ExtendedKalmanFilter<T: RealField, S: Dim, Z: Dim, U: Dim>
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
    r: OMatrix<T, S, S>,
    q: OMatrix<T, Z, Z>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
    motion_model: Box<dyn MotionModel<T, S, Z, U>>,
}

impl<T: RealField, S: Dim, Z: Dim, U: Dim> ExtendedKalmanFilter<T, S, Z, U>
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
        r: OMatrix<T, S, S>,
        q: OMatrix<T, Z, Z>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
        motion_model: Box<dyn MotionModel<T, S, Z, U>>,
    ) -> ExtendedKalmanFilter<T, S, Z, U> {
        ExtendedKalmanFilter {
            r,
            q,
            measurement_model,
            motion_model,
        }
    }

    pub fn estimate(
        &self,
        estimate: &GaussianState<T, S>,
        u: &OVector<T, U>,
        z: &OVector<T, Z>,
        dt: T,
    ) -> GaussianState<T, S> {
        // predict
        let g = self
            .motion_model
            .jacobian_wrt_state(&estimate.x, u, dt.clone());
        let x_pred = self.motion_model.prediction(&estimate.x, u, dt);
        let cov_pred = &g * &estimate.cov * g.transpose() + &self.r;

        // update
        let h = self.measurement_model.jacobian(&x_pred, None);
        let z_pred = self.measurement_model.prediction(&x_pred, None);

        let s = &h * &cov_pred * h.transpose() + &self.q;
        let kalman_gain = &cov_pred * h.transpose() * s.try_inverse().unwrap();
        let x_est = &x_pred + &kalman_gain * (z - z_pred);
        let shape = cov_pred.shape_generic();
        let cov_est = (OMatrix::identity_generic(shape.0, shape.1) - kalman_gain * h) * &cov_pred;
        GaussianState {
            x: x_est,
            cov: cov_est,
        }
    }
}

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ExtendedKalmanFilterKnownCorrespondences<T: RealField, S: Dim, Z: Dim, U: Dim>
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
        + Allocator<T, Const<1>, Z>
        + Allocator<T, U, S>,
{
    r: OMatrix<T, S, S>,
    q: OMatrix<T, Z, Z>,
    landmarks: FxHashMap<u32, OVector<T, S>>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
    motion_model: Box<dyn MotionModel<T, S, Z, U>>,
    fixed_noise: bool,
}

impl<T: RealField, S: Dim, Z: Dim, U: Dim> ExtendedKalmanFilterKnownCorrespondences<T, S, Z, U>
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
        + Allocator<T, Const<1>, Z>
        + Allocator<T, U, S>,
{
    pub fn new(
        r: OMatrix<T, S, S>,
        q: OMatrix<T, Z, Z>,
        landmarks: FxHashMap<u32, OVector<T, S>>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
        motion_model: Box<dyn MotionModel<T, S, Z, U>>,
        fixed_noise: bool,
    ) -> ExtendedKalmanFilterKnownCorrespondences<T, S, Z, U> {
        ExtendedKalmanFilterKnownCorrespondences {
            q,
            r,
            landmarks,
            measurement_model,
            motion_model,
            fixed_noise,
        }
    }

    pub fn estimate(
        &self,
        estimate: &GaussianState<T, S>,
        control: Option<OVector<T, U>>,
        measurements: Option<Vec<(u32, OVector<T, Z>)>>,
        dt: T,
    ) -> GaussianState<T, S> {
        let mut x_out = estimate.x.clone();
        let mut cov_out = estimate.cov.clone();
        // predict
        if let Some(u) = control {
            let g = self
                .motion_model
                .jacobian_wrt_state(&estimate.x, &u, dt.clone());

            let x_est = self.motion_model.prediction(&estimate.x, &u, dt.clone());
            let cov_est = if self.fixed_noise {
                // fixed version
                &g * &estimate.cov * g.transpose() + &self.r
            } else {
                // adaptive version
                let v = self.motion_model.jacobian_wrt_input(&estimate.x, &u, dt);
                let m = self.motion_model.cov_noise_control_space(&u);
                &g * &estimate.cov * g.transpose() + &v * m * v.transpose()
            };
            x_out = x_est;
            cov_out = cov_est;
        }

        // update / correction step
        if let Some(measurements) = measurements {
            let shape = cov_out.shape_generic();
            for (id, z) in measurements
                .iter()
                .filter(|(id, _)| self.landmarks.contains_key(id))
            {
                let landmark = self.landmarks.get(id);
                let z_pred = self.measurement_model.prediction(&x_out, landmark);
                let h = self.measurement_model.jacobian(&x_out, landmark);
                let s = &h * &cov_out * h.transpose() + &self.q;
                let kalman_gain = &cov_out * h.transpose() * s.try_inverse().unwrap();
                x_out += &kalman_gain * (z - z_pred);
                cov_out = (OMatrix::identity_generic(shape.0, shape.1) - kalman_gain * h) * &cov_out
            }
        }

        GaussianState {
            x: x_out,
            cov: cov_out,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::localization::extended_kalman_filter::ExtendedKalmanFilter;
    use crate::models::measurement::SimpleProblemMeasurementModel;
    use crate::models::motion::SimpleProblemMotionModel;
    use crate::utils::deg2rad;
    use crate::utils::state::GaussianState;
    use nalgebra::{Const, Matrix4, Vector2, Vector4};

    #[test]
    fn ekf_runs() {
        // setup ukf
        let q = Matrix4::<f64>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
        let r = nalgebra::Matrix2::identity();
        let motion_model = Box::new(SimpleProblemMotionModel {});
        let measurement_model = Box::new(SimpleProblemMeasurementModel {});
        let ekf = ExtendedKalmanFilter::<f64, Const<4>, Const<2>, Const<2>>::new(
            q,
            r,
            measurement_model,
            motion_model,
        );

        let dt = 0.1;
        let u: Vector2<f64> = Default::default();
        let kalman_state = GaussianState {
            x: Vector4::<f64>::new(0., 0., 0., 0.),
            cov: Matrix4::<f64>::identity(),
        };
        let z: Vector2<f64> = Default::default();

        ekf.estimate(&kalman_state, &u, &z, dt);
    }
}

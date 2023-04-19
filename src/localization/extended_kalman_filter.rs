use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, RealField};
use rustc_hash::FxHashMap;

use crate::localization::{BayesianFilter, BayesianFilterKnownCorrespondences};
use crate::models::measurement::MeasurementModel;
use crate::models::motion::MotionModel;
use crate::utils::state::GaussianState;

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ExtendedKalmanFilter<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Z, Z>,
{
    r: OMatrix<T, S, S>,
    q: OMatrix<T, Z, Z>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
    motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
    state: GaussianState<T, S>,
}

impl<T: RealField, S: Dim, Z: Dim, U: Dim> ExtendedKalmanFilter<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Z, Z>,
{
    pub fn new(
        r: OMatrix<T, S, S>,
        q: OMatrix<T, Z, Z>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
        motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
        initial_state: GaussianState<T, S>,
    ) -> ExtendedKalmanFilter<T, S, Z, U> {
        ExtendedKalmanFilter {
            r,
            q,
            measurement_model,
            motion_model,
            state: initial_state,
        }
    }
}

impl<T: RealField, S: Dim, Z: Dim, U: Dim> BayesianFilter<T, S, Z, U>
    for ExtendedKalmanFilter<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, Z>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Z, S>
        + Allocator<T, S, U>
        + Allocator<T, U, U>
        + Allocator<T, S, Z>,
{
    fn update_estimate(&mut self, u: &OVector<T, U>, z: &OVector<T, Z>, dt: T) {
        // predict
        let g = self
            .motion_model
            .jacobian_wrt_state(&self.state.x, u, dt.clone());
        self.state.x = self.motion_model.prediction(&self.state.x, u, dt);
        self.state.cov = &g * &self.state.cov * g.transpose() + &self.r;

        // update
        let h = self.measurement_model.jacobian(&self.state.x, None);
        let z_pred = self.measurement_model.prediction(&self.state.x, None);

        let s = &h * &self.state.cov * h.transpose() + &self.q;
        let kalman_gain = &self.state.cov * h.transpose() * s.try_inverse().unwrap();
        self.state.x = &self.state.x + &kalman_gain * (z - z_pred);
        let shape = self.state.cov.shape_generic();
        self.state.cov =
            (OMatrix::identity_generic(shape.0, shape.1) - kalman_gain * h) * &self.state.cov;
    }

    fn gaussian_estimate(&self) -> GaussianState<T, S> {
        self.state.clone()
    }
}

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ExtendedKalmanFilterKnownCorrespondences<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Z, Z>,
{
    q: OMatrix<T, Z, Z>,
    landmarks: FxHashMap<u32, OVector<T, S>>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
    motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
    state: GaussianState<T, S>,
}

impl<T: RealField, S: Dim, Z: Dim, U: Dim> ExtendedKalmanFilterKnownCorrespondences<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, U> + Allocator<T, S, S> + Allocator<T, Z, Z>,
{
    pub fn new(
        q: OMatrix<T, Z, Z>,
        landmarks: FxHashMap<u32, OVector<T, S>>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
        motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
        initial_state: GaussianState<T, S>,
    ) -> ExtendedKalmanFilterKnownCorrespondences<T, S, Z, U> {
        ExtendedKalmanFilterKnownCorrespondences {
            q,
            landmarks,
            measurement_model,
            motion_model,
            state: initial_state,
        }
    }
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> BayesianFilterKnownCorrespondences<T, S, Z, U>
    for ExtendedKalmanFilterKnownCorrespondences<T, S, Z, U>
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
        + Allocator<T, U, S>,
{
    fn update_estimate(
        &mut self,
        control: Option<OVector<T, U>>,
        measurements: Option<Vec<(u32, OVector<T, Z>)>>,
        dt: T,
    ) {
        // predict
        if let Some(u) = control {
            let g = self.motion_model.jacobian_wrt_state(&self.state.x, &u, dt);
            let v = self.motion_model.jacobian_wrt_input(&self.state.x, &u, dt);
            let m = self.motion_model.cov_noise_control_space(&u);

            self.state.x = self.motion_model.prediction(&self.state.x, &u, dt);
            self.state.cov = &g * &self.state.cov * g.transpose() + &v * m * v.transpose();
        }

        // update / correction step
        if let Some(measurements) = measurements {
            let shape = self.state.cov.shape_generic();
            for (id, z) in measurements
                .iter()
                .filter(|(id, _)| self.landmarks.contains_key(id))
            {
                let landmark = self.landmarks.get(id);
                let z_pred = self.measurement_model.prediction(&self.state.x, landmark);
                let h = self.measurement_model.jacobian(&self.state.x, landmark);
                let s = &h * &self.state.cov * h.transpose() + &self.q;
                let kalman_gain = &self.state.cov * h.transpose() * s.try_inverse().unwrap();
                self.state.x += &kalman_gain * (z - z_pred);
                self.state.cov = (OMatrix::identity_generic(shape.0, shape.1) - kalman_gain * h)
                    * &self.state.cov;
            }
        }
    }

    fn gaussian_estimate(&self) -> GaussianState<T, S> {
        self.state.clone()
    }
}

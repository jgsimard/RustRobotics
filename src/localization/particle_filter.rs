use nalgebra::{RealField, SMatrix, SVector};
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::{Standard, StandardNormal};
use rustc_hash::FxHashMap;

use crate::models::measurement::MeasurementModel;
use crate::models::motion::MotionModel;
use crate::utils::mvn::MultiVariateNormal;
use crate::utils::state::GaussianStateStatic;

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ParticleFilterKnownCorrespondences<
    T: RealField,
    const S: usize,
    const Z: usize,
    const U: usize,
    const NP: usize,
> {
    q: SMatrix<T, Z, Z>,
    landmarks: FxHashMap<u32, SVector<T, S>>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
    motion_model: Box<dyn MotionModel<T, S, Z, U>>,
    pub particules: [SVector<T, S>; NP],
}

impl<T: RealField + Copy, const S: usize, const Z: usize, const U: usize, const NP: usize>
    ParticleFilterKnownCorrespondences<T, S, Z, U, NP>
where
    StandardNormal: Distribution<T>,
    Standard: Distribution<T>,
{
    pub fn new(
        initial_noise: SMatrix<T, S, S>,
        q: SMatrix<T, Z, Z>,
        landmarks: FxHashMap<u32, SVector<T, S>>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
        motion_model: Box<dyn MotionModel<T, S, Z, U>>,
        initial_state: GaussianStateStatic<T, S>,
    ) -> ParticleFilterKnownCorrespondences<T, S, Z, U, NP> {
        let mvn = MultiVariateNormal::new(&initial_state.x, &initial_noise).unwrap();
        let particules = core::array::from_fn(|_| mvn.sample());

        ParticleFilterKnownCorrespondences {
            q,
            landmarks,
            measurement_model,
            motion_model,
            particules,
        }
    }

    pub fn estimate(
        &mut self,
        control: Option<SVector<T, U>>,
        measurements: Option<Vec<(u32, SVector<T, Z>)>>,
        dt: T,
    ) {
        if let Some(u) = control {
            self.particules =
                core::array::from_fn(|i| self.motion_model.sample(&self.particules[i], &u, dt));
        }

        if let Some(measurements) = measurements {
            let mut weights = [T::one(); NP];
            let mvn = MultiVariateNormal::new(&SVector::<T, Z>::zeros(), &self.q).unwrap();

            for (id, z) in measurements
                .iter()
                .filter(|(id, _)| self.landmarks.contains_key(id))
            {
                let landmark = self.landmarks.get(id);
                for (i, particule) in self.particules.iter().enumerate() {
                    let z_pred = self.measurement_model.prediction(particule, landmark);
                    let error = z - z_pred;
                    let pdf = mvn.pdf(&error);
                    weights[i] *= pdf;
                }
            }
            //  pdf
            let norm: T = weights.iter().fold(T::zero(), |a, b| a + *b);
            let pdf = if norm != T::zero() {
                weights.map(|pdf| pdf / norm)
            } else {
                [T::one() / T::from_usize(NP).unwrap(); NP]
            };

            //cdf
            let mut total_prob = T::zero();
            let cdf: [T; NP] = core::array::from_fn(|i| {
                total_prob += (&pdf)[i];
                total_prob
            });

            // sampling
            let mut rng = rand::thread_rng();
            self.particules = core::array::from_fn(|_| {
                let rng_nb = rng.gen();
                for i in 0..NP {
                    if (&cdf)[i] > rng_nb {
                        return self.particules[i];
                    }
                }
                unreachable!()
            });
        }
    }

    pub fn gaussian_estimate(&self) -> GaussianStateStatic<T, S> {
        let x = self
            .particules
            .iter()
            .fold(SVector::<T, S>::zeros(), |a, b| a + b)
            / T::from_usize(NP).unwrap();
        let cov = self
            .particules
            .iter()
            .map(|p| p - x)
            .map(|dx| dx * dx.transpose())
            .fold(SMatrix::<T, S, S>::zeros(), |a, b| a + b)
            / T::from_usize(NP).unwrap();
        GaussianStateStatic { x, P: cov }
    }
}

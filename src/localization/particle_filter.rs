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
            self.resampling(weights);
            // self.resampling_sort(weights);
        }
    }

    fn resampling(&mut self, weights: [T; NP]) {
        let mut weight_tot = T::zero();
        let cum_weight: [T; NP] = core::array::from_fn(|i| {
            weight_tot += (&weights)[i];
            weight_tot
        });

        // sampling
        let mut rng = rand::thread_rng();
        self.particules = core::array::from_fn(|_| {
            let rng_nb = rng.gen() * weight_tot;
            for i in 0..NP {
                if (&cum_weight)[i] > rng_nb {
                    return self.particules[i];
                }
            }
            unreachable!()
        });
    }

    #[allow(dead_code)]
    fn resampling_sort(&mut self, weights: [T; NP]) {
        let total_weight: T = weights.iter().fold(T::zero(), |a, b| a + *b);
        let mut rng = rand::thread_rng();
        let mut draws: Vec<T> = (0..NP).map(|_| rng.gen() * total_weight).collect();
        draws.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mut index = 0;
        let mut cum_weight = draws[0];
        let new_particules = core::array::from_fn(|i| {
            while cum_weight < draws[i] {
                if index == NP - 1 {
                    // weird precision edge case
                    cum_weight = total_weight;
                    break;
                } else {
                    cum_weight += weights[index];
                    index += 1;
                }
            }
            self.particules[index]
        });
        self.particules = new_particules;
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

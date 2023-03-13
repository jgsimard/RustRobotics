#![allow(non_snake_case)]

use nalgebra::{RealField, SMatrix, SVector};

use crate::models::measurement::MeasurementModel;
use crate::models::motion::MotionModel;
use crate::utils::state::GaussianStateStatic;

/// S : State Size, Z: Observation Size, U: Input Size
pub struct UnscentedKalmanFilter<T: RealField, const S: usize, const Z: usize, const U: usize> {
    Q: SMatrix<T, S, S>,
    R: SMatrix<T, Z, Z>,
    gamma: T,
    observation_model: Box<dyn MeasurementModel<T, S, Z>>,
    motion_model: Box<dyn MotionModel<T, S, Z, U>>,
    mw: Vec<T>,
    cw: Vec<T>,
}

impl<T: RealField + Copy, const S: usize, const Z: usize, const U: usize>
    UnscentedKalmanFilter<T, S, Z, U>
{
    pub fn new(
        Q: SMatrix<T, S, S>,
        R: SMatrix<T, Z, Z>,
        observation_model: Box<dyn MeasurementModel<T, S, Z>>,
        motion_model: Box<dyn MotionModel<T, S, Z, U>>,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> UnscentedKalmanFilter<T, S, Z, U> {
        let (mw, cw, gamma) =
            UnscentedKalmanFilter::<T, S, Z, U>::sigma_weights(alpha, beta, kappa);
        UnscentedKalmanFilter {
            Q,
            R,
            observation_model,
            motion_model,
            gamma,
            mw,
            cw,
        }
    }
    fn sigma_weights(alpha: T, beta: T, kappa: T) -> (Vec<T>, Vec<T>, T) {
        let n = T::from_usize(S).unwrap();
        let lambda = alpha.powi(2) * (n + kappa) - n;

        let v = T::one() / ((T::one() + T::one()) * (n + lambda));
        let mut mw = vec![v; 2 * S + 1];
        let mut cw = vec![v; 2 * S + 1];

        // special cases
        let v = lambda / (n + lambda);
        mw[0] = v;
        cw[0] = v + T::one() - alpha.powi(2) + beta;

        let gamma = (n + lambda).sqrt();
        (mw, cw, gamma)
    }

    fn generate_sigma_points(&self, state: &GaussianStateStatic<T, S>) -> Vec<SVector<T, S>> {
        // use cholesky to compute the matrix square root  // cholesky(A) = L * L^T
        let sigma = state.P.cholesky().expect("unable to sqrt").l() * self.gamma;
        let mut sigma_points = vec![state.x; 2 * S + 1];
        for i in 0..S {
            let sigma_column = sigma.column(i);
            sigma_points[i + 1] += sigma_column;
            sigma_points[i + 1 + S] -= sigma_column;
        }
        sigma_points
    }

    pub fn estimate(
        &self,
        state: &GaussianStateStatic<T, S>,
        u: &SVector<T, U>,
        z: &SVector<T, Z>,
        dt: T,
    ) -> GaussianStateStatic<T, S> {
        // predict
        let sigma_points = self.generate_sigma_points(state);
        let sp_xpred: Vec<SVector<T, S>> = sigma_points
            .iter()
            .map(|x| self.motion_model.prediction(x, u, dt))
            .collect();

        let mean_xpred: SVector<T, S> = sp_xpred
            .iter()
            .zip(self.mw.iter())
            .map(|(x, w)| x * *w)
            .sum();

        let cov_xpred = sp_xpred
            .iter()
            .map(|x| x - mean_xpred)
            .zip(self.cw.iter())
            .map(|(dx, cw)| dx * dx.transpose() * *cw)
            .sum::<SMatrix<T, S, S>>()
            + self.Q;

        let prediction = GaussianStateStatic {
            x: mean_xpred,
            P: cov_xpred,
        };

        // update
        let sp_xpred = self.generate_sigma_points(&prediction);
        let sp_z: Vec<SVector<T, Z>> = sp_xpred
            .iter()
            .map(|x| self.observation_model.prediction(x, None))
            .collect();

        let mean_z: SVector<T, Z> = sp_z.iter().zip(self.mw.iter()).map(|(x, w)| x * *w).sum();

        let cov_z = sp_z
            .iter()
            .map(|x| x - mean_z)
            .zip(self.cw.iter())
            .map(|(dx, cw)| dx * dx.transpose() * *cw)
            .sum::<SMatrix<T, Z, Z>>()
            + self.R;

        let s = sp_xpred
            .iter()
            .zip(sp_z.iter().zip(self.cw.iter()))
            .map(|(x_pred, (z_point, cw))| {
                (x_pred - mean_xpred) * (z_point - mean_z).transpose() * *cw
            })
            .sum::<SMatrix<T, S, Z>>();

        let y = z - mean_z;
        let kalman_gain = s * cov_z.try_inverse().unwrap();

        let x_est = mean_xpred + kalman_gain * y;
        let p_est = cov_xpred - kalman_gain * cov_z * kalman_gain.transpose();
        GaussianStateStatic { x: x_est, P: p_est }
    }
}

#[cfg(test)]
mod tests {
    use crate::localization::unscented_kalman_filter::UnscentedKalmanFilter;
    use crate::models::measurement::SimpleProblemMeasurementModel;
    use crate::models::motion::SimpleProblemMotionModel;
    use crate::utils::deg2rad;
    use crate::utils::state::GaussianStateStatic as GaussianState;
    use nalgebra::{Matrix4, Vector2, Vector4};

    #[test]
    fn ukf_runs() {
        let dt = 0.1;

        // setup ukf
        let q = Matrix4::<f64>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
        let r = nalgebra::Matrix2::identity(); //Observation x,y position covariance
        let ukf = UnscentedKalmanFilter::<f64, 4, 2, 2>::new(
            q,
            r,
            Box::new(SimpleProblemMeasurementModel {}),
            Box::new(SimpleProblemMotionModel {}),
            0.1,
            2.0,
            0.0,
        );

        let u: Vector2<f64> = Default::default();
        let kalman_state = GaussianState {
            x: Vector4::<f64>::new(0., 0., 0., 0.),
            P: Matrix4::<f64>::identity(),
        };
        let z: Vector2<f64> = Default::default();

        ukf.estimate(&kalman_state, &u, &z, dt);
    }
}

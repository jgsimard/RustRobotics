use nalgebra::{RealField, SMatrix, SVector};
use rand::distributions::Distribution;

#[derive(Debug, Clone)]
pub enum ErrorType {
    CovarianceNotSemiDefinitePositive,
}

#[derive(Debug)]
pub struct Error {
    error_type: ErrorType,
}

impl Error {
    pub fn kind(&self) -> &ErrorType {
        &self.error_type
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.error_type)
    }
}

pub struct MultiVariateNormal<T: RealField, const D: usize> {
    mean: SVector<T, D>,
    precision: SMatrix<T, D, D>,
    lower: SMatrix<T, D, D>,
    factor: T,
}

impl<T: RealField, const D: usize> MultiVariateNormal<T, D>
where
    rand_distr::StandardNormal: Distribution<T>,
{
    // {
    pub fn new(mean: &SVector<T, D>, covariance: &SMatrix<T, D, D>) -> Result<Self, Error> {
        let Some(covariance_cholesky) = covariance.clone().cholesky() else {
            return Err(Error{error_type : ErrorType::CovarianceNotSemiDefinitePositive})
        };
        let det = covariance_cholesky.determinant();
        let precision = covariance_cholesky.inverse();
        let factor = T::one() / (T::two_pi().powi(D as i32) * det).sqrt();
        let mvn = MultiVariateNormal {
            mean: mean.clone(),
            precision,
            lower: covariance_cholesky.l(),
            factor,
        };
        Ok(mvn)
    }

    /// Probability density function
    pub fn pdf(&self, x: &SVector<T, D>) -> T {
        let dx = &self.mean - x;
        let neg_half = T::from_f32(-0.5).unwrap();
        let interior = (dx.transpose() * &self.precision * dx).x.clone();
        T::exp(neg_half * interior) * self.factor.clone()
    }

    pub fn sample(&self) -> SVector<T, D> {
        // https://juanitorduz.github.io/multivariate_normal/
        let mut rng = rand::thread_rng();
        let u = SVector::<T, D>::from_distribution(&rand_distr::StandardNormal, &mut rng);
        &self.mean + &self.lower * u
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra as na;

    #[test]
    fn test_density() {
        // parameters for a standard normal (mean=0, sigma=1)
        let mu = na::Vector2::<f64>::new(0.0, 0.0);
        let precision = na::Matrix2::<f64>::new(1.0, 0.0, 0.0, 1.0);

        let mvn = MultiVariateNormal::new(&mu, &precision).unwrap();

        let x0 = na::Vector2::<f64>::new(0.0, 0.0);
        let x1 = na::Vector2::<f64>::new(1.0, 0.0);
        let x2 = na::Vector2::<f64>::new(0.0, 1.0);

        let epsilon = 1e-5;
        // some spot checks with standard normal
        assert_relative_eq!(mvn.pdf(&x0), 0.15915494, epsilon = epsilon);
        assert_relative_eq!(mvn.pdf(&x1), 0.09653235, epsilon = epsilon);
        assert_relative_eq!(mvn.pdf(&x2), 0.09653235, epsilon = epsilon);
    }
}

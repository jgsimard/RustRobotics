use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, OMatrix, OVector, RealField, U1,
};
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

pub struct MultiVariateNormal<T: RealField, D: Dim>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    mean: OVector<T, D>,
    precision: OMatrix<T, D, D>,
    lower: OMatrix<T, D, D>,
    factor: T,
}

impl<T: RealField, D: Dim> MultiVariateNormal<T, D>
where
    rand_distr::StandardNormal: Distribution<T>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D> + Allocator<T, Const<1>, D>,
{
    pub fn new(mean: &OVector<T, D>, covariance: &OMatrix<T, D, D>) -> Result<Self, Error> {
        let Some(covariance_cholesky) = covariance.clone().cholesky() else {
            return Err(Error{error_type : ErrorType::CovarianceNotSemiDefinitePositive})
        };
        let det = covariance_cholesky.determinant();
        let precision = covariance_cholesky.inverse();
        let factor =
            T::one() / (T::two_pi().powi(mean.shape_generic().0.value() as i32) * det).sqrt();
        let mvn = MultiVariateNormal {
            mean: mean.clone(),
            precision,
            lower: covariance_cholesky.l(),
            factor,
        };
        Ok(mvn)
    }

    /// Probability density function
    pub fn pdf(&self, x: &OVector<T, D>) -> T {
        let dx = &self.mean - x;
        let neg_half = T::from_f32(-0.5).unwrap();
        let interior = (&dx.transpose() * &self.precision * dx).x.clone();
        T::exp(neg_half * interior) * self.factor.clone()
    }

    pub fn sample(&self) -> OVector<T, D> {
        // https://juanitorduz.github.io/multivariate_normal/
        let mut rng = rand::thread_rng();
        let dim = self.mean.shape_generic().0;
        let u = OVector::<T, D>::from_distribution_generic(
            dim,
            U1,
            &rand_distr::StandardNormal,
            &mut rng,
        );
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

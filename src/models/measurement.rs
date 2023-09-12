use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, Matrix2x3, Matrix2x4, OMatrix, OVector,
    RealField, Vector2, Vector3, Vector4,
};

pub trait MeasurementModel<T: RealField, S: Dim, Z: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, Z> + Allocator<T, S, S> + Allocator<T, Z, S>,
{
    fn prediction(&self, x: &OVector<T, S>, landmark: Option<&OVector<T, S>>) -> OVector<T, Z>;
    fn jacobian(&self, x: &OVector<T, S>, landmark: Option<&OVector<T, S>>) -> OMatrix<T, Z, S>;
}

/// Measurement = [range, bearing, signature]
/// Probabilistic Robotics p. 177
pub struct RangeBearingMeasurementModel;

impl RangeBearingMeasurementModel {
    pub fn new() -> Box<RangeBearingMeasurementModel> {
        Box::new(RangeBearingMeasurementModel {})
    }
}

impl MeasurementModel<f64, Const<3>, Const<2>> for RangeBearingMeasurementModel {
    fn prediction(&self, x: &Vector3<f64>, landmark: Option<&Vector3<f64>>) -> Vector2<f64> {
        //state
        let x_x = x[0];
        let x_y = x[1];
        let x_theta = x[2];
        // landmark
        let Some(lm) = landmark else {
            panic!("WRONG!!!")
        };
        let l_x = lm[0];
        let l_y = lm[1];

        let q = (l_x - x_x).powi(2) + (l_y - x_y).powi(2);
        let q_sqrt = q.sqrt();

        let range = q_sqrt;
        let bearing = f64::atan2(l_y - x_y, l_x - x_x) - x_theta;
        Vector2::new(range, bearing)
    }

    fn jacobian(&self, x: &Vector3<f64>, landmark: Option<&Vector3<f64>>) -> Matrix2x3<f64> {
        //state
        let x_x = x[0];
        let x_y = x[1];
        // landmark
        let Some(lm) = landmark else {
            panic!("WRONG x2 !!!")
        };
        let l_x = lm[0];
        let l_y = lm[1];

        let q = (l_x - x_x).powi(2) + (l_y - x_y).powi(2);
        let q_sqrt = q.sqrt();

        #[rustfmt::skip]
        let jac = Matrix2x3::<f64>::new(
            -(l_x - x_x) / q_sqrt, -(l_y - x_y) / q_sqrt,   0.,
            (l_y - x_y) / q, (l_x - x_x) / q,              -1.,
        );
        jac
    }
}

pub struct SimpleProblemMeasurementModel;

impl SimpleProblemMeasurementModel {
    pub fn new() -> Box<SimpleProblemMeasurementModel> {
        Box::new(SimpleProblemMeasurementModel {})
    }
}

impl MeasurementModel<f64, Const<4>, Const<2>> for SimpleProblemMeasurementModel {
    fn prediction(&self, x: &Vector4<f64>, _landmark: Option<&Vector4<f64>>) -> Vector2<f64> {
        x.xy()
    }

    fn jacobian(&self, _x: &Vector4<f64>, _landmark: Option<&Vector4<f64>>) -> Matrix2x4<f64> {
        #[rustfmt::skip]
        let jac = Matrix2x4::<f64>::new(
            1., 0., 0., 0.,
            0., 1., 0., 0.
        );
        jac
    }
}

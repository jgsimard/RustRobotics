use nalgebra::{Matrix2x3, Matrix2x4, RealField, SMatrix, SVector, Vector2, Vector3, Vector4};

pub trait MeasurementModel<T: RealField, const S: usize, const Z: usize> {
    fn prediction(&self, x: &SVector<T, S>, landmark: Option<&SVector<T, Z>>) -> SVector<T, Z>;
    fn jacobian(&self, x: &SVector<T, S>, landmark: Option<&SVector<T, Z>>) -> SMatrix<T, Z, S>;
}

/// Measurement = [range, bearing, signature]
/// Probabilistic Robotics p. 177
pub struct RangeBearing;

impl MeasurementModel<f32, 3, 2> for RangeBearing {
    fn prediction(&self, x: &Vector3<f32>, landmark: Option<&Vector2<f32>>) -> Vector2<f32> {
        //state
        let x_x = x[0];
        let x_y = x[1];
        // let x_theta = x[2];
        // landmark
        let l_x = landmark.unwrap()[0];
        let l_y = landmark.unwrap()[1];

        let q = (l_x - x_x).powi(2) + (l_y - x_y).powi(2);
        let q_sqrt = q.sqrt();

        let range = q_sqrt;
        let bearing = f32::atan2(l_y - x_y, l_x - x_x);
        Vector2::new(range, bearing)
    }

    #[allow(clippy::deprecated_cfg_attr)]
    fn jacobian(&self, x: &Vector3<f32>, landmark: Option<&Vector2<f32>>) -> Matrix2x3<f32> {
        //state
        let x_x = x[0];
        let x_y = x[1];
        // landmark
        let l_x = landmark.unwrap()[0];
        let l_y = landmark.unwrap()[1];

        let q = (l_x - x_x).powi(2) + (l_y - x_y).powi(2);
        let q_sqrt = q.sqrt();

        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix2x3::<f32>::new(
            -(l_x - x_x) / q_sqrt, -(l_y - x_y) / q_sqrt,   0.,
            (l_y - x_y) / q, (l_x - x_x) / q,              -1.,
        )
    }
}

pub struct SimpleProblemMeasurementModel;

impl MeasurementModel<f32, 4, 2> for SimpleProblemMeasurementModel {
    fn prediction(&self, x: &Vector4<f32>, _landmark: Option<&Vector2<f32>>) -> Vector2<f32> {
        x.xy()
    }

    #[allow(clippy::deprecated_cfg_attr)]
    fn jacobian(&self, _x: &Vector4<f32>, _landmark: Option<&Vector2<f32>>) -> Matrix2x4<f32> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix2x4::<f32>::new(
            1., 0., 0., 0.,
            0., 1., 0., 0.
        )
    }
}

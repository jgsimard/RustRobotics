// use enum_dispatch::enum_dispatch;
use nalgebra::{
    Matrix2, Matrix3, Matrix3x2, Matrix4, Matrix4x2, RealField, SMatrix, SVector, Vector2, Vector3,
    Vector4,
};

// #[enum_dispatch(MM<T, S, Z, U>)]
pub trait MotionModel<T: RealField, const S: usize, const Z: usize, const U: usize> {
    fn prediction(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SVector<T, S>;
    fn jacobian_wrt_state(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SMatrix<T, S, S>;
    fn jacobian_wrt_input(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SMatrix<T, S, U>;
    fn cov_noise_control_space(&self, u: &SVector<T, U>) -> SMatrix<T, U, U>;
}

pub struct Velocity {
    a1: f64,
    a2: f64,
    a3: f64,
    a4: f64,
}

impl Velocity {
    pub fn new(a1: f64, a2: f64, a3: f64, a4: f64) -> Velocity {
        Velocity { a1, a2, a3, a4 }
    }
}

impl MotionModel<f64, 3, 2, 2> for Velocity {
    fn prediction(&self, x: &Vector3<f64>, u: &Vector2<f64>, dt: f64) -> Vector3<f64> {
        //state
        let theta = x[2];
        //control
        let v = u[0];
        let w = u[1];
        let delta = if w != 0.0 {
            Vector3::new(
                v / w * (-theta.sin() + (theta + w * dt).sin()),
                v / w * (theta.cos() - (theta + w * dt).cos()),
                w * dt,
            )
        } else {
            // no rotation
            Vector3::new(v * theta.cos() * dt, v * theta.sin() * dt, w * dt)
        };

        let mut out = x + delta;

        // Limit theta within [-pi, pi]
        let mut theta_out = out[2];
        if theta_out > std::f64::consts::PI {
            theta_out -= 2.0 * std::f64::consts::PI;
        } else if theta_out < -std::f64::consts::PI {
            theta_out += 2.0 * std::f64::consts::PI;
        }
        out[2] = theta_out;

        out
    }

    #[allow(clippy::deprecated_cfg_attr)]
    fn jacobian_wrt_state(&self, x: &Vector3<f64>, u: &Vector2<f64>, dt: f64) -> Matrix3<f64> {
        //state
        let theta = x[2];
        //control
        let v = u[0];
        let w = u[1];
        if w != 0.0 {
            #[cfg_attr(rustfmt, rustfmt_skip)]
            Matrix3::<f64>::new(
                1., 0., v / w * (-theta.cos() + (theta + w * dt).cos()),
                0., 1., v / w * (-theta.sin() + (theta + w * dt).sin()),
                0., 0., 1.
            )
        } else {
            #[cfg_attr(rustfmt, rustfmt_skip)]
            Matrix3::<f64>::new(
                1., 0., -v * theta.sin() * dt,
                0., 1., -v * theta.cos() * dt,
                0., 0., 1.
            )
        }
    }

    fn jacobian_wrt_input(&self, x: &Vector3<f64>, u: &Vector2<f64>, dt: f64) -> Matrix3x2<f64> {
        //state
        let theta = x[2];
        //control
        let v = u[0];
        let w = x[1];

        if w != 0.0 {
            let sint = theta.sin();
            let cost = theta.cos();
            let sintdt = (theta + w * dt).sin();
            let costdt = (theta + w * dt).cos();
            let w2 = w * w;
            #[allow(clippy::deprecated_cfg_attr)]
            #[cfg_attr(rustfmt, rustfmt_skip)]
            Matrix3x2::<f64>::new(
                (-sint + sintdt) / w, v  * ((sint - sintdt) / w2 + costdt * dt / w),
                (cost - costdt) / w, v  * (-(cost - costdt) / w2 + sintdt * dt / w),
                0., dt
            )
        } else {
            #[allow(clippy::deprecated_cfg_attr)]
            #[cfg_attr(rustfmt, rustfmt_skip)]
            Matrix3x2::<f64>::new(
                theta.cos() * dt, 0.,
                theta.sin() * dt, 0.,
                0., dt
            )
        }
    }
    fn cov_noise_control_space(&self, u: &Vector2<f64>) -> Matrix2<f64> {
        let v2 = u[0].powi(2);
        let w2 = u[1].powi(2);
        #[allow(clippy::deprecated_cfg_attr)]
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix2::<f64>::new(
            self.a1 * v2 + self.a2 * w2, 0.0,
            0.0, self.a3 * v2 + self.a4 * w2,
        )
    }
}

/// motion model
///
/// x_{t+1} = x_t + v * dt * cos(yaw)
///
/// y_{t+1} = y_t + v * dt * sin(yaw)
///
/// yaw_{t+1} = yaw_t + omega * dt
///
/// v_{t+1} = v{t}
///
/// so
///
/// dx/dyaw = -v * dt * sin(yaw)
///
/// dx/dv = dt * cos(yaw)
///
/// dy/dyaw = v * dt * cos(yaw)
///
/// dy/dv = dt * sin(yaw)
pub struct SimpleProblemMotionModel {}

impl MotionModel<f64, 4, 2, 2> for SimpleProblemMotionModel {
    fn prediction(&self, x: &Vector4<f64>, u: &Vector2<f64>, dt: f64) -> Vector4<f64> {
        let yaw = x[2];
        let v = x[3];
        Vector4::new(
            x.x + yaw.cos() * v * dt,
            x.y + yaw.sin() * v * dt,
            yaw + u.y * dt,
            u.x,
        )
    }

    #[allow(clippy::deprecated_cfg_attr)]
    fn jacobian_wrt_state(&self, x: &Vector4<f64>, u: &Vector2<f64>, dt: f64) -> Matrix4<f64> {
        let yaw = x[2];
        let v = u[0];
        #[cfg_attr(rustfmt, rustfmt_skip)]
            Matrix4::<f64>::new(
                1., 0., -dt * v * (yaw).sin(), dt * (yaw).cos(),
                0., 1., dt * v * (yaw).cos(), dt * (yaw).sin(),
                0., 0., 1., 0.,
                0., 0., 0., 0.,
            )
    }
    fn jacobian_wrt_input(&self, _x: &Vector4<f64>, _u: &Vector2<f64>, _dt: f64) -> Matrix4x2<f64> {
        unimplemented!()
    }
    fn cov_noise_control_space(&self, _u: &SVector<f64, 2>) -> SMatrix<f64, 2, 2> {
        unimplemented!()
    }
}

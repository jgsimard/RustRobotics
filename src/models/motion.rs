// use enum_dispatch::enum_dispatch;
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, Matrix2, Matrix3, Matrix3x2, Matrix4,
    Matrix4x2, OMatrix, OVector, RealField, Vector2, Vector3, Vector4,
};

use rand_distr::{Distribution, Normal};

// #[enum_dispatch(MM<T, S, Z, U>)]
pub trait MotionModel<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, S, S>
        + Allocator<T, U, U>
        + Allocator<T, S, U>
        + Allocator<T, Z, S>,
{
    fn prediction(&self, x: &OVector<T, S>, u: &OVector<T, U>, dt: T) -> OVector<T, S>;
    fn jacobian_wrt_state(&self, x: &OVector<T, S>, u: &OVector<T, U>, dt: T) -> OMatrix<T, S, S>;
    fn jacobian_wrt_input(&self, x: &OVector<T, S>, u: &OVector<T, U>, dt: T) -> OMatrix<T, S, U>;
    fn cov_noise_control_space(&self, u: &OVector<T, U>) -> OMatrix<T, U, U>;
    fn sample(&self, x: &OVector<T, S>, u: &OVector<T, U>, dt: T) -> OVector<T, S>;
}

pub struct Velocity {
    a1: f64,
    a2: f64,
    a3: f64,
    a4: f64,
    a5: f64,
    a6: f64,
}

impl Velocity {
    pub fn new(a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64) -> Box<Velocity> {
        Box::new(Velocity {
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
        })
    }
}

impl MotionModel<f64, Const<3>, Const<2>, Const<2>> for Velocity {
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
        let eps = 0.00001;
        #[allow(clippy::deprecated_cfg_attr)]
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix2::<f64>::new(
            self.a1 * v2 + self.a2 * w2 + eps, 0.0,
            0.0, self.a3 * v2 + self.a4 * w2 + eps,
        )
    }

    fn sample(&self, x: &Vector3<f64>, u: &Vector2<f64>, dt: f64) -> Vector3<f64> {
        //state
        let theta = x[2];
        //control
        let v = u[0];
        let w = u[1];

        let v2 = v.powi(2);
        let w2 = w.powi(2);
        let eps = 0.00001;
        let mut rng = rand::thread_rng();
        let v_noisy = Normal::new(v, (self.a1 * v2 + self.a2 * w2 + eps).sqrt())
            .unwrap()
            .sample(&mut rng);
        let w_noisy = Normal::new(w, (self.a3 * v2 + self.a4 * w2 + eps).sqrt())
            .unwrap()
            .sample(&mut rng);
        let gamma_noisy = Normal::new(0.0, (self.a5 * v2 + self.a6 * w2).sqrt())
            .unwrap()
            .sample(&mut rng);

        let delta = Vector3::new(
            v_noisy / w_noisy * (-theta.sin() + (theta + w_noisy * dt).sin()),
            v_noisy / w_noisy * (theta.cos() - (theta + w_noisy * dt).cos()),
            w_noisy * dt + gamma_noisy * dt,
        );

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

impl SimpleProblemMotionModel {
    pub fn new() -> Box<SimpleProblemMotionModel> {
        Box::new(SimpleProblemMotionModel {})
    }
}

impl MotionModel<f64, Const<4>, Const<2>, Const<2>> for SimpleProblemMotionModel {
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
    fn cov_noise_control_space(&self, _u: &Vector2<f64>) -> Matrix2<f64> {
        unimplemented!()
    }
    fn sample(&self, _x: &Vector4<f64>, _u: &Vector2<f64>, _dt: f64) -> Vector4<f64> {
        unimplemented!()
    }
}

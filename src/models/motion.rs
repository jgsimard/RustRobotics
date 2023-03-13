// use enum_dispatch::enum_dispatch;
use nalgebra::{
    Matrix3, Matrix3x2, Matrix4, Matrix4x2, RealField, SMatrix, SVector, Vector2, Vector3, Vector4,
};

// #[enum_dispatch(MM<T, S, Z, U>)]
pub trait MotionModel<T: RealField, const S: usize, const Z: usize, const U: usize> {
    fn prediction(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SVector<T, S>;
    fn jacobian_wrt_state(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SMatrix<T, S, S>;
    fn jacobian_wrt_input(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SMatrix<T, S, U>;
}

pub struct Velocity {}

impl MotionModel<f64, 3, 2, 2> for Velocity {
    fn prediction(&self, x: &Vector3<f64>, u: &Vector2<f64>, dt: f64) -> Vector3<f64> {
        //state
        let theta = x[2];
        //control
        let v = u[0];
        let w = x[1];
        let delta = Vector3::new(
            v / w * (-theta.sin() + (theta + w * dt).sin()),
            v / w * (theta.cos() - (theta + w * dt).cos()),
            w * dt,
        );
        x + delta
    }

    #[allow(clippy::deprecated_cfg_attr)]
    fn jacobian_wrt_state(&self, x: &Vector3<f64>, u: &Vector2<f64>, dt: f64) -> Matrix3<f64> {
        //state
        let theta = x[2];
        //control
        let v = u[0];
        let w = u[1];
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix3::<f64>::new(
            1., 0., v / w * (-theta.cos() + (theta + w * dt).cos()),
            0., 1., v / w * (-theta.sin() + (theta + w * dt).sin()),
            0., 0., 1.
        )
    }

    #[allow(clippy::deprecated_cfg_attr)]
    fn jacobian_wrt_input(&self, x: &Vector3<f64>, u: &Vector2<f64>, dt: f64) -> Matrix3x2<f64> {
        //state
        let theta = x[2];
        //control
        let v = u[0];
        let w = x[1];

        let sint = theta.sin();
        let cost = theta.cos();
        let sintdt = (theta + w * dt).sin();
        let costdt = (theta + w * dt).cos();
        let w2 = w * w;
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix3x2::<f64>::new(
            (-sint + sintdt) / w, v  * ((sint - sintdt) / w2 + costdt * dt / w),
            (cost - costdt) / w, v  * (-(cost - costdt) / w2 + sintdt * dt / w),
            0., 1.
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
}

// impl<T: RealField, const S: usize, const Z: usize, const U: usize> MotionModel<T, S, Z, U> for Vec<T> {
//     fn jacobian_wrt_input(&self,x: &SVector<T,S>,u: &SVector<T,U>,dt:T) -> SMatrix<T,S,U> {
//         unimplemented!()
//     }
//     fn jacobian_wrt_state(&self,x: &SVector<T,S>,u: &SVector<T,U>,dt:T) -> SMatrix<T,S,S> {
//         unimplemented!()
//     }
//     fn prediction(&self,x: &SVector<T,S>,u: &SVector<T,U>,dt:T) -> SVector<T,S> {
//         unimplemented!()
//     }
// }

// // I am unable to make enum_dispatch work. need help
// #[enum_dispatch]
// pub enum MM<T: RealField, const S: usize, const Z: usize, const U: usize> {
//     Velocity(Velocity),
//     SimpleProblemMotionModel(SimpleProblemMotionModel),
// }

// impl <T: RealField, const S: usize, const Z: usize, const U: usize> MotionModel<T, S, Z, U> for MM<T, S, Z, U> {
//     fn prediction(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SVector<T, S> {
//         match self {
//             MM::SimpleProblemMotionModel(mm) => mm.prediction(x, u, dt),
//             MM::Velocity(mm) => mm.prediction(x, u, dt)
//         }
//     }
//     fn jacobian_wrt_input(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SMatrix<T, S, U> {
//         match self {
//             MM::SimpleProblemMotionModel(mm) => mm.jacobian_wrt_input(x, u, dt),
//             MM::Velocity(mm) => mm.jacobian_wrt_input(x, u, dt)
//         }
//     }
//     fn jacobian_wrt_state(&self, x: &SVector<T, S>, u: &SVector<T, U>, dt: T) -> SMatrix<T, S, S> {
//         match self {
//             MM::SimpleProblemMotionModel(mm) => mm.jacobian_wrt_state(x, u, dt),
//             MM::Velocity(mm) => mm.jacobian_wrt_state(x, u, dt)
//         }
//     }
// // }

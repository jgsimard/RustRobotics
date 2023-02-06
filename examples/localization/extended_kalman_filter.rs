// // Extended kalman filter (EKF) localization sample
// // author: Atsushi Sakai (@Atsushi_twi)
// //         Jean-Gabriel Simard (@jgsimard)

extern crate nalgebra;
use nalgebra::Vector2;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::PointStyle;
use plotlib::view::ContinuousView;
use rand_distr::{Distribution, Normal};

extern crate robotics;
use robotics::localization::extended_kalman_filter::ExtendedKalmanFilter;
use robotics::utils::{deg2rad, Matrix, Vector};

struct SimpleProblem {
    // state
    // [x, y, yaw, v]
    // motion model
    // x_{t+1} = x_t+v*dt*cos(yaw)
    // y_{t+1} = y_t+v*dt*sin(yaw)
    // yaw_{t+1} = yaw_t+omega*dt
    // v_{t+1} = v{t}
    // so
    // dx/dyaw = -v*dt*sin(yaw)
    // dx/dv = dt*cos(yaw)
    // dy/dyaw = v*dt*cos(yaw)
    // dy/dv = dt*sin(yaw)
    pub gps_noise: Matrix<2, 2>,
    pub input_noise: Matrix<2, 2>,
    pub q: Matrix<4, 4>,
    pub r: Matrix<2, 2>,
}

impl ExtendedKalmanFilter<4, 2, 2> for SimpleProblem {
    fn f(&self, _x: Vector<4>, _dt: f32) -> Matrix<4, 4> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix::<4,4>::new(
            1., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        )
    }

    fn b(&self, x: Vector<4>, dt: f32) -> Matrix<4, 2> {
        let yaw = x[2];

        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix::<4, 2>::new(
            dt * (yaw).cos(), 0.,
            dt * (yaw).sin(), 0.,
            0., dt,
            1., 0.
        )
    }

    fn h(&self) -> Matrix<2, 4> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix::<2,4>::new(
            1., 0., 0., 0.,
            0., 1., 0., 0.
        )
    }

    fn jacob_f(&self, x: Vector<4>, u: Vector<2>, dt: f32) -> Matrix<4, 4> {
        let yaw = x[2];
        let v = u[0];
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix::<4,4>::new(
            1., 0., -dt * v * (yaw).sin(), dt * (yaw).cos(),
            0., 1., dt * v * (yaw).cos(), dt * (yaw).sin(),
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        )
    }

    fn jacob_h(&self) -> Matrix<2, 4> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix::<2,4>::new(
            1., 0., 0., 0.,
            0., 1., 0., 0.
        )
    }

    fn q(&self) -> Matrix<4, 4> {
        self.q
    }

    fn r(&self) -> Matrix<2, 2> {
        self.r
    }
}

fn observation(
    x_true: Vector<4>,
    xd: Vector<4>,
    u: Vector<2>,
    dt: f32,
    problem: &SimpleProblem,
) -> (Vector<4>, Vector<2>, Vector<4>, Vector<2>) {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0., 1.).unwrap();

    let x_true_next = problem.motion_model(x_true, u, dt);

    // let m = nalgebra::base::Matrix::from_distribution_generic(Const::<2>, Const::<1>, &normal, &mut rng);

    // add noise to gps x-y
    let z_noise =
        problem.gps_noise * Vector2::new(normal.sample(&mut rng), normal.sample(&mut rng));
    let z = problem.observation_model(x_true_next) + z_noise;

    // add noise to input
    let u_noise =
        problem.input_noise * Vector2::new(normal.sample(&mut rng), normal.sample(&mut rng));
    let ud = u + u_noise;

    let xd_next = problem.motion_model(xd, ud, dt);

    (x_true_next, z, xd_next, ud)
}

fn main() {
    let sim_time = 50.0;
    let dt = 0.1;
    let mut time = 0.;

    let mut q = Matrix::<4, 4>::identity();
    q[(0, 0)] = 0.1; // variance of location on x-axis
    q[(1, 1)] = deg2rad(1.0); // variance of location on y-axis
    q[(2, 2)] = 0.1; // variance of yaw angle
    q[(3, 3)] = 1.0; // variance of velocity
    q = q * q; // predict state covariance

    let r = nalgebra::Matrix2::identity(); //Observation x,y position covariance

    let simple_problem = SimpleProblem {
        input_noise: nalgebra::Matrix2::new(1., 0., 0., deg2rad(30.0).powi(2)),
        gps_noise: nalgebra::Matrix2::new(1.0, 0., 0., 1.0),
        q: q,
        r: r,
    };

    let u = Vector::<2>::new(1.0, 0.1);
    let mut ud: Vector<2>;
    let mut x_dr = Vector::<4>::new(0., 0., 0., 0.);
    let mut x_true = Vector::<4>::new(0., 0., 0., 0.);
    let mut x_est = Vector::<4>::new(0., 0., 0., 0.);
    let mut p_est = Matrix::<4, 4>::identity();
    let mut z: Vector<2>;

    let mut history_z = Vec::new();
    let mut history_x_true = Vec::new();
    let mut history_x_dr = Vec::new();
    let mut history_x_est = Vec::new();

    while time < sim_time {
        time += dt;
        (x_true, z, x_dr, ud) = observation(x_true, x_dr, u, dt, &simple_problem);
        (x_est, p_est) = simple_problem.estimation(x_est, p_est, z, ud, dt);

        // record step
        history_z.push((z[0] as f64, z[1] as f64));
        history_x_true.push((x_true[0] as f64, x_true[1] as f64));
        history_x_dr.push((x_dr[0] as f64, x_dr[1] as f64));
        history_x_est.push((x_est[0] as f64, x_est[1] as f64));
    }

    // PLOT
    let s0: Plot = Plot::new(history_z).point_style(PointStyle::new().colour("#DD3355").size(3.)); // RED
    let s1: Plot =
        Plot::new(history_x_true).point_style(PointStyle::new().colour("#0000ff").size(3.)); // blue
    let s2: Plot =
        Plot::new(history_x_dr).point_style(PointStyle::new().colour("#FFFF00").size(3.)); // yellow
    let s3: Plot =
        Plot::new(history_x_est).point_style(PointStyle::new().colour("#35C788").size(3.)); // green

    let v = ContinuousView::new()
        .add(s0)
        .add(s1)
        .add(s2)
        .add(s3)
        .x_range(-15., 15.)
        .y_range(-5., 25.)
        .x_label("x")
        .y_label("y");

    Page::single(&v).save("./img/ekf.svg").unwrap();
}

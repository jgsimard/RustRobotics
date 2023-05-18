use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Matrix1, Matrix4, Matrix4x1, Vector4};
use plotpy::{Curve, Plot};
use std::error::Error;

extern crate robotics;
use robotics::control::lqr::{lqr, LinearModel};

/// [x, x_dot, theta, thetha_dot]
struct InvertedPendulumModel {
    da: Matrix4<f64>,
    db: Matrix4x1<f64>,
    r: Matrix1<f64>,
    q: Matrix4<f64>,
}

impl InvertedPendulumModel {
    fn new(l_bar: f64, mass_cart: f64, mass_ball: f64, g: f64) -> InvertedPendulumModel {
        let q = Matrix4::from_diagonal(&Vector4::new(1.0, 1.0, 1.0, 1.0));
        let r = Matrix1::new(0.01);

        #[rustfmt::skip]
        let da = Matrix4::new(
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, mass_ball * g / mass_cart, 0.0,
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, g * (mass_cart + mass_ball) / (l_bar * mass_cart), 0.0
        );

        let db = Matrix4x1::new(0.0, 1.0 / mass_cart, 0.0, 1.0 / (l_bar * mass_cart));

        InvertedPendulumModel { da, db, r, q }
    }
}

impl<'a> LinearModel<'a, f64, Const<4>, Const<1>> for InvertedPendulumModel {
    fn a(&self, dt: f64) -> Matrix4<f64> {
        Matrix4::identity() + dt * self.da
    }
    fn b(&self, dt: f64) -> Matrix4x1<f64> {
        dt * self.db
    }
    fn q(&'a self) -> &'a Matrix4<f64> {
        &self.q
    }
    fn r(&'a self) -> &'a Matrix1<f64> {
        &self.r
    }
}

fn run() -> Vec<Vector4<f64>> {
    let sim_time = 5.0;
    let dt = 0.1;
    let mut time = 0.;
    let max_iter = 100;
    let epsilon = 0.01;

    let l_bar = 2.0; // length of bar
    let mass_cart = 1.0; // [kg]
    let mass_ball = 0.3; // [kg]
    let g = 9.8; // [m/s^2]

    let linear_model = InvertedPendulumModel::new(l_bar, mass_cart, mass_ball, g);

    let mut x = Vector4::new(0.0, 0.0, -0.5, 0.0);
    let x_goal = Vector4::new(-2.0, 0.0, 0.0, 0.0);

    let mut history = vec![x];

    while time < sim_time {
        time += dt;

        let u = lqr(&(&x - &x_goal), dt, &linear_model, max_iter, epsilon);

        x = linear_model.a(dt) * x + linear_model.b(dt) * u;

        history.push(x.clone());
    }
    history
}

fn main() -> Result<(), Box<dyn Error>> {
    let algos = &[
        "Linear Quadratic Regulator (LQR)",
        "Model Predictive Control (MPC)",
    ];
    let algo_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Pick control algorithm")
        .default(0)
        .items(&algos[..])
        .interact()
        .unwrap();
    // let algo = algos[algo_idx];

    // get data
    let history = run();

    // Create output directory if it didnt exist
    std::fs::create_dir_all("./img")?;

    let path = match algo_idx {
        0 => "lqr",
        1 => "mpc",
        _ => unreachable!(),
    };

    let mut state_x = Curve::new();
    state_x.set_label("x").points_begin();

    let mut state_theta = Curve::new();
    state_theta.set_label("theta").points_begin();

    for (i, x) in history.iter().enumerate() {
        state_x.points_add(i as f64, x.x);
        state_theta.points_add(i as f64, x.z);
    }
    state_x.points_end();
    state_theta.points_end();

    // add features to plot
    let mut plot = Plot::new();

    plot.add(&state_x)
        .add(&state_theta)
        .legend()
        .set_equal_axes(true) // save figure
        .set_figure_size_points(1000.0, 1000.0)
        .save(format!("img/{}-{}.svg", path, "done").as_str())?;
    Ok(())
}

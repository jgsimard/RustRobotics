use nalgebra::{Matrix2, Matrix3, Vector2};
use plotters::coord::Shift;
use plotters::prelude::*;
use std::error::Error;

extern crate robotics;
use robotics::utils::state::GaussianStateStatic as GaussianState;

pub fn ellipse_series(xy: Vector2<f64>, p_xy: Matrix2<f64>) -> Option<Vec<(f64, f64)>> {
    let eigen = p_xy.symmetric_eigen();
    let eigenvectors = eigen.eigenvectors;
    let eigenvalues = eigen.eigenvalues;

    let (a, b, angle) = if eigenvalues.x >= eigenvalues.y {
        (
            eigenvalues.x.sqrt(),
            eigenvalues.y.sqrt(),
            f64::atan2(eigenvectors.m21, eigenvectors.m22),
        )
    } else {
        (
            eigenvalues.y.sqrt(),
            eigenvalues.x.sqrt(),
            f64::atan2(eigenvectors.m22, eigenvectors.m21),
        )
    };

    let rot_mat = Matrix3::new_rotation(angle)
        .fixed_view::<2, 2>(0, 0)
        .clone_owned();

    let ellipse_points = (0..100)
        .map(|x| x as f64 / 100.0 * std::f64::consts::TAU) // map [0..100] -> [0..2pi]
        .map(|t| rot_mat * Vector2::new(a * t.cos(), b * t.sin()) + xy) //map [0..2pi] -> elipse points
        .map(|xy| (xy.x as f64, xy.y as f64))
        .collect();
    Some(ellipse_points)
}

#[derive(Debug, Default)]
pub struct History {
    pub z: Vec<(f64, f64)>,
    pub x_true: Vec<(f64, f64)>,
    pub x_dr: Vec<(f64, f64)>,
    pub x_est: Vec<(f64, f64)>,
    // pub gaussian_state: Vec<GaussianState<f64, Const<4>>>,
    pub gaussian_state: Vec<GaussianState<f64, 4>>,
}

pub fn chart(
    root: &DrawingArea<BitMapBackend, Shift>,
    history: &History,
    i: usize,
    name: &str,
) -> Result<(), Box<dyn Error>> {
    let min_x = history
        .z
        .iter()
        .take(i)
        .map(|(x, _y)| *x)
        .reduce(f64::min)
        .unwrap_or(-1.0)
        - 1.0;
    let min_y = history
        .z
        .iter()
        .take(i)
        .map(|(_x, y)| *y)
        .reduce(f64::min)
        .unwrap_or(-1.0)
        - 1.0;
    let max_x = history
        .z
        .iter()
        .take(i)
        .map(|(x, _y)| *x)
        .reduce(f64::max)
        .unwrap_or(1.0)
        + 1.0;
    let max_y = history
        .z
        .iter()
        .take(i)
        .map(|(_x, y)| *y)
        .reduce(f64::max)
        .unwrap_or(1.0)
        + 1.0;
    // find chart dimensions
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(name, ("sans-serif", 40))
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(
            history
                .z
                .iter()
                .take(i)
                .map(|(x, y)| Circle::new((*x, *y), 3, RED.filled())),
        )?
        .label("observation")
        .legend(|(x, y)| Circle::new((x, y), 3, RED.filled()));
    chart
        .draw_series(
            history
                .x_true
                .iter()
                .take(i)
                .map(|(x, y)| Circle::new((*x, *y), 3, BLUE.filled())),
        )?
        .label("true position")
        .legend(|(x, y)| Circle::new((x, y), 3, BLUE.filled()));
    chart
        .draw_series(
            history
                .x_dr
                .iter()
                .take(i)
                .map(|(x, y)| Circle::new((*x, *y), 3, YELLOW.filled())),
        )?
        .label("Dead reckoning")
        .legend(|(x, y)| Circle::new((x, y), 3, YELLOW.filled()));
    chart
        .draw_series(
            history
                .x_est
                .iter()
                .take(i)
                .map(|(x, y)| Circle::new((*x, *y), 3, GREEN.filled())),
        )?
        .label("kalman estimate")
        .legend(|(x, y)| Circle::new((x, y), 3, GREEN.filled()));

    // Draw ellipse
    let state = history.gaussian_state.get(i).unwrap();
    let p_xy = state.P.fixed_view::<2, 2>(0, 0).clone_owned();
    let xy = state.x.fixed_view::<2, 1>(0, 0).clone_owned();

    chart.draw_series(std::iter::once(Polygon::new(
        ellipse_series(xy, p_xy).unwrap(),
        &GREEN.mix(0.4),
    )))?;
    chart.draw_series(std::iter::once(PathElement::new(
        ellipse_series(xy, p_xy).unwrap(),
        &GREEN,
    )))?;

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .border_style(BLACK)
        .draw()?;
    Ok(())
}

// to make the compiler happy
#[allow(dead_code)]
fn main() {}

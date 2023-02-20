use nalgebra::{Matrix2, Matrix3, Vector2};

pub fn ellipse_series(xy: Vector2<f32>, p_xy: Matrix2<f32>) -> Option<Vec<(f64, f64)>> {
    let eigen = p_xy.symmetric_eigen();
    let eigenvectors = eigen.eigenvectors;
    let eigenvalues = eigen.eigenvalues;

    let (a, b, angle) = if eigenvalues.x >= eigenvalues.y {
        (
            eigenvalues.x.sqrt(),
            eigenvalues.y.sqrt(),
            f32::atan2(eigenvectors.m21, eigenvectors.m22),
        )
    } else {
        (
            eigenvalues.y.sqrt(),
            eigenvalues.x.sqrt(),
            f32::atan2(eigenvectors.m22, eigenvectors.m21),
        )
    };

    let rot_mat = Matrix3::new_rotation(angle)
        .fixed_view::<2, 2>(0, 0)
        .clone_owned();

    let ellipse_points = (0..100)
        .map(|x| x as f32 / 100.0 * std::f32::consts::TAU) // map [0..100] -> [0..2pi]
        .map(|t| rot_mat * Vector2::new(a * t.cos(), b * t.sin()) + xy) //map [0..2pi] -> elipse points
        .map(|xy| (xy.x as f64, xy.y as f64))
        .collect();
    Some(ellipse_points)
}

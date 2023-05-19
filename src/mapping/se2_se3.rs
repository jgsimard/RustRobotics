// rust traduction of code in https://github.com/RainerKuemmerle/g2o/blob/master/g2o/types/slam3d/isometry3d_gradients.h

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use nalgebra::{Matrix3, SMatrix, Vector3};

// pub trait Manifold{
//     // fn box_plus(other: &self) -> self;
//     fn jacobian_boxplus(&self);
// }

// impl Manifold for Isometry3<f64> {
//     fn jacobian_boxplus(&self) {
//         let [x, y, z] = self.translation.vector.as_slice() else {unreachable!()};
//         let [qx, qy, qz, qw] =self.rotation.coords.as_slice() else {unreachable!()};

//         #[rustfmt::skip]
//         let j = [
//             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
//         ];
//     }
// }

/// Source : A tutorial on $\mathbf{SE}(3)$ transformation parameterizations and on-manifold optimization.
///
/// Equation : Jacobian of the SO(3) logarithm.
///
/// No the most numerically stable, might be better to use g2o code
pub fn jacobian_so3(m: &Matrix3<f64>) -> SMatrix<f64, 3, 9> {
    let trace = m.trace();
    let cos = (trace - 1.0).sqrt() * 0.5;
    let mut a1 = 0.0;
    let mut a2 = 0.0;
    let mut a3 = 0.0;
    let mut b = 0.5;

    if cos < 0.9999999 {
        let sin = (1.0 - cos * cos).sqrt();
        let theta = f64::atan2(sin, cos);
        let factor = (theta * cos - sin) / (4.0 * sin * sin * sin);
        a1 = (m.m32 - m.m23) * factor;
        a2 = (m.m13 - m.m31) * factor;
        a3 = (m.m21 - m.m12) * factor;
        b = 0.5 * theta / sin;
    }

    #[rustfmt::skip]
    let res = SMatrix::<f64, 3, 9>::from_column_slice(&[ 
        // transpose of actual matrix
        a1,  a2,  a3,  
        0.0, 0.0,   b,
        0.0,  -b, 0.0,
        0.0, 0.0,  -b,
        a1,  a2,  a3,
        b, 0.0, 0.0,
        0.0,   b, 0.0,
        -b, 0.0, 0.0,
        a1,  a2,  a3
    ]);
    res
}

pub fn skew(t: &Vector3<f64>) -> Matrix3<f64> {
    #[rustfmt::skip]
    let res = Matrix3::new(
        0.0, -t.z,  t.y,
        t.z,  0.0, -t.x,
        -t.y, t.x,  0.0
    );
    res
}

pub fn skew_m_and_mult_parts(m: &Matrix3<f64>, mult: &Matrix3<f64>) -> SMatrix<f64, 9, 3> {
    let top = mult * skew(&m.column(0).clone_owned());
    let mid = mult * skew(&m.column(1).clone_owned());
    let bot = mult * skew(&m.column(2).clone_owned());
    let mut ret = SMatrix::<f64, 9, 3>::zeros();
    ret.index_mut((0..3, ..)).copy_from(&top);
    ret.index_mut((3..6, ..)).copy_from(&mid);
    ret.index_mut((6..9, ..)).copy_from(&bot);
    ret
}

// enum Quat{
//     w, x, y, z
// }

// fn q2m(m: &Matrix3<f64>){
//     let trace = m.trace();

// }

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn compute_dq_dR_correct() {
    //     #[rustfmt::skip]
    //     let err_rot_matrix = Matrix3::new(
    //         1.0,          -7.7835e-07,  5.74141e-08,    // transposed matrix is displayed
    //         7.99504e-07,  1.0,          1.14998e-07,
    //         -7.15592e-08, -1.46825e-07, 1.0
    //     );
    //     let actual = jacobian_so3(&err_rot_matrix);
    //     let a1 = -8.18195e-09;
    //     let a2 = 4.03041e-09;
    //     let a3 = 4.93079e-08;
    //     let b = 0.25;

    //     #[rustfmt::skip]
    //     let expected = SMatrix::<f64, 3, 9>::from_vec(vec![
    //         a1,  a2,  a3,    // transposed matrix is displayed
    //         0.0, 0.0,   b,
    //         0.0,  -b, 0.0,
    //         0.0, 0.0,  -b,
    //         a1,  a2,  a3,
    //         b, 0.0, 0.0,
    //         0.0,   b, 0.0,
    //         -b, 0.0, 0.0,
    //         a1,  a2,  a3
    //     ]);

    //     println!("{}", 0.5 * actual);
    //     println!("{}", expected);

    //     approx::assert_abs_diff_eq!(0.5 * actual, expected, epsilon = 1e-5);
    // }

    #[test]
    fn skew_correct() {
        #[rustfmt::skip]
        let t = Vector3::new(-0.0199389, 2.43871, -0.14102);

        #[rustfmt::skip]
        let expected = Matrix3::new(
            0.0,  -0.282041,   -4.87743,
            0.282041,        0.0, -0.0398779,
             4.87743,  0.0398779,        0.0
        );

        approx::assert_abs_diff_eq!(skew(&(2.0 * t)).transpose(), expected, epsilon = 1e-2);
    }
}

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, RealField};

pub struct LinearTimeInvariantModel<T: RealField, S: Dim, U: Dim>
where
    DefaultAllocator:
        Allocator<T, S, S> + Allocator<T, S, U> + Allocator<T, U, S> + Allocator<T, U, U>,
{
    pub A: OMatrix<T, S, S>,
    pub B: OMatrix<T, S, U>,
    pub R: OMatrix<T, U, U>,
    pub Q: OMatrix<T, S, S>,
}

pub fn lqr<T: RealField + Copy, S: Dim, U: Dim>(
    linear_model: &LinearTimeInvariantModel<T, S, U>,
    max_iter: usize,
    epsilon: T,
) -> Option<OMatrix<T, U, S>>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, S, S>
        + Allocator<T, S, U>
        + Allocator<T, U, S>
        + Allocator<T, U, U>,
{
    let A = &linear_model.A;
    let A_T = &A.transpose();
    let B = &linear_model.B;
    let B_T = &B.transpose();
    let Q = &linear_model.Q;
    let R = &linear_model.R;

    // Discrete time Algebraic Riccati Equation (DARE)
    let mut P = linear_model.Q.clone_owned();
    for _ in 0..max_iter {
        let Pn = A_T * &P * A - A_T * &P * B * (R + B_T * &P * B).try_inverse()? * B_T * &P * A + Q;
        if (&Pn - &P).abs().max() < epsilon {
            // println!("done {i}");
            break;
        }
        P = Pn;
    }
    // LQR gain
    let k = (R + B_T * &P * B).try_inverse()? * B_T * &P * A;
    //
    // // LQR control
    Some(k)
}

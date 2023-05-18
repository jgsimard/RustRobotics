use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, RealField};

pub trait LinearModel<'a, T: RealField, S: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S, S> + Allocator<T, S, U> + Allocator<T, U, U>,
{
    fn a(&'a self, dt: T) -> OMatrix<T, S, S>;
    fn b(&'a self, dt: T) -> OMatrix<T, S, U>;
    fn r(&'a self) -> &'a OMatrix<T, U, U>;
    fn q(&'a self) -> &'a OMatrix<T, S, S>;
}

pub fn lqr<'a, T: RealField + Copy, S: Dim, U: Dim>(
    x: &OVector<T, S>,
    dt: T,
    linear_model: &'a impl LinearModel<'a, T, S, U>,
    max_iter: usize,
    epsilon: T,
) -> OVector<T, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, S, S>
        + Allocator<T, S, U>
        + Allocator<T, U, S>
        + Allocator<T, U, U>,
{
    let a = linear_model.a(dt);
    let at = a.transpose();
    let b = linear_model.b(dt);
    let bt = b.transpose();
    let q = linear_model.q();
    let r = linear_model.r();

    // Discrete time Algebraic Riccati Equation (DARE)
    let mut p = linear_model.q().clone();
    for _ in 0..max_iter {
        let pn = &at * &p * &a
            - &at * &p * &b * (r + &bt * &p * &b).try_inverse().unwrap() * &bt * &p * &a
            + q;
        if (&pn - &p).abs().max() < epsilon {
            break;
        }
        p = pn;
    }
    // LQR gain
    let k = (r + &bt * &p * b).try_inverse().unwrap() * bt * &p * a;

    // LQR control
    -k * x
}

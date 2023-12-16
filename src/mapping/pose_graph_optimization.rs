#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)] // TODO: remove this

use nalgebra::{
    AbstractRotation, DVector, Isometry, Isometry2, Isometry3, Matrix2, Matrix2x3, Matrix3,
    Matrix6, SMatrix, SVector, UnitComplex, Vector2, Vector3,
};
use plotpy::{Curve, Plot};
use rayon::prelude::*;
use russell_lab::Vector;
use russell_sparse::prelude::*;
use rustc_hash::FxHashMap;
use std::error::Error;

use crate::mapping::g2o::parse_g2o;
use crate::mapping::se2_se3::{jacobian_so3, skew, skew_m_and_mult_parts};

#[derive(Debug)]
pub enum Edge<T> {
    SE2_SE2(EdgeSE2<T>),
    SE2_XY(EdgeSE2_XY<T>),
    SE3_SE3(EdgeSE3<T>),
    SE3_XYZ,
}

#[derive(PartialEq, Debug)]
pub enum PoseGraphSolver {
    GaussNewton,
    LevenbergMarquardt,
}

#[derive(Debug)]
pub struct EdgeSE2<T> {
    from: u32,
    to: u32,
    measurement: Isometry2<T>,
    information: Matrix3<T>,
}

impl<T> EdgeSE2<T> {
    pub fn new(
        from: u32,
        to: u32,
        measurement: Isometry2<T>,
        information: Matrix3<T>,
    ) -> EdgeSE2<T> {
        EdgeSE2 {
            from,
            to,
            measurement,
            information,
        }
    }
}

#[derive(Debug)]
pub struct EdgeSE2_XY<T> {
    from: u32,
    to: u32,
    measurement: Vector2<T>,
    information: Matrix2<T>,
}

impl<T> EdgeSE2_XY<T> {
    pub fn new(
        from: u32,
        to: u32,
        measurement: Vector2<T>,
        information: Matrix2<T>,
    ) -> EdgeSE2_XY<T> {
        EdgeSE2_XY {
            from,
            to,
            measurement,
            information,
        }
    }
}

#[derive(Debug)]
pub struct EdgeSE3<T> {
    from: u32,
    to: u32,
    measurement: Isometry3<T>,
    information: Matrix6<T>,
}

impl<T> EdgeSE3<T> {
    pub fn new(
        from: u32,
        to: u32,
        measurement: Isometry3<T>,
        information: Matrix6<T>,
    ) -> EdgeSE3<T> {
        EdgeSE3 {
            from,
            to,
            measurement,
            information,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(PartialEq)]
pub enum Node {
    SE2(Isometry2<f64>),
    SE3(Isometry3<f64>),
    XY(Vector2<f64>),
    XYZ(Vector3<f64>),
}
pub struct PoseGraph {
    len: usize,
    nodes: FxHashMap<u32, Node>,
    edges: Vec<Edge<f64>>,
    lut: FxHashMap<u32, usize>,
    iteration: usize,
    name: String,
    solver: PoseGraphSolver,
}

#[allow(clippy::too_many_arguments)]
fn update_linear_system<const X1: usize, const X2: usize>(
    H: &mut SparseMatrix,
    b: &mut Vector,
    e: &SVector<f64, X2>,
    A: &SMatrix<f64, X2, X1>,
    B: &SMatrix<f64, X2, X2>,
    omega: &SMatrix<f64, X2, X2>,
    from: usize,
    to: usize,
) -> Result<(), Box<dyn Error>> {
    // Compute jacobians
    let H_ii = A.transpose() * omega * A;
    let H_ij = A.transpose() * omega * B;
    let H_ji = H_ij.transpose();
    let H_jj = B.transpose() * omega * B;

    let b_i = A.transpose() * omega * e;
    let b_j = B.transpose() * omega * e;

    // Update the linear system
    set_matrix(H, from, from, &H_ii)?;
    set_matrix(H, from, to, &H_ij)?;
    set_matrix(H, to, from, &H_ji)?;
    set_matrix(H, to, to, &H_jj)?;

    set_vector(b, from, &b_i);
    set_vector(b, to, &b_j);
    Ok(())
}

fn set_matrix<const R: usize, const C: usize>(
    A: &mut SparseMatrix,
    i: usize,
    j: usize,
    m: &SMatrix<f64, R, C>,
) -> Result<(), Box<dyn Error>> {
    for ii in 0..R {
        for jj in 0..C {
            A.put(i + ii, j + jj, m.fixed_view::<1, 1>(ii, jj).x)?;
        }
    }
    Ok(())
}

fn set_vector<const D: usize>(v: &mut Vector, i: usize, source: &SVector<f64, D>) {
    for ii in 0..D {
        v.set(i + ii, source.fixed_view::<1, 1>(ii, 0).x + v.get(i + ii))
    }
}

fn solve_sparse(A: &mut SparseMatrix, b: &Vector) -> Result<DVector<f64>, Box<dyn Error>> {
    // Russell Sparse (SuiteSparse wrapper) is much faster then nalgebra_sparse
    let n = b.dim();
    let mut solution = Vector::new(n);

    // allocate solver
    let mut umfpack = SolverUMFPACK::new()?;

    // parameters
    let mut params = LinSolParams::new();
    params.verbose = false;
    params.compute_determinant = true;

    // call factorize
    umfpack.factorize(A, Some(params))?;

    // calculate the solution
    umfpack.solve(&mut solution, A, b, false)?;

    Ok(DVector::from_vec(solution.as_data().clone()))
}

impl PoseGraph {
    pub fn new(filename: &str, solver: PoseGraphSolver) -> Result<PoseGraph, Box<dyn Error>> {
        let (len, edges, lut, nodes) = parse_g2o(filename)?;
        let name = filename
            .split('/')
            .last()
            .unwrap()
            .split('.')
            .next()
            .unwrap()
            .to_string();
        Ok(PoseGraph {
            len,
            nodes,
            edges,
            lut,
            iteration: 0,
            name,
            solver,
        })
    }

    fn update_nodes(&mut self, dx: &DVector<f64>) {
        self.nodes.par_iter_mut().for_each(|(id, node)| {
            let offset = *self.lut.get(id).unwrap();
            match node {
                Node::SE2(node) => {
                    let diff = dx.fixed_rows::<3>(offset);
                    node.translation.vector += diff.xy();
                    node.rotation *= UnitComplex::from_angle(diff.z);
                }
                Node::XY(node) => {
                    *node += dx.fixed_rows::<2>(offset);
                }
                Node::SE3(_) => todo!(),
                Node::XYZ(_) => todo!(),
            }
        });
    }

    pub fn optimize(
        &mut self,
        num_iterations: usize,
        log: bool,
        plot: bool,
    ) -> Result<Vec<f64>, Box<dyn Error>> {
        let tolerance = 1e-4;
        let mut lambda = 0.01;
        let mut norms = Vec::new();
        let mut last_error = global_error(self);
        let mut errors = vec![last_error];
        if log {
            println!(
                "Loaded graph with {} nodes and {} edges",
                self.nodes.len(),
                self.edges.len()
            );
            println!("initial error :{:.5}", errors.last().unwrap());
        }
        if plot {
            self.plot()?;
        }
        for i in 0..num_iterations {
            self.iteration += 1;
            // let dx = self.linearize_and_solve()?;
            let (mut H, b) = self.build_linear_system(lambda)?;
            let dx = solve_sparse(&mut H, &b)?;
            self.update_nodes(&dx);
            // self.x += &dx;
            let norm_dx = dx.norm();
            let error = global_error(self);
            if self.solver == PoseGraphSolver::LevenbergMarquardt {
                if last_error < error {
                    self.update_nodes(&(-dx)); // get back old state
                    lambda *= 2.0;
                } else {
                    lambda /= 2.0;
                }
            }

            last_error = error;
            norms.push(norm_dx);
            errors.push(error);

            if log {
                println!(
                    "step {i:3} : |dx| = {norm_dx:3.5}, error = {:3.5}",
                    errors.last().unwrap()
                );
            }
            if plot {
                self.plot()?;
            }

            if norm_dx < tolerance {
                break;
            }
        }
        Ok(errors)
    }

    fn build_linear_system(&self, lambda: f64) -> Result<(SparseMatrix, Vector), Box<dyn Error>> {
        let mut H = SparseMatrix::new_coo(
            self.len,
            self.len,
            self.len * self.len,
            Some(Symmetry::PositiveDefinite(Storage::Full)),
            false,
        )?;
        let mut b = Vector::new(self.len);

        let mut need_to_add_prior = true;

        for edge in &self.edges {
            match edge {
                Edge::SE2_SE2(edge) => {
                    let from_idx = *self.lut.get(&edge.from).unwrap();
                    let to_idx = *self.lut.get(&edge.to).unwrap();

                    let Some(Node::SE2(x1)) = self.nodes.get(&edge.from) else {
                        unreachable!()
                    };
                    let Some(Node::SE2(x2)) = self.nodes.get(&edge.to) else {
                        unreachable!()
                    };

                    let z = &edge.measurement;
                    let omega = &edge.information;

                    let e = v3(&pose2D_pose2D_constraint(x1, x2, z));
                    let (A, B) = linearize_pose2D_pose2D_constraint(x1, x2, z);

                    update_linear_system(&mut H, &mut b, &e, &A, &B, omega, from_idx, to_idx)?;

                    if need_to_add_prior {
                        const V: f64 = 10000000.0;
                        H.put(from_idx, from_idx, V)?;
                        H.put(from_idx + 1, from_idx + 1, V)?;
                        H.put(from_idx + 2, from_idx + 2, V)?;
                        need_to_add_prior = false;
                    }
                }
                Edge::SE2_XY(edge) => {
                    let from_idx = *self.lut.get(&edge.from).unwrap();
                    let to_idx = *self.lut.get(&edge.to).unwrap();

                    let Some(Node::SE2(x)) = self.nodes.get(&edge.from) else {
                        unreachable!()
                    };
                    let Some(Node::XY(landmark)) = self.nodes.get(&edge.to) else {
                        unreachable!()
                    };

                    let z = &edge.measurement;
                    let omega = &edge.information;

                    let e = pose2D_landmark2D_constraint(x, landmark, z);
                    let (A, B) = linearize_pose_landmark_constraint(x, landmark);

                    update_linear_system(&mut H, &mut b, &e, &A, &B, omega, from_idx, to_idx)?;
                }
                Edge::SE3_SE3(_) => todo!(),
                Edge::SE3_XYZ => todo!(),
            }
        }
        b.map(|x| -x);
        if self.solver == PoseGraphSolver::LevenbergMarquardt {
            for i in 0..self.len {
                H.put(i, i, lambda)?;
            }
        }

        Ok((H, b))
    }

    fn linearize_and_solve(&self) -> Result<DVector<f64>, Box<dyn Error>> {
        let (mut H, b) = self.build_linear_system(0.0)?;
        solve_sparse(&mut H, &b)
    }

    pub fn plot(&self) -> Result<(), Box<dyn Error>> {
        // poses + landmarks
        let mut landmarks_present = false;
        let mut poses_seq = Vec::new();
        let mut poses = Curve::new();
        poses
            .set_line_style("None")
            .set_marker_color("b")
            .set_marker_style("o");

        let mut landmarks = Curve::new();
        landmarks
            .set_line_style("None")
            .set_marker_color("r")
            .set_marker_style("*");
        poses.points_begin();
        landmarks.points_begin();
        for (id, node) in self.nodes.iter() {
            match *node {
                Node::SE2(n) => {
                    poses.points_add(n.translation.x, n.translation.y);
                    poses_seq.push((id, n.translation.vector));
                }
                Node::XY(n) => {
                    landmarks.points_add(n.x, n.y);
                    landmarks_present = true;
                }
                Node::SE3(_) => todo!(),
                Node::XYZ(_) => todo!(),
            }
        }
        poses.points_end();
        landmarks.points_end();

        poses_seq.sort_by(|a, b| (a.0).partial_cmp(b.0).unwrap());
        let mut poses_seq_curve = Curve::new();
        poses_seq_curve.set_line_color("r");

        poses_seq_curve.points_begin();
        for (_, p) in poses_seq {
            poses_seq_curve.points_add(p.x, p.y);
        }
        poses_seq_curve.points_end();

        // add features to plot
        let mut plot = Plot::new();

        plot.add(&poses).add(&poses_seq_curve);
        if landmarks_present {
            plot.add(&landmarks);
        }
        // save figure
        plot.set_equal_axes(true)
            // .set_figure_size_points(600.0, 600.0)
            .save(format!("img/{}-{}-{:?}.svg", self.name, self.iteration, self.solver).as_str())?;
        Ok(())
    }
}

fn v3(iso2: &Isometry2<f64>) -> Vector3<f64> {
    Vector3::new(
        iso2.translation.x,
        iso2.translation.y,
        iso2.rotation.angle(),
    )
}
fn pose2D_pose2D_constraint<R: AbstractRotation<f64, D>, const D: usize>(
    x1: &Isometry<f64, R, D>,
    x2: &Isometry<f64, R, D>,
    z: &Isometry<f64, R, D>,
) -> Isometry<f64, R, D> {
    z.inverse() * x1.inverse() * x2
}

fn pose2D_landmark2D_constraint(
    x: &Isometry2<f64>,
    landmark: &Vector2<f64>,
    z: &Vector2<f64>,
) -> Vector2<f64> {
    x.rotation.to_rotation_matrix().transpose() * (landmark - x.translation.vector) - z
}

fn linearize_pose2D_pose2D_constraint(
    x1: &Isometry2<f64>,
    x2: &Isometry2<f64>,
    z: &Isometry2<f64>,
) -> (Matrix3<f64>, Matrix3<f64>) {
    let deriv = Matrix2::<f64>::new(0.0, -1.0, 1.0, 0.0);

    let z_rot = z.rotation.to_rotation_matrix();
    let x1_rot = x1.rotation.to_rotation_matrix();
    let a_11 = -(z_rot.inverse() * x1_rot.inverse()).matrix();
    let xr1d = deriv * x1.rotation.to_rotation_matrix().matrix();
    let a_12 =
        z_rot.transpose() * xr1d.transpose() * (x2.translation.vector - x1.translation.vector);

    #[rustfmt::skip]
    let A = Matrix3::new(
        a_11.m11, a_11.m12, a_12.x,
        a_11.m21, a_11.m22, a_12.y,
        0.0, 0.0, -1.0,
    );

    let b_11 = (z_rot.inverse() * x1_rot.inverse()).matrix().to_owned();
    #[rustfmt::skip]
    let B = Matrix3::new(
        b_11.m11, b_11.m12, 0.0,
        b_11.m21, b_11.m22, 0.0,
        0.0, 0.0, 1.0,
    );
    (A, B)
}

fn linearize_pose3D_pose3D_constraint(
    x1: &Isometry3<f64>,
    x2: &Isometry3<f64>,
    z: &Isometry3<f64>,
) -> (Matrix6<f64>, Matrix6<f64>) {
    let a = z.inverse();
    let b = x1.inverse() * x2;
    let error = a * b;
    let a_rot = a.rotation.to_rotation_matrix();
    let b_rot = b.rotation.to_rotation_matrix();
    let error_rot = error.rotation.to_rotation_matrix();
    let dq_dR = jacobian_so3(error_rot.matrix()); // variable name taken over from g2o

    let mut A = Matrix6::zeros();
    A.index_mut((..3, ..3)).copy_from(&(-1.0 * a_rot.matrix()));
    A.index_mut((..3, 3..))
        .copy_from(&(a_rot.matrix() * skew(&b.translation.vector)));
    A.index_mut((3.., 3..))
        .copy_from(&(dq_dR * skew_m_and_mult_parts(b_rot.matrix(), a_rot.matrix())));

    let mut B = Matrix6::zeros();
    B.index_mut((..3, ..3)).copy_from(error_rot.matrix());

    B.index_mut((3.., 3..))
        .copy_from(&(dq_dR * skew_m_and_mult_parts(&Matrix3::identity(), error_rot.matrix())));
    (A, B)
}

fn linearize_pose_landmark_constraint(
    x: &Isometry2<f64>,
    landmark: &Vector2<f64>,
) -> (Matrix2x3<f64>, Matrix2<f64>) {
    let deriv = Matrix2::<f64>::new(0.0, -1.0, 1.0, 0.0);

    let a_1 = -x.rotation.to_rotation_matrix().transpose().matrix();
    let xrd = deriv * *x.rotation.to_rotation_matrix().matrix();
    let a_2 = xrd.transpose() * (landmark - x.translation.vector);

    #[rustfmt::skip]
    let A = Matrix2x3::new(
        a_1.m11, a_1.m12, a_2.x,
        a_1.m21, a_1.m22, a_2.y,
    );

    let B = *x.rotation.to_rotation_matrix().transpose().matrix();

    (A, B)
}

fn global_error(graph: &PoseGraph) -> f64 {
    graph
        .edges
        .iter()
        .map(|edge| match edge {
            Edge::SE2_SE2(edge) => {
                let Some(Node::SE2(x1)) = graph.nodes.get(&edge.from) else {
                    unreachable!()
                };
                let Some(Node::SE2(x2)) = graph.nodes.get(&edge.to) else {
                    unreachable!()
                };

                let z = &edge.measurement;
                let omega = &edge.information;

                let e = v3(&pose2D_pose2D_constraint(x1, x2, z));

                (e.transpose() * omega * e).x
            }
            Edge::SE2_XY(edge) => {
                let Some(Node::SE2(x)) = graph.nodes.get(&edge.from) else {
                    unreachable!()
                };
                let Some(Node::XY(l)) = graph.nodes.get(&edge.to) else {
                    unreachable!()
                };

                let z = &edge.measurement;
                let omega = &edge.information;
                let e = pose2D_landmark2D_constraint(x, l, z);
                (e.transpose() * omega * e).x
            }
            Edge::SE3_SE3(_) => todo!(),
            Edge::SE3_XYZ => todo!(),
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_global_error() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/g2o/simulation-pose-pose.g2o";
        let graph = PoseGraph::new(filename, PoseGraphSolver::GaussNewton)?;
        approx::assert_abs_diff_eq!(138862234.0, global_error(&graph), epsilon = 10.0);

        let filename = "dataset/g2o/simulation-pose-landmark.g2o";
        let graph = PoseGraph::new(filename, PoseGraphSolver::GaussNewton)?;
        approx::assert_abs_diff_eq!(3030.0, global_error(&graph), epsilon = 1.0);

        let filename = "dataset/g2o/intel.g2o";
        let graph = PoseGraph::new(filename, PoseGraphSolver::GaussNewton)?;
        approx::assert_abs_diff_eq!(1795139.0, global_error(&graph), epsilon = 1e-2);

        let filename = "dataset/g2o/dlr.g2o";
        let graph = PoseGraph::new(filename, PoseGraphSolver::GaussNewton)?;
        approx::assert_abs_diff_eq!(369655336.0, global_error(&graph), epsilon = 10.0);
        Ok(())
    }

    #[test]
    fn final_global_error() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/g2o/simulation-pose-pose.g2o";
        let error = *PoseGraph::new(filename, PoseGraphSolver::GaussNewton)?
            .optimize(100, false, false)?
            .last()
            .unwrap();
        approx::assert_abs_diff_eq!(8269.0, error, epsilon = 1.0);

        let filename = "dataset/g2o/simulation-pose-landmark.g2o";
        let error = *PoseGraph::new(filename, PoseGraphSolver::GaussNewton)?
            .optimize(100, false, false)?
            .last()
            .unwrap();
        approx::assert_abs_diff_eq!(474.0, error, epsilon = 1.0);

        let filename = "dataset/g2o/intel.g2o";
        let error = *PoseGraph::new(filename, PoseGraphSolver::GaussNewton)?
            .optimize(100, false, false)?
            .last()
            .unwrap();
        approx::assert_abs_diff_eq!(360.0, error, epsilon = 1.0);

        let filename = "dataset/g2o/dlr.g2o";
        let error = *PoseGraph::new(filename, PoseGraphSolver::GaussNewton)?
            .optimize(100, false, false)?
            .last()
            .unwrap();
        approx::assert_abs_diff_eq!(56860.0, error, epsilon = 1.0);

        Ok(())
    }

    #[test]
    fn linearize_pose_pose_constraint_correct() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/g2o/simulation-pose-landmark.g2o";
        let graph = PoseGraph::new(filename, PoseGraphSolver::GaussNewton)?;

        match &graph.edges[0] {
            Edge::SE2_SE2(e) => {
                let Some(Node::SE2(x1)) = graph.nodes.get(&e.from) else {
                    todo!()
                };
                let Some(Node::SE2(x2)) = graph.nodes.get(&e.to) else {
                    todo!()
                };

                let z = e.measurement;

                let e = v3(&pose2D_pose2D_constraint(&x1, &x2, &z));
                let (A, B) = linearize_pose2D_pose2D_constraint(&x1, &x2, &z);

                let A_expected = Matrix3::new(0.0, 1.0, 0.113, -1., 0., 0.024, 0., 0., -1.);

                let B_expected = Matrix3::new(0.0, -1.0, 0.0, 1., 0., 0.0, 0., 0., 1.);

                approx::assert_abs_diff_eq!(A_expected, A, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(B_expected, B, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(Vector3::zeros(), e, epsilon = 1e-3);
            }
            _ => panic!(),
        }

        match &graph.edges[10] {
            Edge::SE2_SE2(e) => {
                let Some(Node::SE2(x1)) = graph.nodes.get(&e.from) else {
                    todo!()
                };
                let Some(Node::SE2(x2)) = graph.nodes.get(&e.to) else {
                    todo!()
                };

                let z = e.measurement;

                let e = v3(&pose2D_pose2D_constraint(&x1, &x2, &z));
                let (A, B) = linearize_pose2D_pose2D_constraint(&x1, &x2, &z);

                let A_expected =
                    Matrix3::new(0.037, 0.999, 0.138, -0.999, 0.037, -0.982, 0., 0., -1.);

                let B_expected = Matrix3::new(-0.037, -0.999, 0.0, 0.999, -0.037, 0.0, 0., 0., 1.);

                approx::assert_abs_diff_eq!(A_expected, A, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(B_expected, B, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(Vector3::zeros(), e, epsilon = 1e-3);
            }
            _ => panic!(),
        }

        Ok(())
    }

    #[test]
    fn linearize_pose_landmark_constraint_correct() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/g2o/simulation-pose-landmark.g2o";
        let graph = PoseGraph::new(filename, PoseGraphSolver::GaussNewton)?;

        match &graph.edges[1] {
            Edge::SE2_XY(edge) => {
                let Some(Node::SE2(x)) = graph.nodes.get(&edge.from) else {
                    todo!()
                };
                let Some(Node::XY(landmark)) = graph.nodes.get(&edge.to) else {
                    todo!()
                };

                let z = edge.measurement;

                let e = pose2D_landmark2D_constraint(&x, &landmark, &z);
                let (A, B) = linearize_pose_landmark_constraint(&x, &landmark);

                let A_expected = Matrix2x3::new(0.0, 1.0, 0.358, -1., 0., -0.051);

                let B_expected = Matrix2::new(0.0, -1.0, 1., 0.);

                approx::assert_abs_diff_eq!(A_expected, A, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(B_expected, B, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(Vector2::zeros(), e, epsilon = 1e-3);
            }
            _ => panic!(),
        }
        Ok(())
    }

    #[test]
    fn linearize_and_solve_correct() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/g2o/simulation-pose-landmark.g2o";
        let graph = PoseGraph::new(filename, PoseGraphSolver::GaussNewton)?;
        let dx = graph.linearize_and_solve()?;
        let expected_first_5 = DVector::<f64>::from_vec(vec![
            1.68518905e-01,
            5.74311089e-01,
            -5.08805168e-02,
            -3.67482151e-02,
            8.89458085e-01,
        ]);
        let first_5 = dx.rows(0, 5).clone_owned();
        approx::assert_abs_diff_eq!(expected_first_5, first_5, epsilon = 1e-3);
        Ok(())
    }
}

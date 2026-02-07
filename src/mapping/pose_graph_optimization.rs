#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)] // TODO: remove this

use nalgebra::{
    AbstractRotation, DVector, Isometry, Isometry2, Isometry3, Matrix2, Matrix2x3, Matrix3,
    Matrix6, SMatrix, SVector, UnitComplex, Vector2, Vector3,
};
use plotters::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::path::Path;

use faer::linalg::solvers::Solve;
use faer::mat::MatRef;
use faer::sparse::{SparseColMat, Triplet};
use faer::Side;

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

pub struct SparseLinearSystem {
    pub triplets: Vec<Triplet<usize, usize, f64>>,
    pub b: DVector<f64>,
    pub len: usize,
}

impl SparseLinearSystem {
    pub fn new(len: usize) -> SparseLinearSystem {
        SparseLinearSystem {
            triplets: Vec::with_capacity(len * 10), // Heuristic allocation
            b: DVector::zeros(len),
            len,
        }
    }
    pub fn solve(&mut self) -> Result<DVector<f64>, Box<dyn Error>> {
        let H =
            SparseColMat::<usize, f64>::try_new_from_triplets(self.len, self.len, &self.triplets)
                .map_err(|_| "Failed to create sparse matrix")?;

        let b_view = MatRef::<'_, f64>::from_column_major_slice(self.b.as_slice(), self.len, 1);

        let x = H
            .sp_cholesky(Side::Lower)
            .map_err(|_| "Matrix is not positive definite (ensure the graph is constrained)")?
            .solve(b_view);

        let mut solution = DVector::zeros(self.len);
        for i in 0..self.len {
            solution[i] = x[(i, 0)];
        }

        Ok(solution)
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
pub struct PoseGraph<'a> {
    len: usize,
    nodes: FxHashMap<u32, Node>,
    edges: Vec<Edge<f64>>,
    lut: FxHashMap<u32, usize>,
    iteration: usize,
    name: &'a str,
    solver: PoseGraphSolver,
}

fn update_linear_system<const X1: usize, const X2: usize>(
    sls: &mut SparseLinearSystem,
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
    set_matrix(sls, from, from, &H_ii);
    set_matrix(sls, from, to, &H_ij);
    set_matrix(sls, to, from, &H_ji);
    set_matrix(sls, to, to, &H_jj);

    set_vector(sls, from, &b_i);
    set_vector(sls, to, &b_j);
    Ok(())
}

fn set_matrix<const R: usize, const C: usize>(
    sls: &mut SparseLinearSystem,
    i: usize,
    j: usize,
    m: &SMatrix<f64, R, C>,
) {
    for ii in 0..R {
        for jj in 0..C {
            sls.triplets.push(Triplet::new(i + ii, j + jj, m[(ii, jj)]));
        }
    }
}

fn set_vector<const D: usize>(sls: &mut SparseLinearSystem, i: usize, source: &SVector<f64, D>) {
    for ii in 0..D {
        sls.b[i + ii] += source[ii];
    }
}

impl<'a> PoseGraph<'a> {
    pub fn new(file_path: &str, solver: PoseGraphSolver) -> Result<PoseGraph<'_>, Box<dyn Error>> {
        let (len, edges, lut, nodes) = parse_g2o(file_path)?;
        let name = Path::new(file_path).file_stem().unwrap().to_str().unwrap();
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
            let dx = self.build_linear_system(lambda)?.solve()?;
            self.update_nodes(&dx);
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

    fn build_linear_system(&self, lambda: f64) -> Result<SparseLinearSystem, Box<dyn Error>> {
        let mut sls = SparseLinearSystem::new(self.len);
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

                    update_linear_system(&mut sls, &e, &A, &B, omega, from_idx, to_idx)?;

                    if need_to_add_prior {
                        const V: f64 = 10000000.0;
                        sls.triplets.push(Triplet::new(from_idx, from_idx, V));
                        sls.triplets
                            .push(Triplet::new(from_idx + 1, from_idx + 1, V));
                        sls.triplets
                            .push(Triplet::new(from_idx + 2, from_idx + 2, V));
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

                    update_linear_system(&mut sls, &e, &A, &B, omega, from_idx, to_idx)?;
                }
                Edge::SE3_SE3(_) => todo!(),
                Edge::SE3_XYZ => todo!(),
            }
        }
        sls.b *= -1.0;

        if self.solver == PoseGraphSolver::LevenbergMarquardt {
            for i in 0..self.len {
                sls.triplets.push(Triplet::new(i, i, lambda));
            }
        }

        Ok(sls)
    }

    fn linearize_and_solve(&self) -> Result<DVector<f64>, Box<dyn Error>> {
        self.build_linear_system(0.0)?.solve()
    }

    pub fn plot(&self) -> Result<(), Box<dyn Error>> {
        // poses + landmarks
        let path = format!("img/{}-{}-{:?}.svg", self.name, self.iteration, self.solver);
        let root = SVGBackend::new(&path, (800, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        // Find plot bounds (min/max of your nodes)
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        for node in self.nodes.values() {
            let (x, y) = match node {
                Node::SE2(n) => (n.translation.x, n.translation.y),
                Node::XY(n) => (n.x, n.y),
                _ => continue,
            };
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let padding = 1.0;
        let mut chart = ChartBuilder::on(&root).margin(10).build_cartesian_2d(
            (min_x - padding)..(max_x + padding),
            (min_y - padding)..(max_y + padding),
        )?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()?;

        chart.draw_series(self.nodes.values().filter_map(|node| {
            if let Node::XY(n) = node {
                Some(Circle::new((n.x, n.y), 3, RED.filled()))
            } else {
                None
            }
        }))?;

        chart.draw_series(self.nodes.values().filter_map(|node| {
            if let Node::SE2(n) = node {
                Some(Circle::new(
                    (n.translation.x, n.translation.y),
                    2,
                    BLUE.filled(),
                ))
            } else {
                None
            }
        }))?;

        let mut poses_seq: Vec<_> = self
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if let Node::SE2(n) = node {
                    Some((*id, n))
                } else {
                    None
                }
            })
            .collect();
        poses_seq.sort_by_key(|a| a.0);

        chart.draw_series(LineSeries::new(
            poses_seq
                .iter()
                .map(|(_, n)| (n.translation.x, n.translation.y)),
            &RED,
        ))?;

        root.present()?;
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

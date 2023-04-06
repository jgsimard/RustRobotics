#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(clippy::deprecated_cfg_attr)]
use nalgebra::{
    DVector, Isometry2, Matrix2, Matrix2x3, Matrix3, Rotation2, SMatrix, SVector, Translation2,
    Vector2, Vector3,
};
use russell_lab::Vector;
use russell_sparse::{ConfigSolver, Solver, SparseTriplet, Symmetry};
use rustc_hash::FxHashMap;
use std::error::Error;

enum Edge<T> {
    SE2(EdgeSE2<T>),
    SE2_XY(EdgeSE2_XY<T>),
}

struct EdgeSE2<T> {
    from: u32,
    to: u32,
    measurement: Isometry2<T>,
    information: Matrix3<T>,
}

impl<T> EdgeSE2<T> {
    fn new(from: u32, to: u32, measurement: Isometry2<T>, information: Matrix3<T>) -> EdgeSE2<T> {
        EdgeSE2 {
            from,
            to,
            measurement,
            information,
        }
    }
}

struct EdgeSE2_XY<T> {
    from: u32,
    to: u32,
    measurement: Vector2<T>,
    information: Matrix2<T>,
}

impl<T> EdgeSE2_XY<T> {
    fn new(from: u32, to: u32, measurement: Vector2<T>, information: Matrix2<T>) -> EdgeSE2_XY<T> {
        EdgeSE2_XY {
            from,
            to,
            measurement,
            information,
        }
    }
}
pub struct PoseGraph {
    x: DVector<f64>,
    // nodes: FxHashMap<int, >
    n_nodes: u32,
    edges: Vec<Edge<f64>>,
    lut: FxHashMap<u32, usize>,
}

#[allow(clippy::too_many_arguments)]
fn update_linear_system<const X1: usize, const X2: usize>(
    H: &mut SparseTriplet,
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
    trip: &mut SparseTriplet,
    i: usize,
    j: usize,
    m: &SMatrix<f64, R, C>,
) -> Result<(), Box<dyn Error>> {
    for ii in 0..R {
        for jj in 0..C {
            trip.put(i + ii, j + jj, m.fixed_view::<1, 1>(ii, jj).x)?
        }
    }
    Ok(())
}

fn set_vector<const D: usize>(v: &mut Vector, i: usize, source: &SVector<f64, D>) {
    for ii in 0..D {
        v.set(i + ii, source.fixed_view::<1, 1>(ii, 0).x + v.get(i + ii))
    }
}

impl PoseGraph {
    pub fn from_g2o_file(filename: &str) -> Result<PoseGraph, Box<dyn Error>> {
        let mut edges = Vec::new();
        let mut lut = FxHashMap::default();
        // let mut nodes = FxHashMap::default();
        let mut n_nodes = 0;
        let mut offset = 0;
        let mut X = Vec::new();

        for line in std::fs::read_to_string(filename)?.lines() {
            let line: Vec<&str> = line.split(' ').collect();
            match line[0] {
                "VERTEX_SE2" => {
                    let id = line[1].parse::<u32>()?;
                    let x = line[2].parse::<f64>()?;
                    let y = line[3].parse::<f64>()?;
                    let angle = line[4].parse::<f64>()?;
                    n_nodes += 1;
                    lut.insert(id, offset);
                    offset += 3;
                    X.push(x);
                    X.push(y);
                    X.push(angle);
                }
                "VERTEX_XY" => {
                    let id = line[1].parse::<u32>()?;
                    let x = line[2].parse::<f64>()?;
                    let y = line[3].parse::<f64>()?;
                    n_nodes += 1;
                    lut.insert(id, offset);
                    offset += 2;
                    X.push(x);
                    X.push(y);
                }
                "EDGE_SE2" => {
                    let from = line[1].parse::<u32>()?;
                    let to = line[2].parse::<u32>()?;
                    let x = line[3].parse::<f64>()?;
                    let y = line[4].parse::<f64>()?;
                    let angle = line[5].parse::<f64>()?;
                    let tri_0 = line[6].parse::<f64>()?;
                    let tri_1 = line[7].parse::<f64>()?;
                    let tri_2 = line[8].parse::<f64>()?;
                    let tri_3 = line[9].parse::<f64>()?;
                    let tri_4 = line[10].parse::<f64>()?;
                    let tri_5 = line[11].parse::<f64>()?;

                    let translation = Translation2::new(x, y);
                    let rotation = Rotation2::new(angle);
                    let measurement = Isometry2::from_parts(translation, rotation.into());

                    #[allow(clippy::deprecated_cfg_attr)]
                    #[cfg_attr(rustfmt, rustfmt_skip)]
                    let information = Matrix3::new(
                        tri_0, tri_1, tri_2,
                        tri_1, tri_3, tri_4,
                        tri_2, tri_4, tri_5
                    );
                    let edge = Edge::SE2(EdgeSE2::new(from, to, measurement, information));
                    edges.push(edge);
                }
                "EDGE_SE2_XY" => {
                    let from = line[1].parse::<u32>()?;
                    let to = line[2].parse::<u32>()?;
                    let x = line[3].parse::<f64>()?;
                    let y = line[4].parse::<f64>()?;
                    let tri_0 = line[5].parse::<f64>()?;
                    let tri_1 = line[6].parse::<f64>()?;
                    let tri_2 = line[7].parse::<f64>()?;

                    let measurement = Vector2::new(x, y);

                    #[allow(clippy::deprecated_cfg_attr)]
                    #[cfg_attr(rustfmt, rustfmt_skip)]
                    let information = Matrix2::new(
                        tri_0, tri_1,
                        tri_1, tri_2
                    );
                    let edge = Edge::SE2_XY(EdgeSE2_XY::new(from, to, measurement, information));
                    edges.push(edge);
                }
                _ => unimplemented!("{}", line[0]),
            }
        }
        println!(
            "Loaded graph with {n_nodes} nodes and {} edges",
            edges.len()
        );
        Ok(PoseGraph {
            x: DVector::from_vec(X),
            n_nodes,
            edges,
            lut,
        })
    }

    pub fn plot(&self) {}

    pub fn optimize(&mut self, num_iterations: usize) -> Result<(), Box<dyn Error>> {
        let tolerance = 1e-4;
        let mut norms = Vec::new();
        let mut errors = vec![compute_global_error(self)];
        println!("initial error :{:?}", errors.last().unwrap());
        for i in 0..num_iterations {
            let dx = self.linearize_and_solve()?;
            self.x += &dx;
            let norm_dx = dx.norm();
            norms.push(norm_dx);
            errors.push(compute_global_error(self));

            println!("|dx| for step {i} : {norm_dx}");
            println!("errors :{:?}", errors.last().unwrap());

            if norm_dx < tolerance {
                break;
            }
        }
        Ok(())
    }

    fn linearize_and_solve(&self) -> Result<DVector<f64>, Box<dyn Error>> {
        let n = self.x.shape().0;

        let mut H = SparseTriplet::new(n, n, n * n, Symmetry::General).unwrap();
        let mut b = Vector::new(n);

        let mut need_to_add_prior = true;

        // make linear system
        for edge in &self.edges {
            match edge {
                Edge::SE2(edge) => {
                    let from_idx = *self.lut.get(&edge.from).unwrap();
                    let to_idx = *self.lut.get(&edge.to).unwrap();

                    let x1 = isometry(&self.x, from_idx);
                    let x2 = isometry(&self.x, to_idx);

                    let z = edge.measurement;
                    let omega = edge.information;

                    let (e, A, B) = linearize_pose_pose_constraint(&x1, &x2, &z);

                    update_linear_system(&mut H, &mut b, &e, &A, &B, &omega, from_idx, to_idx)?;

                    if need_to_add_prior {
                        H.put(from_idx, from_idx, 1000.0)?;
                        H.put(from_idx + 1, from_idx + 1, 1000.0)?;
                        H.put(from_idx + 2, from_idx + 2, 1000.0)?;

                        need_to_add_prior = false;
                    }
                }
                Edge::SE2_XY(edge) => {
                    let from_idx = *self.lut.get(&edge.from).unwrap();
                    let to_idx = *self.lut.get(&edge.to).unwrap();

                    let x = isometry(&self.x, from_idx);
                    let landmark = self.x.fixed_rows::<2>(to_idx).into();

                    let z = edge.measurement;
                    let omega = edge.information;

                    let (e, A, B) = linearize_pose_landmark_constraint(&x, &landmark, &z);

                    update_linear_system(&mut H, &mut b, &e, &A, &B, &omega, from_idx, to_idx)?;
                }
            }
        }
        // Use Russell Sparse because it is much faster then nalgebra_sparse
        b.map(|x| -x);
        let mut solution = Vector::new(n);
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config)?;
        solver.initialize(&H)?;
        solver.factorize()?;
        solver.solve(&mut solution, &b)?;
        Ok(DVector::from_vec(solution.as_data().clone()))
    }
}

fn pose_pose_constraint(
    x1: &Isometry2<f64>,
    x2: &Isometry2<f64>,
    z: &Isometry2<f64>,
) -> Isometry2<f64> {
    z.inverse() * x1.inverse() * x2
}

fn pose_landmark_constraint(
    x: &Isometry2<f64>,
    landmark: &Vector2<f64>,
    z: &Vector2<f64>,
) -> Vector2<f64> {
    x.rotation.to_rotation_matrix().transpose() * (landmark - x.translation.vector) - z
}

fn linearize_pose_pose_constraint(
    x1: &Isometry2<f64>,
    x2: &Isometry2<f64>,
    z: &Isometry2<f64>,
) -> (Vector3<f64>, Matrix3<f64>, Matrix3<f64>) {
    let e_ = pose_pose_constraint(x1, x2, z);
    let e = Vector3::new(e_.translation.x, e_.translation.y, e_.rotation.angle());

    // let minus1 = f64::from_f32(-1.0).unwrap();
    let deriv = Matrix2::<f64>::new(0.0, -1.0, 1.0, 0.0);

    let zr = z.rotation.to_rotation_matrix();
    let x1r = x1.rotation.to_rotation_matrix();
    let a_11 = -(zr.clone().inverse() * x1r.clone().inverse()).matrix();
    let xr1d = deriv * x1.rotation.to_rotation_matrix().matrix();
    let a_12 =
        zr.clone().transpose() * xr1d.transpose() * (x2.translation.vector - x1.translation.vector);

    #[cfg_attr(rustfmt, rustfmt_skip)]
    let A = Matrix3::new(
        a_11.m11, a_11.m12, a_12.x,
        a_11.m21, a_11.m22, a_12.y,
        0.0, 0.0, -1.0,
    );

    let b_11 = (zr.inverse() * x1r.inverse()).matrix().to_owned();
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let B = Matrix3::new(
        b_11.m11, b_11.m12, 0.0,
        b_11.m21, b_11.m22, 0.0,
        0.0, 0.0, 1.0,
    );
    (e, A, B)
}

fn linearize_pose_landmark_constraint(
    x: &Isometry2<f64>,
    landmark: &Vector2<f64>,
    z: &Vector2<f64>,
) -> (Vector2<f64>, Matrix2x3<f64>, Matrix2<f64>) {
    let e = pose_landmark_constraint(x, landmark, z);

    let deriv = Matrix2::<f64>::new(0.0, -1.0, 1.0, 0.0);

    let a_1 = -x.rotation.to_rotation_matrix().transpose().matrix();
    let xrd = deriv * *x.rotation.to_rotation_matrix().matrix();
    let a_2 = xrd.transpose() * (landmark - x.translation.vector);

    #[cfg_attr(rustfmt, rustfmt_skip)]
    let A = Matrix2x3::new(
        a_1.m11, a_1.m12, a_2.x,
        a_1.m21, a_1.m22, a_2.y,
    );

    let B = *x.rotation.to_rotation_matrix().transpose().matrix();

    (e, A, B)
}

fn isometry(x: &DVector<f64>, idx: usize) -> Isometry2<f64> {
    let p = x.fixed_rows::<3>(idx);
    Isometry2::from_parts(Translation2::new(p.x, p.y), Rotation2::new(p.z).into())
}

// TODO : Fixe pose-landmark : bad tests
fn compute_global_error(graph: &PoseGraph) -> f64 {
    let mut error = 0.0;
    for edge in &graph.edges {
        match edge {
            Edge::SE2(e) => {
                let from_idx = *graph.lut.get(&e.from).unwrap();
                let to_idx = *graph.lut.get(&e.to).unwrap();

                let x1 = isometry(&graph.x, from_idx);
                let x2 = isometry(&graph.x, to_idx);

                let z = e.measurement;

                let constraint = pose_pose_constraint(&x1, &x2, &z);
                let error_vec = Vector3::new(
                    constraint.translation.x,
                    constraint.translation.y,
                    constraint.rotation.angle(),
                );

                error += error_vec.norm();
            }
            Edge::SE2_XY(e) => {
                let from_idx = *graph.lut.get(&e.from).unwrap();
                let to_idx = *graph.lut.get(&e.to).unwrap();

                // TODO: change to only read the rotation, dont use translation anyway
                let x = isometry(&graph.x, from_idx);
                let l = graph.x.fixed_rows::<2>(to_idx);
                let z = e.measurement;

                error += pose_landmark_constraint(&x, &l.into(), &z).norm();
            }
        }
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_g2o_file_runs() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/new_slam_course/simulation-pose-pose.g2o";
        let graph = PoseGraph::from_g2o_file(filename)?;
        assert_eq!(400, graph.n_nodes);
        assert_eq!(1773, graph.edges.len());
        assert_eq!(1200, graph.x.shape().0);

        let filename = "dataset/new_slam_course/simulation-pose-landmark.g2o";
        let graph = PoseGraph::from_g2o_file(filename)?;
        assert_eq!(77, graph.n_nodes);
        assert_eq!(297, graph.edges.len());
        assert_eq!(195, graph.x.shape().0);

        let filename = "dataset/new_slam_course/intel.g2o";
        let graph = PoseGraph::from_g2o_file(filename)?;
        assert_eq!(1728, graph.n_nodes);
        assert_eq!(4830, graph.edges.len());
        assert_eq!(5184, graph.x.shape().0);

        let filename = "dataset/new_slam_course/dlr.g2o";
        let graph = PoseGraph::from_g2o_file(filename)?;
        assert_eq!(3873, graph.n_nodes);
        assert_eq!(17605, graph.edges.len());
        assert_eq!(11043, graph.x.shape().0);
        Ok(())
    }

    #[test]
    fn compute_global_error_correct_pose_pose() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/new_slam_course/simulation-pose-pose.g2o";
        let graph = PoseGraph::from_g2o_file(filename)?;
        approx::assert_abs_diff_eq!(3558.0723, compute_global_error(&graph), epsilon = 1e-2);

        Ok(())
    }

    #[test]
    fn compute_global_error_correct_pose_landmark() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/new_slam_course/simulation-pose-landmark.g2o";
        let graph = PoseGraph::from_g2o_file(filename)?;
        approx::assert_abs_diff_eq!(72.50542, compute_global_error(&graph), epsilon = 1e-2);

        Ok(())
    }

    #[test]
    fn compute_global_error_correct_intel() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/new_slam_course/intel.g2o";
        let graph = PoseGraph::from_g2o_file(filename)?;
        approx::assert_abs_diff_eq!(6109.409, compute_global_error(&graph), epsilon = 1e-2);
        Ok(())
    }

    #[test]
    fn compute_global_error_correct_dlr() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/new_slam_course/dlr.g2o";
        let graph = PoseGraph::from_g2o_file(filename)?;
        approx::assert_abs_diff_eq!(37338.21, compute_global_error(&graph), epsilon = 1e-2);
        Ok(())
    }

    #[test]
    fn linearize_pose_pose_constraint_correct() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/new_slam_course/simulation-pose-landmark.g2o";
        let graph = PoseGraph::from_g2o_file(filename)?;

        match &graph.edges[0] {
            Edge::SE2(e) => {
                let from_idx = *graph.lut.get(&e.from).unwrap();
                let to_idx = *graph.lut.get(&e.to).unwrap();

                let x1 = isometry(&graph.x, from_idx);
                let x2 = isometry(&graph.x, to_idx);

                let z = e.measurement;

                let (e, A, B) = linearize_pose_pose_constraint(&x1, &x2, &z);

                let A_expected = Matrix3::new(0.0, 1.0, 0.113, -1., 0., 0.024, 0., 0., -1.);

                let B_expected = Matrix3::new(0.0, -1.0, 0.0, 1., 0., 0.0, 0., 0., 1.);

                approx::assert_abs_diff_eq!(A_expected, A, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(B_expected, B, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(Vector3::zeros(), e, epsilon = 1e-3);
            }
            Edge::SE2_XY(_) => panic!(),
        }

        match &graph.edges[10] {
            Edge::SE2(e) => {
                let from_idx = *graph.lut.get(&e.from).unwrap();
                let to_idx = *graph.lut.get(&e.to).unwrap();

                let x1 = isometry(&graph.x, from_idx);
                let x2 = isometry(&graph.x, to_idx);

                let z = e.measurement;

                let (e, A, B) = linearize_pose_pose_constraint(&x1, &x2, &z);

                let A_expected =
                    Matrix3::new(0.037, 0.999, 0.138, -0.999, 0.037, -0.982, 0., 0., -1.);

                let B_expected = Matrix3::new(-0.037, -0.999, 0.0, 0.999, -0.037, 0.0, 0., 0., 1.);

                approx::assert_abs_diff_eq!(A_expected, A, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(B_expected, B, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(Vector3::zeros(), e, epsilon = 1e-3);
            }
            Edge::SE2_XY(_) => panic!(),
        }

        Ok(())
    }

    #[test]
    fn linearize_pose_landmark_constraint_correct() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/new_slam_course/simulation-pose-landmark.g2o";
        let graph = PoseGraph::from_g2o_file(filename)?;

        match &graph.edges[1] {
            Edge::SE2(_) => panic!(),
            Edge::SE2_XY(e) => {
                let from_idx = *graph.lut.get(&e.from).unwrap();
                let to_idx = *graph.lut.get(&e.to).unwrap();

                let x = isometry(&graph.x, from_idx);
                let landmark = graph.x.fixed_rows::<2>(to_idx);
                let z = e.measurement;

                let (e, A, B) = linearize_pose_landmark_constraint(&x, &landmark.into(), &z);

                let A_expected = Matrix2x3::new(0.0, 1.0, 0.358, -1., 0., -0.051);

                let B_expected = Matrix2::new(0.0, -1.0, 1., 0.);

                approx::assert_abs_diff_eq!(A_expected, A, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(B_expected, B, epsilon = 1e-3);
                approx::assert_abs_diff_eq!(Vector2::zeros(), e, epsilon = 1e-3);
            }
        }
        Ok(())
    }

    #[test]
    fn linearize_and_solve_correct() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/new_slam_course/simulation-pose-landmark.g2o";
        let graph = PoseGraph::from_g2o_file(filename)?;
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

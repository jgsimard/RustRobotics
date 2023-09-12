# RustRobotics

This package is a rust implementation of robotics algorithms. So far, the main source is the book [Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/). I plan to have algorithms implementations in the `src` folder and the algorithms use cases in the `examples` folder. I plan to have python bindings using [pyo3](https://github.com/PyO3/pyo3)/[maturin](https://github.com/PyO3/maturin). I am also implementing the algorithms in python using [JAX](https://jax.readthedocs.io/en/latest/) in this [repo](https://github.com/jgsimard/jaxrobot).

## Table of Contents

- [RustRobotics](#rustrobotics)
  - [Table of Contents](#table-of-contents)
  - [Localization](#localization)
    - [Extended Kalman Filter + Unscented Kalman Filter + Particle Filter](#extended-kalman-filter--unscented-kalman-filter--particle-filter)
    - [EKF/PF With Landmarks](#ekfpf-with-landmarks)
  - [Mapping](#mapping)
    - [Pose Graph Optimization](#pose-graph-optimization)
  - [Todo](#todo)
  - [Sources](#sources)

## Localization

### Extended Kalman Filter + Unscented Kalman Filter + Particle Filter

[EKF](src/localization/extended_kalman_filter.rs), [UKF](src/localization/unscented_kalman_filter.rs), [PF](src/localization/particle_filter.rs), [Example](examples/localization/bayesian_filter.rs)

```bash
cargo run --example localization
```

### EKF/PF With Landmarks

[Example](examples/localization/localization_landmarks.rs)

```bash
cargo run --example localization_landmarks
```

## Mapping

### Pose Graph Optimization

This algorithm uses the sparse solver in [Russel](https://github.com/cpmech/russell/tree/main/russell_sparse) which wraps [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) so follow the installation instructions. [Algorithm](src/mapping/pose_graph_optimization.rs), [Example](examples/mapping/pose_graph_optimization.rs), [Source](https://www.researchgate.net/profile/Mohamed-Mourad-Lafifi/post/What_is_the_relationship_between_GraphSLAM_and_Pose_Graph_SLAM/attachment/613b3f63647f3906fc978272/AS%3A1066449581928450%401631272802870/download/A+tutorial+on+graph-based+SLAM+_+Grisetti2010.pdf)

```bash
cargo run --example pose_graph_optimization
```

## Todo

- Bayesian Filters
  - Information filter
  - Histogram filter
- Pose Graph Optimization
  - PGO on manifold (3D)
  - Robust Kernels / Adaptive Kernels
- Mapping
  - Occupancy Grid
  - Iterative Closest Point
  - EKF-SLAM
  - FastSLAM 1.0
  - FastSLAM 2.0
- Camera Calibration
  - Direct Linear Transform (DLT)
  - Zhang’s Method
  - Projective 3-Point (P3P)
- Bundle Adjustement
- Triangulation
- Optimal Control
  - LQR
  - LQG
- etc

## Sources

- [Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)
- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
- [Underactuated Robotics](https://underactuated.mit.edu/index.html)
- [Probabilistic-Robotics-Algorithms](https://github.com/ChengeYang/Probabilistic-Robotics-Algorithms)
- [A tutorial on Graph-Based SLAM](https://www.researchgate.net/profile/Mohamed-Mourad-Lafifi/post/What_is_the_relationship_between_GraphSLAM_and_Pose_Graph_SLAM/attachment/613b3f63647f3906fc978272/AS%3A1066449581928450%401631272802870/download/A+tutorial+on+graph-based+SLAM+_+Grisetti2010.pdf)
- [A tutorial on SE(3) transformation parameterizations and on-manifold optimization](https://arxiv.org/abs/2103.15980)
- [Courses from Dr. Cyrill Stachniss](https://www.ipb.uni-bonn.de/teaching)

## Compilation issues

For plotters

```bash
sudo apt install fontconfig libfontconfig libfontconfig1-dev
```

For Russel

```bash
sudo apt install liblapacke-dev libmumps-seq-dev libopenblas-dev libsuitesparse-dev
```

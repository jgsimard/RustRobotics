# RustRobotics

This package is a rust implementation of robotics algorithms. So far, the main source is the book [Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/). I plan to have algorithms implementations in the `src` folder and the algorithms use cases in the `examples` folder. I plan to have python bindings using [pyo3](https://github.com/PyO3/pyo3)/[maturin](https://github.com/PyO3/maturin). I am also implementing the algorithms in python using [JAX](https://jax.readthedocs.io/en/latest/) in this [repo](https://github.com/jgsimard/jaxrobot).

## Table of Contents

- [RustRobotics](#rustrobotics)
  - [Table of Contents](#table-of-contents)
  - [Localization](#localization)
    - [Extended Kalman Filter](#extended-kalman-filter)
    - [Unscented Kalman Filter](#unscented-kalman-filter)
    - [Extended Kalman Filter With Landmarks](#extended-kalman-filter-with-landmarks)
    - [Particle Filter With Landmarks](#particle-filter-with-landmarks)
  - [Sources](#sources)

<!-- * Unscented Kalman filter -->
<!-- * Information filter -->
<!-- * Particle filter -->
<!-- * Hisstogram filter -->
<!-- * Mapping -->
  <!-- * Gaussian Grid -->
  <!-- * Ray Casting Grid -->
  <!-- * Lidar to Grid -->
  <!-- * K-means clustering -->
  <!-- * Gaussian Mixture Model -->
  <!-- * Rectangle Fitting -->
<!-- * SLAM -->
  <!-- * Iterative Closest Point -->
  <!-- * EKF-SLAM -->
  <!-- * GraphSlam -->
  <!-- * SEIF-SLAM -->
  <!-- * FastSLAM 1.0 -->
  <!-- * FastSLAM 2.0 -->
<!-- * Path Planning -->
<!-- * Grid Based Search -->
<!-- * Dijkstra -->
<!-- * A-star -->
<!-- * D-star -->
<!-- * D-star lite -->
<!-- * Potential Field -->
<!-- * Rapidly-Exploring Random Trees (RRT) -->
<!-- * RRT-star -->
<!-- * RRT-star with reeds-shepp path -->
<!-- * Polynomial -->
<!-- * Order 3 -->
<!-- * Order 5 -->

## Localization

### Extended Kalman Filter

[Algorithm](src/localization/extended_kalman_filter.rs), [Example](examples/localization/extended_kalman_filter.rs)

```bash
cargo run --example ekf
```

### Unscented Kalman Filter

[Algorithm](src/localization/unscented_kalman_filter.rs), [Example](examples/localization/unscented_kalman_filter.rs)

```bash
cargo run --example ukf
```

### Extended Kalman Filter With Landmarks

[Algorithm](src/localization/extended_kalman_filter.rs), [Example](examples/localization/extended_kalman_filter_landmarks.rs)

```bash
cargo run --example ekf_lm
```

### Particle Filter With Landmarks

[Algorithm](src/localization/particle_filter.rs), [Example](examples/localization/particle_filter.rs)

```bash
cargo run --example pf_lm
```

## Sources

[Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)
[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
[Underactuated Robotics](https://underactuated.mit.edu/index.html)
[Probabilistic-Robotics-Algorithms](https://github.com/ChengeYang/Probabilistic-Robotics-Algorithms)

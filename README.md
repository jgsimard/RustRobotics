# RustRobotics

This package is a rust implementation of [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics). I plan to have algorithms implementations in the `src` folder and the algorithms use cases in the `examples` folder. I plan to have python bindings using [pyo3](https://github.com/PyO3/pyo3)/[maturin](https://github.com/PyO3/maturin).

## Table of Contents

* [Localization](#localization)
  * Kalman filter
  * [Extended kalman filter](#extended-kalman-filter)
  * Unscented Kalman filter
  * Information filter
  * Particle filter
  * Histogram filter
* Mapping
  * Gaussian Grid
  * Ray Casting Grid
  * Lidar to Grid
  * K-means clustering
  * Gaussian Mixture Model
  * Rectangle Fitting
* SLAM
  * Iterative Closest Point
  * EKF-SLAM
  * GraphSlam
  * SEIF-SLAM
  * FastSLAM 1.0
  * FastSLAM 2.0
* Path Planning
  * Grid Based Search
    * Dijkstra
    * A*
    * D*
    * D* lite
    * Potential Field
  * Rapidly-Exploring Random Trees (RRT)
    * RRT*
    * RRT* with reeds-shepp path
  * Polynomial
    * Order 3
    * Order 5

## Localization

### Extended Kalman Filter

[Algorithm](src/localization/extended_kalman_filter.rs), [Example](examples/localization/extended_kalman_filter.rs)

```bash
cargo run --example ekf
```

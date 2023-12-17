use criterion::{criterion_group, criterion_main, Criterion};

extern crate robotics;
use robotics::mapping::{PoseGraph, PoseGraphSolver};

fn graph_slam_intel(b: &mut Criterion) {
    b.bench_function("graph_slam_intel", |b| {
        b.iter(|| {
            PoseGraph::new("dataset/g2o/intel.g2o", PoseGraphSolver::GaussNewton)?
                .optimize(10, false, false)
        })
    });
}

criterion_group!(benches, graph_slam_intel);
criterion_main!(benches);

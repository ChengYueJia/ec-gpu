#![cfg(any(feature = "cuda", feature = "opencl"))]

use std::char::MAX;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ec_gpu_gen::{
    multiexp::MultiexpKernel, multiexp_cpu::SourceBuilder, rust_gpu_tools::Device,
    threadpool::Worker,
};
use ff::{Field, PrimeField};
use group::{Curve, Group};
use halo2curves::bn256::Bn256;
use halo2curves::pairing::Engine;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// The power that will be used to define the maximum number of elements. The number of elements
/// is `2^MAX_K`.
const MAX_K: usize = 24;
const MIN_K: usize = 19;
/// The maximum number of elements for this benchmark.
const MAX_ELEMENTS: usize = 1 << MAX_K;

fn bench_multiexp(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("multiexp");
    // The difference between runs is so little, hence a low sample size is OK.
    group.sample_size(10);

    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = MultiexpKernel::<<Bn256 as Engine>::G1Affine>::create(programs, &devices)
        .expect("Cannot initialize kernel!");
    let pool = Worker::new();
    let max_bases: Vec<_> = (0..MAX_ELEMENTS)
        .into_par_iter()
        .map(|_| <Bn256 as Engine>::G1::random(rand::thread_rng()).to_affine())
        .collect();
    let max_exponents: Vec<_> = (0..MAX_ELEMENTS)
        .into_par_iter()
        .map(|_| <Bn256 as Engine>::Scalar::random(rand::thread_rng()).to_repr())
        .collect();

    let num_elements: Vec<_> = (MIN_K..=MAX_K).map(|shift| 1 << shift).collect();
    for num in num_elements {
        let (bases, skip) = SourceBuilder::get((Arc::new(max_bases[0..num].to_vec()), 0));
        let exponents = Arc::new(max_exponents[0..num].to_vec());

        group.bench_function(BenchmarkId::new("k", num), |b| {
            b.iter(|| {
                black_box(
                    kern.multiexp(&pool, bases.clone(), exponents.clone(), skip)
                        .unwrap(),
                );
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_multiexp);
criterion_main!(benches);

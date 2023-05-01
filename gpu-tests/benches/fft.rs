#![cfg(any(feature = "cuda", feature = "opencl"))]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ec_gpu_gen::{fft::FftKernel, rust_gpu_tools::Device, threadpool::Worker};
use ff::{Field, PrimeField};
use group::Group;
use halo2curves::bn256::{Bn256, Fr};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// The power that will be used to define the maximum number of elements. The number of elements
/// is `2^MAX_K`.
const MAX_K: u32 = 24;
const MIN_K: u32 = 19;

fn omega<F: PrimeField>(num_coeffs: usize) -> F {
    // Compute omega, the 2^exp primitive root of unity
    let exp = (num_coeffs as f32).log2().floor() as u32;
    let mut omega = F::ROOT_OF_UNITY;
    for _ in exp..F::S {
        omega = omega.square();
    }
    omega
}

fn bench_fft_many(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("fft_many");
    // The difference between runs is so little, hence a low sample size is OK.
    group.sample_size(10);

    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");

    for k in MIN_K..=MAX_K {
        let num = 1 << k;
        let mut coeffs = (0..num)
            .into_par_iter()
            .map(|_| Fr::random(rand::thread_rng()))
            .collect::<Vec<_>>();
        let omega = omega::<Fr>(coeffs.len());

        group.bench_function(BenchmarkId::new("k", k), |b| {
            b.iter(|| {
                black_box(
                    kern.radix_fft_many(&mut [&mut coeffs], &[omega], &[k])
                        .unwrap(),
                );
            })
        });
        drop(coeffs);
    }
    group.finish();
}

fn bench_fft(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("fft");
    // The difference between runs is so little, hence a low sample size is OK.
    group.sample_size(10);

    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");

    for k in MIN_K..=MAX_K {
        let num = 1 << k;
        let mut coeffs = (0..num)
            .into_par_iter()
            .map(|_| Fr::random(rand::thread_rng()))
            .collect::<Vec<_>>();
        let omega = omega::<Fr>(coeffs.len());

        group.bench_function(BenchmarkId::new("k", k), |b| {
            b.iter(|| {
                black_box(kern.radix_fft(&mut coeffs, &omega, k).unwrap());
            })
        });
        drop(coeffs);
    }
    group.finish();
}

criterion_group!(benches, bench_fft, bench_fft_many);
criterion_main!(benches);

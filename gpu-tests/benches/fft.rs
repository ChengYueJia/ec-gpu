#![cfg(any(feature = "cuda", feature = "opencl"))]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ec_gpu_gen::{fft::FftKernel, rust_gpu_tools::Device, threadpool::Worker};
use ff::{Field, PrimeField};
use group::Group;
use halo2curves::bn256::{Bn256, Fr};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// The power that will be used to define the maximum number of elements. The number of elements
/// is `2^MAX_ELEMENTS_POWER`.
const MAX_ELEMENTS_POWER: u32 = 30;

fn omega<F: PrimeField>(num_coeffs: usize) -> F {
    // Compute omega, the 2^exp primitive root of unity
    let exp = (num_coeffs as f32).log2().floor() as u32;
    let mut omega = F::ROOT_OF_UNITY;
    for _ in exp..F::S {
        omega = omega.square();
    }
    omega
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

    let num_elements: Vec<u32> = (10..MAX_ELEMENTS_POWER).map(|shift| 1 << shift).collect();
    for num in num_elements {
        let mut coeffs = (0..num)
            .into_par_iter()
            .map(|_| Fr::random(rand::thread_rng()))
            .collect::<Vec<_>>();
        let omega = omega::<Fr>(coeffs.len());

        group.bench_with_input(BenchmarkId::from_parameter(num), &num, |bencher, &num| {
            bencher.iter(|| {
                black_box(kern.radix_fft(&mut coeffs, &omega, num).unwrap());
            })
        });
        drop(coeffs);
    }
    group.finish();
}

criterion_group!(benches, bench_fft);
criterion_main!(benches);

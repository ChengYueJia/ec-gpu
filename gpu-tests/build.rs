#[cfg(not(any(feature = "cuda", feature = "opencl")))]
fn main() {}

#[cfg(any(feature = "cuda", feature = "opencl"))]
fn main() {
    use ec_gpu_gen::SourceBuilder;
    use halo2curves::bn256::{Fq, Fq2, Fr, G1Affine, G2Affine};

    let source_builder = SourceBuilder::new()
        .add_fft::<Fr>()
        .add_multiexp::<G1Affine, Fq>()
        .add_multiexp::<G2Affine, Fq2>();
    ec_gpu_gen::generate(&source_builder);
}

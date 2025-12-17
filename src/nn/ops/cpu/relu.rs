use crate::nn::{
    tensors::{Tensor, WithGrad},
    TensorFloat,
};
use tensor_optim::TensorOps;

#[cfg(feature = "dyntensor")]
use alloc::vec;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(not(feature = "alloc"))]
use box_closure::{Align8, OpaqueFn};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;

/// Applies a `ReLU` activation function element-wise on the input tensor:
/// `$$ f(x) = \\max(0, x) $$`
///
/// # Returns
///
/// - Output tensor of same shape
/// - Backward function which propagates upstream gradients through `ReLU`:
///   
///     `$$ \\frac{\\partial f}{\\partial x} = 1 \\text{ if } x > 0 \\text{ else } 0 $$`
///
/// # Optimizations
///
/// - Uses SIMD (`AVX2`) for fast element-wise max (if `simd` feature enabled)
/// - Uses `rayon` to parallelize both forward and backward passes
///
/// # Notes
///
/// - Backward function uses input value to compute mask
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn relu(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    let shape = input.get_value().shape();
    let len = input.get_value().data().len();
    let mut data = vec![TensorFloat::default(); len];

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        const LANES: usize = 4;
        data.chunks_mut(LANES)
            .zip(input.value.data.chunks(LANES))
            .for_each(|(out_chunk, in_chunk)| unsafe {
                let mut in_buf = [0.0; LANES];
                in_buf[..in_chunk.len()].copy_from_slice(in_chunk);

                let x = _mm256_loadu_pd(in_buf.as_ptr());
                let zero = _mm256_setzero_pd();
                let y = _mm256_max_pd(x, zero);

                let mut out_buf = [0.0; LANES];
                _mm256_storeu_pd(out_buf.as_mut_ptr(), y);

                out_chunk.copy_from_slice(&out_buf[..in_chunk.len()]);
            });
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        data.iter_mut()
            .zip(input.get_value().data().iter())
            .for_each(|(y, &x)| {
                *y = if x > TensorFloat::default() {
                    x
                } else {
                    TensorFloat::default()
                };
            });
    }

    let out = Tensor::new(shape, &data);
    let input_data = input.get_value().data();

    let back = move |grad_output: Tensor<TensorFloat>| {
        let mut grad = vec![TensorFloat::default(); grad_output.data().len()];

        grad.iter_mut()
            .zip(input_data.iter())
            .zip(grad_output.data().iter())
            .for_each(|((g, &x), &dy)| {
                *g = if x > 0.0 { dy } else { 0.0 };
            });

        Tensor::new(shape, &grad)
    };

    (out, Box::new(back))
}

/// Applies a `ReLU` activation function element-wise on the input tensor:
/// `$$ f(x) = \\max(0, x) $$`
///
/// # Returns
///
/// - Output tensor of same shape
/// - Backward function which propagates upstream gradients through `ReLU`:
///   
///     `$$ \\frac{\\partial f}{\\partial x} = 1 \\text{ if } x > 0 \\text{ else } 0 $$`
///
/// # Optimizations
///
/// - Uses SIMD (`AVX2`) for fast element-wise max (if `simd` feature enabled)
/// - Uses `rayon` to parallelize both forward and backward passes
///
/// # Notes
///
/// - Backward function uses input value to compute mask
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn relu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    Box<dyn Fn(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let mut data = [TensorFloat::default(); N];

    data.iter_mut()
        .zip(input.get_value().data().iter())
        .for_each(|(y, &x)| {
            *y = if x > TensorFloat::default() {
                x
            } else {
                TensorFloat::default()
            };
        });

    let out = Tensor::new(shape, &data);
    let input_data = input.get_value().data(); // &[TensorFloat; N]

    let back = move |grad_output: Tensor<TensorFloat, N, D>| {
        let mut grad = [TensorFloat::default(); N];

        grad.iter_mut()
            .zip(input_data.iter())
            .zip(grad_output.data().iter())
            .for_each(|((g, &x), &dy)| {
                *g = if x > 0.0 { dy } else { 0.0 };
            });

        Tensor::new(shape, &grad)
    };

    (out, Box::new(back))
}

/// Applies a `ReLU` activation function element-wise on the input tensor:
/// `$$ f(x) = \\max(0, x) $$`
///
/// # Returns
///
/// - Output tensor of same shape
/// - Backward function which propagates upstream gradients through `ReLU`:
///   
///     `$$ \\frac{\\partial f}{\\partial x} = 1 \\text{ if } x > 0 \\text{ else } 0 $$`
///
/// # Optimizations
///
/// - Uses SIMD (`AVX2`) for fast element-wise max (if `simd` feature enabled)
/// - Uses `rayon` to parallelize both forward and backward passes
///
/// # Notes
///
/// - Backward function uses input value to compute mask
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn relu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    OpaqueFn<'_, Tensor<TensorFloat, N, D>, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let mut data = [TensorFloat::default(); N];

    data.iter_mut()
        .zip(input.get_value().data().iter())
        .for_each(|(y, &x)| {
            *y = if x > TensorFloat::default() {
                x
            } else {
                TensorFloat::default()
            };
        });

    let out = Tensor::new(shape, &data);
    let input_data = input.get_value().data(); // &[TensorFloat; N]

    let back = move |grad_output: Tensor<TensorFloat, N, D>| {
        let mut grad = [TensorFloat::default(); N];

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            const LANES: usize = 4;
            grad.chunks_mut(LANES)
                .zip(input_data.chunks(LANES))
                .zip(grad_output.data().chunks(LANES))
                .for_each(|((g_out, in_chunk), grad_chunk)| unsafe {
                    let mut in_buf = [0.0; LANES];
                    let mut grad_in_buf = [0.0; LANES];

                    in_buf[..in_chunk.len()].copy_from_slice(in_chunk);
                    grad_in_buf[..grad_chunk.len()].copy_from_slice(grad_chunk);

                    let x = _mm256_loadu_pd(in_buf.as_ptr());
                    let dy = _mm256_loadu_pd(grad_in_buf.as_ptr());
                    let mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ);
                    let gradv = _mm256_and_pd(dy, mask);

                    let mut out_buf = [0.0; LANES];
                    _mm256_storeu_pd(out_buf.as_mut_ptr(), gradv);
                    g_out.copy_from_slice(&out_buf[..in_chunk.len()]);
                });
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            grad.iter_mut()
                .zip(input_data.iter())
                .zip(grad_output.data().iter())
                .for_each(|((g, &x), &dy)| {
                    *g = if x > 0.0 { dy } else { 0.0 };
                });
        }

        Tensor::new(shape, &grad)
    };

    (out, OpaqueFn::new(back))
}

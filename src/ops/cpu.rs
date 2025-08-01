//! Parallel CPU backend tensor operations
//! 
//! # CPU Backend
//!
//! This module provides high-performance CPU implementations of core tensor operations
//! used in neural network training and inference.
//! 
//! These CPU functions are the default when calling `backprop::xyz`; It dispatches to
//! `ops::cpu::xyz` as long as WGPU and CUDA are disabled.
//!
//! ## Features
//!
//! - Parallel execution using [`rayon`](https://docs.rs/rayon)
//! - Optional SIMD acceleration using AVX2 (enabled via `simd` feature flag)
//! - Pure Rust fallback path when SIMD is disabled or unavailable
//!
//! ## Implemented Ops
//!
//! - `matmul`: Matrix multiplication with SIMD and multithreading
//! - `relu`: ReLU activation with forward and backward pass
//! - `mse_loss`: Mean squared error loss with autograd
//! - `sgd`: In-place stochastic gradient descent step
//!
//! ## Design Goals
//!
//! - Deterministic results (given deterministic input and scheduling)
//! - Zero dependencies beyond `rayon`
//! - Modular: CPU functions are separate from backend dispatching
//!
//! ## Safety
//!
//! - SIMD paths use `unsafe` blocks and assume 64-bit AVX2-capable CPUs
//! - Runtime checks are encouraged but not enforced in this module

use rayon::prelude::*;
use crate::{ops::dispatch::{FnF64Ten64, FnTen64To, FnToDoubleTen64}, tensors::{Ten64, Tensor, WithGrad}};

/// Performs a matrix multiplication `C = A × B` on two 2D tensors (`A: m×k`, `B: k×n`),
/// returning the result tensor and a closure for backpropagation.
///
/// # Requirements
/// - Shapes must be compatible: `A.shape = [m, k]` and `B.shape = [k, n]`.
///
/// # Optimizations
/// - Uses `rayon` for parallel row computation
/// - Uses AVX2 SIMD for fused multiply-adds (if enabled via `--features=simd`)
///
/// # Returns
/// - Output tensor of shape `[m, n]`
/// - Backward function computing gradients w.r.t. `A` and `B`
///
/// # Panics
/// - If the inner dimensions of `A` and `B` do not match.
///
/// # Example
/// ```rust
/// use briny_ai::backprop::matmul;
/// use briny_ai::{tensor, tensors::WithGrad};
/// 
/// let a = WithGrad::new(briny_ai::tensor!([[5.0, 1.0], [6.0, 3.0]]));
/// let b = WithGrad::new(briny_ai::tensor!([[1.0, 2.0], [5.0, 1.9]]));
/// let grad_output = tensor!([[1.0, 2.0], [3.0, 2.0]]);
/// let (c, back) = matmul(&a, &b);
/// let (grad_a, grad_b) = back(&grad_output);
/// ```
pub fn matmul(
    a: &WithGrad<Ten64>,
    b: &WithGrad<Ten64>,
) -> (Ten64, Box<FnToDoubleTen64>) {
    let m = a.value.shape[0];
    let k = a.value.shape[1];
    let n = b.value.shape[1];
    assert_eq!(k, b.value.shape[0], "matmul shape mismatch");

    let a_data = &a.value.data;
    let b_data = &b.value.data;

    let mut out_data = vec![0.0; m * n];

    out_data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..n {
                let sum = {
                    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
                    {
                        let mut acc = unsafe { _mm256_setzero_pd() };
                        let mut idx = 0;
                        while idx + 4 <= k {
                            unsafe {
                                let a_chunk = _mm256_loadu_pd(&a_data[i * k + idx]);
                                let b_chunk = _mm256_set_pd(
                                    b_data[(idx + 3) * n + j],
                                    b_data[(idx + 2) * n + j],
                                    b_data[(idx + 1) * n + j],
                                    b_data[(idx) * n + j],
                                );
                                acc = _mm256_fmadd_pd(a_chunk, b_chunk, acc);
                            }
                            idx += 4;
                        }

                        let mut temp = [0.0; 4];
                        unsafe { _mm256_storeu_pd(temp.as_mut_ptr(), acc) };
                        let mut sum: f64 = temp.iter().sum();

                        for l in idx..k {
                            sum += a_data[i * k + l] * b_data[l * n + j];
                        }

                        sum
                    }

                    #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2")))]
                    {
                        let mut sum = 0.0;
                        for l in 0..k {
                            sum += a_data[i * k + l] * b_data[l * n + j];
                        }
                        sum
                    }
                };
                row[j] = sum;
            }
        });

    let out = Tensor::new(vec![m, n], out_data);

    let a_val = a.value.clone();
    let b_val = b.value.clone();

    let back = Box::new(move |grad: &Ten64| {
        grad.matmul(&a_val, &b_val)
    });

    (out, Box::new(back))
}

/// Computes the mean squared error (MSE) between predictions and targets,
/// returning both the scalar loss and a gradient function.
///
/// # Formula
/// $$ L = \\frac{1}{n} \\sum_i (y_i - t_i)^2 $$
///
/// # Returns
/// - Scalar loss `f64`
/// - Backward function mapping upstream scalar gradient `dL` to a tensor of shape `prediction`
///
/// # Notes
/// - Forward and backward passes are fully parallelized with `rayon`
/// - Suitable for batch or scalar regression losses
///
/// # Example
/// ```rust
/// use briny_ai::backprop::mse_loss;
/// use briny_ai::tensors::WithGrad;
/// use briny_ai::tensor;
/// 
/// let y_pred = WithGrad::new(tensor!([1.0, 2.0, 3.0]));
/// let y_true = tensor!([1.0, 3.0, 2.0]);
/// let (loss, back) = mse_loss(&y_pred, &y_true);
/// let grad_tensor = back(1.0); // ∂L/∂y_pred
/// ```
pub fn mse_loss<'a>(
    prediction: &'a WithGrad<Ten64>,
    target: &'a Ten64,
) -> (f64, Box<FnF64Ten64<'a>>) {
    let n = prediction.value.data.len() as f64;

    // parallel forward pass
    let loss = prediction
        .value
        .data
        .par_iter()
        .zip(&target.data)
        .map(|(&y, &t)| (y - t).powi(2))
        .sum::<f64>()
        / n;

    let shape = prediction.value.shape.clone();
    let pred_data = prediction.value.data.clone();
    let target_data = target.data.clone();

    // parallel backward pass
    let back = move |grad_output: f64| {
        let grad: Vec<f64> = pred_data
            .par_iter()
            .zip(&target_data)
            .map(|(&y, &t)| 2.0 * (y - t) * grad_output / n)
            .collect();

        Tensor::new(shape.clone(), grad)
    };

    (loss, Box::new(back))
}

/// Applies the ReLU activation function element-wise on the input tensor:
/// $$ f(x) = \\max(0, x) $$
///
/// # Returns
/// - Output tensor of same shape
/// - Backward function which propagates upstream gradients through ReLU:
///   $$ \\frac{\\partial f}{\\partial x} = 1 \\text{ if } x > 0 \\text{ else } 0 $$
///
/// # Optimizations
/// - Uses SIMD (`AVX2`) for fast element-wise max (if `simd` feature enabled)
/// - Uses `rayon` to parallelize both forward and backward passes
///
/// # Notes
/// - Backward function uses input value to compute mask
///
/// # Example
/// ```rust
/// use briny_ai::backprop::relu;
/// use briny_ai::{tensor, tensors::WithGrad};
/// 
/// let input = WithGrad::new(tensor!([[10.0, 30.0], [20.0, 50.0]]));
/// let grad_output = tensor!([5.0, 2.0, 3.0, 4.0]);
/// let (out, back) = relu(&input);
/// let grad_input = back(&grad_output);
/// ```
pub fn relu(
    input: &WithGrad<Ten64>,
) -> (Ten64, Box<FnTen64To>) {
    let shape = input.value.shape.clone();
    let len = input.value.data.len();
    let mut data = vec![0.0f64; len];

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    {
        const LANES: usize = 4;
        data.par_chunks_mut(LANES)
            .zip(input.value.data.par_chunks(LANES))
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

    #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2")))]
    {
        data.par_iter_mut()
            .zip(input.value.data.par_iter())
            .for_each(|(y, &x)| {
                *y = if x > 0.0 { x } else { 0.0 };
            });
    }

    let out = Tensor::new(shape.clone(), data);
    let input_data = input.value.data.clone();

    let back = move |grad_output: &Ten64| {
        let mut grad = vec![0.0f64; grad_output.data.len()];

        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
        {
            const LANES: usize = 4;
            grad.par_chunks_mut(LANES)
                .zip(input_data.par_chunks(LANES))
                .zip(grad_output.data.par_chunks(LANES))
                .for_each(|((g_out, in_chunk), grad_chunk)| unsafe {
                    let mut in_buf = [0.0; LANES];
                    let mut grad_in_buf = [0.0; LANES];

                    in_buf[..in_chunk.len()].copy_from_slice(in_chunk);
                    grad_in_buf[..grad_chunk.len()].copy_from_slice(grad_chunk);

                    let x = _mm256_loadu_pd(in_buf.as_ptr());
                    let dy = _mm256_loadu_pd(grad_in_buf.as_ptr());
                    let mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ);
                    let grad = _mm256_and_pd(dy, mask);

                    let mut out_buf = [0.0; LANES];
                    _mm256_storeu_pd(out_buf.as_mut_ptr(), grad);
                    g_out.copy_from_slice(&out_buf[..in_chunk.len()]);
                });
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2")))]
        {
            grad.par_iter_mut()
                .zip(input_data.par_iter())
                .zip(grad_output.data.par_iter())
                .for_each(|((g, &x), &dy)| {
                    *g = if x > 0.0 { dy } else { 0.0 };
                });
        }

        Tensor::new(shape.clone(), grad)
    };

    (out, Box::new(back))
}

/// Performs one step of stochastic gradient descent (SGD) on the given parameter tensor.
///
/// # Formula
/// $$ w := w - \\text{lr} \\cdot \\frac{\\partial L}{\\partial w} $$
///
/// # Behavior
/// - Updates `w.value` in-place
/// - Zeros out `w.grad` after update (gradient reset step)
///
/// # Arguments
/// - `w`: Tensor with gradient to be updated
/// - `lr`: Learning rate (step size)
///
/// # Example
/// ```rust
/// use briny_ai::backprop::sgd;
/// use briny_ai::tensor;
/// use briny_ai::tensors::WithGrad;
/// 
/// let mut weights = WithGrad::new(tensor!([3.0, 5.0, 4.0]));
/// sgd(&mut weights, 0.01);
/// ```
pub fn sgd(w: &mut WithGrad<Ten64>, lr: f64) {
    for (param, grad) in w.value.data.iter_mut().zip(&w.grad.data) {
        *param -= lr * *grad;
    }
    for grad in &mut w.grad.data {
        *grad = 0.0;
    }
}

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
//! - `relu`: `ReLU` activation with forward and backward pass
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

use crate::nn::TensorFloat;

mod adam;
pub use self::adam::adam;

mod binary_cross_entropy_loss;
pub use self::binary_cross_entropy_loss::binary_cross_entropy_loss;

mod cross_entropy_loss;
pub use self::cross_entropy_loss::cross_entropy_loss;

mod matmul;
pub use self::matmul::matmul;

mod mse_loss;
pub use self::mse_loss::mse_loss;

mod relu;
pub use self::relu::relu;

mod sgd;
pub use self::sgd::sgd;

mod softmax;
pub use self::softmax::softmax;

mod sigmoid;
pub use self::sigmoid::sigmoid;

mod tanh;
pub use self::tanh::tanh;

mod swish_silu;
pub use self::swish_silu::swish;

mod gelu;
pub use self::gelu::gelu;

#[inline]
fn exp(x: TensorFloat) -> TensorFloat {
    libm::expf(x)
}

#[inline]
fn pow(x: TensorFloat, y: TensorFloat) -> TensorFloat {
    libm::powf(x, y)
}

#[inline]
fn ln(x: TensorFloat) -> TensorFloat {
    libm::logf(x)
}

#[inline]
fn sqrt(x: TensorFloat) -> TensorFloat {
    libm::sqrtf(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::tensors::{Tensor, WithGrad};

    use crate::nn::TensorFloat;
    use tensor_optim::TensorOps;

    #[test]
    fn matmul_forward_and_backward_produces_correct_shapes_and_values() {
        // Setup input tensors
        let a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b_data = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

        let a = WithGrad::new(Tensor::new(&[2, 3], &a_data));
        let b = WithGrad::new(Tensor::new(&[3, 2], &b_data));

        let (out, back) = matmul(&a, &b);

        // Output shape should be [2, 2]
        assert_eq!(out.shape(), &[2, 2]);

        // Forward matmul result correctness
        let expected = [
            1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0,
            1.0 * 8.0 + 2.0 * 10.0 + 3.0 * 12.0,
            4.0 * 7.0 + 5.0 * 9.0 + 6.0 * 11.0,
            4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0,
        ];
        assert_eq!(out.data(), &expected);

        // Backprop gradient tensor (same shape as output)
        let grad_output = Tensor::new(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        #[cfg(not(feature = "alloc"))]
        let (grad_a, grad_b) = back.call(grad_output);
        #[cfg(feature = "alloc")]
        let (grad_a, grad_b) = back(grad_output);

        // grad_a shape should be [2, 3], grad_b shape should be [3, 2]
        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_b.shape(), &[3, 2]);

        // Validate some sanity of gradient values (not exact check)
        assert!(grad_a.data().iter().all(|x: &TensorFloat| x.is_finite()));
        assert!(grad_b.data().iter().all(|x: &TensorFloat| x.is_finite()));
    }

    #[test]
    #[should_panic(expected = "inner dimensions must match for matmul")] // was "matmul dimension mismatch"
    fn matmul_panics_on_invalid_shape() {
        let a = WithGrad::new(Tensor::new(&[2, 3], &[1.0; 6]));
        let b = WithGrad::new(Tensor::new(&[4, 2], &[1.0; 8]));
        #[cfg(feature = "dyntensor")]
        let _ = matmul(&a, &b);
        #[cfg(not(feature = "dyntensor"))]
        let _ = matmul::<6, 8, 4, 2>(&a, &b);
    }

    #[test]
    #[cfg(all(feature = "alloc", not(miri)))] // miri makes loss inaccurate
    fn mse_loss_forward_and_backward_matches_expected() {
        let pred_data = [1.0, 2.0, 3.0];
        let target_data = [1.0, 3.0, 2.0];

        let prediction = WithGrad::new(Tensor::new(&[3], &pred_data));
        let target = Tensor::new(&[3], &target_data);

        let (loss, back) = mse_loss(&prediction, &target);

        // compute expected loss manually
        let expected_loss = ((1.0 as TensorFloat - 1.0).powi(2)
            + (2.0 as TensorFloat - 3.0).powi(2)
            + (3.0 as TensorFloat - 2.0).powi(2))
            / 3.0;
        assert!((loss - expected_loss).abs() < 1e-12);

        // compute gradient tensor
        #[cfg(feature = "alloc")]
        let grad = back(1.0);
        #[cfg(not(feature = "alloc"))]
        let grad = back.call(1.0);

        // expected gradients: 2*(y - t)/n
        let expected_grad: alloc::vec::Vec<_> = pred_data
            .iter()
            .zip(target_data.iter())
            .map(|(&y, &t)| 2.0 * (y - t) / 3.0)
            .collect();

        assert_eq!(grad.shape(), &[3]);
        grad.data()
            .iter()
            .zip(expected_grad.iter())
            .for_each(|(&g, &e)| assert!((g - e).abs() < 1e-12));
    }

    #[test]
    fn relu_forward_and_backward() {
        let input_data = [-1.0, 0.0, 1.0, 2.0];
        let input = WithGrad::new(Tensor::new(&[4], &input_data));

        let (out, back) = relu(&input);

        // forward output is max(0,x)
        let expected_out = [0.0, 0.0, 1.0, 2.0];
        assert_eq!(out.data(), &expected_out);

        // provide upstream gradient all ones
        let grad_output = Tensor::new(&[4], &[1.0; 4]);
        #[cfg(feature = "alloc")]
        let grad_input = back(grad_output);
        #[cfg(not(feature = "alloc"))]
        let grad_input = back.call(grad_output);

        // backward grad is upstream grad if input > 0 else 0
        let expected_grad = [0.0, 0.0, 1.0, 1.0];
        assert_eq!(grad_input.data(), &expected_grad);
    }

    #[test]
    fn sgd_updates_parameters_and_zeros_gradients() {
        let init_params = [10.0, 20.0, 30.0];
        let mut wg = WithGrad::new(Tensor::new(&[3], &init_params));

        // setup gradients manually
        wg.set_grad(Tensor::new(&[3], &[1.0, 2.0, 3.0]));

        // apply SGD with learning rate 0.1
        sgd(&mut wg, 0.1);

        // parameters updated: param_i = param_i - lr * grad_i
        let expected_params = [10.0 - 0.1 * 1.0, 20.0 - 0.1 * 2.0, 30.0 - 0.1 * 3.0];
        assert_eq!(wg.get_value().data(), &expected_params);

        // gradients zeroed out
        assert!(wg.get_grad().data().iter().all(|&g| g == 0.0));
    }
}

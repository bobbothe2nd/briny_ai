//! Differentiable operations and autograd utilities.
//! 
//! # Backpropagation and Optimization Primitives
//!
//! Provides core operations with built-in autograd support for training neural networks and other models.
//!
//! **Key Features:**
//! - **Elementwise Activation (ReLU):** Zero-out negatives and propagate gradients accordingly.
//! - **Matrix Multiplication:** Naïve m×k · k×n implementation with gradient closures.
//! - **Loss Computation (MSE):** Mean Squared Error with gradient generator.
//! - **Optimizer (SGD):** In-place parameter update with gradient reset.
//!
//! ## Autograd Pattern
//!
//! Each operation follows a simple pattern:
//! 1. **Inputs** are references to `WithGrad<Ten64>` for tensor ops.
//! 2. **Forward Pass** computes an output `Ten64`.
//! 3. **Backward Pass** returns a closure capturing minimal cloned data to compute gradients.
//! 4. **Gradient Application** uses these results to update `WithGrad` wrappers.
//!
//! ## Usage Guidelines
//!
//! - Operations **panic** on shape mismatches; ensure consistent tensor dimensions.
//! - The backward closures implement `Fn`, allowing multiple invocations if needed.
//! - For performance-critical use, replace loops with optimized BLAS or SIMD kernels.
//! - Extend with broadcasting or GPU support as future enhancements.
//!

use crate::tensors::{Ten64, WithGrad};

/// Applies the ReLU activation (Rectified Linear Unit): `max(0, x)` elementwise.
///
/// # Returns
/// - `out`: Tensor with negatives zeroed.
/// - `back`: Closure mapping `dL/d(out)` to `dL/d(input)` by passing gradients only where input > 0.
///
/// # Example
/// ```rust
/// use briny_ai::tensor;
/// 
/// let input = briny_ai::tensors::WithGrad::new(tensor!([[3.0, 3.0], [9.0, 0.0]]));
/// let grad_out = tensor!([[2.0, 4.0], [6.0, 3.0]]);
/// let (out, back) = briny_ai::backprop::relu(&input);
/// let grad_in = back(&grad_out);
/// ```
/// 
/// # Performance
/// Uses AVX2 if compiled with `simd` feature. Uses Rayon for outer parallelism.
pub fn relu(
    input: &WithGrad<Ten64>,
) -> (Ten64, impl Fn(&Ten64) -> Ten64) {
    crate::ops::dispatch::relu(input)
}

/// Performs matrix multiplication of two 2D tensors: `a` (m×k) · `b` (k×n).
///
/// # Returns
/// - `out`: Product tensor (m×n).
/// - `back`: Closure that given `dL/d(out)` returns `(dL/d(a), dL/d(b))`.
///
/// # Panics
/// Panics if internal dimensions do not match (`a.shape[1] != b.shape[0]`).
///
/// # Performance
/// Uses AVX2 if compiled with `simd` feature. Uses Rayon for outer parallelism.
/// 
/// When compiled with `wgpu` feature, it accelerates matrix multiplication to perform both forward and backward pass on the GPU.
pub fn matmul(
    a: &WithGrad<Ten64>,
    b: &WithGrad<Ten64>,
) -> (Ten64, impl Fn(&Ten64) -> (Ten64, Ten64)) {
    crate::ops::dispatch::matmul(a, b)
}

/// Computes Mean Squared Error (MSE) loss: `mean((prediction - target)^2)`.
///
/// # Returns
/// - Scalar loss value
/// - Closure that maps `dL/dloss` into gradient tensor shape
///
/// # Panics
/// Panics if shapes of `prediction` and `target` differ.
pub fn mse_loss<'a>(
    prediction: &'a WithGrad<Ten64>,
    target: &'a Ten64,
) -> (f64, impl Fn(f64) -> Ten64 + 'a) {
    assert_eq!(prediction.value.shape, target.shape);
    crate::ops::dispatch::mse_loss(prediction, target)
}

/// Performs an in-place Stochastic Gradient Descent (SGD) update.
///
/// Applies: `param = param - learning_rate * gradient` and then zeros gradient.
pub fn sgd(w: &mut WithGrad<Ten64>, lr: f64) {
    crate::ops::dispatch::sgd(w, lr)
}

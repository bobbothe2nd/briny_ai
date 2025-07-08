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
//! 1. **Inputs** are references to `WithGrad<Tensor<f64>>` for tensor ops.
//! 2. **Forward Pass** computes an output `Tensor<f64>`.
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

use crate::tensors::{Tensor, WithGrad};

/// Applies the ReLU activation (Rectified Linear Unit): `max(0, x)` elementwise.
///
/// # Returns
/// - `out`: Tensor with negatives zeroed.
/// - `back`: Closure mapping `dL/d(out)` to `dL/d(input)` by passing gradients only where input > 0.
///
/// # Example
/// ```ignore
/// let (out, back) = relu(&input);
/// let grad_in = back(&grad_out);
/// ```
pub fn relu(
    input: &WithGrad<Tensor<f64>>,
) -> (Tensor<f64>, impl Fn(&Tensor<f64>) -> Tensor<f64>) {
    let shape = input.value.shape.clone();
    let out_data = input
        .value
        .data
        .iter()
        .map(|&x| x.max(0.0))
        .collect();
    let out = Tensor::new(shape.clone(), out_data);

    let back = move |grad_output: &Tensor<f64>| {
        let grad_data = input
            .value
            .data
            .iter()
            .zip(&grad_output.data)
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect();
        Tensor::new(shape.clone(), grad_data)
    };

    (out, back)
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
/// This naive triple-loop may be too slow for large tensors. Consider optimized libraries.
pub fn matmul<'a>(
    a: &'a WithGrad<Tensor<f64>>,
    b: &'a WithGrad<Tensor<f64>>,
) -> (Tensor<f64>, impl Fn(&Tensor<f64>) -> (Tensor<f64>, Tensor<f64>) + 'a) {
    let (m, k1) = (a.value.shape[0], a.value.shape[1]);
    let (k2, n) = (b.value.shape[0], b.value.shape[1]);
    assert_eq!(k1, k2, "matmul shape mismatch");

    let mut out_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for k in 0..k1 {
                out_data[i * n + j] += a.value.data[i * k1 + k] * b.value.data[k * n + j];
            }
        }
    }
    let out = Tensor::new(vec![m, n], out_data);

    // Clone minimal data for backward pass
    let a_shape = a.value.shape.clone();
    let b_shape = b.value.shape.clone();
    let a_data = a.value.data.clone();
    let b_data = b.value.data.clone();

    let back = move |grad_output: &Tensor<f64>| {
        // dL/dA = dL/dOut · B^T
        let mut da = vec![0.0; m * k1];
        for i in 0..m {
            for k in 0..k1 {
                for j in 0..n {
                    da[i * k1 + k] += grad_output.data[i * n + j] * b_data[k * n + j];
                }
            }
        }
        // dL/dB = A^T · dL/dOut
        let mut db = vec![0.0; k1 * n];
        for k in 0..k1 {
            for j in 0..n {
                for i in 0..m {
                    db[k * n + j] += a_data[i * k1 + k] * grad_output.data[i * n + j];
                }
            }
        }
        (
            Tensor::new(a_shape.clone(), da),
            Tensor::new(b_shape.clone(), db),
        )
    };

    (out, back)
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
    prediction: &'a WithGrad<Tensor<f64>>,
    target: &'a Tensor<f64>,
) -> (f64, impl Fn(f64) -> Tensor<f64> + 'a) {
    assert_eq!(prediction.value.shape, target.shape);
    let n = prediction.value.data.len();

    let loss = prediction
        .value
        .data
        .iter()
        .zip(&target.data)
        .map(|(&y, &t)| (y - t).powi(2))
        .sum::<f64>() / n as f64;

    let shape = prediction.value.shape.clone();
    let pred_data = prediction.value.data.clone();
    let tgt_data = target.data.clone();

    let back = move |grad_out: f64| {
        let grad_vec: Vec<f64> = pred_data
            .iter()
            .zip(&tgt_data)
            .map(|(&y, &t)| 2.0 * (y - t) * grad_out / n as f64)
            .collect();
        Tensor::new(shape.clone(), grad_vec)
    };

    (loss, back)
}

/// Performs an in-place Stochastic Gradient Descent (SGD) update.
///
/// Applies: `param = param - learning_rate * gradient` and then zeros gradient.
pub fn sgd(w: &mut WithGrad<Tensor<f64>>, lr: f64) {
    for (param, grad) in w.value.data.iter_mut().zip(&w.grad.data) {
        *param -= lr * *grad;
    }
    for grad in &mut w.grad.data {
        *grad = 0.0;
    }
}

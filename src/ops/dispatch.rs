//! Operation Dispatch Layer
//!
//! This module selects the correct backend (CPU, WGPU, CUDA, etc.) at runtime
//! for each differentiable operation, based on the global `Backend`.
//!
//! Each function attempts backend-specific implementations in priority order:
//! 1. `Cuda` (if enabled)
//! 2. `Wgpu` (if enabled)
//! 3. Falls back to `Cpu`
//!
//! # Design Highlights
//! - **Pluggable**: Backends are optional and modular
//! - **Minimal overhead**: Function returns immediately upon match
//! - **Fallback logic**: Safe and deterministic fallback to CPU
//!
//! # Example
//! ```ignore
//! use crate::tensors::WithGrad;
//! use crate::backprop::matmul;
//!
//! let (out, back) = matmul(&a, &b); // uses GPU if available
//! let (grad_a, grad_b) = back(&grad_out);
//! ```
//! 

use crate::backend::{get_backend, Backend};
use crate::tensors::{WithGrad, Ten64};

/// Dispatches matrix multiplication to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
/// - `Tensor`: Output tensor (m×n)
/// - `Fn`: Closure computing (∂L/∂a, ∂L/∂b)
///
/// # Behavior
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
pub fn matmul(
    a: &WithGrad<Ten64>,
    b: &WithGrad<Ten64>,
) -> (Ten64, Box<dyn Fn(&Ten64) -> (Ten64, Ten64)>) {
    match get_backend() {
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                if let Some(result) = super::cuda::cuda_matmul(a, b) {
                    return result;
                }
            }
        }
        Backend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                if let Some(result) = super::wgpu::wgpu_matmul(a, b) {
                    return result;
                }
            }
        }
        _ => {}
    }

    super::cpu::matmul(a, b)
}

/// Dispatches MSE loss calculation to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
/// - Scalar loss value
/// - Closure that maps `dL/dloss` into gradient tensor shape
///
/// # Behavior
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
pub fn mse_loss<'a>(
    prediction: &'a WithGrad<Ten64>,
    target: &'a Ten64,
) -> (f64, Box<dyn Fn(f64) -> Ten64 + 'a>) {
    match get_backend() {
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                if let Some(result) = super::cuda::cuda_mse_loss(prediction, target) {
                    return result;
                }
            }
        }
        Backend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                if let Some(result) = super::wgpu::wgpu_mse_loss(prediction, target) {
                    return result;
                }
            }
        }
        _ => {}
    }

    super::cpu::mse_loss(prediction, target)
}

/// Dispatches Stochastic Gradient Descent to the selected backend (CPU, WGPU, or CUDA).
///
/// # Behavior
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
pub fn sgd(w: &mut WithGrad<Ten64>, lr: f64) {
    match get_backend() {
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                if super::cuda::cuda_sgd(w, lr) {
                    return;
                }
            }
        }
        Backend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                if super::wgpu::wgpu_sgd(w, lr) {
                    return;
                }
            }
        }
        _ => {}
    }

    super::cpu::sgd(w, lr)
}

/// Dispatches Rectified Learning Unit calculations to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
/// - `out`: Tensor with negatives zeroed.
/// - `back`: Closure mapping `dL/d(out)` to `dL/d(input)` by passing gradients only where input > 0.
///
/// # Behavior
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
pub fn relu(
    input: &WithGrad<Ten64>,
) -> (Ten64, Box<dyn Fn(&Ten64) -> Ten64 + '_>) {
    match get_backend() {
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                if let Some(result) = super::cuda::cuda_relu(input) {
                    return result;
                }
            }
        }
        Backend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                if let Some(result) = super::wgpu::wgpu_relu(input) {
                    return result;
                }
            }
        }
        _ => {}
    }

    super::cpu::relu(input)
}

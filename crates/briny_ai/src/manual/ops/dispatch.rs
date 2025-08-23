//! Operation Dispatch Layer
//!
//! This module selects the correct backend (CPU, WGPU, CUDA, etc.) at runtime
//! for each differentiable operation, based on the global `Backend`.
//!
//! When `alloc` is enabled:
//!
//! Each function attempts backend-specific implementations in priority order:
//! 1. `Cuda` (if enabled)
//! 2. `Wgpu` (if enabled)
//! 3. Falls back to `Cpu`
//!
//! With `alloc` disabled, all that can be done is run operations on the CPU.
//!
//! # Design Highlights
//! - **Pluggable**: Backends are optional and modular
//! - **Minimal overhead**: Function returns immediately upon match
//! - **Fallback logic**: Safe and deterministic fallback to CPU

#[cfg(not(feature = "alloc"))]
use box_closure::{Align32, OpaqueFn};

#[cfg(feature = "alloc")]
use crate::backend::{Backend, get_backend};
use crate::manual::TensorFloat;
use crate::manual::tensors::{Tensor, WithGrad};
#[cfg(feature = "alloc")]
use alloc::boxed::Box;

/// Dispatches matrix multiplication to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
///
/// - `Tensor`: Output tensor (m×n)
/// - `Fn`: Closure computing (∂L/∂a, ∂L/∂b)
///
/// # Behavior
///
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn matmul<'a>(
    a: &'a WithGrad<Tensor<TensorFloat>>,
    b: &'a WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> (Tensor<TensorFloat>, Tensor<TensorFloat>) + 'a>,
) {
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
        Backend::Cpu => {}
    }

    super::cpu::matmul(a, b)
}
/// Dispatches matrix multiplication to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
///
/// - `Tensor`: Output tensor (m×n)
/// - `Fn`: Closure computing (∂L/∂a, ∂L/∂b)
///
/// # Behavior
///
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn matmul<'a, const A: usize, const B: usize, const OUT: usize>(
    a: &'a WithGrad<Tensor<TensorFloat, A, 2>>,
    b: &'a WithGrad<Tensor<TensorFloat, B, 2>>,
) -> (
    Tensor<TensorFloat, OUT, 2>,
    Box<
        dyn Fn(
            Tensor<TensorFloat, OUT, 2>,
        ) -> (Tensor<TensorFloat, A, 2>, Tensor<TensorFloat, B, 2>),
    >,
) {
    #[cfg(any(feature = "wgpu", feature = "cuda"))]
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
        Backend::Cpu => {}
    }

    super::cpu::matmul(a, b)
}
/// Dispatches matrix multiplication to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
///
/// - `Tensor`: Output tensor (m×n)
/// - `Fn`: Closure computing (∂L/∂a, ∂L/∂b)
///
/// # Behavior
///
/// CUDA + WGPU backends aren't available without `alloc`,
/// so the dispatch really just has a tail call to the CPU backend.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn matmul<'a, const A: usize, const B: usize, const OUT: usize>(
    a: &'a WithGrad<Tensor<TensorFloat, A, 2>>,
    b: &'a WithGrad<Tensor<TensorFloat, B, 2>>,
) -> (
    Tensor<TensorFloat, OUT, 2>,
    OpaqueFn<
        'a,
        Tensor<TensorFloat, OUT, 2>,
        (Tensor<TensorFloat, A, 2>, Tensor<TensorFloat, B, 2>),
        Align32<256>,
    >,
) {
    super::cpu::matmul(a, b)
}

/// Dispatches MSE loss calculation to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
///
/// - Scalar loss value
/// - Closure that maps `dL/dloss` into gradient tensor shape
///
/// # Behavior
///
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
#[must_use]
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn mse_loss<'a>(
    prediction: &'a WithGrad<Tensor<TensorFloat>>,
    target: &'a Tensor<TensorFloat>,
) -> (
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat> + 'a>,
) {
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
        Backend::Cpu => {}
    }

    super::cpu::mse_loss(prediction, target)
}
/// Dispatches MSE loss calculation to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
///
/// - Scalar loss value
/// - Closure that maps `dL/dloss` into gradient tensor shape
///
/// # Behavior
///
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn mse_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> (
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat, N, D> + 'a>,
) {
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
        Backend::Cpu => {}
    }

    super::cpu::mse_loss(prediction, target)
}
/// Dispatches MSE loss calculation to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
///
/// - Scalar loss value
/// - Closure that maps `dL/dloss` into gradient tensor shape
///
/// # Behavior
///
/// CUDA + WGPU backends aren't available without `alloc`,
/// so the dispatch really just has a tail call to the CPU backend.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn mse_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> (
    TensorFloat,
    OpaqueFn<'a, TensorFloat, Tensor<TensorFloat, N, D>, Align32<256>>,
) {
    super::cpu::mse_loss(prediction, target)
}

/// Dispatches Stochastic Gradient Descent to the selected backend (CPU, WGPU, or CUDA).
///
/// # Behavior
///
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn sgd(w: &mut WithGrad<Tensor<TensorFloat>>, lr: TensorFloat) {
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
        Backend::Cpu => {}
    }

    super::cpu::sgd(w, lr);
}
/// Dispatches Stochastic Gradient Descent to the selected backend (CPU, WGPU, or CUDA).
///
/// # Behavior
///
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn sgd<const N: usize, const D: usize>(
    w: &mut WithGrad<Tensor<TensorFloat, N, D>>,
    lr: TensorFloat,
) {
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
        Backend::Cpu => {}
    }

    super::cpu::sgd(w, lr);
}
/// Dispatches Stochastic Gradient Descent to the selected backend (CPU, WGPU, or CUDA).
///
/// # Behavior
///
/// CUDA + WGPU backends aren't available without `alloc`,
/// so the dispatch really just has a tail call to the CPU backend.
#[cfg(not(feature = "alloc"))]
pub fn sgd<const N: usize, const D: usize>(
    w: &mut WithGrad<Tensor<TensorFloat, N, D>>,
    lr: TensorFloat,
) {
    super::cpu::sgd(w, lr);
}

/// Dispatches Rectified Learning Unit calculations to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
///
/// - `out`: Tensor with negatives zeroed.
/// - `back`: Closure mapping `dL/d(out)` to `dL/d(input)` by passing gradients only where input > 0.
///
/// # Behavior
///
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
#[must_use]
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn relu(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
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
        Backend::Cpu => {}
    }

    super::cpu::relu(input)
}
/// Dispatches Rectified Learning Unit calculations to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
///
/// - `out`: Tensor with negatives zeroed.
/// - `back`: Closure mapping `dL/d(out)` to `dL/d(input)` by passing gradients only where input > 0.
///
/// # Behavior
///
/// Attempts CUDA → WGPU → CPU, depending on availability and features.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn relu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    Box<dyn Fn(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
) {
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
        Backend::Cpu => {}
    }

    super::cpu::relu(input)
}
/// Dispatches Rectified Learning Unit calculations to the selected backend (CPU, WGPU, or CUDA).
///
/// # Returns
///
/// - `out`: Tensor with negatives zeroed.
/// - `back`: Closure mapping `dL/d(out)` to `dL/d(input)` by passing gradients only where input > 0.
///
/// # Behavior
///
/// CUDA + WGPU backends aren't available without `alloc`,
/// so the dispatch really just has a tail call to the CPU backend.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn relu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    OpaqueFn<'_, Tensor<TensorFloat, N, D>, Tensor<TensorFloat, N, D>, Align32<256>>,
) {
    super::cpu::relu(input)
}

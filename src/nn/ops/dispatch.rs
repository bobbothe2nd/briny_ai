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

#![allow(clippy::type_complexity)]

#[cfg(not(feature = "alloc"))]
use box_closure::{Align8, OpaqueFn, OpaqueFnOnce};

#[cfg(all(feature = "alloc", any(feature = "wgpu", feature = "cuda")))]
use crate::backend::{get_backend, Backend};
use crate::nn::tensors::{Tensor, WithGrad};
use crate::nn::TensorFloat;
#[cfg(feature = "alloc")]
use alloc::boxed::Box;

/// Performs one step of Adam optimization on the given parameter tensor.
///
/// # Arguments
///
/// - `w`: Tensor with gradient to be updated
/// - `m`: First moment estimate (same shape as `w`)
/// - `v`: Second moment estimate (same shape as `w`)
/// - `t`: Current timestep (1-based)
/// - `lr`: Learning rate
///
/// # Hyperparameters (hardcoded)
///
/// - beta1 = 0.9
/// - beta2 = 0.999
/// - eps = 1e-8
#[cfg(feature = "dyntensor")]
pub fn adam(
    w: &mut WithGrad<Tensor<TensorFloat>>,
    m: &mut Tensor<TensorFloat>,
    v: &mut Tensor<TensorFloat>,
    t: TensorFloat,
    lr: TensorFloat,
) {
    super::cpu::adam(w, m, v, t, lr);
}

/// Performs one step of Adam optimization on the given parameter tensor.
///
/// # Arguments
///
/// - `w`: Tensor with gradient to be updated
/// - `m`: First moment estimate (same shape as `w`)
/// - `v`: Second moment estimate (same shape as `w`)
/// - `t`: Current timestep (1-based)
/// - `lr`: Learning rate
///
/// # Hyperparameters (hardcoded)
///
/// - beta1 = 0.9
/// - beta2 = 0.999
/// - eps = 1e-8
#[cfg(not(feature = "dyntensor"))]
pub fn adam<const N: usize, const D: usize>(
    w: &mut WithGrad<Tensor<TensorFloat, N, D>>,
    m: &mut Tensor<TensorFloat, N, D>,
    v: &mut Tensor<TensorFloat, N, D>,
    t: TensorFloat,
    lr: TensorFloat,
) {
    super::cpu::adam(w, m, v, t, lr);
}

/// BCE loss for identifying one of two classes.
#[must_use]
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn binary_cross_entropy_loss<'a>(
    prediction: &'a WithGrad<Tensor<TensorFloat>>,
    target: &'a Tensor<TensorFloat>,
) -> (
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat> + 'a>,
) {
    super::cpu::binary_cross_entropy_loss(prediction, target)
}
/// BCE loss for identifying one of two classes.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn binary_cross_entropy_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> (
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat, N, D> + 'a>,
) {
    super::cpu::binary_cross_entropy_loss(prediction, target)
}
/// BCE loss for identifying one of two classes.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn binary_cross_entropy_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> (
    TensorFloat,
    OpaqueFn<'a, TensorFloat, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    super::cpu::binary_cross_entropy_loss(prediction, target)
}

/// Performs the cross entropy loss function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn cross_entropy_loss<'a>(
    prediction: &'a WithGrad<Tensor<TensorFloat>>,
    target: &'a Tensor<TensorFloat>,
) -> (
    TensorFloat,
    Box<dyn FnOnce(TensorFloat) -> Tensor<TensorFloat> + 'a>,
) {
    super::cpu::cross_entropy_loss(prediction, target)
}
/// Performs the cross entropy loss function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn cross_entropy_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> (
    TensorFloat,
    Box<dyn FnOnce(TensorFloat) -> Tensor<TensorFloat, N, D> + 'a>,
) {
    super::cpu::cross_entropy_loss(prediction, target)
}
/// Performs the cross entropy loss function, dispatching it among different backends.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn cross_entropy_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> (
    TensorFloat,
    OpaqueFnOnce<'a, TensorFloat, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    super::cpu::cross_entropy_loss(prediction, target)
}

/// Performs matrix multiplication, dispatching it among different backends.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn matmul<'a>(
    a: &'a WithGrad<Tensor<TensorFloat>>,
    b: &'a WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn FnOnce(Tensor<TensorFloat>) -> (Tensor<TensorFloat>, Tensor<TensorFloat>) + 'a>,
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
/// Performs matrix multiplication, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn matmul<'a, const A: usize, const B: usize, const OUT: usize, const D: usize>(
    a: &'a WithGrad<Tensor<TensorFloat, A, D>>,
    b: &'a WithGrad<Tensor<TensorFloat, B, D>>,
) -> (
    Tensor<TensorFloat, OUT, D>,
    Box<
        dyn FnOnce(
                Tensor<TensorFloat, OUT, D>,
            ) -> (Tensor<TensorFloat, A, D>, Tensor<TensorFloat, B, D>)
            + 'a,
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
/// Performs matrix multiplication, dispatching it among different backends.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn matmul<'a, const A: usize, const B: usize, const OUT: usize, const D: usize>(
    a: &'a WithGrad<Tensor<TensorFloat, A, D>>,
    b: &'a WithGrad<Tensor<TensorFloat, B, D>>,
) -> (
    Tensor<TensorFloat, OUT, D>,
    OpaqueFnOnce<
        'a,
        Tensor<TensorFloat, OUT, D>,
        (Tensor<TensorFloat, A, D>, Tensor<TensorFloat, B, D>),
        Align8<128>,
    >,
) {
    super::cpu::matmul(a, b)
}

/// Performs the MSE loss function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn mse_loss<'a>(
    prediction: &'a WithGrad<Tensor<TensorFloat>>,
    target: &'a Tensor<TensorFloat>,
) -> (
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat> + 'a>,
) {
    super::cpu::mse_loss(prediction, target)
}
/// Performs the MSE loss function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn mse_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> (
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat, N, D> + 'a>,
) {
    super::cpu::mse_loss(prediction, target)
}
/// Performs the MSE loss function, dispatching it among different backends.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn mse_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> (
    TensorFloat,
    OpaqueFn<'a, TensorFloat, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    super::cpu::mse_loss(prediction, target)
}

/// Performs the SGD function, dispatching it among different backends.
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn sgd(w: &mut WithGrad<Tensor<TensorFloat>>, lr: TensorFloat) {
    super::cpu::sgd(w, lr);
}
/// Performs the SGD function, dispatching it among different backends.
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn sgd<const N: usize, const D: usize>(
    w: &mut WithGrad<Tensor<TensorFloat, N, D>>,
    lr: TensorFloat,
) {
    super::cpu::sgd(w, lr);
}
/// Performs the SGD function, dispatching it among different backends.
#[cfg(not(feature = "alloc"))]
pub fn sgd<const N: usize, const D: usize>(
    w: &mut WithGrad<Tensor<TensorFloat, N, D>>,
    lr: TensorFloat,
) {
    super::cpu::sgd(w, lr);
}

/// Performs the `ReLU` activation function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn relu(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn FnOnce(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    super::cpu::relu(input)
}
/// Performs the `ReLU` activation function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn relu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    Box<dyn FnOnce(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
) {
    super::cpu::relu(input)
}
/// Performs the `ReLU` activation function, dispatching it among different backends.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn relu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    OpaqueFnOnce<'_, Tensor<TensorFloat, N, D>, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    super::cpu::relu(input)
}

/// Performs the softmax function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn softmax(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn FnOnce(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    super::cpu::softmax(input)
}
/// Performs the softmax function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn softmax<const N1: usize, const N2: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N1, D>>,
) -> (
    Tensor<TensorFloat, N2, D>,
    Box<dyn FnOnce(Tensor<TensorFloat, N2, D>) -> Tensor<TensorFloat, N1, D> + '_>,
) {
    super::cpu::softmax(input)
}
/// Performs the softmax function, dispatching it among different backends.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn softmax<const N1: usize, const N2: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N1, D>>,
) -> (
    Tensor<TensorFloat, N2, D>,
    OpaqueFnOnce<'_, Tensor<TensorFloat, N2, D>, Tensor<TensorFloat, N1, D>, Align8<128>>,
) {
    super::cpu::softmax(input)
}

/// Performs the sigmoid activation function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn sigmoid(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn FnOnce(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    super::cpu::sigmoid(input)
}
/// Performs the sigmoid activation function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn sigmoid<const N1: usize, const N2: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N1, D>>,
) -> (
    Tensor<TensorFloat, N2, D>,
    Box<dyn FnOnce(Tensor<TensorFloat, N2, D>) -> Tensor<TensorFloat, N1, D> + '_>,
) {
    super::cpu::sigmoid(input)
}
/// Performs the sigmoid activation function, dispatching it among different backends.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn sigmoid<const N1: usize, const N2: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N1, D>>,
) -> (
    Tensor<TensorFloat, N2, D>,
    OpaqueFnOnce<'_, Tensor<TensorFloat, N2, D>, Tensor<TensorFloat, N1, D>, Align8<128>>,
) {
    super::cpu::sigmoid(input)
}

/// Performs the GELU activation function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn gelu(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<[TensorFloat]>,
    Box<dyn FnOnce(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    super::cpu::gelu(input)
}
/// Performs the GELU activation function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn gelu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    Box<[TensorFloat; N]>,
    Box<dyn FnOnce(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
) {
    super::cpu::gelu(input)
}
/// Performs the GELU activation function, dispatching it among different backends.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn gelu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    [TensorFloat; N],
    OpaqueFnOnce<'_, Tensor<TensorFloat, N, D>, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    super::cpu::gelu(input)
}

/// Performs the Swish (`SiLU`) activation function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn swish(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<[TensorFloat]>,
    Box<dyn FnOnce(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    super::cpu::swish(input)
}
/// Performs the Swish (`SiLU`) activation function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn swish<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    Box<[TensorFloat; N]>,
    Box<dyn FnOnce(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
) {
    super::cpu::swish(input)
}
/// Performs the Swish (`SiLU`) activation function, dispatching it among different backends.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn swish<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    [TensorFloat; N],
    OpaqueFnOnce<'_, Tensor<TensorFloat, N, D>, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    super::cpu::swish(input)
}

/// Performs the `tanh` activation function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", feature = "dyntensor"))]
pub fn tanh(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn FnOnce(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    super::cpu::tanh(input)
}
/// Performs the `tanh` activation function, dispatching it among different backends.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn tanh<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    Box<dyn FnOnce(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
) {
    super::cpu::tanh(input)
}
/// Performs the `tanh` activation function, dispatching it among different backends.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn tanh<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    OpaqueFnOnce<'_, Tensor<TensorFloat, N, D>, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    super::cpu::tanh(input)
}

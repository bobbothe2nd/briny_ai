//! Ultra fast Nvidia-only CUDA acceleration.

use crate::manual::TensorFloat;
use crate::manual::tensors::{Tensor, WithGrad};
use alloc::boxed::Box;

/// Performs matrix multiplication between two tensors, falling back to WGPU.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn cuda_matmul<'a>(
    a: &WithGrad<Tensor<TensorFloat>>,
    b: &WithGrad<Tensor<TensorFloat>>,
) -> Option<(
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> (Tensor<TensorFloat>, Tensor<TensorFloat>) + 'a>,
)> {
    // TODO: implement using `cust` crate
    super::wgpu::wgpu_matmul(a, b) // wgpu fallback
}
/// Performs matrix multiplication between two tensors, falling back to WGPU.
#[must_use]
#[cfg(not(feature = "dyntensor"))]
pub fn cuda_matmul<'a, const A: usize, const B: usize, const OUT: usize>(
    a: &'a WithGrad<Tensor<TensorFloat, A, 2>>,
    b: &'a WithGrad<Tensor<TensorFloat, B, 2>>,
) -> Option<(
    Tensor<TensorFloat, OUT, 2>,
    Box<
        dyn Fn(
            Tensor<TensorFloat, OUT, 2>,
        ) -> (Tensor<TensorFloat, A, 2>, Tensor<TensorFloat, B, 2>),
    >,
)> {
    // TODO: implement using `cust` crate
    super::wgpu::wgpu_matmul(a, b) // wgpu fallback
}

/// Calculates MSE loss, falling back to WGPU.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn cuda_mse_loss<'a>(
    prediction: &'a WithGrad<Tensor<TensorFloat>>,
    target: &'a Tensor<TensorFloat>,
) -> Option<(
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat> + 'a>,
)> {
    // TODO: implement using CUDA kernel
    super::wgpu::wgpu_mse_loss(prediction, target) // wgpu fallback
}
/// Calculates MSE loss, falling back to WGPU.
#[must_use]
#[cfg(not(feature = "dyntensor"))]
pub fn cuda_mse_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> Option<(
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat, N, D> + 'a>,
)> {
    // TODO: implement using CUDA kernel
    super::wgpu::wgpu_mse_loss(prediction, target) // wgpu fallback
}

/// Calculates `ReLU`, falling back to WGPU.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn cuda_relu(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> Option<(
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
)> {
    // TODO: implement CUDA ReLU
    super::wgpu::wgpu_relu(input) // wgpu fallback
}
/// Calculates `ReLU`, falling back to WGPU.
#[must_use]
#[cfg(not(feature = "dyntensor"))]
pub fn cuda_relu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> Option<(
    Tensor<TensorFloat, N, D>,
    Box<dyn Fn(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
)> {
    // TODO: implement CUDA ReLU
    super::wgpu::wgpu_relu(input) // wgpu fallback
}

/// Performs optimizer SGD, falling back to WGPU.
#[cfg(feature = "dyntensor")]
pub fn cuda_sgd(w: &mut WithGrad<Tensor<TensorFloat>>, lr: TensorFloat) -> bool {
    // TODO: implement with CUDA gradient application
    super::wgpu::wgpu_sgd(w, lr) // wgpu fallback
}
/// Performs optimizer SGD, falling back to WGPU.
#[cfg(not(feature = "dyntensor"))]
pub fn cuda_sgd<const N: usize, const D: usize>(
    w: &mut WithGrad<Tensor<TensorFloat, N, D>>,
    lr: TensorFloat,
) -> bool {
    // TODO: implement with CUDA gradient application
    super::wgpu::wgpu_sgd(w, lr) // wgpu fallback
}

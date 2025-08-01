//! Ultra fast Nvidia-only CUDA acceleration.

use crate::tensors::{WithGrad, Ten64};
use crate::ops::dispatch::{FnToDoubleTen64, FnF64Ten64, FnTen64To};

/// Performs matrix multiplication between two tensors, falling back to WGPU.
pub fn cuda_matmul(
    a: &WithGrad<Ten64>,
    b: &WithGrad<Ten64>,
) -> Option<(Ten64, Box<FnToDoubleTen64>)> {
    // TODO: implement using `cust` crate
    super::wgpu::wgpu_matmul(a, b) // wgpu fallback
}

/// Calculates MSE loss, falling back to WGPU.
pub fn cuda_mse_loss<'a>(
    prediction: &'a WithGrad<Ten64>,
    target: &'a Ten64,
) -> Option<(f64, Box<FnF64Ten64<'a>>)> {
    // TODO: implement using GPU kernel
    super::wgpu::wgpu_mse_loss(prediction, target) // wgpu fallback
}

/// Calculates ReLU, falling back to WGPU.
pub fn cuda_relu(
    input: &WithGrad<Ten64>,
) -> Option<(Ten64, Box<FnTen64To>)> {
    // TODO: implement GPU ReLU
    super::wgpu::wgpu_relu(input) // wgpu fallback
}

/// Performs optimizer SGD, falling back to WGPU.
pub fn cuda_sgd(
    w: &mut WithGrad<Ten64>,
    lr: f64,
) -> bool {
    // TODO: implement with GPU gradient application
    super::wgpu::wgpu_sgd(w, lr) // wgpu fallback
}

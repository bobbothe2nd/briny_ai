use crate::tensors::{WithGrad, Ten64};

pub fn cuda_matmul(
    a: &WithGrad<Ten64>,
    b: &WithGrad<Ten64>,
) -> Option<(
    Ten64,
    Box<dyn Fn(&Ten64) -> (Ten64, Ten64)>,
)> {
    // TODO: implement using `cust` crate
    super::wgpu::wgpu_matmul(a, b) // wgpu fallback
}

pub fn cuda_mse_loss<'a>(
    prediction: &'a WithGrad<Ten64>,
    target: &'a Ten64,
) -> Option<(f64, Box<dyn Fn(f64) -> Ten64 + 'a>)> {
    // TODO: implement using GPU kernel
    super::wgpu::wgpu_mse_loss(prediction, target) // wgpu fallback
}

pub fn cuda_relu(
    input: &WithGrad<Ten64>,
) -> Option<(Ten64, Box<dyn Fn(&Ten64) -> Ten64 + '_>)> {
    // TODO: implement GPU ReLU
    super::wgpu::wgpu_relu(input) // wgpu fallback
}

pub fn cuda_sgd(
    w: &mut WithGrad<Ten64>,
    lr: f64,
) -> bool {
    // TODO: implement with GPU gradient application
    super::wgpu::wgpu_sgd(w, lr) // wgpu fallback
}

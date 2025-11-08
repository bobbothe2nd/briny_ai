use crate::nn::tensors::{Tensor, WithGrad};
use crate::nn::TensorFloat;
use tensor_optim::TensorOps;

/// Performs one step of stochastic gradient descent (SGD) on the given parameter tensor.
///
/// # Formula
///
/// $$ w := w - \\text{lr} \\cdot \\frac{\\partial L}{\\partial w} $$
///
/// # Behavior
///
/// - Updates `w.value` in-place
/// - Zeros out `w.grad` after update (gradient reset step)
///
/// # Arguments
///
/// - `w`: Tensor with gradient to be updated
/// - `lr`: Learning rate (step size)
#[cfg(feature = "dyntensor")]
pub fn sgd(w: &mut WithGrad<Tensor<TensorFloat>>, lr: TensorFloat) {
    let (params, grads) = w.split_mut();
    let params_data = params.data_mut();
    let grads_data = grads.data();

    for (param, grad) in params_data.iter_mut().zip(grads_data.iter()) {
        *param -= lr * *grad;
    }

    for grad in grads.data_mut() {
        *grad = 0.0;
    }
}

/// Performs one step of stochastic gradient descent (SGD) on the given parameter tensor.
///
/// # Formula
///
/// $$ w := w - \\text{lr} \\cdot \\frac{\\partial L}{\\partial w} $$
///
/// # Behavior
///
/// - Updates `w.value` in-place
/// - Zeros out `w.grad` after update (gradient reset step)
///
/// # Arguments
///
/// - `w`: Tensor with gradient to be updated
/// - `lr`: Learning rate (step size)
#[cfg(not(feature = "dyntensor"))]
pub fn sgd<const N: usize, const D: usize>(
    w: &mut WithGrad<Tensor<TensorFloat, N, D>>,
    lr: TensorFloat,
) {
    let (params, grads) = w.split_mut();
    let params_data = params.data_mut();
    let grads_data = grads.data();

    for (param, grad) in params_data.iter_mut().zip(grads_data.iter()) {
        *param -= lr * *grad;
    }

    for grad in grads.data_mut() {
        *grad = 0.0;
    }
}

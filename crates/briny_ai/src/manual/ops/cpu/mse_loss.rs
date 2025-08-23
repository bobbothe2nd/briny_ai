use crate::manual::{
    TensorFloat,
    tensors::{Tensor, WithGrad},
};
use tensor_optim::TensorOps;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "dyntensor")]
use alloc::vec::Vec;
#[cfg(not(feature = "alloc"))]
use box_closure::{Align32, OpaqueFn};

/// Computes the mean squared error (MSE) between predictions and targets,
/// returning both the scalar loss and a gradient function.
///
/// # Formula
///
/// `$$ L = \\frac{1}{n} \\sum_i (y_i - t_i)^2 $$`
///
/// # Returns
///
/// - Scalar loss `T`
/// - Backward function mapping upstream scalar gradient `dL` to a tensor of shape `prediction`
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn mse_loss<'a>(
    prediction: &'a WithGrad<Tensor<TensorFloat>>,
    target: &'a Tensor<TensorFloat>,
) -> (
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat> + 'a>,
) {
    #[allow(clippy::cast_precision_loss)]
    let n = prediction.get_value().data().len() as TensorFloat;

    let loss = prediction
        .get_value()
        .data()
        .iter()
        .zip(target.data())
        .map(|(&y, &t)| {
            let diff = y - t;
            diff * diff
        })
        .sum::<TensorFloat>()
        / n;

    let shape = prediction.get_value().shape();
    let pred_data = prediction.get_value().data();
    let target_data = target.data();

    // parallel backward pass
    let back = move |grad_output: TensorFloat| {
        let grad: Vec<TensorFloat> = pred_data
            .iter()
            .zip(target_data)
            .map(|(&y, &t)| 2.0 * (y - t) * grad_output / n)
            .collect();

        Tensor::new(shape, &grad)
    };

    (loss, Box::new(back))
}
/// Computes the mean squared error (MSE) between predictions and targets,
/// returning both the scalar loss and a gradient function.
///
/// # Formula
///
/// `$$ L = \\frac{1}{n} \\sum_i (y_i - t_i)^2 $$`
///
/// # Returns
///
/// - Scalar loss `T`
/// - Backward function mapping upstream scalar gradient `dL` to a tensor of shape `prediction`
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
#[must_use]
pub fn mse_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> (
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat, N, D> + 'a>,
) {
    use tensor_optim::ConstTensorOps;

    #[allow(clippy::cast_precision_loss)]
    let n = N as TensorFloat;

    let pred_data = prediction.get_value().data();
    let target_data = target.data();

    // forward loss
    let loss = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&y, &t)| {
            let diff = y - t;
            diff * diff
        })
        .sum::<TensorFloat>()
        / n;

    // convert shape slice to fixed array reference
    let shape: &[usize; D] = prediction.get_value().shape_array();

    // backward closure
    let back = move |grad_output: TensorFloat| {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = 2.0 * (pred_data[i] - target_data[i]) * grad_output / n;
        }
        Tensor::new(shape, &grad)
    };

    (loss, Box::new(back))
}
/// Computes the mean squared error (MSE) between predictions and targets,
/// returning both the scalar loss and a gradient function.
///
/// # Formula
///
/// `$$ L = \\frac{1}{n} \\sum_i (y_i - t_i)^2 $$`
///
/// # Returns
///
/// - Scalar loss `T`
/// - Backward function mapping upstream scalar gradient `dL` to a tensor of shape `prediction`
#[cfg(not(feature = "alloc"))]
#[must_use]
pub fn mse_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> (
    TensorFloat,
    OpaqueFn<'a, TensorFloat, Tensor<TensorFloat, N, D>, Align32<256>>,
) {
    use tensor_optim::ConstTensorOps;

    #[allow(clippy::cast_precision_loss)]
    let n = N as TensorFloat;

    let pred_data = prediction.get_value().data();
    let target_data = target.data();

    // forward loss
    let loss = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&y, &t)| {
            let diff = y - t;
            diff * diff
        })
        .sum::<TensorFloat>()
        / n;

    // convert shape slice to fixed array reference
    let shape: &[usize; D] = prediction.get_value().shape_array();

    // backward closure
    let back = move |grad_output: TensorFloat| {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = 2.0 * (pred_data[i] - target_data[i]) * grad_output / n;
        }
        Tensor::new(shape, &grad)
    };

    (loss, OpaqueFn::new(back))
}

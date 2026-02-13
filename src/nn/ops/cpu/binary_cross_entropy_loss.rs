#![allow(clippy::cast_precision_loss)]

use super::{exp, ln};
use crate::nn::{
    tensors::{Tensor, WithGrad},
    TensorFloat,
};
use tensor_optim::TensorOps;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "dyntensor")]
use alloc::vec;
#[cfg(not(feature = "alloc"))]
use box_closure::{Align8, OpaqueFn};

/// BCE loss for identifying one of two classes.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn binary_cross_entropy_loss<'a>(
    prediction: &'a WithGrad<Tensor<TensorFloat>>,
    target: &'a Tensor<TensorFloat>,
) -> (
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat> + 'a>,
) {
    let shape = prediction.get_value().shape();
    let rank = shape.len();
    let last_dim = shape[rank - 1];
    let outer_size: usize = shape[..rank - 1].iter().product();

    let pred_data = prediction.get_value().data();
    let target_data = target.data();

    let mut sigmoid = vec![0.0; pred_data.len()];
    let mut loss_sum = 0.0;

    for i in 0..outer_size {
        let offset = i * last_dim;

        for j in 0..last_dim {
            let idx = offset + j;
            let x = pred_data[idx];
            let t = target_data[idx];

            let s = 1.0 / (1.0 + exp(-x));
            sigmoid[idx] = s;

            loss_sum -= t * ln(s) + (1.0 - t) * ln(1.0 - s);
        }
    }

    let n_samples = outer_size as TensorFloat;
    let loss = loss_sum / n_samples;

    let back = move |grad_output: TensorFloat| {
        let mut grad = vec![0.0; pred_data.len()];

        for i in 0..outer_size {
            let offset = i * last_dim;
            for j in 0..last_dim {
                let idx = offset + j;
                grad[idx] = (sigmoid[idx] - target_data[idx]) * grad_output / n_samples;
            }
        }

        Tensor::new(shape, &grad)
    };

    (loss, Box::new(back))
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
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = prediction.get_value().shape_array();
    let pred_data = prediction.get_value().data();
    let target_data = target.data();

    let last_dim = shape[D - 1];
    let outer_size: usize = shape[..D - 1].iter().product();

    let mut sigmoid = {
        #[cfg(feature = "no_stack")]
        {
            alloc::vec![0.0; N]
        }
        #[cfg(not(feature = "no_stack"))]
        {
            [0.0; N]
        }
    };
    let mut loss_sum = 0.0;

    for i in 0..outer_size {
        let offset = i * last_dim;

        for j in 0..last_dim {
            let idx = offset + j;
            let x = pred_data[idx];
            let t = target_data[idx];

            let s = 1.0 / (1.0 + exp(-x));
            sigmoid[idx] = s;

            loss_sum -= t * ln(s) + (1.0 - t) * ln(1.0 - s);
        }
    }

    let n_samples = outer_size as TensorFloat;
    let loss = loss_sum / n_samples;

    let back = move |grad_output: TensorFloat| {
        let mut grad = {
            #[cfg(feature = "no_stack")]
            {
                alloc::vec![0.0; N]
            }
            #[cfg(not(feature = "no_stack"))]
            {
                [0.0; N]
            }
        };

        for i in 0..outer_size {
            let offset = i * last_dim;
            for j in 0..last_dim {
                let idx = offset + j;
                grad[idx] = (sigmoid[idx] - target_data[idx]) * grad_output / n_samples;
            }
        }

        {
            #[cfg(feature = "no_stack")]
            {
                use crate::nn::tensors::VecTensor;

                unsafe {
                    VecTensor::from_vec(shape, grad)
                        .into_tensor()
                        .unwrap_unchecked()
                }
            }
            #[cfg(not(feature = "no_stack"))]
            {
                Tensor::new(shape, &grad)
            }
        }
    };

    (loss, Box::new(back))
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
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = prediction.get_value().shape_array();
    let pred_data = prediction.get_value().data();
    let target_data = target.data();

    let last_dim = shape[D - 1];
    let outer_size: usize = shape[..D - 1].iter().product();

    let mut sigmoid = [0.0; N];
    let mut loss_sum = 0.0;

    for i in 0..outer_size {
        let offset = i * last_dim;

        for j in 0..last_dim {
            let idx = offset + j;
            let x = pred_data[idx];
            let t = target_data[idx];

            let s = 1.0 / (1.0 + exp(-x));
            sigmoid[idx] = s;

            loss_sum -= t * ln(s) + (1.0 - t) * ln(1.0 - s);
        }
    }

    let n_samples = outer_size as TensorFloat;
    let loss = loss_sum / n_samples;

    let back = move |grad_output: TensorFloat| {
        let mut grad = [0.0; N];

        for i in 0..outer_size {
            let offset = i * last_dim;
            for j in 0..last_dim {
                let idx = offset + j;
                grad[idx] = (sigmoid[idx] - target_data[idx]) * grad_output / n_samples;
            }
        }

        Tensor::new(shape, &grad)
    };

    (loss, OpaqueFn::new(back))
}

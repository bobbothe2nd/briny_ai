use crate::nn::{
    tensors::{Tensor, WithGrad},
    TensorFloat,
};
use tensor_optim::TensorOps;
use super::{exp, ln};

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "dyntensor")]
use alloc::vec;
#[cfg(not(feature = "alloc"))]
use box_closure::{Align8, OpaqueFn};

/// Cross-entropy loss for arbitrary-rank tensors along last axis.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn cross_entropy_loss<'a>(
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

    // Compute softmax and loss
    let mut softmax = vec![TensorFloat::default(); pred_data.len()];
    let mut loss_sum = TensorFloat::default();

    for i in 0..outer_size {
        let offset = i * last_dim;
        let slice = &pred_data[offset..offset + last_dim];
        let t_slice = &target_data[offset..offset + last_dim];

        let max_val = slice.iter().copied().fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
        let exp_sum: TensorFloat = slice.iter().map(|&x| exp(x - max_val)).sum();

        for j in 0..last_dim {
            let s = exp(slice[j] - max_val) / exp_sum;
            softmax[offset + j] = s;
            loss_sum -= t_slice[j] * ln(s);
        }
    }

    let n_samples = outer_size as TensorFloat;
    let loss = loss_sum / n_samples;

    let back = {
        let softmax = softmax;
        move |grad_output: TensorFloat| {
            let mut grad = vec![TensorFloat::default(); pred_data.len()];
            for i in 0..outer_size {
                let offset = i * last_dim;
                let t_slice = &target_data[offset..offset + last_dim];
                let s_slice = &softmax[offset..offset + last_dim];

                for j in 0..last_dim {
                    grad[offset + j] = (s_slice[j] - t_slice[j]) * grad_output / n_samples;
                }
            }
            Tensor::new(shape, &grad)
        }
    };

    (loss, Box::new(back))
}

/// Cross-entropy loss for arbitrary-rank tensors along last axis.
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
#[must_use]
pub fn cross_entropy_loss<'a, const N: usize, const D: usize>(
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

    let mut softmax = [TensorFloat::default(); N];
    let mut loss_sum = TensorFloat::default();

    for i in 0..outer_size {
        let offset = i * last_dim;
        let slice = &pred_data[offset..offset + last_dim];
        let t_slice = &target_data[offset..offset + last_dim];

        let max_val = slice.iter().copied().fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
        let exp_sum: TensorFloat = slice.iter().map(|&x| exp(x - max_val)).sum();

        for j in 0..last_dim {
            let s = exp(slice[j] - max_val) / exp_sum;
            softmax[offset + j] = s;
            loss_sum -= t_slice[j] * ln(s);
        }
    }

    let n_samples = outer_size as TensorFloat;
    let loss = loss_sum / n_samples;

    let back = {
        let softmax = softmax;
        move |grad_output: TensorFloat| {
            let mut grad = [TensorFloat::default(); N];
            for i in 0..outer_size {
                let offset = i * last_dim;
                let t_slice = &target_data[offset..offset + last_dim];
                let s_slice = &softmax[offset..offset + last_dim];

                for j in 0..last_dim {
                    grad[offset + j] = (s_slice[j] - t_slice[j]) * grad_output / n_samples;
                }
            }
            Tensor::new(shape, &grad)
        }
    };

    (loss, Box::new(back))
}

/// Cross-entropy loss for arbitrary-rank tensors along last axis.
#[cfg(not(feature = "alloc"))]
#[must_use]
pub fn cross_entropy_loss<'a, const N: usize, const D: usize>(
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

    let mut softmax = [TensorFloat::default(); N];
    let mut loss_sum = TensorFloat::default();

    for i in 0..outer_size {
        let offset = i * last_dim;
        let slice = &pred_data[offset..offset + last_dim];
        let t_slice = &target_data[offset..offset + last_dim];

        let max_val = slice.iter().copied().fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
        let exp_sum: TensorFloat = slice.iter().map(|&x| exp(x - max_val)).sum();

        for j in 0..last_dim {
            let s = exp(slice[j] - max_val) / exp_sum;
            softmax[offset + j] = s;
            loss_sum -= t_slice[j] * ln(s);
        }
    }

    let n_samples = outer_size as TensorFloat;
    let loss = loss_sum / n_samples;

    let back = {
        let softmax = softmax;
        move |grad_output: TensorFloat| {
            let mut grad = [TensorFloat::default(); N];
            for i in 0..outer_size {
                let offset = i * last_dim;
                let t_slice = &target_data[offset..offset + last_dim];
                let s_slice = &softmax[offset..offset + last_dim];

                for j in 0..last_dim {
                    grad[offset + j] = (s_slice[j] - t_slice[j]) * grad_output / n_samples;
                }
            }
            Tensor::new(shape, &grad)
        }
    };

    (loss, OpaqueFn::new(back))
}

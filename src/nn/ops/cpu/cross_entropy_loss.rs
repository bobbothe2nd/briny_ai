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
use box_closure::{Align8, OpaqueFnOnce};

/// Cross-entropy loss for arbitrary-rank tensors along last axis.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn cross_entropy_loss<'a>(
    prediction: &'a WithGrad<Tensor<TensorFloat>>,
    target: &'a Tensor<TensorFloat>,
) -> (
    TensorFloat,
    Box<dyn FnOnce(TensorFloat) -> Tensor<TensorFloat> + 'a>,
) {
    let shape = prediction.get_value().shape();
    let rank = shape.len();
    let last_dim = shape[rank - 1];
    let outer_size: usize = shape[..rank - 1].iter().product();

    let pred_data = prediction.get_value().data();
    let target_data = target.data();

    let mut softmax = vec![0.0; pred_data.len()];
    let mut loss_sum = 0.0;
    let mut mask = vec![0.0; outer_size];
    let mut valid_tokens = 0.0;

    // compute mask for PAD tokens
    for (i, val) in mask.iter_mut().enumerate().take(outer_size) {
        let offset = i * last_dim;
        let factor = f32::from(u8::from(
            target_data[offset..offset + last_dim]
                .iter()
                .all(|&x| x == 0.0),
        ));
        *val = factor;
        valid_tokens += factor;
    }

    for (i, val) in mask.iter().enumerate().take(outer_size) {
        let offset = i * last_dim;
        let slice = &pred_data[offset..offset + last_dim];
        let t_slice = &target_data[offset..offset + last_dim];

        let max_val = slice
            .iter()
            .copied()
            .fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
        let exp_sum: TensorFloat = slice.iter().map(|&x| exp(x - max_val)).sum();

        for j in 0..last_dim {
            let s = exp(slice[j] - max_val) / exp_sum;
            softmax[offset + j] = s;
            loss_sum -= val * t_slice[j] * ln(s);
        }
    }

    let loss = loss_sum / valid_tokens;

    let back = move |grad_output: TensorFloat| {
        let mut grad = vec![0.0; pred_data.len()];
        for (i, val) in mask.iter().enumerate().take(outer_size) {
            let offset = i * last_dim;
            let t_slice = &target_data[offset..offset + last_dim];
            let s_slice = &softmax[offset..offset + last_dim];

            for j in 0..last_dim {
                grad[offset + j] = val * (s_slice[j] - t_slice[j]) * grad_output / valid_tokens;
            }
        }
        Tensor::new(shape, &grad)
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
    Box<dyn FnOnce(TensorFloat) -> Tensor<TensorFloat, N, D> + 'a>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = prediction.get_value().shape_array();
    let pred_data = prediction.get_value().data();
    let target_data = target.data();

    let last_dim = shape[D - 1];
    let outer_size: usize = shape[..D - 1].iter().product();

    let mut softmax = {
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
    let mut mask = {
        #[cfg(feature = "no_stack")]
        {
            alloc::vec![0.0; N]
        }
        #[cfg(not(feature = "no_stack"))]
        {
            [0.0; N]
        }
    };
    let mut valid_tokens = 0.0;

    for (i, val) in mask.iter_mut().enumerate().take(outer_size) {
        let offset = i * last_dim;
        let factor = f32::from(u8::from(
            !target_data[offset..offset + last_dim]
                .iter()
                .all(|&x| x == 0.0),
        ));
        *val = factor;
        valid_tokens += factor;
    }

    for (i, val) in mask.iter().enumerate().take(outer_size) {
        let offset = i * last_dim;
        let slice = &pred_data[offset..offset + last_dim];
        let t_slice = &target_data[offset..offset + last_dim];

        let max_val = slice
            .iter()
            .copied()
            .fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
        let exp_sum: TensorFloat = slice.iter().map(|&x| exp(x - max_val)).sum();

        for j in 0..last_dim {
            let s = exp(slice[j] - max_val) / exp_sum;
            softmax[offset + j] = s;
            loss_sum -= val * t_slice[j] * ln(s);
        }
    }

    let loss = loss_sum / valid_tokens;

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

        for (i, val) in mask.iter().enumerate().take(outer_size) {
            let offset = i * last_dim;
            let t_slice = &target_data[offset..offset + last_dim];
            let s_slice = &softmax[offset..offset + last_dim];

            for j in 0..last_dim {
                grad[offset + j] = val * (s_slice[j] - t_slice[j]) * grad_output / valid_tokens;
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

/// Cross-entropy loss for arbitrary-rank tensors along last axis.
#[cfg(not(feature = "alloc"))]
#[must_use]
pub fn cross_entropy_loss<'a, const N: usize, const D: usize>(
    prediction: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> (
    TensorFloat,
    OpaqueFnOnce<'a, TensorFloat, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = prediction.get_value().shape_array();
    let pred_data = prediction.get_value().data();
    let target_data = target.data();

    let last_dim = shape[D - 1];
    let outer_size: usize = shape[..D - 1].iter().product();

    let mut softmax = [0.0; N];
    let mut loss_sum = 0.0;
    let mut mask = [0.0; N];
    let mut valid_tokens = 0.0;

    for (i, val) in mask.iter_mut().enumerate().take(outer_size) {
        let offset = i * last_dim;
        let factor = f32::from(u8::from(
            target_data[offset..offset + last_dim]
                .iter()
                .all(|&x| x == 0.0),
        ));
        *val = factor;
        valid_tokens += factor;
    }

    for (i, val) in mask.iter().enumerate().take(outer_size) {
        let offset = i * last_dim;
        let slice = &pred_data[offset..offset + last_dim];
        let t_slice = &target_data[offset..offset + last_dim];

        let max_val = slice
            .iter()
            .copied()
            .fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
        let exp_sum: TensorFloat = slice.iter().map(|&x| exp(x - max_val)).sum();

        for j in 0..last_dim {
            let s = exp(slice[j] - max_val) / exp_sum;
            softmax[offset + j] = s;
            loss_sum -= val * t_slice[j] * ln(s);
        }
    }

    let loss = loss_sum / valid_tokens;

    let back = move |grad_output: TensorFloat| {
        let mut grad = [0.0; N];
        for (i, val) in mask.iter().enumerate().take(outer_size) {
            let offset = i * last_dim;
            let t_slice = &target_data[offset..offset + last_dim];
            let s_slice = &softmax[offset..offset + last_dim];

            for j in 0..last_dim {
                grad[offset + j] = val * (s_slice[j] - t_slice[j]) * grad_output / valid_tokens;
            }
        }
        Tensor::new(shape, &grad)
    };

    (loss, OpaqueFnOnce::new(back))
}

use crate::nn::{
    tensors::{Tensor, WithGrad},
    TensorFloat,
};
use tensor_optim::TensorOps;
use super::exp;

#[cfg(feature = "dyntensor")]
use alloc::vec;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(not(feature = "alloc"))]
use box_closure::{Align8, OpaqueFn};

/// Softmax along the last axis of an arbitrary-rank tensor.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn softmax(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    let shape = input.get_value().shape();
    let rank = shape.len();
    let last_dim = shape[rank - 1];
    let outer_size: usize = shape[..rank - 1].iter().product();
    let input_data = input.get_value().data();

    let mut out_data = vec![TensorFloat::default(); input_data.len()];

    // Forward pass
    for i in 0..outer_size {
        let offset = i * last_dim;
        let slice = &input_data[offset..offset + last_dim];

        let max_val = slice.iter().copied().fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
        let exp_sum: TensorFloat = slice.iter().map(|&x| exp(x - max_val)).sum();

        for j in 0..last_dim {
            out_data[offset + j] = exp(slice[j] - max_val) / exp_sum;
        }
    }

    let out = Tensor::new(shape, &out_data);

    let back = {
        let out_clone = out_data;
        move |grad_output: Tensor<TensorFloat>| {
            let grad_data = grad_output.data();
            let mut grad = vec![TensorFloat::default(); grad_data.len()];

            for i in 0..outer_size {
                let offset = i * last_dim;
                let y = &out_clone[offset..offset + last_dim];
                let dy = &grad_data[offset..offset + last_dim];

                let dot: TensorFloat = y.iter().zip(dy.iter()).map(|(&yi, &dyi)| yi * dyi).sum();

                for j in 0..last_dim {
                    grad[offset + j] = y[j] * (dy[j] - dot);
                }
            }

            Tensor::new(shape, &grad)
        }
    };

    (out, Box::new(back))
}

/// Softmax along the last axis of an arbitrary-rank tensor.
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
#[must_use]
pub fn softmax<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    Box<dyn Fn(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();
    let last_dim = shape[D - 1];
    let outer_size: usize = shape[..D - 1].iter().product();

    let mut out_data = [TensorFloat::default(); N];

    for i in 0..outer_size {
        let offset = i * last_dim;
        let slice = &input_data[offset..offset + last_dim];

        let max_val = slice.iter().copied().fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
        let exp_sum: TensorFloat = slice.iter().map(|&x| exp(x - max_val)).sum();

        for j in 0..last_dim {
            out_data[offset + j] = exp(slice[j] - max_val) / exp_sum;
        }
    }

    let out = Tensor::new(shape, &out_data);

    let back = {
        let out_clone = out_data;
        move |grad_output: Tensor<TensorFloat, N, D>| {
            let grad_data = grad_output.data();
            let mut grad = [TensorFloat::default(); N];

            for i in 0..outer_size {
                let offset = i * last_dim;
                let y = &out_clone[offset..offset + last_dim];
                let dy = &grad_data[offset..offset + last_dim];

                let dot: TensorFloat = y.iter().zip(dy.iter()).map(|(&yi, &dyi)| yi * dyi).sum();

                for j in 0..last_dim {
                    grad[offset + j] = y[j] * (dy[j] - dot);
                }
            }

            Tensor::new(shape, &grad)
        }
    };

    (out, Box::new(back))
}

/// Softmax along the last axis of an arbitrary-rank tensor.
#[cfg(not(feature = "alloc"))]
#[must_use]
pub fn softmax<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
        OpaqueFn<'_, Tensor<TensorFloat, N, D>, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();
    let last_dim = shape[D - 1];
    let outer_size: usize = shape[..D - 1].iter().product();

    let mut out_data = [TensorFloat::default(); N];

    for i in 0..outer_size {
        let offset = i * last_dim;
        let slice = &input_data[offset..offset + last_dim];

        let max_val = slice.iter().copied().fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
        let exp_sum: TensorFloat = slice.iter().map(|&x| exp(x - max_val)).sum();

        for j in 0..last_dim {
            out_data[offset + j] = exp(slice[j] - max_val) / exp_sum;
        }
    }

    let out = Tensor::new(shape, &out_data);

    let back = {
        let out_clone = out_data;
        move |grad_output: Tensor<TensorFloat, N, D>| {
            let grad_data = grad_output.data();
            let mut grad = [TensorFloat::default(); N];

            for i in 0..outer_size {
                let offset = i * last_dim;
                let y = &out_clone[offset..offset + last_dim];
                let dy = &grad_data[offset..offset + last_dim];

                let dot: TensorFloat = y.iter().zip(dy.iter()).map(|(&yi, &dyi)| yi * dyi).sum();

                for j in 0..last_dim {
                    grad[offset + j] = y[j] * (dy[j] - dot);
                }
            }

            Tensor::new(shape, &grad)
        }
    };

    (out, OpaqueFn::new(back))
}

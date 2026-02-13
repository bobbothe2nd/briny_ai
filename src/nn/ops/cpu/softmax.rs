#![allow(clippy::type_complexity)]

use super::exp;
use crate::nn::{
    tensors::{Tensor, WithGrad},
    TensorFloat,
};
use tensor_optim::TensorOps;

#[cfg(feature = "dyntensor")]
use alloc::vec;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(not(feature = "alloc"))]
use box_closure::{Align8, OpaqueFnOnce};

/// Softmax along the last axis of an arbitrary-rank tensor.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn softmax(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn FnOnce(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    let shape = input.get_value().shape();
    let rank = shape.len();
    let last_dim = shape[rank - 1];
    let outer_size: usize = shape[..rank - 1].iter().product();
    let input_data = input.get_value().data();

    let mut out_data = vec![0.0; input_data.len()];

    // Forward pass
    for i in 0..outer_size {
        let offset = i * last_dim;
        let slice = &input_data[offset..offset + last_dim];

        let max_val = slice
            .iter()
            .copied()
            .fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
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
            let mut grad = vec![0.0; grad_data.len()];

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
pub fn softmax<const N1: usize, const N2: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N1, D>>,
) -> (
    Tensor<TensorFloat, N2, D>,
    Box<dyn FnOnce(Tensor<TensorFloat, N2, D>) -> Tensor<TensorFloat, N1, D> + '_>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();
    let last_dim = shape[D - 1];
    let outer_size: usize = shape[..D - 1].iter().product();

    let mut out_data = {
        #[cfg(feature = "no_stack")]
        {
            alloc::vec![0.0; N2]
        }
        #[cfg(not(feature = "no_stack"))]
        {
            [0.0; N2]
        }
    };

    for i in 0..outer_size {
        let offset = i * last_dim;
        let slice = &input_data[offset..offset + last_dim];

        let max_val = slice
            .iter()
            .copied()
            .fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
        let exp_sum: TensorFloat = slice.iter().map(|&x| exp(x - max_val)).sum();

        for j in 0..last_dim {
            out_data[offset + j] = exp(slice[j] - max_val) / exp_sum;
        }
    }

    let out = {
        #[cfg(feature = "no_stack")]
        {
            Tensor::new(shape, unsafe {
                out_data.as_slice().try_into().unwrap_unchecked()
            })
        }
        #[cfg(not(feature = "no_stack"))]
        {
            Tensor::new(shape, &out_data)
        }
    };

    let back = move |grad_output: Tensor<TensorFloat, N2, D>| {
        let grad_data = grad_output.data();
        let mut grad = {
            #[cfg(feature = "no_stack")]
            {
                alloc::vec![0.0; N1]
            }
            #[cfg(not(feature = "no_stack"))]
            {
                [0.0; N1]
            }
        };

        for i in 0..outer_size {
            let offset = i * last_dim;
            let y = &out_data[offset..offset + last_dim];
            let dy = &grad_data[offset..offset + last_dim];

            let dot: TensorFloat = y.iter().zip(dy.iter()).map(|(&yi, &dyi)| yi * dyi).sum();

            for j in 0..last_dim {
                grad[offset + j] = y[j] * (dy[j] - dot);
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

    (out, Box::new(back))
}

/// Softmax along the last axis of an arbitrary-rank tensor.
#[cfg(not(feature = "alloc"))]
#[must_use]
pub fn softmax<const N1: usize, const N2: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N1, D>>,
) -> (
    Tensor<TensorFloat, N2, D>,
    OpaqueFnOnce<'_, Tensor<TensorFloat, N2, D>, Tensor<TensorFloat, N1, D>, Align8<128>>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();
    let last_dim = shape[D - 1];
    let outer_size: usize = shape[..D - 1].iter().product();

    let mut out_data = [0.0; N2];

    for i in 0..outer_size {
        let offset = i * last_dim;
        let slice = &input_data[offset..offset + last_dim];

        let max_val = slice
            .iter()
            .copied()
            .fold(TensorFloat::NEG_INFINITY, TensorFloat::max);
        let exp_sum: TensorFloat = slice.iter().map(|&x| exp(x - max_val)).sum();

        for j in 0..last_dim {
            out_data[offset + j] = exp(slice[j] - max_val) / exp_sum;
        }
    }

    let out = Tensor::new(shape, &out_data);

    let back = {
        move |grad_output: Tensor<TensorFloat, N2, D>| {
            let grad_data = grad_output.data();
            let mut grad = [0.0; N1];

            for i in 0..outer_size {
                let offset = i * last_dim;
                let y = &out_data[offset..offset + last_dim];
                let dy = &grad_data[offset..offset + last_dim];

                let dot: TensorFloat = y.iter().zip(dy.iter()).map(|(&yi, &dyi)| yi * dyi).sum();

                for j in 0..last_dim {
                    grad[offset + j] = y[j] * (dy[j] - dot);
                }
            }

            Tensor::new(shape, &grad)
        }
    };

    (out, OpaqueFnOnce::new(back))
}

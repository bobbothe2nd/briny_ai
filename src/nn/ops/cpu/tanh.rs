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

/// Applies a `tanh` activation function element-wise.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn tanh(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn FnOnce(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    let shape = input.get_value().shape();
    let input_data = input.get_value().data();

    let mut out_data = vec![0.0; input_data.len()];

    // Forward
    out_data
        .iter_mut()
        .zip(input_data.iter())
        .for_each(|(y, &x)| {
            // numerically stable tanh
            let e2x = exp(x + x);
            *y = (e2x - 1.0) / (e2x + 1.0);
        });

    let out = Tensor::new(shape, &out_data);

    // Backward
    let back = move |grad_output: Tensor<TensorFloat>| {
        let dy = grad_output.data();
        let mut grad = vec![0.0; dy.len()];

        grad.iter_mut()
            .zip(out_data.iter())
            .zip(dy.iter())
            .for_each(|((g, &y), &dyi)| {
                *g = dyi * (1.0 - y * y);
            });

        Tensor::new(shape, &grad)
    };

    (out, Box::new(back))
}

/// Applies a `tanh` activation function element-wise.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn tanh<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    Box<dyn FnOnce(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();

    let mut out_data = [0.0; N];

    // Forward
    out_data
        .iter_mut()
        .zip(input_data.iter())
        .for_each(|(y, &x)| {
            let e2x = exp(x + x);
            *y = (e2x - 1.0) / (e2x + 1.0);
        });

    let out = Tensor::new(shape, &out_data);

    // Backward
    let back = move |grad_output: Tensor<TensorFloat, N, D>| {
        let dy = grad_output.data();
        let mut grad = [0.0; N];

        grad.iter_mut()
            .zip(out_data.iter())
            .zip(dy.iter())
            .for_each(|((g, &y), &dyi)| {
                *g = dyi * (1.0 - y * y);
            });

        Tensor::new(shape, &grad)
    };

    (out, Box::new(back))
}

/// Applies a `tanh` activation function element-wise.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn tanh<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    OpaqueFnOnce<'_, Tensor<TensorFloat, N, D>, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();

    let mut out_data = [0.0; N];

    // Forward
    out_data
        .iter_mut()
        .zip(input_data.iter())
        .for_each(|(y, &x)| {
            let e2x = exp(x + x);
            *y = (e2x - 1.0) / (e2x + 1.0);
        });

    let out = Tensor::new(shape, &out_data);

    let back = move |grad_output: Tensor<TensorFloat, N, D>| {
        let dy = grad_output.data();
        let mut grad = [0.0; N];

        grad.iter_mut()
            .zip(out_data.iter())
            .zip(dy.iter())
            .for_each(|((g, &y), &dyi)| {
                *g = dyi * (1.0 - y * y);
            });

        Tensor::new(shape, &grad)
    };

    (out, OpaqueFnOnce::new(back))
}

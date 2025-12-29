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
use box_closure::{Align8, OpaqueFn};

/// Performs Sigmoid activation.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn sigmoid(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    let shape = input.get_value().shape();
    let input_data = input.get_value().data();

    let mut out_data = vec![TensorFloat::default(); input_data.len()];

    out_data
        .iter_mut()
        .zip(input_data.iter())
        .for_each(|(y, &x)| {
            *y = TensorFloat::from(1.0) / (TensorFloat::from(1.0) + exp(-x));
        });

    let out = Tensor::new(shape, &out_data);

    let back = {
        let y_saved = out_data;
        move |grad_output: Tensor<TensorFloat>| {
            let dy = grad_output.data();
            let mut grad = vec![TensorFloat::default(); dy.len()];

            grad.iter_mut()
                .zip(y_saved.iter())
                .zip(dy.iter())
                .for_each(|((g, &y), &dyi)| {
                    *g = dyi * y * (TensorFloat::from(1.0) - y);
                });

            Tensor::new(shape, &grad)
        }
    };

    (out, Box::new(back))
}

/// Performs Sigmoid activation.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn sigmoid<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    Box<dyn Fn(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();

    let mut out_data = [TensorFloat::default(); N];

    out_data
        .iter_mut()
        .zip(input_data.iter())
        .for_each(|(y, &x)| {
            *y = TensorFloat::from(1.0) / (TensorFloat::from(1.0) + exp(-x));
        });

    let out = Tensor::new(shape, &out_data);

    let back = {
        let y_saved = out_data;
        move |grad_output: Tensor<TensorFloat, N, D>| {
            let dy = grad_output.data();
            let mut grad = [TensorFloat::default(); N];

            grad.iter_mut()
                .zip(y_saved.iter())
                .zip(dy.iter())
                .for_each(|((g, &y), &dyi)| {
                    *g = dyi * y * (TensorFloat::from(1.0) - y);
                });

            Tensor::new(shape, &grad)
        }
    };

    (out, Box::new(back))
}

/// Performs Sigmoid activation.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn sigmoid<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    OpaqueFn<'_, Tensor<TensorFloat, N, D>, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();

    let mut out_data = [TensorFloat::default(); N];

    out_data
        .iter_mut()
        .zip(input_data.iter())
        .for_each(|(y, &x)| {
            *y = TensorFloat::from(1.0) / (TensorFloat::from(1.0) + exp(-x));
        });

    let out = Tensor::new(shape, &out_data);

    let back = move |grad_output: Tensor<TensorFloat, N, D>| {
        let dy = grad_output.data();
        let mut grad = [TensorFloat::default(); N];

        grad.iter_mut()
            .zip(out_data.iter())
            .zip(dy.iter())
            .for_each(|((g, &y), &dyi)| {
                *g = dyi * y * (TensorFloat::from(1.0) - y);
            });

        Tensor::new(shape, &grad)
    };

    (out, OpaqueFn::new(back))
}

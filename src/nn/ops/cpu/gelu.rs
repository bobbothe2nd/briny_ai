use super::exp;
use libm::erff;

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

/// Performs GELU activation.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn gelu(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    let shape = input.get_value().shape();
    let input_data = input.get_value().data();

    let mut out_data = vec![0.0; input_data.len()];
    let mut phi_data = vec![0.0; input_data.len()];

    let inv_sqrt2 = 0.707_106_781_186_547_6; // 1 / sqrt(2)

    out_data
        .iter_mut()
        .zip(phi_data.iter_mut())
        .zip(input_data.iter())
        .for_each(|((y, phi), &x)| {
            let cdf = 0.5 * TensorFloat::from(1.0 + erff((x * inv_sqrt2) as f32));
            *phi = cdf;
            *y = x * cdf;
        });

    let out = Tensor::new(shape, &out_data);

    let back = move |grad_output: Tensor<TensorFloat>| {
        let dy = grad_output.data();
        let mut grad = vec![TensorFloat::default(); dy.len()];

        let inv_sqrt2pi = TensorFloat::from(0.398_942_280_401_432_7); // 1 / sqrt(2π)

        grad.iter_mut()
            .zip(input_data.iter())
            .zip(phi_data.iter())
            .zip(dy.iter())
            .for_each(|(((g, &x), &phi), &dyi)| {
                let pdf = inv_sqrt2pi * exp(-TensorFloat::from(0.5) * x * x);
                *g = dyi * (phi + x * pdf);
            });

        Tensor::new(shape, &grad)
    };

    (out, Box::new(back))
}

/// Performs GELU activation.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn gelu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    Box<dyn Fn(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();

    let mut out_data = [TensorFloat::default(); N];
    let mut phi_data = [TensorFloat::default(); N];

    let inv_sqrt2 = TensorFloat::from(0.707_106_781_186_547_6);

    out_data
        .iter_mut()
        .zip(phi_data.iter_mut())
        .zip(input_data.iter())
        .for_each(|((y, phi), &x)| {
            let cdf = 0.5 * TensorFloat::from(1.0 + erff((x * inv_sqrt2) as f32));
            *phi = cdf;
            *y = x * cdf;
        });

    let out = Tensor::new(shape, &out_data);

    let back = move |grad_output: Tensor<TensorFloat, N, D>| {
        let dy = grad_output.data();
        let mut grad = [TensorFloat::default(); N];

        let inv_sqrt2pi = TensorFloat::from(0.398_942_280_401_432_7);

        grad.iter_mut()
            .zip(input_data.iter())
            .zip(phi_data.iter())
            .zip(dy.iter())
            .for_each(|(((g, &x), &phi), &dyi)| {
                let pdf = inv_sqrt2pi * exp(-TensorFloat::from(0.5) * x * x);
                *g = dyi * (phi + x * pdf);
            });

        Tensor::new(shape, &grad)
    };

    (out, Box::new(back))
}

/// Performs GELU activation.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn gelu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    OpaqueFn<'_, Tensor<TensorFloat, N, D>, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();

    let mut out_data = [0.0; N];
    let mut phi_data = [0.0; N];

    let inv_sqrt2 = 0.707_106_781_186_547_6;

    out_data
        .iter_mut()
        .zip(phi_data.iter_mut())
        .zip(input_data.iter())
        .for_each(|((y, phi), &x)| {
            let cdf = 0.5 * TensorFloat::from(1.0 + erff((x * inv_sqrt2) as f32));
            *phi = cdf;
            *y = x * cdf;
        });

    let out = Tensor::new(shape, &out_data);

    let back = move |grad_output: Tensor<TensorFloat, N, D>| {
        let dy = grad_output.data();
        let mut grad = [0.0; N];

        let inv_sqrt2pi = 0.398_942_280_401_432_7;

        grad.iter_mut()
            .zip(input_data.iter())
            .zip(phi_data.iter())
            .zip(dy.iter())
            .for_each(|(((g, &x), &phi), &dyi)| {
                let pdf = inv_sqrt2pi * exp(-0.5 * x * x);
                *g = dyi * (phi + x * pdf);
            });

        Tensor::new(shape, &grad)
    };

    (out, OpaqueFn::new(back))
}

#![allow(clippy::type_complexity, clippy::excessive_precision)]

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
use box_closure::{Align8, OpaqueFnOnce};

/// Performs GELU activation.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn gelu(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<[TensorFloat]>,
    Box<dyn FnOnce(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    let shape = input.get_value().shape();
    let input_data = input.get_value().data();

    let mut out_data = vec![0.0; input_data.len()];
    let mut phi_data = vec![0.0; input_data.len()];

    out_data
        .iter_mut()
        .zip(phi_data.iter_mut())
        .zip(input_data.iter())
        .for_each(|((y, phi), &x)| {
            let cdf = 0.5 * 1.0 + erff(x * core::f32::consts::FRAC_1_SQRT_2);
            *phi = cdf;
            *y = x * cdf;
        });

    let out = Tensor::new(shape, &out_data);

    let phi_data_box = phi_data.clone().into_boxed_slice();
    let back = move |grad_output: Tensor<TensorFloat>| {
        let dy = grad_output.data();
        let mut grad = vec![0.0; dy.len()];

        let inv_sqrt2pi = 0.398_942_280_401_432_7; // 1 / sqrt(2π)

        grad.iter_mut()
            .zip(input_data.iter())
            .zip(phi_data.clone().iter())
            .zip(dy.iter())
            .for_each(|(((g, &x), &phi), &dyi)| {
                let pdf = inv_sqrt2pi * exp(-0.5) * x * x;
                *g = dyi * (phi + x * pdf);
            });

        Tensor::new(shape, &grad)
    };

    (out, phi_data_box, Box::new(back))
}

/// Performs GELU activation.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn gelu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    Box<[TensorFloat; N]>,
    Box<dyn FnOnce(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();

    let mut out_data = {
        #[cfg(feature = "no_stack")]
        {
            alloc::vec![0.0; N]
        }
        #[cfg(not(feature = "no_stack"))]
        {
            [0.0; N]
        }
    };
    let mut phi_data = Box::<[TensorFloat; N]>::new_uninit();
    unsafe {
        phi_data.as_mut_ptr().write_bytes(0, 1);
    }
    let mut phi_data = unsafe { phi_data.assume_init() };

    out_data
        .iter_mut()
        .zip(phi_data.iter_mut())
        .zip(input_data.iter())
        .for_each(|((y, phi), &x)| {
            let cdf = 0.5 * 1.0 + erff(x * core::f32::consts::FRAC_1_SQRT_2);
            *phi = cdf;
            *y = x * cdf;
        });

    let out_data = {
        #[cfg(feature = "no_stack")]
        {
            unsafe { out_data.as_slice().try_into().unwrap_unchecked() }
        }
        #[cfg(not(feature = "no_stack"))]
        {
            &out_data
        }
    };
    let out = Tensor::new(shape, out_data);

    let phi_clone = phi_data.clone();

    let back = move |grad_output: Tensor<TensorFloat, N, D>| {
        let dy = grad_output.data();
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

        #[allow(clippy::excessive_precision)]
        let inv_sqrt2pi = 0.398_942_280_401_432_7;

        grad.iter_mut()
            .zip(input_data.iter())
            .zip(phi_clone.iter())
            .zip(dy.iter())
            .for_each(|(((g, &x), &phi), &dyi)| {
                let pdf = inv_sqrt2pi * exp(-0.5) * x * x;
                *g = dyi * (phi + x * pdf);
            });

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

    (out, phi_data, Box::new(back))
}

/// Performs GELU activation.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn gelu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> (
    Tensor<TensorFloat, N, D>,
    [TensorFloat; N],
    OpaqueFnOnce<'_, Tensor<TensorFloat, N, D>, Tensor<TensorFloat, N, D>, Align8<128>>,
) {
    use tensor_optim::ConstTensorOps;

    let shape: &[usize; D] = input.get_value().shape_array();
    let input_data = input.get_value().data();

    let mut out_data = [0.0; N];
    let mut phi_data = [0.0; N];

    out_data
        .iter_mut()
        .zip(phi_data.iter_mut())
        .zip(input_data.iter())
        .for_each(|((y, phi), &x)| {
            let cdf = 0.5 * 1.0 + erff(x * core::f32::consts::FRAC_1_SQRT_2);
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

    (out, phi_data, OpaqueFnOnce::new(back))
}

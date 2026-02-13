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

/// Performs Swish/SiLU activation.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn swish(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<[TensorFloat]>,
    Box<dyn FnOnce(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
) {
    let shape = input.get_value().shape();
    let input_data = input.get_value().data();

    let mut out_data = vec![0.0; input_data.len()];
    let mut sig_data = vec![0.0; input_data.len()];

    // Forward
    out_data
        .iter_mut()
        .zip(sig_data.iter_mut())
        .zip(input_data.iter())
        .for_each(|((y, s), &x)| {
            let sig = 1.0 / (1.0 + exp(-x));
            *s = sig;
            *y = x * sig;
        });

    let out = Tensor::new(shape, &out_data);

    let sig_data_box = sig_data.clone().into_boxed_slice();
    let back = move |grad_output: Tensor<TensorFloat>| {
        let dy = grad_output.data();
        let mut grad = vec![0.0; dy.len()];

        grad.iter_mut()
            .zip(input_data.iter())
            .zip(sig_data.iter())
            .zip(dy.iter())
            .for_each(|(((g, &x), &s), &dyi)| {
                *g = dyi * (s + x * s * (1.0 - s));
            });

        Tensor::new(shape, &grad)
    };

    (out, sig_data_box, Box::new(back))
}

/// Performs Swish/SiLU activation.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn swish<const N: usize, const D: usize>(
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
    let mut sig_data = Box::<[TensorFloat; N]>::new_uninit();
    unsafe {
        sig_data.as_mut_ptr().write_bytes(0, 1);
    }
    let mut sig_data = unsafe { sig_data.assume_init() };

    out_data
        .iter_mut()
        .zip(sig_data.iter_mut())
        .zip(input_data.iter())
        .for_each(|((y, s), &x)| {
            let sig = 1.0 / (1.0 + exp(-x));
            *s = sig;
            *y = x * sig;
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

    let sig_clone = sig_data.clone();

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

        grad.iter_mut()
            .zip(input_data.iter())
            .zip(sig_clone.iter())
            .zip(dy.iter())
            .for_each(|(((g, &x), &s), &dyi)| {
                *g = dyi * (s + x * s * (1.0 - s));
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

    (out, sig_data, Box::new(back))
}

/// Performs Swish/SiLU activation.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn swish<const N: usize, const D: usize>(
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
    let mut sig_data = [0.0; N];

    // Forward
    out_data
        .iter_mut()
        .zip(sig_data.iter_mut())
        .zip(input_data.iter())
        .for_each(|((y, s), &x)| {
            let sig = 1.0 / (1.0 + exp(-x));
            *s = sig;
            *y = x * sig;
        });

    let out = Tensor::new(shape, &out_data);

    let back = move |grad_output: Tensor<TensorFloat, N, D>| {
        let dy = grad_output.data();
        let mut grad = [0.0; N];

        grad.iter_mut()
            .zip(input_data.iter())
            .zip(sig_data.iter())
            .zip(dy.iter())
            .for_each(|(((g, &x), &s), &dyi)| {
                *g = dyi * (s + x * s * (1.0 - s));
            });

        Tensor::new(shape, &grad)
    };

    (out, sig_data, OpaqueFnOnce::new(back))
}

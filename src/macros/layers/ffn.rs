#![allow(clippy::excessive_precision)]

use core::marker::PhantomData;

use super::{Backward, ClosureOnce, IntoWithGrad, Layer, Tensor, TensorFloat, WithGrad};
use crate::{
    macros::{optim::Optim, ActivationFn, ActivationKind, BackFn, BuildLayer, LayerOpHeavy},
    nn::ops::dispatch::{gelu, relu, swish, tanh},
};
use alloc::boxed::Box;
use tensor_optim::TensorOps;

#[cfg(feature = "no_stack")]
use crate::nn::tensors::VecTensor;

/// A representation of a feed-forward layer builder.
///
/// Performs a right-hand multiplication: `output = W * input`
pub struct FeedForward<const RANK: usize, const SIZE: usize, O, Activation> {
    /// The shape of the layer.
    pub shape: [usize; RANK],

    /// The data of the layer.
    pub data: Box<[TensorFloat; SIZE]>,

    /// Optimizer for weights.
    pub _optim: PhantomData<O>,

    /// Intermediate activation function.
    pub _activation: PhantomData<Activation>,
}

impl<const RANK: usize, const SIZE: usize, Activation: ActivationFn, O: Optim<RANK, SIZE>>
    FeedForward<RANK, SIZE, O, Activation>
{
    /// The amount of tensors in the layer.
    pub const TENSORS: usize = 2;

    /// Builds the structure into a compute layer.
    #[must_use]
    #[allow(clippy::explicit_auto_deref)]
    pub fn build(self) -> FeedForwardLayer<RANK, SIZE, O> {
        let mut b_shape = self.shape;
        b_shape[RANK - 1] = self.shape[RANK - 2];
        b_shape[RANK - 2] = self.shape[RANK - 1];
        FeedForwardLayer {
            w_a: Tensor::new(&self.shape, &*self.data).with_grad(),
            w_b: Tensor::new(&b_shape, &*self.data).with_grad(),
            optim: O::new(&self.shape),
            actfn: Activation::kind(),
        }
    }
}

impl<const RANK: usize, const SIZE: usize, Activation: ActivationFn, O: Optim<RANK, SIZE>>
    BuildLayer<RANK, SIZE, 2> for FeedForward<RANK, SIZE, O, Activation>
{
    type Layer = FeedForwardLayer<RANK, SIZE, O>;

    fn build(self) -> FeedForwardLayer<RANK, SIZE, O> {
        self.build()
    }
}

/// An expanded representation of a feed-forward (implicitly transposed dense) layer.
///
/// Performs a right-hand multiplication: `output = W * input`
pub struct FeedForwardLayer<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> {
    w_a: WithGrad<Tensor<RANK, SIZE>>,
    w_b: WithGrad<Tensor<RANK, SIZE>>,
    optim: O,
    actfn: ActivationKind,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> FeedForwardLayer<RANK, SIZE, O> {
    #[inline]
    #[must_use]
    #[allow(clippy::too_many_lines, clippy::useless_conversion)]
    fn __forward<'a, const IN_SIZE: usize, const W_SIZE: usize, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
        [(); W_SIZE],
    ) {
        let a: WithGrad<Tensor<RANK, W_SIZE>> =
            input.get_value().matmul(self.w_a.get_value()).with_grad();
        let (a_act, act_grad_fn): (_, BackFn<'a, Tensor<RANK, W_SIZE>, Tensor<RANK, W_SIZE>>) =
            match self.actfn {
                ActivationKind::ReLU => {
                    let out = relu(&a).0;

                    #[cfg(feature = "no_stack")]
                    let back = move |grad_output: Tensor<RANK, W_SIZE>| {
                        let mut grad = alloc::vec![0.0; grad_output.data().len()];

                        grad.iter_mut()
                            .zip(a.get_value().data().iter())
                            .zip(grad_output.data().iter())
                            .for_each(|((g, &x), &dy)| {
                                *g = if x > 0.0 { dy } else { 0.0 };
                            });

                        unsafe {
                            VecTensor::from_vec(a.get_value().shape().try_into().unwrap(), grad)
                                .into_tensor()
                                .unwrap_unchecked()
                        }
                    };
                    #[cfg(not(feature = "no_stack"))]
                    let back = move |grad_output: Tensor<RANK, W_SIZE>| {
                        let mut grad = [0.0; W_SIZE];

                        grad.iter_mut()
                            .zip(a.get_value().data().iter())
                            .zip(grad_output.data().iter())
                            .for_each(|((g, &x), &dy)| {
                                *g = if x > 0.0 { dy } else { 0.0 };
                            });

                        Tensor::new(a.get_value().shape().try_into().unwrap(), &grad)
                    };

                    #[cfg(feature = "alloc")]
                    {
                        (out, alloc::boxed::Box::new(back))
                    }
                    #[cfg(not(feature = "alloc"))]
                    {
                        (out, box_closure::OpaqueFnOnce::new(back))
                    }
                }
                ActivationKind::GELU => {
                    let (out, phi, _) = gelu(&a);

                    #[cfg(feature = "no_stack")]
                    let back = move |grad_output: Tensor<RANK, W_SIZE>| {
                        let dy = grad_output.data();
                        let mut grad = alloc::vec![0.0; grad_output.data().len()];

                        let inv_sqrt2pi = 0.398_942_280_401_432_7; // 1 / sqrt(2π)

                        grad.iter_mut()
                            .zip(a.get_value().data().iter())
                            .zip(phi.iter())
                            .zip(dy.iter())
                            .for_each(|(((g, &x), &phi), &dyi)| {
                                let pdf = inv_sqrt2pi * libm::expf(-0.5) * x * x;
                                *g = dyi * (phi + x * pdf);
                            });

                        unsafe {
                            VecTensor::from_vec(a.get_value().shape().try_into().unwrap(), grad)
                                .into_tensor()
                                .unwrap_unchecked()
                        }
                    };
                    #[cfg(not(feature = "no_stack"))]
                    let back = move |grad_output: Tensor<RANK, W_SIZE>| {
                        let dy = grad_output.data();
                        let mut grad = [0.0; W_SIZE];

                        let inv_sqrt2pi = 0.398_942_280_401_432_7;

                        grad.iter_mut()
                            .zip(a.get_value().data().iter())
                            .zip(phi.iter())
                            .zip(dy.iter())
                            .for_each(|(((g, &x), &phi), &dyi)| {
                                let pdf = inv_sqrt2pi * libm::expf(-0.5) * x * x;
                                *g = dyi * (phi + x * pdf);
                            });

                        Tensor::new(a.get_value().shape().try_into().unwrap(), &grad)
                    };

                    #[cfg(feature = "alloc")]
                    {
                        (out, alloc::boxed::Box::new(back))
                    }
                    #[cfg(not(feature = "alloc"))]
                    {
                        (out, box_closure::OpaqueFnOnce::new(back))
                    }
                }
                ActivationKind::Swish => {
                    let (out, sig, _) = swish(&a);

                    #[cfg(feature = "no_stack")]
                    let back = move |grad_output: Tensor<RANK, W_SIZE>| {
                        let dy = grad_output.data();
                        let mut grad = alloc::vec![0.0; grad_output.data().len()];

                        grad.iter_mut()
                            .zip(a.get_value().data().iter())
                            .zip(sig.iter())
                            .zip(dy.iter())
                            .for_each(|(((g, &x), &s), &dyi)| {
                                *g = dyi * (s + x * s * (1.0 - s));
                            });

                        unsafe {
                            VecTensor::from_vec(a.get_value().shape().try_into().unwrap(), grad)
                                .into_tensor()
                                .unwrap_unchecked()
                        }
                    };
                    #[cfg(not(feature = "no_stack"))]
                    let back = move |grad_output: Tensor<RANK, W_SIZE>| {
                        let dy = grad_output.data();
                        let mut grad = [0.0; W_SIZE];

                        grad.iter_mut()
                            .zip(a.get_value().data().iter())
                            .zip(sig.iter())
                            .zip(dy.iter())
                            .for_each(|(((g, &x), &s), &dyi)| {
                                *g = dyi * (s + x * s * (1.0 - s));
                            });

                        Tensor::new(a.get_value().shape().try_into().unwrap(), &grad)
                    };

                    #[cfg(feature = "alloc")]
                    {
                        (out, alloc::boxed::Box::new(back))
                    }
                    #[cfg(not(feature = "alloc"))]
                    {
                        (out, box_closure::OpaqueFnOnce::new(back))
                    }
                }
                ActivationKind::Tanh => {
                    let out = tanh(&a).0;

                    let out_clone = out.clone();

                    #[cfg(feature = "no_stack")]
                    let back = move |grad_output: Tensor<RANK, W_SIZE>| {
                        let dy = grad_output.data();
                        let mut grad = alloc::vec![0.0; grad_output.data().len()];

                        grad.iter_mut()
                            .zip(out_clone.data().iter())
                            .zip(dy.iter())
                            .for_each(|((g, &y), &dyi)| {
                                *g = dyi * (1.0 - y * y);
                            });

                        unsafe {
                            VecTensor::from_vec(a.get_value().shape().try_into().unwrap(), grad)
                                .into_tensor()
                                .unwrap_unchecked()
                        }
                    };
                    #[cfg(not(feature = "no_stack"))]
                    let back = move |grad_output: Tensor<RANK, W_SIZE>| {
                        let dy = grad_output.data();
                        let mut grad = [0.0; W_SIZE];

                        grad.iter_mut()
                            .zip(out_clone.data().iter())
                            .zip(dy.iter())
                            .for_each(|((g, &y), &dyi)| {
                                *g = dyi * (1.0 - y * y);
                            });

                        Tensor::new(a.get_value().shape().try_into().unwrap(), &grad)
                    };

                    #[cfg(feature = "alloc")]
                    {
                        (out, alloc::boxed::Box::new(back))
                    }
                    #[cfg(not(feature = "alloc"))]
                    {
                        (out, box_closure::OpaqueFnOnce::new(back))
                    }
                }
            };
        let out = a_act.matmul(self.w_b.get_value());

        let back = move |grad_out| {
            // grad w_b
            let grad_w_b = a_act.transpose().matmul(&grad_out);

            // grad a_act
            let grad_a_act = grad_out.matmul(&self.w_b.get_value().transpose());

            // grad a through activation
            let grad_a = act_grad_fn.invoke_once(grad_a_act);

            // grad w_a
            let grad_w_a = input.get_value().transpose().matmul(&grad_a);

            // grad input
            let grad_input = grad_a.matmul(&self.w_a.get_value().transpose());

            (grad_w_a, grad_w_b, grad_input)
        };

        #[cfg(feature = "alloc")]
        {
            (
                out,
                Backward::Ternary(alloc::boxed::Box::new(back)),
                [(); W_SIZE],
            )
        }
        #[cfg(not(feature = "alloc"))]
        {
            (
                out,
                Backward::Ternary(box_closure::OpaqueFnOnce::new(back)),
                [(); W_SIZE],
            )
        }
    }

    /// Forwards the feed-forward layer.
    #[must_use]
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn forward<'a, const W_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, 0>>,
    ) -> (
        Tensor<RANK, 0>,
        Backward<'a, RANK, 0, SIZE, 0>,
        [(); W_SIZE],
    ) {
        self.__forward(input)
    }
    /// Forwards the feed-forward layer.
    #[must_use]
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn forward<'a, const IN_SIZE: usize, const W_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, IN_SIZE>,
        Backward<'a, RANK, IN_SIZE, SIZE, IN_SIZE>,
        [(); W_SIZE],
    ) {
        self.__forward(input)
    }

    #[must_use]
    #[inline]
    #[allow(clippy::unnecessary_wraps, clippy::unused_self)] // for consistent API
    fn __backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 2]) {
        // call the backward closure and return the results
        match back {
            Backward::Ternary(f) => {
                let (grad_a, grad_b, grad_in) = f.invoke_once(grad_output);

                (grad_in, [grad_a, grad_b])
            }
            _ => {
                unreachable!("FeedForward always has a binary closure");
            }
        }
    }

    /// Differentiates the `FeedForward` layer with the provided closure.
    #[must_use]
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn backward(
        &self,
        grad_output: Tensor<0, 0>,
        back: Backward<'_, RANK, 0, SIZE, 0>,
    ) -> (Tensor<RANK, 0>, [Tensor<RANK, 0>; 2]) {
        self.__backward(grad_output, back)
    }
    /// Differentiates the `FeedForward` layer with the provided closure.
    #[must_use]
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 2]) {
        self.__backward(grad_output, back)
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Layer<RANK, SIZE, 2>
    for FeedForwardLayer<RANK, SIZE, O>
{
    #[inline]
    fn optim_weights(
        &mut self,
    ) -> (
        &mut impl Optim<RANK, SIZE>,
        [&mut WithGrad<Tensor<RANK, SIZE>>; 2],
    ) {
        (&mut self.optim, [&mut self.w_a, &mut self.w_b])
    }

    #[inline]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, SIZE>>; 2] {
        [&self.w_a, &self.w_b]
    }

    #[inline]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, SIZE>>; 2] {
        [&mut self.w_a, &mut self.w_b]
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> LayerOpHeavy<RANK, SIZE, 2>
    for FeedForwardLayer<RANK, SIZE, O>
{
    #[inline]
    fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_out: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 2]) {
        self.__backward(grad_out, back)
    }

    #[inline]
    fn forward<'a, const IN_SIZE: usize, const W_SIZE: usize, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) {
        let (out, back, _) = self.__forward::<IN_SIZE, W_SIZE, OUT_SIZE>(input);
        (out, back)
    }
}

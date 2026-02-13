#![cfg_attr(feature = "dyntensor", allow(clippy::useless_conversion))]
#![allow(clippy::cast_precision_loss)]

use super::{Backward, ClosureOnce, IntoWithGrad, Layer, Tensor, TensorFloat, WithGrad};
use crate::macros::{optim::Optim, BuildLayer, LayerOpHeavy};
use alloc::boxed::Box;
use core::marker::PhantomData;
use tensor_optim::TensorOps;

const EPSILON: TensorFloat = 1e-5;

/// `LayerNorm` layer builder.
pub struct LayerNorm<const RANK: usize, const SIZE: usize, O> {
    /// Shape of layer.
    pub shape: [usize; RANK],

    /// Data of layer.
    pub data: Box<[TensorFloat; SIZE]>,

    /// Optimizer of layer.
    pub _optim: PhantomData<O>,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> LayerNorm<RANK, SIZE, O> {
    /// Count of tensors on expanded layer.
    pub const TENSORS: usize = 2;

    /// Builds operation layer.
    #[must_use]
    #[allow(clippy::explicit_auto_deref)]
    pub fn build(self) -> LayerNormLayer<RANK, SIZE, O> {
        LayerNormLayer {
            gamma: Tensor::new(&self.shape, &*self.data).with_grad(),
            beta: (Tensor::new(&self.shape, &*self.data) - 0.1).with_grad(),
            optim: O::new(&self.shape),
        }
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> BuildLayer<RANK, SIZE, 2>
    for LayerNorm<RANK, SIZE, O>
{
    type Layer = LayerNormLayer<RANK, SIZE, O>;

    fn build(self) -> Self::Layer {
        self.build()
    }
}

/// Expanded `LayerNorm` function.
pub struct LayerNormLayer<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> {
    gamma: WithGrad<Tensor<RANK, SIZE>>,
    beta: WithGrad<Tensor<RANK, SIZE>>,
    optim: O,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> LayerNormLayer<RANK, SIZE, O> {
    #[inline]
    fn __forward<'a, const IN_SIZE: usize, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
        [(); 1],
    ) {
        let shape = input.get_value().shape();
        let norm_dim = RANK - 1;
        let n = shape[norm_dim];

        debug_assert_eq!(SIZE, n);

        let mut out = Tensor::<RANK, OUT_SIZE>::zeros(shape.try_into().unwrap());

        let outer_size = IN_SIZE / n;
        for o in 0..outer_size {
            let base = o * n;

            let mut mean = 0.0;
            for i in 0..n {
                mean += input.get_value().data()[base + i];
            }
            mean /= n as TensorFloat;

            let mut var = 0.0;
            for i in 0..n {
                let d = input.get_value().data()[base + i] - mean;
                var += d * d;
            }
            var /= n as TensorFloat;

            let inv_std = libm::sqrtf(var + EPSILON).recip();

            for i in 0..n {
                let x = input.get_value().data()[base + i];
                let x_hat = (x - mean) * inv_std;
                out.data_mut()[base + i] =
                    x_hat * self.gamma.get_value().data()[i] + self.beta.get_value().data()[i];
            }
        }

        let back = move |grad_out: Tensor<RANK, OUT_SIZE>| {
            let mut grad_x = Tensor::zeros(shape.try_into().unwrap());
            let mut grad_gamma = Tensor::zeros(self.gamma.get_value().shape().try_into().unwrap());
            let mut grad_beta = Tensor::zeros(self.beta.get_value().shape().try_into().unwrap());

            let outer_size = IN_SIZE / n;

            for o in 0..outer_size {
                let base = o * n;

                let mut mean = 0.0;
                for i in 0..n {
                    mean += input.get_value().data()[base + i];
                }
                mean /= n as TensorFloat;

                let mut var = 0.0;
                for i in 0..n {
                    let d = input.get_value().data()[base + i] - mean;
                    var += d * d;
                }
                var /= n as TensorFloat;

                let inv_std = libm::sqrtf(var + EPSILON).recip();

                let mut sum_dy = 0.0;
                let mut sum_dy_xhat = 0.0;

                for i in 0..n {
                    let x_hat = (input.get_value().data()[base + i] - mean) * inv_std;
                    let dy = grad_out.data()[base + i];
                    sum_dy += dy;
                    sum_dy_xhat += dy * x_hat;

                    grad_gamma.data_mut()[i] += dy * x_hat;
                    grad_beta.data_mut()[i] += dy;
                }

                for i in 0..n {
                    let x_hat = (input.get_value().data()[base + i] - mean) * inv_std;
                    let dy = grad_out.data()[base + i];
                    let g = self.gamma.get_value().data()[i];

                    grad_x.data_mut()[base + i] = (g * inv_std / n as TensorFloat)
                        * (n as TensorFloat * dy - sum_dy - x_hat * sum_dy_xhat);
                }
            }

            (grad_gamma, grad_beta, grad_x)
        };

        #[cfg(feature = "alloc")]
        {
            (
                out,
                Backward::Ternary(alloc::boxed::Box::new(back)),
                [(); 1],
            )
        }
        #[cfg(not(feature = "alloc"))]
        {
            (
                out,
                Backward::Ternary(box_closure::OpaqueFnOnce::new(back)),
                [(); 1],
            )
        }
    }

    /// Forwards the `LayerNorm`.
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, 0>>,
    ) -> (Tensor<RANK, 0>, Backward<'a, RANK, 0, SIZE, 0>, [(); 1]) {
        self.__forward(input)
    }

    /// Forwards the `LayerNorm`.
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn forward<'a, const IN_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, IN_SIZE>,
        Backward<'a, RANK, IN_SIZE, SIZE, IN_SIZE>,
        [(); 1],
    ) {
        self.__forward(input)
    }

    #[inline]
    #[allow(clippy::unused_self)]
    fn __backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_out: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 2]) {
        match back {
            Backward::Ternary(f) => {
                let (g_gamma, g_beta, g_x) = f.invoke_once(grad_out);
                (g_x, [g_gamma, g_beta])
            }
            _ => unreachable!("LayerNorm always has a ternary closure"),
        }
    }

    /// Differentiates the `LayerNorm`.
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn backward(
        &self,
        grad_output: Tensor<RANK, 0>,
        back: Backward<'_, RANK, 0, SIZE, 0>,
    ) -> (Tensor<RANK, 0>, [Tensor<RANK, SIZE>; 2]) {
        self.__backward(grad_output, back)
    }

    /// Differentiates the `LayerNorm`.
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn backward<const IN_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, IN_SIZE>,
        back: Backward<'_, RANK, IN_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 2]) {
        self.__backward(grad_output, back)
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Layer<RANK, SIZE, 2>
    for LayerNormLayer<RANK, SIZE, O>
{
    #[inline]
    fn optim_weights(
        &mut self,
    ) -> (
        &mut impl Optim<RANK, SIZE>,
        [&mut WithGrad<Tensor<RANK, SIZE>>; 2],
    ) {
        (&mut self.optim, [&mut self.gamma, &mut self.beta])
    }

    #[inline]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, SIZE>>; 2] {
        [&self.gamma, &self.beta]
    }

    #[inline]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, SIZE>>; 2] {
        [&mut self.gamma, &mut self.beta]
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> LayerOpHeavy<RANK, SIZE, 2>
    for LayerNormLayer<RANK, SIZE, O>
{
    #[inline]
    fn forward<'a, const IN_SIZE: usize, const W_SIZE: usize, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) {
        let (out, back, _) = self.__forward(input);
        (out, back)
    }

    #[inline]
    fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_out: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 2]) {
        self.__backward(grad_out, back)
    }
}

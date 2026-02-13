use alloc::boxed::Box;

use super::{Backward, ClosureOnce, IntoWithGrad, Layer, Tensor, TensorFloat, WithGrad};
use crate::macros::optim::Optim;
use core::marker::PhantomData;

/// A representation of a bias layer builder.
///
/// Performs: `output = input + b`
pub struct Bias<const RANK: usize, const SIZE: usize, O> {
    /// Shape of the bias tensor.
    pub shape: [usize; RANK],

    /// Bias data.
    pub data: Box<[TensorFloat; SIZE]>,

    /// Optimizer of weights.
    pub _optim: PhantomData<O>,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Bias<RANK, SIZE, O> {
    /// The amount of tensors in the layer.
    pub const TENSORS: usize = 1;

    /// Expands the layer.
    #[must_use]
    #[allow(clippy::explicit_auto_deref)]
    pub fn build(self) -> BiasLayer<RANK, SIZE, O> {
        BiasLayer {
            bias: Tensor::new(&self.shape, &*self.data).with_grad(),
            optim: O::new(&self.shape),
        }
    }
}

/// Expanded bias layer.
///
/// Performs: `output = input + b`
pub struct BiasLayer<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> {
    bias: WithGrad<Tensor<RANK, SIZE>>,
    optim: O,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> BiasLayer<RANK, SIZE, O> {
    #[inline]
    fn __forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, SIZE>>,
    ) -> (
        Tensor<RANK, SIZE>,
        Backward<'a, RANK, SIZE, SIZE, SIZE>,
        [(); 1],
    ) {
        let out = input.get_value().clone() + self.bias.get_value().clone();

        let back = move |grad_out: Tensor<RANK, SIZE>| {
            // dL/dx = grad_out
            // dL/db = grad_out
            (grad_out.clone(), grad_out)
        };

        #[cfg(feature = "alloc")]
        {
            (out, Backward::Binary(alloc::boxed::Box::new(back)), [(); 1])
        }
        #[cfg(not(feature = "alloc"))]
        {
            (
                out,
                Backward::Binary(box_closure::OpaqueFnOnce::new(back)),
                [(); 1],
            )
        }
    }

    /// Forwards the bias layer (`x + b`).
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, 0>>,
    ) -> (
        Tensor<RANK, 0>,
        Backward<'a, RANK, SIZE, SIZE, SIZE>,
        [(); 1],
    ) {
        self.__forward(input)
    }

    /// Forwards the bias layer (`x + b`).
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, SIZE>>,
    ) -> (
        Tensor<RANK, SIZE>,
        Backward<'a, RANK, SIZE, SIZE, SIZE>,
        [(); 1],
    ) {
        self.__forward(input)
    }

    #[inline]
    #[allow(clippy::unused_self)]
    fn __backward(
        &self,
        grad_output: Tensor<RANK, SIZE>,
        back: Backward<'_, RANK, SIZE, SIZE, SIZE>,
    ) -> (Tensor<RANK, SIZE>, [Tensor<RANK, SIZE>; 1]) {
        match back {
            Backward::Binary(f) => {
                let (grad_in, grad_b) = f.invoke_once(grad_output);
                (grad_in, [grad_b])
            }
            _ => unreachable!("Bias always has a binary closure"),
        }
    }

    /// Differentiates the bias layer (`x`).
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn backward(
        &self,
        grad_output: Tensor<RANK, SIZE>,
        back: Backward<'_, RANK, SIZE, SIZE, SIZE>,
    ) -> (Tensor<RANK, SIZE>, [Tensor<RANK, SIZE>; 1]) {
        self.__backward(grad_output, back)
    }

    /// Differentiates the bias layer (`x`).
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn backward(
        &self,
        grad_output: Tensor<RANK, SIZE>,
        back: Backward<'_, RANK, SIZE, SIZE, SIZE>,
    ) -> (Tensor<RANK, SIZE>, [Tensor<RANK, SIZE>; 1]) {
        self.__backward(grad_output, back)
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Layer<RANK, SIZE, 1>
    for BiasLayer<RANK, SIZE, O>
{
    #[inline]
    fn optim_weights(
        &mut self,
    ) -> (
        &mut impl Optim<RANK, SIZE>,
        [&mut WithGrad<Tensor<RANK, SIZE>>; 1],
    ) {
        (&mut self.optim, [&mut self.bias])
    }

    #[inline]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, SIZE>>; 1] {
        [&self.bias]
    }

    #[inline]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, SIZE>>; 1] {
        [&mut self.bias]
    }
}

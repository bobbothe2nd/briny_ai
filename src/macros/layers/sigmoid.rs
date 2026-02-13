use alloc::boxed::Box;

use crate::macros::optim::Optim;
use core::marker::PhantomData;

use super::{sigmoid, Backward, ClosureOnce, Layer, Tensor, TensorFloat, WithGrad};

/// Defines a layer that uses sigmoid to propagate probability distributions.
pub struct Sigmoid<const RANK: usize, const SIZE: usize, O> {
    /// The shape of the layer.
    pub shape: [usize; RANK],

    /// The data of the layer.
    pub data: Box<[TensorFloat; SIZE]>,

    /// Unused: no weights.
    pub _optim: PhantomData<O>,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Sigmoid<RANK, SIZE, O> {
    /// The amount of tensors in the layer.
    pub const TENSORS: usize = 0;

    /// Builds the structure into a compute layer.
    #[must_use]
    pub fn build(self) -> SigmoidLayer<RANK, SIZE, O> {
        SigmoidLayer {
            optim: O::new(&self.shape),
        }
    }
}

/// A layer that uses sigmoid to propagate probability distributions.
pub struct SigmoidLayer<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> {
    optim: O,
}

#[cfg(feature = "dyntensor")]
impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> SigmoidLayer<RANK, SIZE, O> {
    /// Forwards the sigmoid layer.
    #[inline]
    #[must_use]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, 0>>,
    ) -> (Tensor<RANK, 0>, Backward<'a, RANK, 0, 0, 0>, [(); 1]) {
        let (out, f) = sigmoid(input);

        (out, Backward::Unary(f), [(); 1])
    }

    /// Differentiates the sigmoid layer with the provided closure.
    #[inline]
    #[must_use]
    #[allow(clippy::unnecessary_wraps)]
    pub fn backward(
        &self,
        grad_output: Tensor<RANK, 0>,
        back: Backward<'_, RANK, 0, 0, 0>,
    ) -> (Tensor<RANK, 0>, Option<Tensor<RANK, 0>>) {
        // call the backward closure and return the results
        match back {
            Backward::Unary(f) => {
                let grad_in = f.invoke_once(grad_output);
                (grad_in, None)
            }
            _ => {
                unreachable!("Sigmoid always has an unary closure");
            }
        }
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> SigmoidLayer<RANK, SIZE, O> {
    /// Forwards the sigmoid layer.
    #[inline]
    #[must_use]
    pub fn forward<'a, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, OUT_SIZE, SIZE>,
        [(); 1],
    ) {
        let (out, f) = sigmoid(input);

        (out, Backward::Unary(f), [(); 1])
    }

    /// Differentiates the sigmoid layer with the provided closure.
    #[inline]
    #[must_use]
    #[allow(clippy::unnecessary_wraps)]
    pub fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, IN_SIZE>,
        back: Backward<'_, RANK, SIZE, IN_SIZE, OUT_SIZE>,
    ) -> (Tensor<RANK, OUT_SIZE>, Option<Tensor<RANK, OUT_SIZE>>) {
        // call the backward closure and return the results
        match back {
            Backward::Unary(f) => {
                let grad_in = f.invoke_once(grad_output);
                (grad_in, None)
            }
            _ => {
                unreachable!("Sigmoid always has an unary closure");
            }
        }
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Layer<RANK, SIZE, 0>
    for SigmoidLayer<RANK, SIZE, O>
{
    #[inline]
    fn optim_weights(
        &mut self,
    ) -> (
        &mut impl Optim<RANK, SIZE>,
        [&mut WithGrad<Tensor<RANK, SIZE>>; 0],
    ) {
        (&mut self.optim, [])
    }

    #[inline]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, SIZE>>; 0] {
        []
    }

    #[inline]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, SIZE>>; 0] {
        []
    }
}

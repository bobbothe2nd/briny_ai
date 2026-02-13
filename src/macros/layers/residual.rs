use alloc::boxed::Box;

use super::{Backward, Layer, Tensor, TensorFloat, WithGrad};
use crate::macros::{optim::Optim, BuildLayer, LayerOpHeavy};
use core::marker::PhantomData;

/// A representation of a residual layer builder.
///
/// Performs: `output = input + f(input)`
pub struct Residual<const RANK: usize, const SIZE: usize, O, const N: usize, L> {
    /// Shape of the residual tensor.
    pub shape: [usize; RANK],

    /// Residual data.
    pub data: Box<[TensorFloat; SIZE]>,

    /// `f(input)` part of algorithm.
    pub layer: L,

    /// Unused: no weights.
    pub _optim: PhantomData<O>,
}

impl<
        const RANK: usize,
        const SIZE: usize,
        const N: usize,
        O: Optim<RANK, SIZE>,
        L: BuildLayer<RANK, SIZE, N>,
    > Residual<RANK, SIZE, O, N, L>
{
    /// The amount of tensors in the layer.
    pub const TENSORS: usize = N;

    /// Builds a residual operation layer.
    #[must_use]
    pub fn build(self) -> ResidualLayer<RANK, SIZE, O, N, L::Layer> {
        ResidualLayer {
            layer: self.layer.build(),
            _phantom: PhantomData,
        }
    }
}

/// Expanded residual layer.
///
/// Performs: `output = input + b`
pub struct ResidualLayer<
    const RANK: usize,
    const SIZE: usize,
    O: Optim<RANK, SIZE>,
    const N: usize,
    L,
> {
    layer: L,
    _phantom: PhantomData<O>,
}

impl<
        const RANK: usize,
        const SIZE: usize,
        O: Optim<RANK, SIZE>,
        const N: usize,
        L: LayerOpHeavy<RANK, SIZE, N>,
    > ResidualLayer<RANK, SIZE, O, N, L>
{
    #[inline]
    fn __forward<'a, const IN_SIZE: usize, const W_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, IN_SIZE>,
        Backward<'a, RANK, IN_SIZE, SIZE, IN_SIZE>,
        [(); W_SIZE],
    ) {
        let (rhs, back) = self.layer.forward::<IN_SIZE, W_SIZE, IN_SIZE>(input);
        let out = input.get_value().clone() + rhs;

        #[cfg(feature = "alloc")]
        {
            (out, back, [(); W_SIZE])
        }
        #[cfg(not(feature = "alloc"))]
        {
            (out, back, [(); W_SIZE])
        }
    }

    /// Forwards the residual (`x + f(x)`).
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

    /// Forwards the residual (`x + f(x)`).
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

    #[inline]
    fn __backward<const IN_SIZE: usize>(
        &self,
        grad_out: Tensor<RANK, IN_SIZE>,
        back: Backward<'_, RANK, IN_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; N]) {
        let (grad_f, grad_w) = self.layer.backward(grad_out.clone(), back);
        (grad_out + grad_f, grad_w)
    }

    /// Differentiates the residual (`x + f'(x)`).
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn backward(
        &self,
        grad_out: Tensor<RANK, 0>,
        back: Backward<'_, RANK, 0, SIZE, 0>,
    ) -> (Tensor<RANK, 0>, [Tensor<RANK, SIZE>; N]) {
        self.__backward(grad_out, back)
    }

    /// Differentiates the residual (`x + f'(x)`).
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn backward<const IN_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, IN_SIZE>,
        back: Backward<'_, RANK, IN_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; N]) {
        self.__backward(grad_output, back)
    }
}

impl<
        const RANK: usize,
        const SIZE: usize,
        O: Optim<RANK, SIZE>,
        const N: usize,
        L: Layer<RANK, SIZE, N>,
    > Layer<RANK, SIZE, N> for ResidualLayer<RANK, SIZE, O, N, L>
{
    #[inline]
    fn optim_weights(
        &mut self,
    ) -> (
        &mut impl Optim<RANK, SIZE>,
        [&mut WithGrad<Tensor<RANK, SIZE>>; N],
    ) {
        self.layer.optim_weights()
    }

    #[inline]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, SIZE>>; N] {
        self.layer.weights()
    }

    #[inline]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, SIZE>>; N] {
        self.layer.weights_mut()
    }
}

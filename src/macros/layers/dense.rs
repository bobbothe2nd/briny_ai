use core::marker::PhantomData;

use alloc::boxed::Box;
#[cfg(debug_assertions)]
use tensor_optim::TensorOps;

use crate::macros::{optim::Optim, BuildLayer, LayerOpHeavy};

use super::{matmul, Backward, ClosureOnce, IntoWithGrad, Layer, Tensor, TensorFloat, WithGrad};

/// A representation of a dense layer builder.
///
/// Performs a left-hand multiplication: `output = input * W`
pub struct Dense<const RANK: usize, const SIZE: usize, O> {
    /// The shape of the layer.
    pub shape: [usize; RANK],

    /// The data of the layer.
    pub data: Box<[TensorFloat; SIZE]>,

    /// Optimizer for weights.
    pub _optim: PhantomData<O>,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> BuildLayer<RANK, SIZE, 1>
    for Dense<RANK, SIZE, O>
{
    type Layer = DenseLayer<RANK, SIZE, O>;

    fn build(self) -> DenseLayer<RANK, SIZE, O> {
        self.build()
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Dense<RANK, SIZE, O> {
    /// The amount of tensors in the layer.
    pub const TENSORS: usize = 1;

    /// Expands the layer.
    #[must_use]
    #[allow(clippy::explicit_auto_deref)]
    pub fn build(self) -> DenseLayer<RANK, SIZE, O> {
        DenseLayer {
            weights: Tensor::new(&self.shape, &*self.data).with_grad(),
            optim: O::new(&self.shape),
        }
    }
}

/// An expanded representation of a dense layer.
///
/// Performs a left-hand multiplication: `output = input * W`
pub struct DenseLayer<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> {
    weights: WithGrad<Tensor<RANK, SIZE>>,
    optim: O,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> DenseLayer<RANK, SIZE, O> {
    #[inline]
    #[must_use]
    fn __forward<'a, const OUT_SIZE: usize, const IN_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
        [(); 1],
    ) {
        #[cfg(debug_assertions)]
        {
            let in_shape = input.get_value().shape();
            let w_shape = self.weights.get_value().shape();
            let k = w_shape[w_shape.len() - 2];
            let n = in_shape[in_shape.len() - 1];
            let m = w_shape[w_shape.len() - 1];
            let j = in_shape[in_shape.len() - 2];
            if RANK == 2 && OUT_SIZE != 0 {
                assert!(
                    k * n == OUT_SIZE,
                    "output size ({}) is invalid for shape ({:?})",
                    OUT_SIZE,
                    [k, n],
                );
            }
            assert!(
                m == j,
                "inner dimensions must match for matmul ({m} == {j})"
            );
        }
        let (out, back) = matmul(&self.weights, input);
        (out, Backward::Binary(back), [(); 1])
    }

    /// Forwards the activation layer.
    #[inline]
    #[must_use]
    #[cfg(feature = "dyntensor")]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, 0>>,
    ) -> (Tensor<RANK, 0>, Backward<'a, RANK, 0, SIZE, 0>, [(); 1]) {
        self.__forward(input)
    }
    /// Forwards the activation layer.
    #[must_use]
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn forward<'a, const OUT_SIZE: usize, const IN_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
        [(); 1],
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
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 1]) {
        // call the backward closure and return the results
        match back {
            Backward::Binary(f) => {
                let (grad_in, grad_w) = f.invoke_once(grad_output);

                (grad_w, [grad_in])
            }
            _ => {
                unreachable!("Dense always has a binary closure");
            }
        }
    }

    /// Differentiates the Dense layer with the provided closure.
    #[must_use]
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn backward(
        &self,
        grad_output: Tensor<0, 0>,
        back: Backward<'_, RANK, 0, SIZE, 0>,
    ) -> (Tensor<RANK, 0>, [Tensor<RANK, SIZE>; 1]) {
        self.__backward(grad_output, back)
    }
    /// Differentiates the Dense layer with the provided closure.
    #[must_use]
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 1]) {
        self.__backward(grad_output, back)
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Layer<RANK, SIZE, 1>
    for DenseLayer<RANK, SIZE, O>
{
    #[inline]
    fn optim_weights(
        &mut self,
    ) -> (
        &mut impl crate::macros::optim::Optim<RANK, SIZE>,
        [&mut WithGrad<Tensor<RANK, SIZE>>; 1],
    ) {
        (&mut self.optim, [&mut self.weights])
    }

    #[inline]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, SIZE>>; 1] {
        [&self.weights]
    }

    #[inline]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, SIZE>>; 1] {
        [&mut self.weights]
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> LayerOpHeavy<RANK, SIZE, 1>
    for DenseLayer<RANK, SIZE, O>
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
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 1]) {
        self.__backward(grad_out, back)
    }
}

use core::marker::PhantomData;

use super::{matmul, Backward, ClosureOnce, IntoWithGrad, Layer, Tensor, TensorFloat, WithGrad};
use crate::macros::{optim::Optim, BuildLayer, LayerOpHeavy};

use alloc::boxed::Box;
#[cfg(debug_assertions)]
use tensor_optim::TensorOps;

/// A representation of a collapse layer builder.
///
/// Performs a right-hand multiplication: `output = W * input`
pub struct Collapse<const RANK: usize, const SIZE: usize, O> {
    /// The shape of the layer.
    pub shape: [usize; RANK],

    /// The data of the layer.
    pub data: Box<[TensorFloat; SIZE]>,

    /// Optimizer for weights.
    pub _optim: PhantomData<O>,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Collapse<RANK, SIZE, O> {
    /// The amount of tensors in the layer.
    pub const TENSORS: usize = 1;

    /// Builds the structure into a compute layer.
    #[must_use]
    #[allow(clippy::explicit_auto_deref)]
    pub fn build(self) -> CollapseLayer<RANK, SIZE, O> {
        CollapseLayer {
            weights: Tensor::new(&self.shape, &*self.data).with_grad(),
            optim: O::new(&self.shape),
        }
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> BuildLayer<RANK, SIZE, 1>
    for Collapse<RANK, SIZE, O>
{
    type Layer = CollapseLayer<RANK, SIZE, O>;

    fn build(self) -> CollapseLayer<RANK, SIZE, O> {
        self.build()
    }
}

/// An expanded representation of a collapse (implicitly transposed dense) layer.
///
/// Performs a right-hand multiplication: `output = W * input`
pub struct CollapseLayer<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> {
    weights: WithGrad<Tensor<RANK, SIZE>>,
    optim: O,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> CollapseLayer<RANK, SIZE, O> {
    #[must_use]
    #[inline]
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
            let k = in_shape[RANK - 2];
            let n = w_shape[RANK - 1];
            let m = in_shape[RANK - 1];
            let j = w_shape[RANK - 2];
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
        let (out, back) = matmul(input, &self.weights);
        let back = |grad_out| {
            let (a, b) = back.invoke_once(grad_out);
            (b, a)
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

    /// Forwards the activation layer.
    #[must_use]
    #[inline]
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
                let (grad_w, grad_in) = f.invoke_once(grad_output);

                (grad_in, [grad_w])
            }
            _ => {
                unreachable!("Collapse always has a binary closure");
            }
        }
    }

    /// Differentiates the Collapse layer with the provided closure.
    #[must_use]
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn backward(
        &self,
        grad_output: Tensor<0, 0>,
        back: Backward<'_, RANK, 0, SIZE, 0>,
    ) -> (Tensor<RANK, 0>, [Tensor<RANK, 0>; 1]) {
        self.__backward(grad_output, back)
    }
    /// Differentiates the Collapse layer with the provided closure.
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
    for CollapseLayer<RANK, SIZE, O>
{
    #[inline]
    fn optim_weights(
        &mut self,
    ) -> (
        &mut impl Optim<RANK, SIZE>,
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
    for CollapseLayer<RANK, SIZE, O>
{
    #[inline]
    fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_out: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 1]) {
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
        let (out, back, _) = self.__forward(input);
        (out, back)
    }
}

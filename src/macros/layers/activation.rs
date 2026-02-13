use alloc::boxed::Box;

use crate::macros::optim::Optim;

use super::{
    gelu, relu, swish, tanh, ActivationFn, ActivationKind, Backward, ClosureOnce, Layer,
    PhantomData, Tensor, TensorFloat, WithGrad,
};

/// Defines a layer that propagates gradients via an activation function.
pub struct Activation<const RANK: usize, const SIZE: usize, ActivationFn, O> {
    /// The shape of the layer.
    pub shape: [usize; RANK],

    /// The data of the layer.
    pub data: Box<[TensorFloat; SIZE]>,

    /// Optimizer of weights.
    pub _optim: PhantomData<O>,

    /// The activation function used by the layer.
    pub _activation: PhantomData<ActivationFn>,
}

impl<const RANK: usize, const SIZE: usize, Activator: ActivationFn, O: Optim<RANK, SIZE>>
    Activation<RANK, SIZE, Activator, O>
{
    /// The amount of tensors in the layer.
    pub const TENSORS: usize = 0;

    /// Builds the structure into a compute layer.
    #[must_use]
    pub fn build(self) -> ActivationLayer<RANK, SIZE, O> {
        ActivationLayer {
            optim: Optim::new(&self.shape),
            actfn: Activator::kind(),
        }
    }
}

/// A layer that uses the provided activation function to propagate gradients.
pub struct ActivationLayer<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> {
    optim: O,
    actfn: ActivationKind,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> ActivationLayer<RANK, SIZE, O> {
    /// Forwards the activation layer.
    #[inline]
    #[must_use]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, SIZE>>,
    ) -> (
        Tensor<RANK, SIZE>,
        Backward<'a, RANK, SIZE, SIZE, SIZE>,
        [(); 1],
    ) {
        match self.actfn {
            ActivationKind::ReLU => {
                let (out, back) = relu(input);
                (out, Backward::Unary(back), [(); 1])
            }
            ActivationKind::GELU => {
                let (out, _, back) = gelu(input);
                (out, Backward::Unary(back), [(); 1])
            }
            ActivationKind::Swish => {
                let (out, _, back) = swish(input);
                (out, Backward::Unary(back), [(); 1])
            }
            ActivationKind::Tanh => {
                let (out, back) = tanh(input);
                (out, Backward::Unary(back), [(); 1])
            }
        }
    }

    /// Differentiates the activation layer with the provided closure.
    #[inline]
    #[must_use]
    #[allow(clippy::unnecessary_wraps)]
    pub fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, IN_SIZE>,
        back: Backward<'_, RANK, SIZE, IN_SIZE, OUT_SIZE>,
    ) -> (Tensor<RANK, OUT_SIZE>, [Tensor<RANK, OUT_SIZE>; 0]) {
        // call the backward closure and return the results
        match back {
            Backward::Unary(f) => {
                let grad_in = f.invoke_once(grad_output);
                (grad_in, [])
            }
            _ => {
                unreachable!("Activation never has a binary closure");
            }
        }
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Layer<RANK, SIZE, 0>
    for ActivationLayer<RANK, SIZE, O>
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
